from __future__ import annotations

import argparse
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from typing import Optional

import json
import datetime

from api_reference import get_api_reference_text
from failed_store import append_failed, get_past_failures_for_prompt
from rag_generator import (
    build_retriever,
    propose_new_task,
    generate_new_task_code,
)
from verifier import verify_task_code, extract_task_name_literal
from settings import load_config, resolve_path


def save_task_code(
    code: str,
    save_dir: str,
    filename: Optional[str] = None,
) -> str:
    """Save generated task code to a local directory."""
    os.makedirs(save_dir, exist_ok=True)
    task_name = extract_task_name_literal(code) or "generated_task"
    if filename is None:
        filename = f"{task_name}.py"
    path = os.path.join(save_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="两步生成 VIMA 新任务：① 提出任务名与描述 ② 生成代码并验证。"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="运行配置文件路径（默认 VIMA_Gen/config.yaml）。",
    )
    parser.add_argument(
        "--brief",
        type=str,
        default=None,
        help="可选。对第一步的提示（希望新任务的方向），不填则完全由模型根据现有任务列表提出。",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="生成候选任务数量（覆盖 config.yaml）。",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="RAG 检索时使用的文档个数（覆盖 config.yaml）。",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=None,
        help="将通过验证的任务代码保存到 save_dir（覆盖 config.yaml）。",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM 模型名称（覆盖 .env / config.yaml）。",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="生成随机性（覆盖 .env / config.yaml）。",
    )
    parser.add_argument(
        "--max-repair",
        type=int,
        default=None,
        help="验证失败后带着报错重写几次（覆盖 config.yaml；0 = 关闭）。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "运行设备：auto（默认，检测到 CUDA 就用 GPU）/ cpu / cuda / cuda:0。"
            "也可用环境变量 VIMA_DEVICE 指定。"
        ),
    )

    args = parser.parse_args()

    # ---- 配置合并：CLI 参数 > .env / 环境变量 > config.yaml > 内置默认 ----
    cfg = load_config(args.config)
    if args.model is not None:
        cfg.llm.model = args.model
    if args.temperature is not None:
        cfg.llm.temperature = args.temperature
    if args.n is not None:
        cfg.n = args.n
    if args.k is not None:
        cfg.k = args.k
    if args.save is not None:
        cfg.save = args.save
    if args.brief is not None:
        cfg.brief = args.brief
    if args.max_repair is not None:
        cfg.max_repair = args.max_repair
    if args.device is not None:
        cfg.device = args.device

    print(f"[RAG] 配置：{cfg.summary()}")

    # 解析并启用运行设备：auto 时若检测到 CUDA 则使用 GPU
    from vima_bench.tasks.utils.device import configure_device

    device = configure_device(cfg.device)
    try:
        cfg.llm.require_api_key()
    except RuntimeError as exc:
        print(f"[RAG] 配置错误：{exc}")
        raise SystemExit(2) from None

    failed_store_path = resolve_path(cfg.failed_store_path)
    save_dir = resolve_path(cfg.save_dir)
    run_results_dir = resolve_path(cfg.run_results_dir)

    print("[RAG] 构建任务检索器（内置 + 已生成任务）...")
    retriever = build_retriever(k=cfg.k, llm=cfg.llm, embedding=cfg.embedding)
    api_reference = get_api_reference_text()
    past_failures_text = get_past_failures_for_prompt(str(failed_store_path))

    results = []
    seen_names: list[str] = []
    for i in range(cfg.n):
        print(f"\n========== 候选任务 #{i + 1} ==========")

        attempt_record = {
            "index": i,
            "task_name": None,
            # verify sub-steps (Step1: syntax/struct, Step2: reset, Step3: oracle)
            "verify_step1": False,
            "verify_step2": False,
            "verify_step3": False,
            "verify_ok": False,
            "failed_step": None,
            "error_msg": None,
        }

        # Step 1: Propose task name and description
        print("[RAG] Step 1: 提出任务名与描述...")
        try:
            proposal = propose_new_task(
                retriever=retriever,
                llm=cfg.llm,
                hint_brief=cfg.brief,
                seen_names=seen_names,
            )
            # proposal succeeded; but we do not record generation rates here per request
        except Exception as e:
            print(f"[RAG] Step 1 失败：{e}")
            attempt_record["error_msg"] = str(e)
            # record and continue to next candidate
            attempt_record["task_name"] = None
            results.append(attempt_record)
            continue

        task_name = proposal["task_name"]
        group = proposal["group"]
        task_description = proposal["task_description"]
        attempt_record["task_name"] = task_name
        seen_names.append(task_name)
        print(f"[RAG] 提议: task_name={task_name}, group={group}")
        print(f"[RAG] 描述: {task_description}")

        # Step 2 + 验证（失败则带着确定性报错重写，最多 cfg.max_repair 次）
        print("[RAG] Step 2: 生成代码...")
        code = None
        ok = False
        failed_step = None
        error_msg = None
        repairs = 0
        initial_failed_step = None

        for attempt in range(cfg.max_repair + 1):
            try:
                code = generate_new_task_code(
                    task_name=task_name,
                    task_description=task_description,
                    group=group,
                    retriever=retriever,
                    api_reference=api_reference,
                    past_failures_text=past_failures_text,
                    llm=cfg.llm,
                    repair_error=error_msg if attempt > 0 else None,
                )
            except Exception as e:
                print(f"[RAG] Step 2 失败：{e}")
                attempt_record["error_msg"] = str(e)
                code = None
                break

            print(f"[RAG] 生成任务的 task_name: {extract_task_name_literal(code) or task_name}")
            if attempt == 0:
                print("\n----- 代码预览-----")
                print("\n".join(code.splitlines()))
                print("----- 预览结束 -----\n")

            ok, failed_step, error_msg = verify_task_code(code, verbose=True)
            if ok:
                break
            if initial_failed_step is None:
                initial_failed_step = failed_step
            if attempt < cfg.max_repair:
                repairs += 1
                print(
                    f"[RAG] 修复重试 #{repairs}：上一次失败于 Step {failed_step}，"
                    "带着该报错重新生成..."
                )
            else:
                break

        if code is None:
            results.append(attempt_record)
            continue

        if not ok and failed_step is not None and error_msg is not None:
            append_failed(
                code,
                failed_step,
                error_msg,
                task_name=task_name,
                path=str(failed_store_path),
            )
            print("[RAG] 已将该次失败记录到 failed_generations.json。")
        attempt_record["repairs"] = repairs
        attempt_record["initial_failed_step"] = initial_failed_step
        attempt_record["verify_ok"] = bool(ok)
        attempt_record["failed_step"] = failed_step
        attempt_record["error_msg"] = error_msg
        # derive per-verify-step pass/fail from failed_step
        if ok and failed_step is None:
            attempt_record["verify_step1"] = True
            attempt_record["verify_step2"] = True
            attempt_record["verify_step3"] = True
        else:
            if failed_step == 1:
                attempt_record["verify_step1"] = False
                attempt_record["verify_step2"] = False
                attempt_record["verify_step3"] = False
            elif failed_step == 2:
                attempt_record["verify_step1"] = True
                attempt_record["verify_step2"] = False
                attempt_record["verify_step3"] = False
            elif failed_step == 3:
                attempt_record["verify_step1"] = True
                attempt_record["verify_step2"] = True
                attempt_record["verify_step3"] = False
            else:
                # unknown failure mode -> mark all as False
                attempt_record["verify_step1"] = False
                attempt_record["verify_step2"] = False
                attempt_record["verify_step3"] = False
        print(f"[RAG] 验证结果：{'通过' if ok else '失败'}")

        if cfg.save and ok:
            path = save_task_code(code, save_dir=save_dir)
            print(f"[RAG] 已保存到：{path}")
        elif cfg.save and not ok:
            print("[RAG] 验证未通过，未保存代码。")
        else:
            print("[RAG] 未保存（如需保存请加 --save）")
        # append attempt record to results
        results.append(attempt_record)


    # Summarize results and save to JSON
    try:
        all_task_names = [r.get("task_name") for r in results]
        total = len(results)
        # verification sub-step pass counts
        step1_pass = sum(1 for r in results if r.get("verify_step1"))
        step2_pass = sum(1 for r in results if r.get("verify_step2"))
        step3_pass = sum(1 for r in results if r.get("verify_step3"))
        # pass rates relative to total attempts (not conditional)
        step1_rate = step1_pass / total if total else 0.0
        step2_rate = step2_pass / total if total else 0.0
        step3_rate = step3_pass / total if total else 0.0
        summary = {
            "timestamp": datetime.datetime.now().isoformat(),
            "device": str(device),
            "total_attempts": total,
            "verify_step1_pass_count": step1_pass,
            "verify_step2_pass_count": step2_pass,
            "verify_step3_pass_count": step3_pass,
            "verify_step1_pass_rate": step1_rate,
            "verify_step2_pass_rate": step2_rate,
            "verify_step3_pass_rate": step3_rate,
            # 修复重试统计
            "max_repair": cfg.max_repair,
            "repair_attempts": sum(int(r.get("repairs") or 0) for r in results),
            "candidates_needing_repair": sum(1 for r in results if r.get("repairs")),
            "first_try_pass_count": sum(
                1 for r in results if r.get("verify_ok") and not r.get("repairs")
            ),
            "repair_rescued_count": sum(
                1 for r in results if r.get("verify_ok") and r.get("repairs")
            ),
            "all_task_names": all_task_names,
            "passed_task_names": [r.get("task_name") for r in results if r.get("verify_ok")],
            "attempts": results,
        }
        out_dir = str(run_results_dir)
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(out_dir, f"run_{ts}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[RAG] 运行摘要已保存到: {out_path}")
        print("[RAG] 通过的任务:", summary["passed_task_names"])
    except Exception as e:
        print("[RAG] 无法保存运行摘要：", e)

if __name__ == "__main__":
    main()
