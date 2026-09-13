#!/usr/bin/env python3
"""
汇总 VIMA_Gen 的运行质量指标（三步通过率 + 失败归因），输出 Markdown。

数据来源：
  * ``VIMA_Gen/run_results/run_*.json`` —— 每轮验证摘要
  * ``VIMA_Gen/failed_generations.json`` —— 失败样本池

用法::

  python scripts/run_metrics.py                          # 打印 Markdown
  python scripts/run_metrics.py --json                   # 打印机器可读 JSON
  python scripts/run_metrics.py --write doc/metrics.md   # 写入文件（追加生成段落）

注意：``run_results/`` 中的文件是**累积**的，多次运行会叠加。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter, OrderedDict
from typing import Any, Dict, List, Optional, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_RESULTS_DIR = os.path.join(_ROOT, "VIMA_Gen", "run_results")
FAILED_STORE = os.path.join(_ROOT, "VIMA_Gen", "failed_generations.json")


# ---------------------------------------------------------------------------
# 统计工具
# ---------------------------------------------------------------------------


def wilson_ci(successes: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson 区间（小样本下比正态近似更可靠）。total=0 时返回 (0, 0)。"""
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


# ---------------------------------------------------------------------------
# 失败归因
# ---------------------------------------------------------------------------

# 顺序敏感：先匹配更具体的模式。
FAILURE_RULES: "OrderedDict[str, List[str]]" = OrderedDict(
    [
        (
            "百科条目误用",
            [
                r"Cannot find provided color",
                r"^[A-Z][A-Z0-9_]*$",  # Enum 缺成员 -> AttributeError("<NAME>")
                # 把枚举类当成成员用：ObjPedia.size_range / TexturePedia.color_value
                r"'?(ObjPedia|TexturePedia|ProfilePedia)'? object has no attribute",
                r"not a valid",
            ],
        ),
        (
            "幻觉 API 签名",
            [
                r"unexpected keyword argument",
                r"unexpected argument",
                r"takes \d+ positional argument",
                r"missing \d+ required positional",
            ],
        ),
        (
            "None 传播",
            [
                r"NoneType",
                r"Failed to sample",
                r"cannot unpack non-iterable NoneType",
            ],
        ),
        (
            "结构不合规(Step1)",
            [
                r"Should not override oracle",
                r"No self\.goals\.append",
                r"Missing self\._all_goals",
                r"无法从生成代码中找到 class 定义",
                r"没有继承 BaseTask",
            ],
        ),
        (
            "Oracle 不可解(Step3)",
            [r"步内未成功完成任务", r"oracle 返回 None"],
        ),
        (
            "代码缺陷",
            [
                r"not defined",
                r"has no attribute",
                r"is not subscriptable",
                r"KeyError",
                r"IndexError",
            ],
        ),
    ]
)


def classify(error: Optional[str], failed_step: Optional[int] = None) -> str:
    """把一条错误信息归入类别。"""
    text = (error or "").strip()
    if not text:
        return "未知" if failed_step else "未知"
    for category, patterns in FAILURE_RULES.items():
        for pat in patterns:
            if re.search(pat, text, flags=re.MULTILINE):
                return category
    return "其他"


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------


def load_runs(path: str = RUN_RESULTS_DIR) -> List[Dict[str, Any]]:
    """按时间戳排序加载所有 run_*.json。"""
    if not os.path.isdir(path):
        return []
    runs: List[Dict[str, Any]] = []
    for name in sorted(os.listdir(path)):
        if not (name.startswith("run_") and name.endswith(".json")):
            continue
        full = os.path.join(path, name)
        try:
            with open(full, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        data["_file"] = name
        runs.append(data)
    runs.sort(key=lambda r: r.get("timestamp") or "")
    return runs


def load_failed(path: str = FAILED_STORE) -> List[Dict[str, Any]]:
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return []
    return data if isinstance(data, list) else []


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------


def aggregate(runs: List[Dict[str, Any]], failed: List[Dict[str, Any]]) -> Dict[str, Any]:
    attempts: List[Dict[str, Any]] = []
    for run in runs:
        for a in run.get("attempts", []) or []:
            attempts.append(a)

    total = len(attempts)
    result: Dict[str, Any] = {
        "n_rounds": len(runs),
        "n_attempts": total,
        "first_timestamp": runs[0].get("timestamp") if runs else None,
        "last_timestamp": runs[-1].get("timestamp") if runs else None,
        "devices": sorted({r.get("device") for r in runs if r.get("device")}),
        "steps": {},
        "per_round": [],
        "failure_categories": {},
        "distinct_tasks": sorted({a.get("task_name") for a in attempts if a.get("task_name")}),
    }

    for step in (1, 2, 3):
        key = f"verify_step{step}"
        ok = sum(1 for a in attempts if a.get(key))
        lo, hi = wilson_ci(ok, total)
        result["steps"][key] = {
            "pass": ok,
            "total": total,
            "rate": (ok / total) if total else 0.0,
            "ci95": [lo, hi],
        }

    for run in runs:
        atts = run.get("attempts", []) or []
        n = len(atts)
        result["per_round"].append(
            {
                "file": run.get("_file"),
                "timestamp": run.get("timestamp"),
                "device": run.get("device"),
                "n": n,
                "step1": sum(1 for a in atts if a.get("verify_step1")),
                "step2": sum(1 for a in atts if a.get("verify_step2")),
                "step3": sum(1 for a in atts if a.get("verify_step3")),
                "task_names": [a.get("task_name") for a in atts],
            }
        )

    cats = Counter()
    for a in attempts:
        if a.get("verify_ok"):
            continue
        cats[classify(a.get("error_msg"), a.get("failed_step"))] += 1
    # 失败池中尚未出现在 run_results 的历史样本也计入（单独标注）
    hist_cats = Counter()
    for e in failed:
        hist_cats[classify(e.get("error"), e.get("failed_step"))] += 1

    result["failure_categories"] = dict(cats.most_common())
    result["failure_categories_recorded_pool"] = dict(hist_cats.most_common())
    result["failed_pool_size"] = len(failed)
    return result


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------


def render_markdown(result: Dict[str, Any]) -> str:
    lines: List[str] = []
    a = lines.append

    a("## 运行指标（自动生成）")
    a("")
    a(f"- 轮次：**{result['n_rounds']}**，候选总数：**{result['n_attempts']}**")
    a(f"- 时间范围：`{result['first_timestamp']}` → `{result['last_timestamp']}`")
    a(f"- 设备：{', '.join(str(d) for d in result['devices']) or 'n/a'}")
    a("")
    a("### 三步通过率")
    a("")
    a("| 验证步骤 | 通过 / 总数 | 通过率 | 95% Wilson CI |")
    a("|---|---|---|---|")
    labels = {
        "verify_step1": "Step 1 语法 / 结构",
        "verify_step2": "Step 2 运行时 reset",
        "verify_step3": "Step 3 Oracle 可解",
    }
    for key, label in labels.items():
        s = result["steps"][key]
        lo, hi = s["ci95"]
        a(
            f"| {label} | {s['pass']} / {s['total']} | **{pct(s['rate'])}** | "
            f"[{pct(lo)}, {pct(hi)}] |"
        )
    a("")
    a("### 逐轮明细")
    a("")
    a("| # | 轮次文件 | 候选 | Step1 | Step2 | Step3 | 提议的任务名 |")
    a("|---|---|---|---|---|---|---|")
    for i, r in enumerate(result["per_round"], 1):
        names = ", ".join(str(n) for n in r["task_names"])
        a(
            f"| {i} | `{r['file']}` | {r['n']} | {r['step1']}/{r['n']} | "
            f"{r['step2']}/{r['n']} | {r['step3']}/{r['n']} | {names} |"
        )
    a("")
    a("### 失败归因（本轮）")
    a("")
    if result["failure_categories"]:
        a("| 类别 | 次数 |")
        a("|---|---|")
        for k, v in result["failure_categories"].items():
            a(f"| {k} | {v} |")
    else:
        a("_本轮没有失败候选。_")
    a("")
    a(f"（历史失败池 `failed_generations.json` 共 {result['failed_pool_size']} 条，累计归因：")
    for k, v in result.get("failure_categories_recorded_pool", {}).items():
        a(f"- {k}: {v}")
    a("）")
    a("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="汇总 VIMA_Gen 运行质量指标")
    parser.add_argument("--run-results", default=RUN_RESULTS_DIR, help="run_*.json 所在目录")
    parser.add_argument("--failed-store", default=FAILED_STORE, help="failed_generations.json 路径")
    parser.add_argument("--json", action="store_true", help="输出 JSON 而不是 Markdown")
    parser.add_argument("--write", default=None, help="把 Markdown 追加写入指定文件")
    args = parser.parse_args()

    runs = load_runs(args.run_results)
    failed = load_failed(args.failed_store)
    result = aggregate(runs, failed)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    md = render_markdown(result)
    if args.write:
        os.makedirs(os.path.dirname(os.path.abspath(args.write)), exist_ok=True)
        with open(args.write, "a", encoding="utf-8") as fh:
            fh.write(md)
        print(f"已写入 {args.write}")
    else:
        print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
