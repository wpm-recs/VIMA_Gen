#!/usr/bin/env python3
"""
验证新建基本任务是否可行。

用法:
  python scripts/verify_task.py <task_name>
  python scripts/verify_task.py visual_manipulation
  python scripts/verify_task.py instruction_following/visual_manipulation

可选:
  --oracle  跑几轮 oracle 并检查 success（需要显示窗口）
  --seed N  随机种子
"""

import argparse
import sys

# 确保能 import 到 vima_bench
sys.path.insert(0, ".")


def verify_task_import_and_reset(task_name: str, seed: int = 42) -> bool:
    """只做：创建环境、reset，不弹窗、不跑 oracle。"""
    import vima_bench

    print(f"[1/2] 创建环境 task_name={task_name} ...")
    env = vima_bench.make(
        task_name=task_name,
        task_kwargs=None,
        modalities=["rgb", "segm"],
        seed=seed,
        display_debug_window=False,  # 不弹窗，方便 CI/无头环境
    )
    task = env.task
    print(f"      任务类: {type(task).__name__}, task_name={task.task_name}")

    print(f"[2/2] env.reset() ...")
    env.seed(seed)
    obs = env.reset()
    prompt, prompt_assets = env.get_prompt_and_assets()
    print(f"      reset 成功. prompt 长度={len(prompt)}, assets keys={list(prompt_assets.keys())}")
    env.close()
    return True


def verify_task_with_oracle(task_name: str, seed: int = 42, num_episodes: int = 2) -> bool:
    """创建环境、reset、用 oracle 跑几步，检查是否 done 且 success。"""
    import numpy as np
    import vima_bench

    print(f"[1/3] 创建环境 (display_debug_window=True) ...")
    env = vima_bench.make(
        task_name=task_name,
        task_kwargs=None,
        modalities=["rgb", "segm"],
        seed=seed,
        display_debug_window=True,
        hide_arm_rgb=False,
    )
    task = env.task
    oracle_fn = task.oracle(env)

    print(f"[2/3] 运行 {num_episodes} 个 episode (oracle) ...")
    for ep in range(num_episodes):
        s = seed + ep
        env.seed(s)
        obs = env.reset()
        prompt, _ = env.get_prompt_and_assets()
        print(f"      Episode {ep} seed={s}, prompt: {prompt[:80]}...")

        for step in range(task.oracle_max_steps):
            action = oracle_fn.act(obs)
            if action is None:
                print(f"      Episode {ep}: oracle 返回 None at step {step}")
                env.close()
                return False
            action = {
                k: np.clip(v, env.action_space[k].low, env.action_space[k].high)
                for k, v in action.items()
            }
            obs, reward, done, info = env.step(action=action, skip_oracle=False)
            if done:
                ok = "OK" if info.get("success") else "FAIL"
                print(f"      Episode {ep}: done at step {step+1}, success={info.get('success')} [{ok}]")
                if not info.get("success"):
                    env.close()
                    return False
                break
        else:
            print(f"      Episode {ep}: 未在 oracle_max_steps={task.oracle_max_steps} 内 done")
            env.close()
            return False

    print(f"[3/3] 全部 {num_episodes} 个 episode 成功完成")
    env.close()
    return True


def main():
    parser = argparse.ArgumentParser(description="验证新建基本任务")
    parser.add_argument("task_name", type=str, help="任务名，如 visual_manipulation 或 instruction_following/visual_manipulation")
    parser.add_argument("--oracle", action="store_true", help="用 oracle 跑几轮并检查 success（会弹窗）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--episodes", type=int, default=2, help="--oracle 时跑的 episode 数")
    args = parser.parse_args()

    try:
        if args.oracle:
            ok = verify_task_with_oracle(args.task_name, seed=args.seed, num_episodes=args.episodes)
        else:
            ok = verify_task_import_and_reset(args.task_name, seed=args.seed)
    except Exception as e:
        print(f"验证失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    if not ok:
        sys.exit(1)
    print("验证通过。")


if __name__ == "__main__":
    main()
