"""
Verify generated task code in an environment that mirrors vima_bench/tasks:
- Only project root is on sys.path , so code must use correct imports
  (e.g. from vima_bench.tasks.task_suite.base import BaseTask).
"""
from __future__ import annotations

import os
import re
import sys
from typing import Optional, Type

import numpy as np

# 模拟“放在 tasks 文件夹中”的环境：保证项目根在 sys.path，使
# from vima_bench.tasks... 能正确解析，不注入任何 shim 模块。
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_THIS_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from vima_bench.env import VIMAEnvBase
from vima_bench.tasks.task_suite.base import BaseTask


def _extract_class_name(code: str) -> str:
    m = re.search(r"class\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", code)
    if not m:
        raise ValueError("无法从生成代码中找到 class 定义。")
    return m.group(1)


def extract_task_name_literal(code: str) -> Optional[str]:
    m = re.search(r'task_name\s*=\s*["\']([^"\']+)["\']', code)
    return m.group(1) if m else None


def _structural_checks(code: str) -> tuple[bool, str]:
    """快速静态检查，尽早过滤明显结构错误。"""
    # 不允许自定义 oracle，避免破坏 BaseTask.oracle 的逻辑
    if "def oracle(" in code:
        return False, "Should not override oracle(); inherit BaseTask.oracle instead."
    # 必须有 goals scaffold
    if "self.goals.append(" not in code:
        return False, "No self.goals.append(...) found in reset(); oracle has no goals to follow."
    if "self._all_goals = self.goals.copy()" not in code:
        return False, "Missing self._all_goals = self.goals.copy() after setting goals."
    return True, ""


def load_task_class_from_code(code: str) -> Type[BaseTask]:
    """
    在“模拟 tasks 文件夹”的环境中 exec 代码并返回任务类。
    """
    local_ns: dict = {"__builtins__": __builtins__}
    exec(code, local_ns, local_ns)

    cls_name = _extract_class_name(code)
    TaskCls = local_ns.get(cls_name)
    if TaskCls is None:
        raise ValueError(f"在 exec 环境中找不到类 {cls_name}")
    if not issubclass(TaskCls, BaseTask):
        raise TypeError(f"{cls_name} 没有继承 BaseTask")
    return TaskCls


def verify_task_code(code: str, verbose: bool = True) -> tuple[bool, Optional[int], Optional[str]]:
    """
    三步验证：
    1. 语法/导入：exec 成功且得到继承 BaseTask 的类
    2. 运行时：构造实例 + env.reset() 成功
    3. Oracle：在 oracle_max_steps 内能完成且 info['success'] 为 True

    Returns:
        (success, failed_step, error_message)
        - success=True 时 failed_step 与 error_message 为 None
        - success=False 时 failed_step 为 1/2/3，error_message 为报错内容
    """
    # ---------- Step 1: 语法 / exec + 结构检查 ----------
    ok_struct, struct_msg = _structural_checks(code)
    if not ok_struct:
        if verbose:
            print(f"[VERIFY][Step 1] 结构检查失败：{struct_msg}")
        return False, 1, struct_msg
    try:
        TaskCls = load_task_class_from_code(code)
        if verbose:
            print("[VERIFY][Step 1] 语法 / exec 检查通过。")
    except Exception as e:
        err_msg = str(e)
        if verbose:
            print(f"[VERIFY][Step 1] 语法 / exec 检查失败：{err_msg}")
        return False, 1, err_msg

    # ---------- Step 2: 运行时 reset ----------
    env = None
    try:
        task_instance = TaskCls(debug=False)
        env = VIMAEnvBase(
            task=task_instance,
            modalities=["rgb", "segm"],
            seed=42,
            debug=False,
            display_debug_window=False,
            hide_arm_rgb=True,
        )
        obs = env.reset()
        task = env.task
        # 额外结构检查：reset 后 goals 必须非空，_all_goals 已初始化
        if not getattr(task, "goals", None):
            raise RuntimeError("reset() 后 self.goals 为空，oracle 无法工作。")
        if getattr(task, "_all_goals", None) is None or len(task._all_goals) == 0:
            raise RuntimeError("reset() 后 self._all_goals 未正确初始化。")
        prompt, assets = env.prompt, env.prompt_assets
        if verbose:
            print("[VERIFY][Step 2] 运行时 reset 检查通过。")
            print(f"[VERIFY]  prompt 预览：{prompt[:80]!r}")
            print(f"[VERIFY]  prompt_assets keys: {list(assets.keys())}")
    except Exception as e:
        err_msg = str(e)
        if verbose:
            print(f"[VERIFY][Step 2] 运行时（构造实例 / env.reset）失败：{err_msg}")
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        return False, 2, err_msg

    # ---------- Step 3: Oracle 完成度 ----------
    try:
        task = env.task
        oracle_fn = task.oracle(env)
        if oracle_fn is None:
            raise RuntimeError(
                f"task.oracle(env) 返回了 None！"
                f"检查：goals={task.goals}, _all_goals={getattr(task, '_all_goals', 'NOT SET')}"
            )
        success = False
        info = {}

        for step in range(getattr(task, "oracle_max_steps", 10)):
            # DEBUG: inspect environment / masks before calling oracle
            try:
                _, hmap, obj_mask = task.get_true_image(env)
                print("[DEBUG] goals:", task.goals)
                print("[DEBUG] obj_id_reverse_mapping keys:", list(env.obj_id_reverse_mapping.keys()))
                print("[DEBUG] obj_mask unique ids:", np.unique(obj_mask)[:20])
                print("[DEBUG] obj_mask nonzero count:", np.count_nonzero(obj_mask))
            except Exception as _e:
                print("[DEBUG] get_true_image() failed:", _e)
            
            action = oracle_fn.act(obs)
            if action is None:
                raise RuntimeError("oracle 返回 None，无法继续。")
            action = {
                k: np.clip(v, env.action_space[k].low, env.action_space[k].high)
                for k, v in action.items()
            }
            obs, reward, done, info = env.step(action=action, skip_oracle=False)
            if done:
                success = bool(info.get("success"))
                break

        if not success:
            err_msg = f"在 {getattr(task, 'oracle_max_steps', 10)} 步内未成功完成任务。 info={info}"
            if verbose:
                print("[VERIFY][Step 3] Oracle 检查失败：" + err_msg)
            env.close()
            return False, 3, err_msg

        if verbose:
            print("[VERIFY][Step 3] Oracle 检查通过，任务可被 oracle 完成。")
        env.close()
        return True, None, None

    except Exception as e:
        err_msg = str(e)
        if verbose:
            print(f"[VERIFY][Step 3] Oracle 运行中出错：{err_msg}")
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        return False, 3, err_msg
