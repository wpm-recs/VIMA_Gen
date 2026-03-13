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
from PIL import Image
import datetime

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
    def _save_debug_data(obs_obj, hmap_obj, obj_mask_obj, env_obj, task_obj, tag: str):
        """Save debug artifacts to disk: rgb per view (PNG), hmap.npy, obj_mask.npy, and a small metadata txt.

        Files are saved under <project_root>/verifier_debug/<task_name>_<timestamp>_<tag>/
        """
        try:
            task_name_safe = getattr(task_obj, "task_name", task_obj.__class__.__name__)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            base = os.path.join(_ROOT_DIR, "verifier_debug", f"{task_name_safe}_{ts}_{tag}")
            os.makedirs(base, exist_ok=True)

            # Save hmap and obj_mask as numpy
            if hmap_obj is not None:
                np.save(os.path.join(base, "hmap.npy"), hmap_obj)
            if obj_mask_obj is not None:
                np.save(os.path.join(base, "obj_mask.npy"), obj_mask_obj)

            # Save rgb frames if present in obs
            try:
                if obs_obj is not None and "rgb" in obs_obj:
                    rgb_dict = obs_obj["rgb"]
                    # choose first view/frame available and save
                    for view, arr in rgb_dict.items():
                        a = np.asarray(arr)
                        # handle (C,H,W) -> (H,W,C)
                        if a.ndim == 3 and a.shape[0] == 3:
                            a = np.transpose(a, (1, 2, 0))
                        # If there is a time dimension (T, C, H, W) or (T, H, W, C), take first
                        if a.ndim == 4:
                            a = a[0]
                        if a.dtype != np.uint8:
                            a = np.clip(a, 0, 255).astype(np.uint8)
                        Image.fromarray(a).save(os.path.join(base, f"rgb_{view}.png"))
            except Exception:
                # non-fatal
                pass

            # Save a small metadata text
            try:
                meta_path = os.path.join(base, "meta.txt")
                with open(meta_path, "w") as mf:
                    mf.write(f"task: {task_name_safe}\n")
                    mf.write(f"goals: {getattr(task_obj, 'goals', None)}\n")
                    mf.write(f"obj_id_reverse_mapping keys: {list(getattr(env_obj, 'obj_id_reverse_mapping', {}).keys())}\n")
            except Exception:
                pass
            if verbose:
                print(f"[VERIFY][DEBUG] Saved debug artifacts to: {base}")
        except Exception as _e:
            if verbose:
                print("[VERIFY][DEBUG] Failed to save debug data:", _e)
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

        # task.oracle_max_steps 步内循环调用 oracle_fn.act(obs)，遇到 None 视为 oracle 失败
        #（返回失败），否则 clip 动作并执行 env.step，直到 done 或用尽步数。
        obs_curr = obs
        for _ in range(getattr(task, "oracle_max_steps", 10)):
            # 可选调试：在调用 oracle 前查看真值图像/掩码，帮助定位不可见问题
            try:
                _, hmap, obj_mask = task.get_true_image(env)
                if verbose:
                    print("[DEBUG] goals:", task.goals)
                    print("[DEBUG] obj_id_reverse_mapping keys:", list(env.obj_id_reverse_mapping.keys()))
                    print("[DEBUG] obj_mask unique ids:", np.unique(obj_mask)[:20])
                    print("[DEBUG] obj_mask nonzero count:", np.count_nonzero(obj_mask))
            except Exception as _e:
                if verbose:
                    print("[DEBUG] get_true_image() failed:", _e)

            oracle_action = oracle_fn.act(obs_curr)
            if oracle_action is None:
                err_msg = "oracle 返回 None，无法继续。"
                if verbose:
                    print("[VERIFY][Step 3] Oracle 失败：", err_msg)
                # save debug artifacts: ensure we have latest hmap/obj_mask/obs
                try:
                    _hmap = locals().get("hmap", None)
                    _obj_mask = locals().get("obj_mask", None)
                    if _hmap is None or _obj_mask is None:
                        try:
                            _, _hmap, _obj_mask = task.get_true_image(env)
                        except Exception:
                            _hmap, _obj_mask = None, None
                except Exception:
                    _hmap, _obj_mask = None, None
                _save_debug_data(obs_curr, _hmap, _obj_mask, env, task, tag="oracle_none")
                env.close()
                return False, 3, err_msg

            # clip 并执行一步环境
            oracle_action = {
                k: np.clip(v, env.action_space[k].low, env.action_space[k].high)
                for k, v in oracle_action.items()
            }
            obs_curr, _, done, info = env.step(action=oracle_action, skip_oracle=False)
            if done:
                success = bool(info.get("success"))
                break

        if not success:
            err_msg = f"在 {getattr(task, 'oracle_max_steps', 10)} 步内未成功完成任务。 info={info}"
            if verbose:
                print("[VERIFY][Step 3] Oracle 检查失败：" + err_msg)
            # save debug artifacts on overall failure as well
            try:
                try:
                    _, _hmap, _obj_mask = task.get_true_image(env)
                except Exception:
                    _hmap, _obj_mask = None, None
                _save_debug_data(obs_curr, _hmap, _obj_mask, env, task, tag="oracle_not_success")
            except Exception:
                pass
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
