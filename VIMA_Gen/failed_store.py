"""
维护未通过验证的生成代码与报错信息，供后续生成时注入提示词以规避相似错误。
"""
from __future__ import annotations

import json
import os
from typing import List

# 存储文件放在 VIMA_Gen 目录下
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PATH = os.path.join(_THIS_DIR, "failed_generations.json")

# 单条代码预览最大行数；最多保留的失败条数
CODE_SNIPPET_LINES = 35
MAX_ENTRIES = 25


def _load_raw(path: str) -> List[dict]:
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []
    return data if isinstance(data, list) else []


def append_failed(
    code: str,
    failed_step: int,
    error_message: str,
    task_name: str = "",
    path: str = DEFAULT_PATH,
) -> None:
    """记录一次未通过的生成：代码片段、失败步骤、报错信息、任务名。"""
    lines = code.strip().splitlines()
    snippet = "\n".join(lines[:CODE_SNIPPET_LINES])
    if len(lines) > CODE_SNIPPET_LINES:
        snippet += "\n# ... (truncated)"

    entry = {
        "task_name": task_name or "",
        "failed_step": failed_step,
        "error": error_message.strip(),
        "code_snippet": snippet,
    }
    data = _load_raw(path)
    data.append(entry)
    if len(data) > MAX_ENTRIES:
        data = data[-MAX_ENTRIES:]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_past_failures_for_prompt(path: str = DEFAULT_PATH) -> str:
    """
    读取历史失败记录并格式化为可注入提示词的一段文本。
    若没有记录则返回空字符串。
    """
    data = _load_raw(path)
    if not data:
        return ""

    parts = [
        "========== Past failures (avoid similar mistakes) ==========",
        "The following generated code failed verification. Do NOT repeat these errors.",
        "",
    ]
    for i, entry in enumerate(data[-15:], 1):  # 最多注入最近 15 条
        step = entry.get("failed_step", 0)
        err = entry.get("error", "")
        snippet = entry.get("code_snippet", "")
        name = entry.get("task_name", "")
        parts.append(f"--- Failure #{i} (task_name={name}, failed at Step {step}) ---")
        parts.append(f"Error: {err[:500]}" + ("..." if len(err) > 500 else ""))
        parts.append("Code snippet:")
        parts.append(snippet)
        parts.append("")

    parts.append("========== End of past failures ==========")
    return "\n".join(parts)
