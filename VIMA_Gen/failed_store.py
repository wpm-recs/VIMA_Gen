"""
维护未通过验证的生成代码与报错信息，供后续生成时注入提示词以规避相似错误。
"""
from __future__ import annotations

import json
import os
import re
from typing import List

# 存储文件放在 VIMA_Gen 目录下
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PATH = os.path.join(_THIS_DIR, "failed_generations.json")

# 单条代码预览最大行数；最多保留的失败条数
CODE_SNIPPET_LINES = 35
# 按「错误指纹」聚合后条数上限可以放宽：同一种错误只占 1 条
MAX_ENTRIES = 60
# 注入提示词时最多展示多少种不同的错误指纹
MAX_PROMPT_FINGERPRINTS = 12


def fingerprint(error: str) -> str:
    """把错误信息归一化成指纹，用于把同类失败聚合到一起。

    去掉引号内的具体名字与数字，使
    ``ResultTuple.__new__() got an unexpected keyword argument 'distance'`` 与
    ``... argument 'extra'`` 归为同一条。
    """
    text = (error or "").strip().splitlines()[0] if error else ""
    text = re.sub(r"'[^']*'", "'X'", text)
    text = re.sub(r'"[^"]*"', "'X'", text)
    text = re.sub(r"\b\d+\b", "N", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:160] or "unknown"


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

    fp = fingerprint(error_message)
    entry = {
        "task_name": task_name or "",
        "failed_step": failed_step,
        "error": error_message.strip(),
        "code_snippet": snippet,
        "fingerprint": fp,
        "count": 1,
    }
    data = _load_raw(path)

    # 同类错误聚合：同指纹只保留一条，累加 count 并刷新为最新样本
    for i, existing in enumerate(data):
        same = existing.get("fingerprint") == fp or (
            not existing.get("fingerprint")
            and fingerprint(existing.get("error", "")) == fp
        )
        if same:
            entry["count"] = int(existing.get("count", 1)) + 1
            data.pop(i)
            break

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

    # 按「出现次数优先、其次失败步骤」排序，只注入最典型的高频错误
    ranked = sorted(
        data,
        key=lambda e: (int(e.get("count", 1)), e.get("failed_step") or 0),
        reverse=True,
    )[:MAX_PROMPT_FINGERPRINTS]

    parts = [
        "========== Past failures (avoid similar mistakes) ==========",
        "The following generated code failed verification. Do NOT repeat these errors.",
        "They are ranked by how often they occurred. Fix the ROOT CAUSE, not the symptom.",
        "",
    ]
    for i, entry in enumerate(ranked, 1):
        step = entry.get("failed_step", 0)
        err = entry.get("error", "")
        snippet = entry.get("code_snippet", "")
        name = entry.get("task_name", "")
        count = int(entry.get("count", 1))
        parts.append(
            f"--- Failure #{i} (occurred {count}x, task_name={name}, failed at Step {step}) ---"
        )
        parts.append(f"Error: {err[:500]}" + ("..." if len(err) > 500 else ""))
        parts.append("Code snippet:")
        parts.append(snippet)
        parts.append("")

    parts.append("========== End of past failures ==========")
    return "\n".join(parts)
