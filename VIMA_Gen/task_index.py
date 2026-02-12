from __future__ import annotations

import inspect
import os
import re
import sys
from dataclasses import dataclass
from typing import List, Optional

# Ensure project root (containing vima_bench) is on sys.path when running
# scripts from inside VIMA_Gen directly.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_THIS_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from vima_bench.tasks import ALL_TASKS as _ALL_TASKS


@dataclass
class TaskDoc:
    """Light-weight representation of a task for retrieval."""

    id: str
    origin: str  # "builtin" or "generated"
    group: str
    task_name: str
    class_name: Optional[str]
    module: Optional[str]
    text: str


def _split_full_task_name(full_name: str) -> tuple[str, str]:
    """Split 'group/task_name' into (group, task_name)."""
    if "/" in full_name:
        group, task_name = full_name.split("/", 1)
    else:
        group, task_name = "unknown", full_name
    return group, task_name


def load_builtin_task_docs() -> List[TaskDoc]:
    """Collect the 17 built-in VIMA tasks as text documents."""
    docs: List[TaskDoc] = []
    for full_name, cls in _ALL_TASKS.items():
        group, task_name = _split_full_task_name(full_name)
        try:
            module = inspect.getmodule(cls)
            source = inspect.getsource(cls)
            doc = inspect.getdoc(cls) or ""
            module_name = module.__name__ if module is not None else "unknown"
        except OSError:
            # Source may be unavailable in some environments.
            source = ""
            module_name = "unknown"
            doc = inspect.getdoc(cls) or ""

        text_parts = [
            f"[BUILTIN TASK] {group}/{task_name} ({cls.__name__})",
            f"Module: {module_name}",
            "",
            "Docstring:",
            doc,
            "",
            "Source code:",
            source,
        ]
        docs.append(
            TaskDoc(
                id=f"builtin::{group}/{task_name}",
                origin="builtin",
                group=group,
                task_name=task_name,
                class_name=cls.__name__,
                module=module_name,
                text="\n".join(text_parts),
            )
        )
    return docs


def _infer_group_from_text(text: str) -> str:
    """Best-effort guess of a group name from a generated file's contents."""
    # Allow a convention at the top of generated files:
    #   # group: require_memory
    m = re.search(r"^#\s*group\s*:\s*([A-Za-z0-9_\/\-]+)", text, flags=re.MULTILINE)
    if m:
        return m.group(1).strip()
    # Fallback generic label.
    return "generated"


def _infer_task_name_from_text(path: str, text: str) -> str:
    """Try to infer task_name from code, fall back to filename stem."""
    m = re.search(r'task_name\s*=\s*["\']([^"\']+)["\']', text)
    if m:
        return m.group(1).strip()
    # Fallback to filename without extension.
    return os.path.splitext(os.path.basename(path))[0]


def _infer_class_name_from_text(text: str) -> Optional[str]:
    m = re.search(r"class\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", text)
    return m.group(1) if m else None


def load_generated_task_docs(
    base_dir: Optional[str] = None,
) -> List[TaskDoc]:
    """
    Load any user-generated task Python files under VIMA_Gen/generated_tasks.

    These files are treated as text documents for retrieval; they do NOT need
    to be registered in vima_bench.ALL_TASKS to be useful for RAG.
    """
    if base_dir is None:
        base_dir = os.path.dirname(__file__)
    gen_dir = os.path.join(base_dir, "generated_tasks")
    if not os.path.isdir(gen_dir):
        return []

    docs: List[TaskDoc] = []
    for fname in sorted(os.listdir(gen_dir)):
        if not fname.endswith(".py"):
            continue
        path = os.path.join(gen_dir, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        except OSError:
            continue

        group = _infer_group_from_text(text)
        task_name = _infer_task_name_from_text(path, text)
        class_name = _infer_class_name_from_text(text)
        docs.append(
            TaskDoc(
                id=f"generated::{fname}",
                origin="generated",
                group=group,
                task_name=task_name,
                class_name=class_name,
                module=None,
                text=text,
            )
        )
    return docs


def load_all_task_docs() -> List[TaskDoc]:
    """Return both builtin (17 original) and any saved generated tasks."""
    return load_builtin_task_docs() + load_generated_task_docs()


def get_existing_task_names_and_docs() -> List[tuple[str, str]]:
    """Return [(full_name, first_line_of_doc), ...] for all builtin tasks (for step-1 prompt)."""
    result: List[tuple[str, str]] = []
    for full_name, cls in _ALL_TASKS.items():
        doc = (inspect.getdoc(cls) or "").strip()
        first_line = doc.split("\n")[0] if doc else ""
        result.append((full_name, first_line))
    return result

