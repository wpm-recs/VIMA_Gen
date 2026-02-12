"""
Build API reference text from vima_bench for use in LLM prompts.
Only lists modules/symbols that exist so the model does not call non-existent tools.
"""
from __future__ import annotations

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_THIS_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)


def get_api_reference_text() -> str:
    """
    Build a comprehensive reference string including:
    - Allowed imports
    - ObjPedia/TexturePedia entries
    - Code reference from base.py, utils, components
    """
    import vima_bench.tasks.components.encyclopedia as enc
    from code_reference import get_code_reference_text

    obj_names = [m.name for m in enc.ObjPedia]
    tex_names = [m.name for m in enc.TexturePedia]

    # 基础 API 列表
    basic_lines = [
        "========== Allowed imports (use ONLY these, no other paths) ==========",
        "",
        "from vima_bench.tasks.task_suite.base import BaseTask",
        "from vima_bench.tasks.components.encyclopedia import ObjPedia, TexturePedia",
        "from vima_bench.tasks.components.encyclopedia.definitions import ObjEntry, TextureEntry, SizeRange",
        "from vima_bench.tasks.components.placeholders import PlaceholderObj, PlaceholderText",
        "from vima_bench.tasks.utils.pybullet_utils import (",
        "    add_any_object,",
        "    add_object_id_reverse_mapping_info,",
        "    p_change_texture,",
        "    if_in_hollow_object,",
        ")",
        "from vima_bench.tasks.utils import misc_utils as utils",
        "import numpy as np",
        "import pybullet as p",
        "",
        "========== ObjPedia entries (use ObjPedia.XXX, only these exist) ==========",
        ", ".join(obj_names),
        "",
        "========== TexturePedia entries (use TexturePedia.XXX or .lookup_color_by_name) ==========",
        ", ".join(tex_names[:30]) + (" ..." if len(tex_names) > 30 else ""),
        "",
        "Do NOT import from vima_bench.utils.obj_pedia, vima_bench.utils.tex_pedia, or vima_bench.utils.pybullet_utils (those paths do not exist).",
        "",
    ]

    # 添加详细代码参考
    code_ref = get_code_reference_text()

    return "\n".join(basic_lines) + "\n\n" + code_ref
