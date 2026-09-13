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
        "========== ObjPedia entries (use ObjPedia.XXX.value, ONLY these exist) ==========",
        ", ".join(obj_names),
        "",
        "========== TexturePedia entries (use TexturePedia.XXX.value, ONLY these exist) ==========",
        ", ".join(tex_names),
        "",
        "========== CRITICAL RULES — read before writing any code ==========",
        "",
        "[R1] ObjPedia.XXX / TexturePedia.XXX are ENUM MEMBERS, not entry objects.",
        "     An enum member has NO `.size_range` / `.color_value` attribute, so accessing one raises",
        "       AttributeError: 'ObjPedia' object has no attribute 'size_range'",
        "     You MUST append `.value` to get the entry object:",
        "       obj_entry     = ObjPedia.BOWL.value      # -> ObjEntry     (has .size_range)",
        "       texture_entry = TexturePedia.RED.value   # -> TextureEntry (has .color_value)",
        "     Never pass an enum member where an entry is expected:",
        "       self.add_object_to_env(env, ObjPedia.BOWL, ...)        # WRONG",
        "       self.add_object_to_env(env, ObjPedia.BOWL.value, ...)  # RIGHT",
        "       color=TexturePedia.RED                                 # WRONG",
        "       color=TexturePedia.RED.value                           # RIGHT",
        "",
        "[R2] FORBIDDEN helper methods. Despite their type hints (which claim to return",
        "     ObjEntry/TextureEntry) they actually return ENUM MEMBERS, so do NOT call them:",
        "       ObjPedia.all_entries() / ObjPedia.all_entries_no_rotational_symmetry()",
        "       TexturePedia.all_entries() / TexturePedia.all_light_dark_entries()",
        "       ObjPedia.lookup_object_by_name(...) / TexturePedia.lookup_color_by_name(...)",
        "     Instead write the allowed names explicitly (the complete lists are above), e.g.",
        "       self.possible_obj = [ObjPedia.BOWL.value, ObjPedia.BLOCK.value]",
        "",
        "[R3] Use ONLY the names listed above. Do NOT invent colour or object names.",
        "     (e.g. BROWN / GRAY / MAGENTA / BLACK do NOT exist in TexturePedia.)",
        "",
        "[R4] ResultTuple accepts ONLY the two keywords `success` and `failure`.",
        "     Never add extra keywords such as distance=/extra= (they raise TypeError).",
        "",
        "Do NOT import from vima_bench.utils.obj_pedia, vima_bench.utils.tex_pedia, or vima_bench.utils.pybullet_utils (those paths do not exist).",
        "",
    ]

    # 添加详细代码参考
    code_ref = get_code_reference_text()

    return "\n".join(basic_lines) + "\n\n" + code_ref
