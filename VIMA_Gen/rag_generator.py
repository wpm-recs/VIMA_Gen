"""
Two-step RAG generation: (1) propose task name + description, (2) generate code.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

from task_index import load_all_task_docs, get_existing_task_names_and_docs, TaskDoc


# ---------- Step 1: Propose task name and description ----------

PROMPT_STEP1_SYSTEM = """You are an expert in the VIMA-Bench task suite.

Given the list of EXISTING task names and their short descriptions below, propose ONE new task that:
- Has a unique task_name (snake_case, not in the existing list).
- Has a clear one- or two-sentence task_description (what the agent must do).
- Fits one of the existing groups: instruction_following, constraint_satisfaction, novel_concept_grounding, one_shot_imitation, rearrangement, require_memory, require_reasoning.

Output format (use exactly this structure, no extra text):
TASK_NAME: <snake_case_name>
GROUP: <one of the groups above>
TASK_DESCRIPTION: <one or two sentences>
"""


def propose_new_task(
    retriever,
    model_name: str = "gpt-4.1-mini",
    temperature: float = 0.7,
    hint_brief: str | None = None,
) -> Dict[str, str]:
    """
    Step 1: Propose a new task name and description from existing task list.
    Returns dict with keys: task_name, group, task_description.
    """
    existing = get_existing_task_names_and_docs()
    existing_text = "\n".join(
        [f"- {full_name}: {doc or '(no doc)'}" for full_name, doc in existing]
    )

    user_content = f"""Existing VIMA-Bench tasks (do not duplicate these names):
{existing_text}
"""
    if hint_brief:
        user_content += f"\nUser hint for the new task: {hint_brief}\n"
    user_content += "\nPropose one new task (TASK_NAME, GROUP, TASK_DESCRIPTION):"

    llm = ChatOpenAI(model=model_name, temperature=temperature)
    resp = llm.invoke(
        [{"role": "system", "content": PROMPT_STEP1_SYSTEM}, {"role": "user", "content": user_content}]
    )
    content = resp.content if isinstance(resp.content, str) else str(resp.content)

    # Parse TASK_NAME:, GROUP:, TASK_DESCRIPTION:
    task_name = ""
    group = "instruction_following"
    task_description = ""
    for line in content.strip().split("\n"):
        line = line.strip()
        if line.upper().startswith("TASK_NAME:"):
            task_name = line.split(":", 1)[1].strip()
        elif line.upper().startswith("GROUP:"):
            group = line.split(":", 1)[1].strip()
        elif line.upper().startswith("TASK_DESCRIPTION:"):
            task_description = line.split(":", 1)[1].strip()

    if not task_name:
        task_name = "generated_task"
    if not task_description:
        task_description = content[:200]

    return {"task_name": task_name, "group": group, "task_description": task_description}


# ---------- Step 2: Generate code (with API reference) ----------

def _system_prompt_step2(api_reference: str) -> str:
    skeleton = """
```python
class TEMPLATE_Task(BaseTask):
    task_name = "<TO_FILL_TASK_NAME>"

    def __init__(self, *args, **kwargs):
        # TODO: fill prompt_template, task_meta, placeholder_expression, oracle_max_steps
        prompt_template = "..."
        task_meta = {...}
        placeholder_expression = {...}
        oracle_max_steps = 3
        super().__init__(
            prompt_template=prompt_template,
            task_meta=task_meta,
            placeholder_expression=placeholder_expression,
            oracle_max_steps=oracle_max_steps,
            *args,
            **kwargs,
        )

    def reset(self, env):
        super().reset(env)
        # TODO: sample objects/textures using ObjPedia / TexturePedia
        # Example scaffold:
        #   obj_entry = ObjPedia.BLOCK.value
        #   tex_entry = TexturePedia.RED.value
        #   size = self.get_random_size(obj_entry.size_range)
        #   obj_id, urdf_path, pose = self.add_object_to_env(env, obj_entry, tex_entry, size, category="rigid")
        #   env.obj_ids["rigid"].append(obj_id)
        #   self.placeholders["dragged_obj"] = PlaceholderObj(
        #       name=obj_entry.name,
        #       obj_id=obj_id,
        #       urdf=urdf_path,
        #       alias=obj_entry.alias,
        #       novel_name=obj_entry.novel_name,
        #       color=tex_entry,
        #       image_size=self._placeholder_img_size,
        #       seed=self.seed,
        #   )
        #
        # REQUIRED goals scaffold (do NOT change structure, only fill ... parts):
        #   target_pose = (target_pos, target_quat)
        #   self.goals.append((
        #       [(obj_id, (obj_entry.symmetry, None))],     # objs
        #       np.ones((1, 1)),                            # matches
        #       [target_pose],                              # targs
        #       False,                                      # replace
        #       True,                                       # rotations
        #       "pose",                                     # metric
        #       None,                                       # params
        #       1.0,                                        # max_progress
        #   ))
        #   self._all_goals = self.goals.copy()

    def check_success(self, *args, **kwargs) -> ResultTuple:
        # TODO: implement using self.goals and env state
        ...
```
"""
    return f"""You are an expert in the VIMA-Bench codebase.

You MUST use ONLY the following imports and tools. Do NOT use any other module path (e.g. do NOT use vima_bench.utils.obj_pedia or vima_bench.utils.pybullet_utils).

{api_reference}

You are also given example task implementations from VIMA-Bench, and the following canonical skeleton that you MUST follow closely (only fill TODO sections, do not remove required lines like self._all_goals = self.goals.copy()):

{skeleton}

Goal:
- Implement a NEW task class that matches the given task name and description.
- Subclass BaseTask (or a specialized base like RotateTheObjBase, SweepObjectsToZoneBase if appropriate).
- Define class attribute task_name = "<task_name>" (snake_case string).
- Implement __init__(...) calling super().__init__(prompt_template=..., task_meta=..., placeholder_expression=..., oracle_max_steps=...).
- Implement reset(self, env): spawn objects via self.add_object_to_env(env, obj_entry, color, size, ...), append obj_id to env.obj_ids['rigid'], create PlaceholderObj for prompt, append a goal tuple EXACTLY following the scaffold, and call self._all_goals = self.goals.copy().
- Implement check_success(self, ...) returning a NamedTuple(success: bool, failure: bool, ...).
- Do NOT override oracle(); always inherit BaseTask.oracle.
- Use only ObjPedia.XXX and TexturePedia.XXX entries listed above; do not invent new ones.

Constraints:
- Output ONLY a Python code block (```python ... ```) containing the class definition.
- No explanations outside the code block.
- First line of the code can be a comment: # group: <group_name>
"""


def _build_documents(task_docs: List[TaskDoc]) -> List[Document]:
    docs: List[Document] = []
    for td in task_docs:
        docs.append(
            Document(
                page_content=td.text,
                metadata={
                    "id": td.id,
                    "origin": td.origin,
                    "group": td.group,
                    "task_name": td.task_name,
                    "class_name": td.class_name,
                    "module": td.module,
                },
            )
        )
    return docs


def build_retriever(k: int = 5):
    """Build a FAISS retriever over builtin + generated tasks. k is the number of docs to retrieve."""
    task_docs = load_all_task_docs()
    docs = _build_documents(task_docs)
    embeddings = OpenAIEmbeddings()
    vs = FAISS.from_documents(docs, embedding=embeddings)
    return vs.as_retriever(search_kwargs={"k": k})


def generate_new_task_code(
    task_name: str,
    task_description: str,
    group: str,
    retriever,
    api_reference: str,
    past_failures_text: str = "",
    model_name: str = "gpt-4.1-mini",
    temperature: float = 0.7,
) -> str:
    """
    Step 2: Generate Python code for the task given (task_name, task_description, group).
    Uses api_reference so the model only uses allowed imports and tools.
    past_failures_text: 历史未通过代码与报错，注入提示词以规避相似错误。
    """
    brief = f"task_name: {task_name}\ngroup: {group}\ntask_description: {task_description}"

    related_docs = retriever.get_relevant_documents(brief)
    context_snippets = []
    for d in related_docs:
        tn = d.metadata.get("task_name")
        cn = d.metadata.get("class_name")
        gr = d.metadata.get("group")
        header = f"=== Example: {gr}/{tn} ({cn}) ==="
        context_snippets.append(header + "\n" + d.page_content)
    context = "\n\n".join(context_snippets)

    failures_block = ""
    if past_failures_text:
        failures_block = f"""
{past_failures_text}

Avoid the above errors in your new code.

"""

    user_prompt = f"""New task to implement:
{brief}
{failures_block}
Example tasks for reference:
{context}

Generate the Python class for this task. Use ONLY the imports and tools from the API reference. Output only a ```python ... ``` block.
""".strip()

    llm = ChatOpenAI(model=model_name, temperature=temperature)
    resp = llm.invoke(
        [
            {"role": "system", "content": _system_prompt_step2(api_reference)},
            {"role": "user", "content": user_prompt},
        ]
    )
    content = resp.content if isinstance(resp.content, str) else "".join(
        [c.get("text", "") for c in (resp.content if isinstance(resp.content, list) else []) if isinstance(c, dict)]
    )
    if not content:
        content = str(resp.content)

    code_match = re.search(r"```python(.*?)```", content, re.DOTALL | re.IGNORECASE)
    if not code_match:
        code_match = re.search(r"```(.*?)```", content, re.DOTALL)
    if not code_match:
        raise ValueError("LLM 没有返回可解析的 code block。\n完整回复：\n" + content)

    code = code_match.group(1).strip()
    return code
