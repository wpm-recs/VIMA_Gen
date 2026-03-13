# VIMA_Gen

`VIMA_Gen` is an automatic task generation toolkit for `VIMA_Bench`, focused on:

1. Retrieval-augmented generation (RAG) over existing tasks
2. Using an LLM to propose new tasks and generate task code
3. Automatically verifying code executability and oracle solvability
4. Recording failed samples and feeding them back to reduce repeated errors


## VIMA_Gen Workflow

`VIMA_Gen/cli.py` follows a two-step pipeline:

1. `propose_new_task`: generates a new `task_name/group/task_description` from existing tasks
2. `generate_new_task_code`: generates task class code using API constraints and retrieved context

After generation, it automatically calls `verifier.verify_task_code` for three-step verification:

1. Step 1: syntax and structure checks (must include goals scaffold, must not override `oracle`)
2. Step 2: instantiate task and run `env.reset()`
3. Step 3: verify whether oracle can complete the task within `oracle_max_steps`

If verification fails, the case is written to `failed_generations.json`. Future generations inject these historical failures into prompts to avoid similar mistakes.

## Directory and Module Overview

```text
VIMA_Gen/
|-- __init__.py
|-- cli.py                # Main entry: propose, generate, verify, save, summarize
|-- rag_generator.py      # RAG retrieval + LLM generation
|-- task_index.py         # Indexes built-in and generated tasks for retrieval
|-- api_reference.py      # Builds allowed API references (import whitelist)
|-- code_reference.py     # Provides key code references for the model
|-- verifier.py           # Three-step verification logic
|-- failed_store.py       # Failed sample persistence and prompt injection
|-- failed_generations.json
`-- run_results/          # Per-run summary JSON files
```

## Quick Start

### 1) Run the generation pipeline

```bash
python VIMA_Gen/cli.py --brief "design a new reasoning task" --n 3 --k 5 --save
```

Common arguments:

- `--brief`: high-level hint for Step 1
- `--n`: number of candidate tasks
- `--k`: number of retrieved documents
- `--model`: model name (default: `gpt-4.1-mini`)
- `--temperature`: sampling temperature
- `--save`: save only code that passes verification

### 2) Outputs

- `VIMA_Gen/run_results/run_*.json`: verification summaries for all candidates in a run
- `VIMA_Gen/failed_generations.json`: failed sample pool (error message + code snippet)
- `VIMA_Gen/generated_tasks/*.py`: saved when `--save` is enabled and verification passes

## Key Design Choices

- `task_index.py` unifies built-in and previously generated tasks for retrieval, reducing duplicate proposals.
- `api_reference.py` + `code_reference.py` strongly constrain available APIs to reduce bad imports and invalid calls.
- `verifier.py` executes generated code in an environment close to real `vima_bench/tasks`, exposing structural/runtime issues early.
- `failed_store.py` enables failure-feedback injection so future prompts can explicitly avoid known error patterns.

## Prerequisites

`VIMA_Gen` depends on project Python packages and OpenAI API access:

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=your_key_here
```

