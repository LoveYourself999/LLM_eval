# Local Eval Pipeline (MMLU‑Pro → MultiTurn → MT‑Bench‑101)

This repo contains a small, **fully local** evaluation pipeline for OpenAI-compatible model endpoints (e.g., `llama_cpp.server`, vLLM OpenAI server). It runs:
1) **MMLU‑Pro** (single-turn multiple choice accuracy),
2) a lightweight **MultiTurn** suite (ability-focused regression tests),
3) **MT‑Bench‑101** (multi-turn dialogues scored by an LLM judge, e.g., Prometheus).

The goal is to make it easy to compare two or more models (e.g., `fast-model` vs `qwen-1p5b`) with consistent, reproducible settings.

---

## Contents

- [Prerequisites](###Prerequisites)
- [Architecture](#architecture)
- [1) Serve models locally](#1-serve-models-locally)
- [2) Configure endpoints](#2-configure-endpoints)
- [3) Run MMLU‑Pro eval](#3-run-mmlu-pro-eval)
- [4) Run multiturn_eval](#4-run-multiturn_eval)
- [5) Run mtbench101_eval](#5-run-mtbench101_eval)
- [Outputs](#outputs)
- [Common issues](#common-issues)
- [Notes on judging](#notes-on-judging)

---

### Prerequisites
- Python 3.11+
- An OpenAI-compatible server per model (we use `llama_cpp.server`)
- A judge server (recommend Prometheus)

### Serve models locally
```bash
# In three terminals, run:

# 1) Start judge (Prometheus)
python -m llama_cpp.server \
  --host 127.0.0.1 \
  --port 8001 \
  --hf_model_repo_id Qwen/Qwen2-0.5B-Instruct-GGUF \
  --model "*q4_0.gguf" \
  --chat_format chatml \
  --model_alias judge-model \
  --n_ctx 512

# 2) Start model A
python -m llama_cpp.server \
  --host 127.0.0.1 --port 8003 \
  --hf_model_repo_id Qwen/Qwen2-0.5B-Instruct-GGUF \
  --model "*q4_0.gguf" \
  --chat_format chatml \
  --n_ctx 2048 \
  --model_alias phi-4

# 3) Start model B
python -m llama_cpp.server \
  --host 127.0.0.1 --port 8004 \
  --hf_model_repo_id Qwen/Qwen2.5-1.5B-Instruct-GGUF \
  --model "*q4_0.gguf" \
  --chat_format chatml \
  --n_ctx 2048 \
  --model_alias qwen-1p5b

#  4) Run evals
python -m eval.multiturn_eval fast-model qwen-1p5b
python -m eval.mtbench101_eval fast-model qwen-1p5b --limit 260

```
---
### Architecture
At a high level:

1. Dataset → Messages

- Each benchmark converts raw dataset records into OpenAI chat messages=[{role, content}, ...].

2. Model Inference

- eval/client.py calls your model endpoint:

- POST {endpoint}/chat/completions

3. Scoring

- MMLU‑Pro: parse the multiple-choice letter and compute accuracy.

- MultiTurn: LLM-as-judge score per test case.

- MT‑Bench‑101: LLM-as-judge score per dialogue item.

4. Aggregation

- Metrics aggregated by domain/ability/task and saved to JSON/JSONL.


### 2) Configure endpoints
Edit eval/config.py (or your YAML->CONFIG equivalent). Example:
```python
models:
  fast-model:
    endpoint: "http://127.0.0.1:8003/v1"
    model: "phi-4"
  qwen-1p5b:
    endpoint: "http://127.0.0.1:8004/v1"
    model: "qwen-1p5b"

judge:
  endpoint: "http://127.0.0.1:8001/v1"
  model: "judge-model"
```

### 3) Run mmlu-pro-eval

Python packages for evaluation scripts:

- From the repo root: matplotlib in requirements.txt
- From /src/training/model_eval: requirements.txt

```bash
# We will work at this dir in this guide
cd /src/training/model_eval
pip install -r requirements.txt
```

run mmlu_pro_vllm_eval.py
```bash
python mmlu_pro_vllm_eval.py \
  --endpoint http://localhost:8003/v1 \
  --samples-per-category 10
```
## 4) Multi-turn Eval Benchmark
A lightweight benchmark for evaluating how well a model handles **multi-turn conversations**.

This benchmark measures whether a model can:
- remember earlier context,
- adapt to user corrections or reframing,
- ask useful follow-up questions,
- follow constraints across turns,
- stay safe when the user requests unsafe content.

It is designed to be simple to run locally with:
- a target model endpoint,
- a judge model endpoint,
- a small set of curated multi-turn test cases,
- score aggregation into a compact model profile.

---

## What it evaluates

The benchmark groups cases into three high-level abilities:

- **Perceptivity**  
  Can the model track context, resolve references, notice topic shifts, and preserve constraints across turns?

- **Adaptability**  
  Can the model revise its answer, follow changed instructions, rephrase content, and respond safely under pressure?

- **Interactivity**  
  Can the model ask clarifying questions, gather requirements, collaborate step-by-step, and provide safe alternatives when needed?

Each test case contains:
- a short multi-turn dialogue,
- an ability label,
- one or more behavior tags,
- the model’s generated response,
- a judge score from 1 to 5.

---

## Example abilities and tags

### Perceptivity
- `context_memory`
- `anaphora_resolution`
- `topic_shift`
- `separate_input`
- `constraint_following`
- `multi_turn_reasoning`

### Adaptability
- `self_correction`
- `self_affirmation`
- `format_rephrasing`
- `content_rephrasing`
- `feedback_incorporation`
- `coding_fix`
- `policy_following`

### Interactivity
- `instruction_clarification`
- `proactive_interaction`
- `requirements_elicitation`
- `collaborative_reasoning`
- `guardrails`

---

## How it works

For each benchmark case:

1. The target model receives the full multi-turn conversation.
2. The model generates a reply to the final user turn.
3. A judge model evaluates that reply using a rubric.
4. The judge returns a score from 1 to 5.
5. The score is normalized to `0.0 - 1.0`.
6. Scores are averaged by ability category.

In /vllm, run:
```bash
python -m eval.multiturn_eval phi-4 qwen-1p5b
```
Results for Phi-4
<img width="1510" height="96" alt="image" src="https://github.com/user-attachments/assets/6f217c12-e6d0-477d-a204-fb90d1bf8c30" />

Results for Qwen:
<img width="1556" height="96" alt="image" src="https://github.com/user-attachments/assets/1c121d53-71a6-4c45-803b-c001155e52bd" />
#### Visualization for comparison

<img width="630" height="470" alt="image" src="https://github.com/user-attachments/assets/f5af598d-6103-43cc-a86e-45c0fb28c2ec" />
