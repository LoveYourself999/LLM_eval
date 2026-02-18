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
  --host 127.0.0.1 --port 8001 \
  --hf_model_repo_id prometheus-eval/prometheus-7b-v2.0-GGUF \
  --model "*Q4_K_M.gguf" \
  --n_ctx 2048 \
  --model_alias judge

# 2) Start model A
python -m llama_cpp.server \
  --host 127.0.0.1 --port 8003 \
  --hf_model_repo_id Qwen/Qwen2-0.5B-Instruct-GGUF \
  --model "*q4_0.gguf" \
  --chat_format chatml \
  --n_ctx 2048 \
  --model_alias fast-model

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
    model: "fast-model"
  qwen-1p5b:
    endpoint: "http://127.0.0.1:8004/v1"
    model: "qwen-1p5b"

judge:
  endpoint: "http://127.0.0.1:8001/v1"
  model: "judge-model"
```

### 3-Run mmlu-pro-eval

# We will work at this dir in this guide
Python packages for evaluation scripts:

- From the repo root: matplotlib in requirements.txt
- From /src/training/model_eval: requirements.txt

```bash
cd /src/training/model_eval
pip install -r requirements.txt
```

run mmlu_pro_vllm_eval.py
```bash
python mmlu_pro_vllm_eval.py \
  --endpoint http://localhost:8003/v1 \
  --samples-per-category 10
```

### 4) Run multiturn_eval
What it measures
A small ability-focused suite (your curated YAML) for:

- perceptivity (context memory, anaphora, topic shift…),
- adaptability (self-correction, rephrasing…),
- interactivity (clarification questions, guardrails…).

How it runs
For each case:
1. Send the turns to the model (OpenAI chat format).
2. Collect the model reply.
3. Ask the judge to score it.
4. Aggregate mean score by ability and save eval/model_profiles.json.

In /vllm, run:
```bash
python -m eval.multiturn_eval fast-model qwen-1p5b
```
