# Local Eval Pipeline (MMLU‑Pro → MultiTurn → MT‑Bench‑101)

This repo contains a small, **fully local** evaluation pipeline for OpenAI-compatible model endpoints (e.g., `llama_cpp.server`, vLLM OpenAI server). It runs:
1) **MMLU‑Pro** (single-turn multiple choice accuracy),
2) a lightweight **MultiTurn** suite (ability-focused regression tests),
3) **MT‑Bench‑101** (multi-turn dialogues scored by an LLM judge, e.g., Prometheus).

The goal is to make it easy to compare two or more models (e.g., `fast-model` vs `qwen-1p5b`) with consistent, reproducible settings.

---

## Contents

- [Quick start](#quick-start)
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

## Quick start

### Prerequisites
- Python 3.11+
- An OpenAI-compatible server per model (we use `llama_cpp.server`)
- A judge server (recommend Prometheus)

### Typical flow
```bash
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

# 4) Run evals
python -m eval.multiturn_eval fast-model qwen-1p5b
python -m eval.mtbench101_eval fast-model qwen-1p5b --limit 260
