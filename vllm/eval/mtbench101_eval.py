import asyncio
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import re
from collections import defaultdict, Counter

from .client import call_model, call_judge

DATA_PATH = Path("eval/data/mtbench101.jsonl")
OUT_PATH = Path("eval/out/mtbench101_results.jsonl")

# MT-Bench-101 task abbreviations (13 tasks) and their top-level abilities. [web:563]
TASK_TO_ABILITY = {
    # Perceptivity
    "CM": "perceptivity",  # Context Memory
    "SI": "perceptivity",  # Separate Input
    "AR": "perceptivity",  # Anaphora Resolution
    "TS": "perceptivity",  # Topic Shift
    "CC": "perceptivity",  # Content Confusion
    # Adaptability
    "CR": "adaptability",  # Content Rephrasing
    "FR": "adaptability",  # Format Rephrasing
    "SC": "adaptability",  # Self-correction
    "SA": "adaptability",  # Self-affirmation
    "MR": "adaptability",  # Mathematical Reasoning
    "GR": "adaptability",  # General Reasoning
    # Interactivity
    "IC": "interactivity",  # Instruction Clarification
    "PI": "interactivity",  # Proactive Interaction
}


def _shorten(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[: n - 12] + " …(truncated)"


def _avg(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _as_chat_messages(item: Dict[str, Any]) -> List[Dict[str, str]]:
    # Your MT-Bench-101 JSONL format:
    # {"task": "GR", "id": 1, "history": [{"user": "...", "bot": "..."}, ...]}
    if "history" in item and isinstance(item["history"], list):
        msgs: List[Dict[str, str]] = []
        for turn in item["history"]:
            u = (turn.get("user") or "").strip()
            if u:
                msgs.append({"role": "user", "content": u})
        if not msgs:
            raise ValueError(f"Empty history for item id={item.get('id')}")
        return msgs

    raise ValueError(f"Unknown schema for item id={item.get('id')}, keys={list(item.keys())}")


def _reference_answer(item: Dict[str, Any]) -> str:
    if "history" in item and isinstance(item["history"], list) and item["history"]:
        return (item["history"][-1].get("bot") or "").strip()
    return ""


def _balanced_subset(items: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    """
    Take ~balanced sample across tasks rather than first-N lines.
    Round-robin one item per task until reaching limit.
    """
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for it in items:
        t = str(it.get("task") or "UNK")
        buckets[t].append(it)

    # Stable deterministic order: sort tasks, keep file order within each task bucket
    tasks = sorted(buckets.keys())
    idx = {t: 0 for t in tasks}

    out: List[Dict[str, Any]] = []
    while len(out) < limit:
        progressed = False
        for t in tasks:
            i = idx[t]
            if i < len(buckets[t]):
                out.append(buckets[t][i])
                idx[t] += 1
                progressed = True
                if len(out) >= limit:
                    break
        if not progressed:
            break  # ran out of data
    return out


async def _call_model(model_name: str, turns: List[Dict[str, str]]) -> str:
    return await asyncio.wait_for(call_model(model_name, turns), timeout=180)


async def _call_judge_text(judge_prompt: str) -> str:
    return await asyncio.wait_for(
        call_judge([{"role": "user", "content": judge_prompt}], timeout_s=120),
        timeout=120,
    )


def _prometheus_score_0_to_1(judge_text: str) -> float:
    m = re.search(r"\[RESULT\]\s*([1-5])", judge_text)
    if not m:
        raise RuntimeError(f"Judge output missing [RESULT] 1-5. Got:\n{judge_text[:800]}")
    return int(m.group(1)) / 5.0


async def eval_mtbench101(models: List[str], limit: Optional[int] = None):
    items: List[Dict[str, Any]] = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    if not items:
        raise RuntimeError(f"No items loaded from {DATA_PATH}")

    # If limiting, do balanced sample across tasks (instead of head of file)
    if limit is not None:
        items = _balanced_subset(items, limit)

    # Print dataset composition for sanity
    task_counts = Counter(str(it.get("task") or "UNK") for it in items)
    print(f"Loaded {len(items)} items. Task distribution: {dict(task_counts)}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_f = open(OUT_PATH, "a", encoding="utf-8")

    try:
        for model_name in models:
            print(f"\n🧠 MT-Bench-101 eval: {model_name}")

            scores_all: List[float] = []
            scores_by_ability = {"perceptivity": [], "adaptability": [], "interactivity": [], "unknown": []}
            scores_by_task: Dict[str, List[float]] = {}

            for idx, item in enumerate(items):
                task = str(item.get("task") or "UNK")
                ability = TASK_TO_ABILITY.get(task, "unknown")

                turns = _as_chat_messages(item)
                ref = _reference_answer(item)

                reply = await _call_model(model_name, turns)

                judge_prompt = f"""### Task
You are an evaluator. Provide feedback then output a final score.

### Instruction
Score from 1 to 5:
1 = Very poor (wrong/ignores constraints/unsafe/incoherent)
3 = OK (partially correct)
5 = Excellent (fully correct, follows constraints, coherent)

### Input
TASK: {task}
ID: {item.get("id")}
DIALOGUE (messages):
{json.dumps(turns, ensure_ascii=False)}

### Response
{_shorten(reply, 1800)}

### Reference Answer
{ref if ref else "N/A"}

### Output Format
Feedback: <your feedback>
[RESULT] <an integer score 1-5>
"""
                judge_text = await _call_judge_text(judge_prompt)
                s = _prometheus_score_0_to_1(judge_text)

                scores_all.append(s)
                scores_by_ability.setdefault(ability, []).append(s)
                scores_by_task.setdefault(task, []).append(s)

                out_obj = {
                    "model": model_name,
                    "task": task,
                    "ability": ability,
                    "id": item.get("id"),
                    "score": s,
                    "turns": turns,
                    "reply": reply,
                    "reference_answer": ref,
                    "judge_text": judge_text,
                }
                out_f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                out_f.flush()

                if (idx + 1) % 25 == 0:
                    print(f"  {idx+1}/{len(items)} done, avg={_avg(scores_all):.3f}")

            print(f"\n📊 {model_name}: overall avg={_avg(scores_all):.4f} (n={len(scores_all)})")
            print("By ability:")
            for ab in ["perceptivity", "adaptability", "interactivity", "unknown"]:
                print(f"  - {ab}: avg={_avg(scores_by_ability.get(ab, [])):.4f} (n={len(scores_by_ability.get(ab, []))})")

            print("By task:")
            for t in sorted(scores_by_task.keys()):
                print(f"  - {t}: avg={_avg(scores_by_task[t]):.4f} (n={len(scores_by_task[t])})")

    finally:
        out_f.close()


if __name__ == "__main__":
    import sys

    # Usage:
    #   python -m eval.mtbench101_eval fast-model qwen-1p5b --limit 200
    args = sys.argv[1:]
    limit = None
    if "--limit" in args:
        i = args.index("--limit")
        limit = int(args[i + 1])
        args = args[:i] + args[i + 2 :]

    models = args or ["fast-model"]
    asyncio.run(eval_mtbench101(models, limit=limit))
