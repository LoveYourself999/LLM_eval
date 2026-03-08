import json
from pathlib import Path
from typing import Dict

from .schema import CapabilityScores, ModelProfile
from .storage import load_profiles, save_profiles

# Your exact results folder
MODEL_FOLDERS: Dict[str, str] = {
    "qwen3-0.6B_direct": "qwen3-0.6B",
    # Add more like "phi4-mini_direct": "phi4-mini" when you run those
}

def ingest_capabilities(results_root: str = "~/dev/semantic-router/src/training/model_eval/results"):
    results_dir = Path(results_root).expanduser()
    profiles = load_profiles()

    for folder_name, logical_name in MODEL_FOLDERS.items():
        analysis_path = results_dir / folder_name / "analysis.json"
        if not analysis_path.exists():
            print(f"[WARN] Missing {analysis_path}")
            continue

        with open(analysis_path) as f:
            analysis = json.load(f)

        # Handle common analysis.json structures
        mmlu = (analysis.get("mmlu_pro", {}) or {}).get("category_accuracy", {})
        if not mmlu:
            mmlu = analysis.get("category_accuracy", {})

        arc = analysis.get("arc_accuracy", 0.0) or analysis.get("arc", {}).get("accuracy", 0.0)

        prof = profiles.get(logical_name) or ModelProfile(name=logical_name)
        prof.capability = CapabilityScores(mmlu=mmlu, arc=float(arc))
        profiles[logical_name] = prof

        print(f"[OK] Ingested {logical_name} from {analysis_path}")
        print(f"  MMLU keys: {list(mmlu.keys())}")
        print(f"  ARC: {arc:.3f}")

    save_profiles(profiles)
    print(f"[OK] Saved to eval/model_profiles.json")

if __name__ == "__main__":
    ingest_capabilities()
