import json
from pathlib import Path
from typing import Dict
from .schema import AllProfiles, ModelProfile

DEFAULT_PATH = Path("eval/model_profiles.json")

def load_profiles(path: Path = DEFAULT_PATH) -> Dict[str, ModelProfile]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    profiles = AllProfiles(**data)
    return {m.name: m for m in profiles.models}

def save_profiles(profiles: Dict[str, ModelProfile], path: Path = DEFAULT_PATH) -> None:
    all_profiles = AllProfiles(models=list(profiles.values()))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(all_profiles.model_dump(), indent=2))
