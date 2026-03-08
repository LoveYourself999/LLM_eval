from pydantic import BaseModel
from typing import Dict, List

class CapabilityScores(BaseModel):
    mmlu: Dict[str, float] = {}
    arc: float = 0.0

class DialogueScores(BaseModel):
    perceptivity: float = 0.0
    adaptability: float = 0.0
    interactivity: float = 0.0

class ResearchScores(BaseModel):
    quality: float = 0.0
    factual_ratio: float = 0.0

class OpsScores(BaseModel):
    latency_ms_p50: float = 0.0
    cost_per_1k_tokens: float = 0.0
    safety_score: float = 0.0

class ModelProfile(BaseModel):
    name: str
    capability: CapabilityScores = CapabilityScores()
    dialogue: DialogueScores = DialogueScores()
    research: ResearchScores = ResearchScores()
    ops: OpsScores = OpsScores()

class AllProfiles(BaseModel):
    models: List[ModelProfile] = []
