from typing import Optional
from pydantic import BaseModel


class Symptom(BaseModel):
    name: str  # e.g. "bloating", "fatigue", "joint pain"
    severity: int  # 1-5
    onset: Optional[str] = None  # e.g. "evening", "next morning"


class Confounders(BaseModel):
    """Explicit non-food factors that could explain a symptom on their own"""
    sleep_hours: Optional[float] = None
    stress_level: Optional[int] = None  # 1-5
    travel: bool = False
    other: Optional[str] = None


class Entry(BaseModel):
    day: str  # YYYY-MM-DD
    raw_text: str
    foods: list[str]
    symptoms: list[Symptom]
    confounders: Confounders


class FlaggedPattern(BaseModel):
    day_flagged: str
    hypothesis: str  # e.g. oats may be linked to bloating
    evidence_days: list[str]
    confidence: str  # low | medium | high
    confounders_not_ruled_out: list[str]
    note: str = "This is a pattern worth discussing with your doctor, not a diagnosis."


class StateFile(BaseModel):
    entries: list[Entry] = []
    flagged_patterns: list[FlaggedPattern] = []
