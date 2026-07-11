"""Pydantic models for the persisted state: daily entries, symptoms,
confounders, flagged patterns, and the top-level state file."""
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
    # Resolved "was this X gluten-free?" exchanges, saved as human-readable
    # "Q: ... A: ..." strings so a later re-run can see how an ambiguous
    # food was disambiguated. Optional/additive: old state.json files that
    # predate this field load fine (see state_store.load_state).
    clarifications: list[str] = []


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
