"""
Load/save the persistent state file (data/state.json).

This is the agent's "memory" across runs — without it, every day would be
a fresh conversation with no history to detect lagged correlations against.
"""
import json
import os

from .schemas import StateFile, Entry, FlaggedPattern, Symptom, Confounders

STATE_PATH = os.path.join(os.path.dirname(__file__), "../data/state.json")


def load_state() -> StateFile:
    """Load persisted entries and flagged patterns, or an empty state if none."""
    if not os.path.exists(STATE_PATH):
        return StateFile()

    with open(STATE_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    entries = []
    for e in raw.get("entries", []):
        symptoms = [Symptom(**s) for s in e.get("symptoms", [])]
        confounders = Confounders(**e.get("confounders", {}))
        entries.append(Entry(
            day=e["day"],
            raw_text=e["raw_text"],
            foods=e.get("foods", []),
            symptoms=symptoms,
            confounders=confounders,
            clarifications=e.get("clarifications", []),
        ))

    patterns = [FlaggedPattern(**p) for p in raw.get("flagged_patterns", [])]

    return StateFile(entries=entries, flagged_patterns=patterns)


def save_state(state: StateFile) -> None:
    """Persist entries and flagged patterns to data/state.json."""
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        data = {
            "entries": [e.model_dump() for e in state.entries],
            "flagged_patterns": [p.model_dump() for p in state.flagged_patterns],
        }
        f.write(json.dumps(data))
