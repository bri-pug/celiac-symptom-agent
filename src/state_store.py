"""
Load/save the persistent state file (data/state.json).

This is the agent's "memory" across runs — without it, every day would be
a fresh conversation with no history to detect lagged correlations against.
"""
import json
import os
import tempfile

from .schemas import StateFile

STATE_PATH = os.path.join(os.path.dirname(__file__), "../data/state.json")


def load_state() -> StateFile:
    """Load persisted entries and flagged patterns, or an empty state if none."""
    if not os.path.exists(STATE_PATH):
        return StateFile()

    with open(STATE_PATH, "r", encoding="utf-8") as f:
        return StateFile.model_validate(json.load(f))


def save_state(state: StateFile) -> None:
    """Persist entries and flagged patterns to data/state.json, atomically.

    This file is the agent's entire memory, so a torn write is catastrophic.
    Write to a temp file, fsync it, and os.replace() it into place.
    Output is indented so the committed state stays diff-friendly.
    """
    directory = os.path.dirname(STATE_PATH)
    os.makedirs(directory, exist_ok=True)
    data = {
        "entries": [e.model_dump() for e in state.entries],
        "flagged_patterns": [p.model_dump() for p in state.flagged_patterns],
    }

    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".state-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, STATE_PATH)
    except BaseException:
        # Never leave a stray temp file behind if the write failed partway.
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
