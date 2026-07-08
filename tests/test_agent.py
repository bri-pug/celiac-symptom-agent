"""
Tests focused on the parts that don't require hitting the live API:
state persistence and tool execution. The confounder-aware confidence
judgment itself lives in the model's reasoning (guided by the system
prompt), not in deterministic code, so it's validated via the demo script
and manual review of the transcript rather than a unit test — see README.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import state_store
from src.tools import run_tool


def test_record_and_read_entry(tmp_path, monkeypatch):
    state_path = tmp_path / "state.json"
    monkeypatch.setattr(state_store, "STATE_PATH", str(state_path))

    result = run_tool(
        "record_parsed_entry",
        {
            "day": "2026-07-01",
            "foods": ["soy sauce", "rice"],
            "symptoms": [{"name": "bloating", "severity": 3, "onset": "evening"}],
            "confounders": {"sleep_hours": 7.5, "stress_level": 2},
        },
        today="2026-07-01",
    )
    assert "Recorded entry" in result

    state = state_store.load_state()
    assert len(state.entries) == 1
    assert state.entries[0].foods == ["soy sauce", "rice"]
    assert state.entries[0].symptoms[0].name == "bloating"


def test_reentry_same_day_is_idempotent(tmp_path, monkeypatch):
    state_path = tmp_path / "state.json"
    monkeypatch.setattr(state_store, "STATE_PATH", str(state_path))

    for _ in range(2):
        run_tool(
            "record_parsed_entry",
            {"day": "2026-07-01", "foods": ["eggs"]},
            today="2026-07-01",
        )

    state = state_store.load_state()
    assert len(state.entries) == 1  # second call replaces, doesn't duplicate


def test_clarifications_round_trip(tmp_path, monkeypatch):
    """A clarified entry (e.g. 'the bagel was a regular one') persists and
    survives a save/load round trip, so the correction isn't lost."""
    state_path = tmp_path / "state.json"
    monkeypatch.setattr(state_store, "STATE_PATH", str(state_path))

    run_tool(
        "record_parsed_entry",
        {
            "day": "2026-07-01",
            "foods": ["regular bagel (contains gluten)", "cream cheese"],
            "clarifications": ["Q: Was the bagel gluten-free? A: No, a regular bagel."],
        },
        today="2026-07-01",
    )

    state = state_store.load_state()
    assert len(state.entries) == 1
    assert state.entries[0].clarifications == [
        "Q: Was the bagel gluten-free? A: No, a regular bagel."
    ]
    # Older entries with no clarifications still default cleanly.
    assert "regular bagel (contains gluten)" in state.entries[0].foods


def test_flag_pattern_records_confounders(tmp_path, monkeypatch):
    state_path = tmp_path / "state.json"
    monkeypatch.setattr(state_store, "STATE_PATH", str(state_path))

    run_tool(
        "flag_pattern",
        {
            "hypothesis": "Soy sauce may correlate with bloating",
            "evidence_days": ["2026-07-01", "2026-07-03"],
            "confidence": "medium",
            "confounders_not_ruled_out": ["stress"],
        },
        today="2026-07-04",
    )

    state = state_store.load_state()
    assert len(state.flagged_patterns) == 1
    assert state.flagged_patterns[0].confidence == "medium"
    assert "stress" in state.flagged_patterns[0].confounders_not_ruled_out
