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
from src.agent import _ensure_entry_recorded
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


def test_safety_net_records_missing_day(tmp_path, monkeypatch):
    """If the model never recorded the day, the safety net saves a minimal
    fallback entry that preserves the raw text — the day must not vanish."""
    state_path = tmp_path / "state.json"
    monkeypatch.setattr(state_store, "STATE_PATH", str(state_path))

    _ensure_entry_recorded("2026-07-01", "Had a mystery bagel, felt off.")

    state = state_store.load_state()
    assert len(state.entries) == 1
    assert state.entries[0].day == "2026-07-01"
    assert state.entries[0].raw_text == "Had a mystery bagel, felt off."
    assert state.entries[0].foods == []  # minimal — nothing parsed


def test_safety_net_does_not_clobber_recorded_day(tmp_path, monkeypatch):
    """If the model already recorded the day, the safety net is a no-op and
    must not overwrite the parsed entry with an empty fallback."""
    state_path = tmp_path / "state.json"
    monkeypatch.setattr(state_store, "STATE_PATH", str(state_path))

    run_tool(
        "record_parsed_entry",
        {"day": "2026-07-01", "foods": ["GF bagel"]},
        today="2026-07-01",
    )
    _ensure_entry_recorded("2026-07-01", "raw text that should be ignored")

    state = state_store.load_state()
    assert len(state.entries) == 1  # not duplicated
    assert state.entries[0].foods == ["GF bagel"]  # parsed entry preserved


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


def test_flag_pattern_dedups_and_merges(tmp_path, monkeypatch):
    """Re-flagging the same hypothesis updates the existing record and merges
    evidence days, rather than appending a duplicate."""
    state_path = tmp_path / "state.json"
    monkeypatch.setattr(state_store, "STATE_PATH", str(state_path))

    run_tool(
        "flag_pattern",
        {
            "hypothesis": "Soy sauce may correlate with bloating",
            "evidence_days": ["2026-07-01", "2026-07-03"],
            "confidence": "medium",
        },
        today="2026-07-04",
    )
    # Same hypothesis (different casing/whitespace), new evidence day + higher
    # confidence — should merge into the single existing pattern.
    run_tool(
        "flag_pattern",
        {
            "hypothesis": "  soy sauce may correlate with bloating  ",
            "evidence_days": ["2026-07-03", "2026-07-07"],
            "confidence": "high",
        },
        today="2026-07-07",
    )

    state = state_store.load_state()
    assert len(state.flagged_patterns) == 1  # merged, not duplicated
    p = state.flagged_patterns[0]
    assert p.evidence_days == ["2026-07-01", "2026-07-03", "2026-07-07"]  # union, sorted
    assert p.confidence == "high"  # latest judgment wins
    assert p.day_flagged == "2026-07-07"  # updated to last-touched day


def test_flag_pattern_keeps_distinct_hypotheses_separate(tmp_path, monkeypatch):
    """Genuinely different hypotheses are not collapsed into one."""
    state_path = tmp_path / "state.json"
    monkeypatch.setattr(state_store, "STATE_PATH", str(state_path))

    for hyp in ("Soy sauce may correlate with bloating",
                "Oats may correlate with fatigue"):
        run_tool(
            "flag_pattern",
            {"hypothesis": hyp, "evidence_days": ["2026-07-01", "2026-07-03"],
             "confidence": "medium"},
            today="2026-07-04",
        )

    state = state_store.load_state()
    assert len(state.flagged_patterns) == 2


def test_create_with_retry_recovers_from_transient_error(monkeypatch):
    """The API retry wrapper retries transient errors with backoff and
    returns the eventual success, instead of throwing away the turn."""
    import httpx
    from anthropic import APIConnectionError

    from src import agent

    monkeypatch.setattr(agent.time, "sleep", lambda _s: None)  # no real waiting

    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    calls = {"n": 0}

    class FakeMessages:
        def create(self, **kwargs):
            calls["n"] += 1
            if calls["n"] < 3:
                raise APIConnectionError(message="boom", request=req)
            return "ok"

    class FakeClient:
        messages = FakeMessages()

    result = agent._create_with_retry(FakeClient())
    assert result == "ok"
    assert calls["n"] == 3  # failed twice, succeeded on the third attempt
