"""
Tools available to the agent, plus their Python implementations.

Design note: `record_parsed_entry` and `flag_pattern` are "structured output"
tools — the model does the actual reasoning (extraction, correlation,
confounder-checking) and calls these tools to commit its conclusions to
state in a fixed shape. `read_history` is a real data-access tool. This
mirrors how you'd design tools in a production agent: let the model reason
in natural language, but force anything that gets persisted through a
strict schema.
"""

from .schemas import Entry, Symptom, Confounders, FlaggedPattern
from .state_store import load_state, save_state

TOOL_SCHEMAS = [
    {
        "name": "read_history",
        "description": (
            "Read past logged entries, most recent first, to check for lagged "
            "correlations (celiac reactions can appear 1-3 days after exposure, "
            "so always check more than just yesterday). Also returns any "
            "previously flagged patterns so you don't repeat yourself."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "window_days": {
                    "type": "integer",
                    "description": "How many past days of entries to retrieve.",
                    "default": 7,
                }
            },
        },
    },
    {
        "name": "record_parsed_entry",
        "description": (
            "Record today's log entry in structured form. Always call this "
            "once per day, even if nothing seems noteworthy — a full history "
            "is what makes lagged pattern detection possible later."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "day": {"type": "string", "description": "ISO date, e.g. 2026-07-01"},
                "foods": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Foods/ingredients mentioned in the entry.",
                },
                "symptoms": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "severity": {"type": "integer", "minimum": 1, "maximum": 5},
                            "onset": {"type": "string"},
                        },
                        "required": ["name", "severity"],
                    },
                },
                "confounders": {
                    "type": "object",
                    "properties": {
                        "sleep_hours": {"type": "number"},
                        "stress_level": {"type": "integer", "minimum": 1, "maximum": 5},
                        "travel": {"type": "boolean"},
                        "menstrual_cycle_relevant": {"type": "boolean"},
                        "other": {"type": "string"},
                    },
                },
            },
            "required": ["day"],
        },
    },
    {
        "name": "flag_pattern",
        "description": (
            "Record a candidate trigger/symptom hypothesis. Only call this when "
            "there is real recurring evidence across multiple days. You MUST "
            "list any confounders (poor sleep, high stress, travel, cycle) "
            "present on the evidence days that you could NOT rule out, and "
            "your confidence must be lower when such confounders are present. "
            "Never phrase this as a diagnosis — always frame it as something "
            "worth discussing with a doctor or dietitian."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "hypothesis": {"type": "string"},
                "evidence_days": {"type": "array", "items": {"type": "string"}},
                "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                "confounders_not_ruled_out": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["hypothesis", "evidence_days", "confidence"],
        },
    },
]


def run_tool(name: str, tool_input: dict, today: str) -> str:
    """Execute a tool call and return a string result for the model."""
    state = load_state()

    if name == "read_history":
        window = tool_input.get("window_days", 7)
        recent = state.entries[-window:]
        history = [e.model_dump() for e in recent]
        patterns = [p.model_dump() for p in state.flagged_patterns]
        import json
        return json.dumps({"recent_entries": history, "previously_flagged": patterns})

    if name == "record_parsed_entry":
        symptoms = [Symptom(**s) for s in tool_input.get("symptoms", [])]
        confounders = Confounders(**tool_input.get("confounders", {}))
        entry = Entry(
            day=tool_input.get("day", today),
            raw_text="(recorded via record_parsed_entry)",
            foods=tool_input.get("foods", []),
            symptoms=symptoms,
            confounders=confounders,
        )
        # replace any existing entry for the same day (idempotent re-runs)
        state.entries = [e for e in state.entries if e.day != entry.day]
        state.entries.append(entry)
        state.entries.sort(key=lambda e: e.day)
        save_state(state)
        return f"Recorded entry for {entry.day}."

    if name == "flag_pattern":
        pattern = FlaggedPattern(
            day_flagged=today,
            hypothesis=tool_input["hypothesis"],
            evidence_days=tool_input.get("evidence_days", []),
            confidence=tool_input["confidence"],
            confounders_not_ruled_out=tool_input.get("confounders_not_ruled_out", []),
        )
        state.flagged_patterns.append(pattern)
        save_state(state)
        return f"Flagged pattern: {pattern.hypothesis} (confidence: {pattern.confidence})"

    return f"Unknown tool: {name}"
