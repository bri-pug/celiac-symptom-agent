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

import json

from .schemas import Entry, Symptom, Confounders, FlaggedPattern
from .state_store import load_state, save_state

# Cap on how many flagged patterns read_history returns to the model. state.json
# grows across months, so anything echoed back every call must be bounded.
MAX_PATTERNS = 10

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
                        "other": {"type": "string"},
                    },
                },
                "clarifications": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Resolved clarifying exchanges from ask_user, one per "
                        "string as 'Q: ... A: ...'. Include these when a food "
                        "was ambiguous and the user told you what it actually "
                        "was, so the correction is saved to history."
                    ),
                },
            },
            "required": ["day"],
        },
    },
    {
        "name": "ask_user",
        "description": (
            "Ask the user ONE short clarifying question about an ambiguous item "
            "in today's entry and get their answer back before you finish. Use "
            "this only when a logged food or exposure is ambiguous in a way that "
            "materially changes gluten risk — e.g. whether a 'bagel' was gluten-"
            "free or a regular one, whether soy sauce was gluten-free tamari, or "
            "whether a restaurant dish was ordered from a gluten-free menu. Do "
            "NOT ask about things that don't affect gluten exposure. After you "
            "get the answer, call record_parsed_entry again for the same day "
            "with the clarified foods and the resolved 'Q: ... A: ...' added to "
            "`clarifications`, so the correction is persisted. This tool is for "
            "factual clarification of what was eaten only — never use it to ask "
            "for, or to give, medical advice."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The single clarifying question to ask the user.",
                }
            },
            "required": ["question"],
        },
    },
    {
        "name": "flag_pattern",
        "description": (
            "Record a candidate trigger/symptom hypothesis. Only call this when "
            "there is real recurring evidence across multiple days. You MUST "
            "list any confounders (poor sleep, high stress, travel) "
            "present on the evidence days that you could NOT rule out, and "
            "your confidence must be lower when such confounders are present. "
            "Keep `hypothesis` a SHORT, STABLE claim of the form "
            "'X may be linked to Y'. "
            "Do NOT list specific dates, foods, or a running evidence log in "
            "the hypothesis text — put every supporting date in `evidence_days` "
            "instead. Likewise, `confounders_not_ruled_out` must be SHORT, STABLE "
            "category labels ('poor sleep', 'high stress', 'travel', 'illness'), "
            "NOT dates or per-day narratives — the per-day specifics belong in each "
            "day's recorded entry, not here. Re-flagging an existing hypothesis "
            "UPDATES that record in "
            "place (merging in the new evidence days) only when the hypothesis "
            "text matches, so reuse the SAME wording as more evidence accrues "
            "rather than rephrasing it, or you will create a near-duplicate. "
            "Never phrase this as a diagnosis — always frame it as something "
            "worth discussing with a doctor or dietitian."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "hypothesis": {
                    "type": "string",
                    "description": (
                        "A short, stable 'X may be linked to Y' claim. No "
                        "dates or evidence log here — use evidence_days for that."
                    ),
                },
                "evidence_days": {"type": "array", "items": {"type": "string"}},
                "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                "confounders_not_ruled_out": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Short, stable confounder category labels present on the "
                        "evidence days that you could NOT rule out, e.g. 'poor "
                        "sleep', 'high stress', 'travel', 'illness'. One label per "
                        "confounder type. Do NOT put dates or per-day narratives "
                        "here — those belong in each day's recorded entry."
                    ),
                },
            },
            "required": ["hypothesis", "evidence_days", "confidence"],
        },
    },
]


def _clean_confounder_labels(labels: list[str]) -> list[str]:
    """Normalise confounder labels to short, deduplicated category strings.

    `confounders_not_ruled_out` is meant to hold a handful of stable category
    labels ("poor sleep", "high stress"), but the model has been prone to
    cramming date-stamped, per-day narratives in here — and because the merge
    below unions raw strings, each slightly-different narrative slipped past
    dedup and the field grew without bound (see the hypothesis-framing skill;
    this is the same failure mode on a sibling field). Collapsing whitespace
    and deduping case-insensitively means equivalent labels actually merge, so
    a flagged pattern accretes evidence days without its confounder list
    ballooning. First-seen display form is preserved; result is sorted stable.
    """
    seen: dict[str, str] = {}
    for raw in labels:
        label = " ".join(raw.split())
        if not label:
            continue
        seen.setdefault(label.lower(), label)
    return sorted(seen.values())


def run_tool(name: str, tool_input: dict, today: str) -> str:
    """Execute a tool call and return a string result for the model."""
    state = load_state()

    if name == "read_history":
        window = tool_input.get("window_days", 7)
        recent = state.entries[-window:]
        history = [e.model_dump() for e in recent]
        # Return only a bounded, compact view of flagged patterns. The purpose
        # of surfacing these is so the model doesn't re-flag a hypothesis it
        # already flagged, which only needs the hypothesis text, confidence,
        # and date — not the full evidence_days/confounders/note payload.
        # Returning every field of every pattern grew this response without
        # bound as the log accrued weeks of history.
        all_patterns = state.flagged_patterns
        patterns = [
            {
                "hypothesis": p.hypothesis,
                "confidence": p.confidence,
                "day_flagged": p.day_flagged,
            }
            for p in all_patterns[-MAX_PATTERNS:]
        ]
        omitted = max(0, len(all_patterns) - MAX_PATTERNS)
        result = {"recent_entries": history, "previously_flagged": patterns}
        if omitted:
            # Don't silently truncate — tell the model older patterns exist.
            result["previously_flagged_omitted_older"] = omitted
        return json.dumps(result)

    if name == "record_parsed_entry":
        symptoms = [Symptom(**s) for s in tool_input.get("symptoms", [])]
        confounders = Confounders(**tool_input.get("confounders", {}))
        entry = Entry(
            day=tool_input.get("day", today),
            raw_text="(recorded via record_parsed_entry)",
            foods=tool_input.get("foods", []),
            symptoms=symptoms,
            confounders=confounders,
            clarifications=tool_input.get("clarifications", []),
        )
        # replace any existing entry for the same day (idempotent re-runs)
        state.entries = [e for e in state.entries if e.day != entry.day]
        state.entries.append(entry)
        state.entries.sort(key=lambda e: e.day)
        save_state(state)
        return f"Recorded entry for {entry.day}."

    if name == "flag_pattern":
        hypothesis = tool_input["hypothesis"]
        evidence_days = tool_input.get("evidence_days", [])
        confounders = tool_input.get("confounders_not_ruled_out", [])

        # Dedup by normalized hypothesis text. Re-running a day, or the model
        # refining a hypothesis as evidence accrues, must UPDATE the existing
        # pattern in place — appending would fill the weekly report with
        # near-duplicate entries. Evidence days and confounders merge as a
        # union; the latest confidence is taken as the model's current call.
        key = hypothesis.strip().lower()
        existing = next(
            (p for p in state.flagged_patterns if p.hypothesis.strip().lower() == key),
            None,
        )
        if existing is not None:
            existing.evidence_days = sorted(set(existing.evidence_days) | set(evidence_days))
            existing.confounders_not_ruled_out = _clean_confounder_labels(
                existing.confounders_not_ruled_out + confounders
            )
            existing.confidence = tool_input["confidence"]
            existing.day_flagged = today
            save_state(state)
            return (
                f"Updated existing pattern: {existing.hypothesis} "
                f"(confidence: {existing.confidence}, "
                f"{len(existing.evidence_days)} evidence day(s))"
            )

        pattern = FlaggedPattern(
            day_flagged=today,
            hypothesis=hypothesis,
            evidence_days=sorted(set(evidence_days)),
            confidence=tool_input["confidence"],
            confounders_not_ruled_out=_clean_confounder_labels(confounders),
        )
        state.flagged_patterns.append(pattern)
        save_state(state)
        return f"Flagged pattern: {pattern.hypothesis} (confidence: {pattern.confidence})"

    return f"Unknown tool: {name}"
