# Skill: Adding a new tool to the agent

This project's tools live in two places that must stay in sync:

1. **Schema** — an entry in `TOOL_SCHEMAS` in `src/tools.py`. This is what
   gets sent to the Claude API so the model knows the tool exists and what
   arguments it takes.
2. **Implementation** — a branch in `run_tool()` in the same file, which
   actually executes the tool and returns a string result.

## Steps for adding a new tool

1. Add the schema to `TOOL_SCHEMAS`. Write the `description` field as if
   you're briefing a new team member — the model only knows what this
   string tells it. Be explicit about *when* to call it and any hard
   constraints (see the existing `flag_pattern` description for an example
   of encoding a business rule directly into the tool description, not
   just a docstring).
2. Add a branch to `run_tool()` that reads `tool_input`, does the work
   (usually reading/writing `data/state.json` via `state_store.py`), and
   returns a short string — this string is what the model sees as the
   tool result, so keep it structured (JSON) if the model needs to reason
   over it, or plain text if it's just a confirmation.
3. If the tool writes to `data/state.json`, update `src/schemas.py` first
   so the new field has a proper dataclass shape — don't let ad-hoc dicts
   into `state.json`, it breaks `state_store.load_state()`.
4. Add a test in `tests/test_agent.py` following the existing pattern
   (`monkeypatch` the `STATE_PATH`, call `run_tool` directly, assert on
   the resulting state — no live API call needed for this layer).
5. If the tool changes what gets persisted or displayed to the user,
   check it against `.claude/skills/hypothesis-framing/SKILL.md` — that
   convention applies to any user-facing output, not just the two tools
   that existed when it was written.

## Why this is a Skill

The schema/implementation split is a common source of drift — it's easy
to update one and forget the other, or to write a vague tool description
that leaves the model guessing about when to call it. Encoding the steps
here means every future tool addition (by a person or by Claude Code)
follows the same shape, instead of reinventing conventions each time.
