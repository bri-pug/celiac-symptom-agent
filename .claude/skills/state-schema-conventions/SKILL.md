# Skill: state.json schema stability

`data/state.json` is the only persistent memory this agent has across runs.
Every design decision here should assume the file will keep growing across
weeks or months of real logs, and that a single format mistake could
silently corrupt history the user can't get back.

## Rules when changing `src/schemas.py` or `src/state_store.py`

1. Prefer **additive** changes. Add new optional fields with sensible
   defaults (see `Confounders.other: Optional[str] = None` for the
   pattern) rather than renaming or removing existing fields.
2. If a field must be renamed or removed, write a small migration in
   `state_store.load_state()` that reads the old key if present and maps
   it to the new shape, rather than breaking on old `state.json` files.
3. `load_state()` must never throw on a missing key — every `.get(key, ...)`
   call needs a sensible default, since real logs will have gaps (a day
   the user forgot to log, an early entry made before a new field existed).
4. Any new nested object (like `Confounders`, `Symptom`) needs both: a
   dataclass in `schemas.py`, and explicit (de)serialization handling in
   `state_store.py` — dataclasses don't nest themselves automatically
   through `json.load`.
5. After any schema change, run `tests/test_agent.py` — it exercises
   round-trip save/load, which is exactly where nested-dataclass bugs show
   up.

## Why this is a Skill

State-schema changes are the highest-blast-radius change in this codebase
— a mistake doesn't just break a feature, it can corrupt a user's actual
health log history. This is the kind of context that's easy to forget
under demo-deadline time pressure, so it's worth having Claude Code pull
this in automatically whenever `schemas.py` or `state_store.py` is being
touched.
