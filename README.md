# Symptom-Trigger Pattern Agent

An agentic tool for people managing celiac disease (or similar chronic conditions) who
want to spot possible symptom/trigger correlations in their own daily logs — without
pretending to be a doctor.

**This is not a medical device and does not diagnose anything.** It surfaces
correlational hypotheses ("3 of your last 4 flare-ups followed meals containing X")
for you to raise with your doctor or dietitian. It explicitly tracks confounders
(sleep, stress, travel, etc.) and lowers its confidence when it can't rule
them out, rather than presenting a bare conclusion.

## What a flagged pattern looks like

Running the bundled demo over the sample log, the agent works one day at a time,
reads back over its history before drawing any conclusion, and ends each turn
with a plain-language summary. Here it picks up a planted correlation — regular
(non–gluten-free) soy sauce tracking with bloating and fatigue — while being
careful about a confounder and about the days that *don't* fit (abridged; exact
wording varies run to run):

```
============================================================
DAY 2026-07-04
============================================================
Log: Big work deadline today, very stressed, only slept 4 hours last
night. Ate plain grilled fish and rice. Felt bloated and fatigued again
this afternoon.

Agent: Recorded today's entry. Bloating + fatigue recurred, but today's meals
were gluten-free and you noted only 4 hours of sleep and high stress — both of
which can cause these symptoms on their own. Checking the last few days, the
same symptoms followed regular soy sauce on 07-01 and 07-03. I'm flagging a
possible link, but keeping confidence at medium because today's flare is
confounded by poor sleep and stress.

============================================================
DAY 2026-07-05
============================================================
Log: Slept 8 hours, low stress, no travel. Had GF soy sauce again with
sushi at lunch. Felt totally fine, no symptoms.

Agent: Recorded today's entry. This is a useful counter-example: gluten-free
soy sauce with good sleep and low stress, and no symptoms — consistent with the
hypothesis being about regular soy sauce specifically, not soy sauce in general.
Nothing new to flag today.

============================================================
WEEKLY REPORT
============================================================
Flagged patterns to discuss with your doctor/dietitian:

- Regular (non-GF) soy sauce may be linked to bloating and fatigue (confidence: medium)
  Evidence days: 2026-07-01, 2026-07-03, 2026-07-04
  Confounders not ruled out: poor sleep, high stress
```

Note what it does *not* do: it doesn't blame the gluten-free tamari day, and it
doesn't claim certainty — the poor-sleep/high-stress day is listed as a
confounder it couldn't rule out, which is exactly why confidence stays at
`medium` rather than `high`.

## Why this is "agentic" and not just a script

A plain script could compute food/symptom correlations in a spreadsheet. This project
instead runs a genuine perceive → plan → act → reflect loop:

1. **Perceive** — reads today's free-text log entry plus a lookback window of history
   (celiac reactions can lag 1–3 days behind exposure, so "yesterday vs. today" isn't
   enough).
2. **Plan** — the model decides *which* tools it needs: does this entry need a
   reference lookup (e.g., "is soy sauce typically gluten-free")? Does a symptom
   need to be checked against history at all, or is there nothing new to flag?
3. **Act** — it calls tools to read history, look things up, and record findings.
4. **Reflect** — before flagging anything, it explicitly checks for confounders
   (bad sleep, high stress, travel) that could explain the symptom instead, and
   downgrades its confidence when it can't rule them out. This step is what
   prevents the agent from being a naive correlation printer.

## Architecture

```
Daily log entry (plain text)
        │
        ▼
  agent.py orchestrator — one perceive → plan → act → reflect loop per day
        │
   ┌────┴─────────────────────────────────┐
   │ Claude API (model: claude-sonnet-5)  │
   │ Tools the model can call:            │
   │   - read_history(window_days)        │  past entries + prior flags (bounded, compact view)
   │   - record_parsed_entry(...)         │  structured entry -> state.json
   │   - flag_pattern(...)                │  hypothesis + evidence_days + confidence + confounders
   │   - ask_user(question)               │  one clarifying Q (interactive single-entry mode only)
   │   - web_search (native Claude tool)  │  general reference facts only, never personalized advice
   └────┬─────────────────────────────────┘
        │   client side: retries w/ backoff, per-day token/cost log,
        │   safety-net entry so no day is ever lost
        ▼
  data/state.json  — atomic writes; all entries + running flagged patterns
        │
        ▼
  Weekly report (console) — flagged patterns, evidence, confidence,
  framed as "discuss with your doctor", never a diagnosis
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# then add your ANTHROPIC_API_KEY to .env
```

## Running the demo

Feed it a sequence of days from the bundled synthetic log (safe to run live —
no real personal data):

```bash
python -m src.cli --demo data/sample_entries.txt
```

Or log a single day interactively:

```bash
python -m src.cli --entry "Had ramen with soy sauce for lunch, bloating and
fatigue started this evening, slept 7hrs last night, no unusual stress."

python -m src.cli --date "2026-07-01" --entry "Had ramen with soy sauce for lunch,
bloating and fatigue started this evening, slept 7hrs last night, no unusual stress."
```

In `--entry` mode the agent may ask **one** clarifying question (the `ask_user`
tool) when a food's gluten status is genuinely ambiguous — e.g. a plain "bagel"
with no GF note — and folds your answer into the recorded entry. The `--demo`
replay runs non-interactively and skips this.

At the end of a run, print the accumulated report:

```bash
python -m src.cli --report
```

### Logs and cost

Add `-v` / `--verbose` to any command for low-level per-call diagnostics
(cache hits, stop reasons). Every day also logs a one-line token/cost summary,
so you can see what a run cost:
```
[usage] 2026-07-01 cost ~7940 tokens (input=1350 cache_read=3700 cache_write=2500 output=390), ~$0.0204
```

## Project layout

```
src/
  agent.py         the orchestrator loop (perceive/plan/act/reflect)
  cli.py           command-line entry point + colored, app-only logging
  schemas.py       Pydantic models for entries, state, and flagged patterns
  state_store.py   atomic load/save of data/state.json
  tools.py         tool schemas (JSON) + Python implementations
data/
  demo-output.txt        captured console transcript from a demo run
  historical-state.json  frozen snapshot of a populated run, kept for reference
  sample_entries.txt     synthetic multi-day log with a planted correlation, for a safe, repeatable live demo
  state.json             persistent history (the agent's working memory)
tests/
  test_agent.py       tools + confounder/confidence logic
  test_agent_loop.py  the multi-step tool-use loop (scripted fake client)
.claude/skills/    Skills used with Claude Code while building this repo
                   (adding-a-tool, hypothesis-framing, state-schema-conventions)
```
