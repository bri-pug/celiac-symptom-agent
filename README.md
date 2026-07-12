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
  agent.py orchestrator (loop, one run per day)
        │
   ┌────┴───────────────────────────────┐
   │  Claude API, tool use               │
   │  Tools:                             │
   │   - read_history(window_days)       │  reads data/state.json
   │   - record_parsed_entry(...)        │  writes structured entry to state
   │   - flag_pattern(...)               │  writes a hypothesis + evidence +
   │                                      │  confidence + confounders NOT ruled out
   │   - web_search (native Claude tool) │  general reference facts only,
   │                                      │  never personalized medical advice
   └────┬───────────────────────────────┘
        │
        ▼
   data/state.json  (all entries + running list of flagged patterns)
        │
        ▼
   Weekly report (console) — flagged patterns, evidence, confidence,
   "discuss with your doctor" framing
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

At the end of a run, print the accumulated report:

```bash
python -m src.cli --report
```

## Project layout

```
src/
  schemas.py       dataclasses for entries, state, and flagged patterns
  state_store.py   load/save data/state.json
  tools.py         tool schemas (JSON) + Python implementations
  agent.py         the orchestrator loop (perceive/plan/act/reflect)
  cli.py           command-line entry point used in the demo
data/
  state.json       persistent history (starts empty)
  sample_entries.txt  synthetic multi-day log with a planted correlation,
                       for a safe, repeatable live demo
.claude/skills/    Skills used with Claude Code while building this repo
tests/
  test_agent.py    tests for the confounder/confidence logic
```
