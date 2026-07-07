# Skill: Hypothesis framing conventions

This project makes a hard commitment: it never diagnoses, it only surfaces
correlational hypotheses for the user to raise with a doctor or dietitian.
This convention has to survive every future change to the codebase — new
tools, new prompts, new output formats — or the project quietly turns into
something that gives medical advice, which it is explicitly not designed
to do.

When touching any of the following, apply these rules:

- `SYSTEM_PROMPT` in `src/agent.py`
- the `flag_pattern` tool schema/description in `src/tools.py`
- any new report/output renderer (e.g. `weekly_report` in `src/agent.py`)
- any new tool that produces user-facing text

## Rules

1. Never phrase output as causal ("X causes Y"). Always correlational
   ("X may be linked to Y", "worth checking whether...").
2. Every flagged pattern must carry a `confidence` level and, if any
   confounder (sleep, stress, travel, cycle, illness) was present on an
   evidence day and not ruled out, it must be listed in
   `confounders_not_ruled_out`. Confidence must be lower when confounders
   are present — never `"high"` if a confounder wasn't ruled out.
3. Never call `flag_pattern` on a single day of evidence. Minimum 2
   supporting days for `"medium"`, 3+ for `"high"`.
4. Any new tool or report must end with, or clearly convey, that this is
   something to discuss with a medical professional — not a conclusion.
5. `web_search` (or any research tool) is for general reference facts only
   (e.g. "is soy sauce typically gluten-free"). It must never be used to
   generate personalized medical advice for the specific user.

## Why this is a Skill and not just a code comment

Anyone (or any coding agent) making a quick edit to the system prompt or
adding a new output path is likely to accidentally soften this framing
without realizing it's a deliberate safety decision, not a stylistic one.
Keeping it as a Skill means it gets pulled into context automatically
whenever Claude Code is asked to touch these files, instead of relying on
someone remembering to re-read the README.
