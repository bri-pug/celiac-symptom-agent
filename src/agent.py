"""
The orchestrator loop: perceive -> plan -> act -> reflect.

Each call to `process_day` is one full agentic turn: the model receives
today's raw entry, decides which tools it needs (read history? look
something up? record findings? flag a pattern?), we execute those tool
calls, feed results back, and let it keep going until it produces a final
text response. This is a genuine multi-step tool-use loop, not a single
function-calling round trip.
"""
import os
from datetime import date

from anthropic import Anthropic

from .tools import TOOL_SCHEMAS, run_tool

MODEL = "claude-sonnet-5"

SYSTEM_PROMPT = """You are a symptom-and-trigger pattern assistant for someone \
managing a chronic condition (e.g. celiac disease). Your job is to help them \
spot POSSIBLE correlations between what they eat/experience and their symptoms \
over time, and nothing more.

Hard rules:
- You never diagnose. You never say a food "caused" a symptom. You surface \
  correlational hypotheses framed as things worth discussing with a doctor \
  or dietitian.
- Celiac reactions can lag 1-3 days behind exposure. Always check history \
  across a multi-day window before concluding there is no pattern, and \
  before flagging one.
- Celiac reactions can also continue for several days after exposure. If a \
  symptom is still present, check for foods eaten in the last 1-3 days, not \
  just today.
- Check for foods that are commonly cross-contaminated with gluten (soy sauce, \
  malt vinegar, Worcestershire sauce) or foods that are likely to contain gluten.
- Before flagging any pattern, explicitly check for confounders (poor sleep, \
  high stress, travel, illness) on the evidence days. If a \
  confounder is present and not ruled out, your confidence must be lower, \
  and you must list it in confounders_not_ruled_out.
- Only call flag_pattern when there is real recurring evidence (at least 2 \
  supporting days for medium confidence, 3+ for high). A single day is never \
  enough.
- Always call record_parsed_entry once for today's entry, even if nothing \
  seems noteworthy.
- Use web_search only for general reference facts (e.g. "is soy sauce \
  typically gluten-free", "typical symptom lag time for celiac exposure"), \
  never to give this specific person personalized medical advice.
- End your turn with a short plain-language summary of what you did and any \
  hypothesis you flagged (or "nothing new to flag today" if applicable).
"""


def process_day(raw_entry: str, day: str | None = None) -> str:
    """Run one full agent turn for a single day's log entry."""
    day = day or date.today().isoformat()

    client = Anthropic()

    messages = [
        {
            "role": "user",
            "content": (
                f"Today's date: {day}\n"
                f"Today's log entry:\n{raw_entry}"
            ),
        }
    ]

    tools = TOOL_SCHEMAS + [{"type": "web_search_20250305", "name": "web_search"}]

    final_text_parts = []

    # The loop: keep going as long as the model wants to call tools.
    for _ in range(8):  # hard cap so a misbehaving loop can't run forever
        response = client.messages.create(
            model=MODEL,
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=messages,
        )

        tool_uses = [b for b in response.content if b.type == "tool_use"]
        text_blocks = [b.text for b in response.content if b.type == "text"]
        final_text_parts.extend(text_blocks)

        # Log every block type so tool usage is observable, including
        # server-side tools like web_search which do NOT show up as
        # type == "tool_use" (they're "server_tool_use" /
        # "web_search_tool_result"). Without this, web_search calls happen
        # invisibly inside the API response and you can't tell whether the
        # model is using it judiciously or not at all.
        for b in response.content:
            if b.type == "server_tool_use":
                print(f"  [web_search called] query: {b.input.get('query')!r}")
            elif b.type == "web_search_tool_result":
                n = len(b.content) if isinstance(b.content, list) else "?"
                print(f"  [web_search result] {n} result(s) returned")
            elif b.type == "tool_use":
                print(f"  [tool call] {b.name}({b.input})")

        if not tool_uses:
            break  # model is done, no more tools requested

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for call in tool_uses:
            if call.name == "web_search":
                # native tool: Claude executes this server-side, nothing to do
                continue
            result_text = run_tool(call.name, call.input, today=day)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": call.id,
                "content": result_text,
            })

        if tool_results:
            messages.append({"role": "user", "content": tool_results})
        elif response.stop_reason == "end_turn":
            break

    return "\n".join(final_text_parts) if final_text_parts else "(no summary produced)"


def weekly_report() -> str:
    """Render the accumulated flagged patterns as a plain-text report."""
    from .state_store import load_state
    state = load_state()

    if not state.flagged_patterns:
        return "No patterns flagged yet."

    lines = ["Flagged patterns to discuss with your doctor/dietitian:\n"]
    for p in state.flagged_patterns:
        lines.append(f"- {p.hypothesis} (confidence: {p.confidence})")
        lines.append(f"  Evidence days: {', '.join(p.evidence_days)}")
        if p.confounders_not_ruled_out:
            lines.append(
                f"  Confounders not ruled out: {', '.join(p.confounders_not_ruled_out)}"
            )
        lines.append("")
    return "\n".join(lines)
