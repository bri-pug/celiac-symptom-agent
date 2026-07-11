"""
The orchestrator loop: perceive -> plan -> act -> reflect.

Each call to `process_day` is one full agentic turn: the model receives
today's raw entry, decides which tools it needs (read history? look
something up? record findings? flag a pattern?), we execute those tool
calls, feed results back, and let it keep going until it produces a final
text response. This is a genuine multi-step tool-use loop, not a single
function-calling round trip.

The turn is decomposed into small, independently testable pieces:
  _build_tools_and_system  -> assemble the (cacheable) prompt prefix
  _dispatch_tool_calls      -> execute one batch of client-side tool calls
  _run_turn                 -> drive the create -> tools -> feed-back loop
`process_day` wires them together. The Anthropic client is injectable so
the loop can be exercised against a scripted fake in tests, with no network.
"""
import time
from datetime import date
from typing import Callable

from anthropic import (
    Anthropic,
    APIConnectionError,
    APIError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)

from .schemas import Confounders, Entry
from .state_store import load_state, save_state
from .tools import TOOL_SCHEMAS, run_tool

MODEL = "claude-sonnet-5"

# A single turn must fit reasoning + a full record_parsed_entry call (many
# foods/symptoms/confounders) and possibly a flag_pattern call. 1500 truncated
# mid-generation on busy days, yielding no complete tool_use block -> the loop
# saw no tools and cut the turn off before recording (stop_reason=max_tokens).
MAX_TOKENS = 4096
# Hard cap so a misbehaving loop can't run forever. Leaves room for a
# clarify -> re-record round on top of the usual read/record/flag sequence.
MAX_LOOP_ITERATIONS = 10

# Transient API failures worth retrying. Other APIErrors (bad request, auth,
# etc.) won't succeed on retry and are re-raised immediately.
_RETRYABLE_ERRORS = (
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
)
_MAX_API_ATTEMPTS = 4

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
  and you must list it in confounders_not_ruled_out as a SHORT category label \
  (e.g. "poor sleep", "high stress"), NOT a date-stamped or per-day \
  description — the per-day detail belongs in that day's recorded entry.
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

ASK_USER_BULLET="""- If today's entry contains an item whose gluten status is genuinely \
  ambiguous in a way that would change your assessment (a plain "bagel" with \
  no GF/regular note, an unspecified restaurant dish, a sauce that may or may \
  not contain gluten), call ask_user to ask ONE short clarifying question, \
  then re-record the day with the clarified detail and the resolved question \
  in `clarifications`. Only ask when the answer would actually change what \
  you record or flag; never ask for or offer medical advice.
"""


def _create_with_retry(client: Anthropic, **kwargs):
    """Call messages.create with exponential backoff on transient errors.

    A single rate-limit or 5xx shouldn't throw away a whole day's turn.
    SDK-level retries are disabled where the client is built, so this is the
    one place retry policy lives. Non-retryable errors (bad request, auth)
    propagate immediately — retrying them would just waste time.
    """
    delay = 1.0
    for attempt in range(1, _MAX_API_ATTEMPTS + 1):
        try:
            return client.messages.create(**kwargs)
        except _RETRYABLE_ERRORS as e:
            if attempt == _MAX_API_ATTEMPTS:
                raise
            print(
                f"  [retry] transient API error ({type(e).__name__}); "
                f"attempt {attempt}/{_MAX_API_ATTEMPTS}, retrying in {delay:.0f}s"
            )
            time.sleep(delay)
            delay *= 2


def _ensure_entry_recorded(day: str, raw_entry: str) -> None:
    """Guarantee an entry exists for `day`, recording a fallback if not.

    The entire lagged-pattern premise depends on a COMPLETE daily history,
    but "always call record_parsed_entry" is only a prompt instruction — the
    model can skip it, or the loop can hit its iteration cap before it does.
    This makes the invariant deterministic: if nothing was persisted for the
    day, save a minimal entry that preserves the raw text (so it can be
    re-parsed later) rather than letting the day vanish from history.
    """
    state = load_state()
    if any(e.day == day for e in state.entries):
        return  # model already recorded it — nothing to do

    print(
        f"  [safety net] model recorded no entry for {day}; saving raw text "
        f"as a minimal fallback so the day isn't lost from history."
    )
    state.entries.append(Entry(
        day=day,
        raw_text=raw_entry,
        foods=[],
        symptoms=[],
        confounders=Confounders(),
    ))
    state.entries.sort(key=lambda e: e.day)
    save_state(state)


def terminal_ask(question: str) -> str:
    """Default ask_user handler: prompt the person at the terminal.

    Returns "" if there is no interactive stdin (e.g. piped input, CI), so
    the agent loop never hangs on input() — it just proceeds without the
    clarification instead.
    """
    print(f"\n  [agent asks] {question}")
    try:
        return input("  your answer > ").strip()
    except (EOFError, KeyboardInterrupt):
        return ""


def _missing_required_fields(tool_name: str, tool_input: dict) -> list[str]:
    """Required fields (per the tool's input_schema) that are absent or empty
    in a tool call.

    The Anthropic API treats `required` as a strong hint, not a hard
    constraint, so the model can still emit an incomplete tool call. We check
    it at the dispatch boundary so every tool implementation can assume its
    required inputs are present and non-empty — e.g. flag_pattern indexes
    tool_input["hypothesis"] directly and must never receive a call without it.
    """
    schema = next((t for t in TOOL_SCHEMAS if t["name"] == tool_name), None)
    if schema is None:
        return []
    required = schema.get("input_schema", {}).get("required", [])
    return [
        field
        for field in required
        if tool_input.get(field) in (None, "", [], {})
    ]


def _build_tools_and_system(ask_user: bool) -> tuple[list, list]:
    """Assemble the tool list and the cacheable system prompt for a turn.

    When `ask_user` is False we drop the ask_user tool entirely (so the model
    can't try to prompt a non-interactive caller); when it's True we append the
    bullet that tells the model when to use it.

    The system prompt is wrapped in a single cache_control block. Render order
    is tools -> system -> messages, so a breakpoint on the last system block
    caches the tools AND the system prompt together. This prefix is
    byte-identical across every loop iteration (and across days, since neither
    the prompt nor the tools change), while the volatile per-day content lives
    in `messages` after it. Each day makes several tool round-trips that all
    re-send this prefix, so the cache is read far more than the ~2-request
    break-even: a net cost saving, not an increase.
    """
    tools = TOOL_SCHEMAS + [{"type": "web_search_20250305", "name": "web_search"}]
    system_prompt = SYSTEM_PROMPT
    if ask_user:
        system_prompt = system_prompt + ASK_USER_BULLET
    else:
        tools = [t for t in tools if t["name"] != "ask_user"]

    system = [
        {"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}
    ]
    return tools, system


def _log_response(response) -> None:
    """Emit per-call observability: cache hits, stop reason, and every content
    block (including server-side web_search, which does NOT appear as a
    client-side tool_use block).

    Cache: `cache_read_input_tokens` should be 0 on the first call (prefix
    written) and >0 on every later call (prefix served at ~0.1x cost). If it
    stays 0, the prefix is either below Sonnet's ~2048-token minimum or a
    silent invalidator is changing it between calls.

    Stop reason: distinguishing end_turn (genuinely done) from pause_turn
    (paused for a server-side tool) and max_tokens (truncated) is essential —
    they look identical in tool output but demand different loop handling, and
    conflating them silently cut turns short before record_parsed_entry ran.
    """
    usage = response.usage
    print(
        f"  [cache] read={usage.cache_read_input_tokens} "
        f"write={usage.cache_creation_input_tokens} "
        f"uncached_input={usage.input_tokens}"
    )
    print(f"  [stop_reason] {response.stop_reason}")
    for b in response.content:
        if b.type == "server_tool_use":
            print(f"  [web_search called] query: {b.input.get('query')!r}")
        elif b.type == "web_search_tool_result":
            n = len(b.content) if isinstance(b.content, list) else "?"
            print(f"  [web_search result] {n} result(s) returned")
        elif b.type == "tool_use":
            print(f"  [tool call] {b.name}({b.input})")


def _dispatch_tool_calls(
    tool_uses: list,
    day: str,
    ask_user_handler: Callable[[str], str],
) -> list:
    """Execute one batch of client-side tool calls, returning tool_result
    blocks to feed back to the model.

    web_search is skipped: it's a native tool Claude runs server-side, so there
    is nothing for us to execute. Calls missing a schema-required field are
    rejected at this boundary (an is_error result) so run_tool never crashes on
    a missing key and the model can re-issue a complete call within the turn.
    """
    tool_results = []
    for call in tool_uses:
        if call.name == "web_search":
            continue

        missing = _missing_required_fields(call.name, call.input)
        if missing:
            print(f"  [tool rejected] {call.name} missing required: {missing}")
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": call.id,
                "content": (
                    f"Error: {call.name} was called without required "
                    f"field(s): {', '.join(missing)}. Re-call {call.name} "
                    f"with every required field present and non-empty."
                ),
                "is_error": True,
            })
            continue

        if call.name == "ask_user":
            # Client-side interactive tool: put the model's question to the
            # user and feed their answer back as the tool result.
            answer = ask_user_handler(call.input.get("question", ""))
            result_text = answer or "(no answer provided)"
        else:
            result_text = run_tool(call.name, call.input, today=day)

        tool_results.append({
            "type": "tool_result",
            "tool_use_id": call.id,
            "content": result_text,
        })
    return tool_results


def _run_turn(
    client: Anthropic,
    system: list,
    tools: list,
    messages: list,
    day: str,
    ask_user_handler: Callable[[str], str],
) -> tuple[list[str], APIError | None]:
    """Drive the create -> execute tools -> feed results loop until the model
    stops requesting tools (or the iteration cap is hit).

    Returns the accumulated final text blocks and, if the API gave up after
    retries, the error (so the caller can still run the safety net and report).
    """
    final_text_parts: list[str] = []
    api_error: APIError | None = None

    for _ in range(MAX_LOOP_ITERATIONS):
        try:
            response = _create_with_retry(
                client,
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=system,
                tools=tools,
                messages=messages,
            )
        except APIError as e:
            # Retries exhausted (or a non-retryable error). Don't crash the
            # whole run — break out so the safety net still preserves the raw
            # entry, and a demo replaying many days can continue.
            api_error = e
            print(f"  [api error] giving up on {day} after retries: {e}")
            break

        _log_response(response)

        tool_uses = [b for b in response.content if b.type == "tool_use"]
        final_text_parts.extend(b.text for b in response.content if b.type == "text")

        # A pause_turn means the model interrupted its OWN turn to let a
        # server-side tool (e.g. web_search) run — it has NOT finished. Feed
        # the partial assistant turn back and continue so it resumes; otherwise
        # the `not tool_uses` check below would treat it as "done" (server
        # tools emit no client-side tool_use block) and cut the turn off before
        # it records the day. This was the root cause of days being backfilled
        # empty by the safety net.
        if response.stop_reason == "pause_turn":
            messages.append({"role": "assistant", "content": response.content})
            continue

        if not tool_uses:
            break  # model is done, no more tools requested

        messages.append({"role": "assistant", "content": response.content})

        tool_results = _dispatch_tool_calls(tool_uses, day, ask_user_handler)
        if tool_results:
            messages.append({"role": "user", "content": tool_results})
        elif response.stop_reason == "end_turn":
            break

    return final_text_parts, api_error


def process_day(
    raw_entry: str,
    day: str | None = None,
    ask_user: bool = True,
    client: Anthropic | None = None,
    ask_user_handler: Callable[[str], str] | None = None,
) -> str:
    """Run one full agent turn for a single day's log entry.

    `ask_user` determines if the agent should prompt the user for additional
    information. `client` and `ask_user_handler` are injectable for testing —
    in normal use they default to a fresh Anthropic client and the terminal
    prompt.
    """
    day = day or date.today().isoformat()
    # Retries are handled by _create_with_retry, so disable the SDK's own retry
    # layer to keep retry policy in one place.
    client = client or Anthropic(max_retries=0)
    ask_user_handler = ask_user_handler or terminal_ask

    messages = [
        {
            "role": "user",
            "content": (
                f"Today's date: {day}\n"
                f"Today's log entry:\n{raw_entry}"
            ),
        }
    ]

    tools, system = _build_tools_and_system(ask_user)

    final_text_parts, api_error = _run_turn(
        client, system, tools, messages, day, ask_user_handler
    )

    # Safety net: never let a day silently drop out of history, even if the
    # model failed to call record_parsed_entry above (including when the API
    # was unreachable).
    _ensure_entry_recorded(day, raw_entry)

    if not final_text_parts and api_error is not None:
        return (
            f"(API error on {day}; raw entry saved to history but not "
            f"analyzed — {type(api_error).__name__})"
        )
    return "\n".join(final_text_parts) if final_text_parts else "(no summary produced)"


def weekly_report() -> str:
    """Render the accumulated flagged patterns as a plain-text report."""
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
