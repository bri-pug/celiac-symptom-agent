# pylint: disable=missing-function-docstring,redefined-outer-name,unused-argument,too-few-public-methods
"""
Tests for the agent orchestration loop itself (`process_day` / `_run_turn`).

These exercise the multi-step tool-use loop without touching the network: a
scripted fake Anthropic client returns a canned sequence of responses, so we
can assert the loop drives tools, threads messages back, honours stop reasons,
rejects malformed tool calls, and respects its iteration cap — the behaviour
that used to be checked only by manually eyeballing a live transcript.
"""

from types import SimpleNamespace

import httpx
import pytest
from anthropic import APIError

from src import agent, state_store


# --- fake response construction --------------------------------------------

def _usage():
    return SimpleNamespace(
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
        input_tokens=0,
        output_tokens=0,
    )


def _text(text):
    return SimpleNamespace(type="text", text=text)


def _tool_use(name, tool_input, id_="t1"):
    return SimpleNamespace(type="tool_use", name=name, input=tool_input, id=id_)


def _response(content, stop_reason="end_turn"):
    return SimpleNamespace(content=content, stop_reason=stop_reason, usage=_usage())


class ScriptedMessages:
    """messages.create() that returns a pre-scripted response per call."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        assert self.responses, "create() called more times than scripted"
        return self.responses.pop(0)


class ScriptedClient:
    def __init__(self, responses):
        self.messages = ScriptedMessages(responses)


class AlwaysToolClient:
    """messages.create() that never stops asking for a tool — used to prove
    the loop's hard iteration cap actually terminates it."""

    def __init__(self):
        self.calls = []

        client = self

        class _Msgs:
            def create(self, **kwargs):
                client.calls.append(kwargs)
                return _response(
                    [_tool_use("record_parsed_entry", {"day": "2026-07-01"})],
                    stop_reason="tool_use",
                )

        self.messages = _Msgs()


@pytest.fixture
def state_path(tmp_path, monkeypatch):
    monkeypatch.setattr(state_store, "STATE_PATH", str(tmp_path / "state.json"))
    return tmp_path / "state.json"


# --- tests ------------------------------------------------------------------

def test_loop_dispatches_tools_then_feeds_results_and_finishes(state_path):
    """A tool_use turn is executed, its result threaded back into messages,
    and the following end_turn text is returned as the summary."""
    client = ScriptedClient([
        _response(
            [_tool_use("record_parsed_entry",
                       {"day": "2026-07-01", "foods": ["oats"]})],
            stop_reason="tool_use",
        ),
        _response([_text("Recorded today. Nothing new to flag.")]),
    ])

    summary = agent.process_day(
        "Ate oats.", day="2026-07-01", ask_user=False, client=client,
    )

    assert "Nothing new to flag" in summary
    assert len(client.messages.calls) == 2  # tool round-trip, then finish

    # The second call must include the assistant tool_use turn AND the
    # tool_result fed back to it.
    second_messages = client.messages.calls[1]["messages"]
    assert second_messages[-2]["role"] == "assistant"
    fed_back = second_messages[-1]
    assert fed_back["role"] == "user"
    assert fed_back["content"][0]["type"] == "tool_result"

    # And the tool actually ran: the entry is now in state.
    state = state_store.load_state()
    assert len(state.entries) == 1
    assert state.entries[0].foods == ["oats"]


def test_loop_respects_iteration_cap(state_path):
    """A model that asks for a tool forever is stopped at the hard cap rather
    than looping without bound."""
    client = AlwaysToolClient()

    agent.process_day("Ate oats.", day="2026-07-01", ask_user=False, client=client)

    assert len(client.calls) == agent.MAX_LOOP_ITERATIONS


def test_loop_resumes_after_pause_turn(state_path):
    """pause_turn (a server-side tool ran) must NOT end the turn: the loop
    feeds the partial turn back and continues to record the day."""
    client = ScriptedClient([
        _response([_text("Let me check whether soy sauce has gluten.")],
                  stop_reason="pause_turn"),
        _response(
            [_tool_use("record_parsed_entry",
                       {"day": "2026-07-01", "foods": ["soy sauce"]})],
            stop_reason="tool_use",
        ),
        _response([_text("Recorded.")]),
    ])

    agent.process_day("Ate soy sauce.", day="2026-07-01", ask_user=False, client=client)

    assert len(client.messages.calls) == 3  # did not stop at pause_turn
    state = state_store.load_state()
    assert len(state.entries) == 1
    assert state.entries[0].foods == ["soy sauce"]


def test_loop_rejects_tool_call_missing_required_field(state_path):
    """A flag_pattern call missing required fields is rejected at the boundary
    with an is_error result — run_tool never runs, no pattern is written."""
    client = ScriptedClient([
        _response(
            # flag_pattern requires evidence_days + confidence; omit both.
            [_tool_use("flag_pattern", {"hypothesis": "oats may be linked to bloating"})],
            stop_reason="tool_use",
        ),
        _response([_text("Could not flag yet.")]),
    ])

    agent.process_day("Ate oats.", day="2026-07-01", ask_user=False, client=client)

    # The error was handed back to the model...
    fed_back = client.messages.calls[1]["messages"][-1]["content"][0]
    assert fed_back["type"] == "tool_result"
    assert fed_back["is_error"] is True
    assert "evidence_days" in fed_back["content"]
    # ...and nothing was persisted.
    assert state_store.load_state().flagged_patterns == []


def test_ask_user_handler_is_invoked_and_answer_fed_back(state_path):
    """When the model calls ask_user, the injected handler is asked the
    question and its answer is threaded back as the tool result."""
    asked = {}

    def fake_handler(question):
        asked["question"] = question
        return "It was a regular gluten bagel."

    client = ScriptedClient([
        _response([_tool_use("ask_user", {"question": "Was the bagel gluten-free?"})],
                  stop_reason="tool_use"),
        _response([_text("Thanks, recorded.")]),
    ])

    agent.process_day(
        "Ate a bagel.", day="2026-07-01", ask_user=True,
        client=client, ask_user_handler=fake_handler,
    )

    assert asked["question"] == "Was the bagel gluten-free?"
    fed_back = client.messages.calls[1]["messages"][-1]["content"][0]
    assert fed_back["content"] == "It was a regular gluten bagel."


def test_api_error_returns_message_and_safety_net_still_records(state_path):
    """If the API gives up, process_day returns a graceful message and the raw
    entry is still preserved in history by the safety net."""
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")

    class ErrorMessages:
        def create(self, **kwargs):
            raise APIError("boom", request=req, body=None)

    client = SimpleNamespace(messages=ErrorMessages())

    summary = agent.process_day(
        "Ate a mystery bagel.", day="2026-07-01", ask_user=False, client=client,
    )

    assert "API error" in summary
    state = state_store.load_state()
    assert len(state.entries) == 1
    assert state.entries[0].raw_text == "Ate a mystery bagel."
    assert state.entries[0].foods == []  # minimal fallback
