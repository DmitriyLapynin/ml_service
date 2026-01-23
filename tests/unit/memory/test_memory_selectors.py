import inspect

import pytest

from ai_domain.agent.nodes.final_answer import generate_node
from ai_domain.graphs.state import GraphState
from ai_domain.utils.memory import select_memory_messages


class CaptureLLM:
    def __init__(self) -> None:
        self.calls = []

    async def invoke_text(self, messages, *, config, context=None):
        _ = config, context
        self.calls.append(messages)
        return "ok"


def _make_state():
    return GraphState(
        trace_id="trace",
        tenant_id="tenant",
        conversation_id="conv",
        channel="chat",
        messages=[
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "latest"},
        ],
        route=None,
        versions={"system_prompt": "v1", "analysis_prompt": "v1", "model": "gpt"},
        policies={"max_output_tokens": 64, "temperature": 0.2},
        credentials={},
        runtime={"degraded": False, "errors": [], "executed": []},
    )


def test_select_memory_messages_defaults_all():
    state = _make_state()
    assert select_memory_messages(state) == state.messages


def test_select_memory_messages_buffer_k():
    state = _make_state()
    state.memory_strategy = "buffer"
    state.memory_params = {"k": 1}

    assert select_memory_messages(state) == [state.messages[-1]]


def test_select_memory_messages_buffer_invalid_k():
    state = _make_state()
    state.memory_strategy = "buffer"
    state.memory_params = {"k": 0}

    assert select_memory_messages(state) == state.messages


def test_select_memory_messages_summary_includes_summary_and_recent():
    state = _make_state()
    state.memory_strategy = "summary"
    state.memory_params = {"k": 1}
    state.memory.summary = "previous summary"

    selected = select_memory_messages(state)

    assert selected[0]["role"] == "system"
    assert "previous summary" in selected[0]["content"]
    assert selected[-1]["content"] == "latest"


def test_select_memory_messages_token_limit_optional():
    if "max_tokens" not in inspect.getsource(select_memory_messages):
        pytest.skip("token-based trimming is not implemented")

    state = _make_state()
    state.memory_strategy = "buffer"
    state.memory_params = {"max_tokens": 1}

    selected = select_memory_messages(state)
    assert selected == [state.messages[-1]]


@pytest.mark.asyncio
async def test_generate_node_trims_messages_with_buffer_strategy():
    llm = CaptureLLM()
    state = {
        "messages": [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "last"},
        ],
        "memory_strategy": "buffer",
        "memory_params": {"k": 1},
        "llm": llm,
        "is_rag": False,
        "model": "gpt-4.1-mini",
        "model_params": {},
        "trace_id": "trace",
        "graph": "agent_graph",
        "channel": "chat",
        "tenant_id": "tenant",
        "request_id": "req",
    }

    await generate_node(state)
    assert llm.calls
    sent = llm.calls[-1]
    non_system = [m for m in sent if m.get("role") != "system"]
    assert [m.get("content") for m in non_system] == ["last"]
