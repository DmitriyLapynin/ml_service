import pytest

from ai_domain.graphs.state import GraphState
from ai_domain.utils.memory import select_memory_messages


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
