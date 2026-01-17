import pytest

from ai_domain.agent.nodes.final_answer import generate_node


class CaptureLLM:
    def __init__(self) -> None:
        self.calls = []

    async def invoke_text(self, messages, *, config, context=None):
        _ = config, context
        self.calls.append(messages)
        return "ok"


@pytest.mark.asyncio
async def test_agent_generate_trims_messages_with_buffer_strategy():
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
