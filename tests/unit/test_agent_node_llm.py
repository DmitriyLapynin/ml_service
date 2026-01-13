import pytest

from ai_domain.agent.nodes import agent_node


class FakeLLMClient:
    async def invoke_text(self, messages, config, context=None):  # noqa: ARG002
        _ = (messages, config, context)
        return "ok-from-llm"


@pytest.mark.asyncio
async def test_agent_node_appends_llm_response():
    state = {
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [],
        "is_rag": False,
        "llm": FakeLLMClient(),
    }

    out = await agent_node(state)

    assert out["messages"][-1]["role"] == "assistant"
    assert out["messages"][-1]["content"] == "ok-from-llm"
