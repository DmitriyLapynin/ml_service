import pytest

from ai_domain.agent.nodes import agent_node


class FakeToolResponse:
    def __init__(self):
        self.content = ""
        self.tool_calls = [{"name": "knowledge_search", "args": {"query": "hi"}, "id": "call-1"}]


class FakeLLMClient:
    async def invoke_tool_calls(self, messages, tools, config, context=None):  # noqa: ARG002
        _ = (messages, tools, config, context)
        return FakeToolResponse()


@pytest.mark.asyncio
async def test_agent_node_appends_llm_response():
    state = {
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"name": "knowledge_search", "description": "search"}],
        "is_rag": True,
        "llm": FakeLLMClient(),
    }

    out = await agent_node(state)

    assert out["messages"][-1]["role"] == "assistant"
    assert out["messages"][-1]["tool_calls"][0]["name"] == "knowledge_search"
    assert out["wants_retrieve"] is True
