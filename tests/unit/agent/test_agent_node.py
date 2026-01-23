import pytest

from ai_domain.agent.nodes import agent_node, create_agent_prompt


class FakeToolResponse:
    def __init__(self):
        self.content = ""
        self.tool_calls = [{"name": "knowledge_search", "args": {"query": "hi"}, "id": "call-1"}]


class FakeLLMClient:
    async def invoke_tool_calls(self, messages, tools, config, context=None):  # noqa: ARG002
        _ = (messages, tools, config, context)
        return FakeToolResponse()


def _make_state(tools=None, is_rag=True):
    return {
        "messages": [{"role": "user", "content": "привет"}],
        "tools": tools or [],
        "is_rag": is_rag,
    }


def test_create_agent_prompt_includes_tools():
    prompt = create_agent_prompt([{"name": "knowledge_search", "description": "search"}], is_rag=True)
    assert "knowledge_search" in prompt


@pytest.mark.asyncio
async def test_agent_node_filters_tools_when_rag_disabled():
    state = _make_state(
        tools=[
            {"name": "knowledge_search", "description": "doc"},
            {"name": "calendar", "description": "schedule"},
        ],
        is_rag=False,
    )

    out = await agent_node(state)
    assert "knowledge_search" not in [t["name"] for t in out["filtered_tools"]]
    assert out["wants_retrieve"] is False


@pytest.mark.asyncio
async def test_agent_node_builds_prompt_based_on_tools():
    state = _make_state(
        tools=[{"name": "knowledge_search", "description": "doc"}],
        is_rag=True,
    )
    out = await agent_node(state)
    assert out["wants_retrieve"] is True
    assert "knowledge_search" in out["agent_prompt"]


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
