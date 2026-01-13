import json
import pytest

from ai_domain.agent.nodes import generate_node, tool_executor_node
from ai_domain.tools.registry import ToolRegistry, ToolSpec


@pytest.mark.asyncio
async def test_tool_executor_node_executes_calls():
    async def handler(args):
        return {"echo": args["query"]}

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="knowledge_search",
            description="stub",
            schema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            handler=handler,
        )
    )

    state = {
        "tool_calls": [{"name": "knowledge_search", "args": {"query": "hi"}, "id": "call-1"}],
        "tool_registry": registry,
    }

    out = await tool_executor_node(state)

    assert out["tool_results"][0].ok is True
    msg = out["tool_messages"][0]
    assert msg["role"] == "tool"
    assert msg["tool_call_id"] == "call-1"
    payload = json.loads(msg["content"])
    assert payload["echo"] == "hi"


@pytest.mark.asyncio
async def test_native_loop_messages_include_tool_calls_and_results():
    class FakeLLMClient:
        def __init__(self):
            self.last_messages = None

        async def invoke_tool_response(self, messages, config, context=None):  # noqa: ARG002
            self.last_messages = messages
            return "ok-final"

    async def handler(args):
        return {"echo": args["query"]}

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="knowledge_search",
            description="stub",
            schema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            handler=handler,
        )
    )

    tool_calls = [{"name": "knowledge_search", "args": {"query": "hi"}, "id": "call-1"}]
    state = {
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "tool_calls": tool_calls},
        ],
        "tool_calls": tool_calls,
        "tool_registry": registry,
        "is_rag": True,
        "model": "gpt-4.1-mini",
        "llm": FakeLLMClient(),
    }

    state = await tool_executor_node(state)
    out = await generate_node(state)

    assert out["answer"]["text"] == "ok-final"
    sent = state["llm"].last_messages
    assert sent is not None
    assistant_idx = next(i for i, m in enumerate(sent) if m.get("role") == "assistant")
    tool_idx = next(i for i, m in enumerate(sent) if m.get("role") == "tool")
    assert assistant_idx < tool_idx
    assert sent[assistant_idx].get("tool_calls")[0]["id"] == "call-1"
    assert sent[tool_idx].get("tool_call_id") == "call-1"
