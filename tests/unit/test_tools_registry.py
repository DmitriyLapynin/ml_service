import pytest

from ai_domain.tools.registry import ToolRegistry, ToolSpec


@pytest.mark.asyncio
async def test_tool_registry_execute_async_handler():
    async def handler(args, state):  # noqa: ARG001
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

    res = await registry.execute("knowledge_search", {"query": "hi"}, state={})
    assert res.ok is True
    assert res.result["echo"] == "hi"


@pytest.mark.asyncio
async def test_tool_registry_execute_sync_handler():
    def handler(args, state):  # noqa: ARG001
        return {"ok": True, "value": args["x"]}

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="sync_tool",
            description="stub",
            schema={"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]},
            handler=handler,
        )
    )

    res = await registry.execute("sync_tool", {"x": 3}, state={})
    assert res.ok is True
    assert res.result["value"] == 3


@pytest.mark.asyncio
async def test_tool_registry_invalid_args():
    async def handler(args, state):  # noqa: ARG001
        return {"ok": True}

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="tool",
            description="stub",
            schema={"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
            handler=handler,
        )
    )

    res = await registry.execute("tool", {"q": 1}, state={})
    assert res.ok is False
    assert res.error["code"] == "invalid_args"


@pytest.mark.asyncio
async def test_tool_registry_missing_handler():
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="tool",
            description="stub",
            schema={"type": "object"},
            handler=None,
        )
    )

    res = await registry.execute("tool", {"x": 1}, state={})
    assert res.ok is False
    assert res.error["code"] == "tool_not_implemented"


def test_tool_registry_duplicate_register():
    registry = ToolRegistry()
    registry.register(ToolSpec(name="tool", description="stub"))
    with pytest.raises(ValueError):
        registry.register(ToolSpec(name="tool", description="dup"))
