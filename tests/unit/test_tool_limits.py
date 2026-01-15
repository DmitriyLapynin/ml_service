import asyncio
import json

import pytest

from ai_domain.agent.nodes import agent_node, tool_executor_node
from ai_domain.tools.registry import ToolRegistry, ToolSpec


class InflightCounter:
    def __init__(self) -> None:
        self.inflight = 0
        self.max_inflight = 0
        self._lock = asyncio.Lock()

    async def enter(self) -> None:
        async with self._lock:
            self.inflight += 1
            if self.inflight > self.max_inflight:
                self.max_inflight = self.inflight

    async def exit(self) -> None:
        async with self._lock:
            self.inflight -= 1


def _make_tool_calls(count: int, *, name: str = "slow_tool") -> list[dict]:
    return [
        {"id": f"call_{idx}", "name": name, "args": {"query": f"q{idx}"}}
        for idx in range(count)
    ]


@pytest.mark.asyncio
async def test_max_tool_calls_trims() -> None:
    class FakeLLM:
        async def invoke_tool_calls(self, *args, **kwargs):
            _ = args, kwargs
            return {"content": "", "tool_calls": _make_tool_calls(7)}

    registry = ToolRegistry(max_concurrency_global=50)
    calls_count = 0

    async def fast_tool(args, state):
        _ = args, state
        nonlocal calls_count
        calls_count += 1
        return {"ok": True}

    registry.register(
        ToolSpec(
            name="slow_tool",
            description="test tool",
            schema={"type": "object", "properties": {"query": {"type": "string"}}},
            handler=fast_tool,
        )
    )

    state = {
        "messages": [{"role": "user", "content": "hi"}],
        "llm": FakeLLM(),
        "tools": [{"name": "slow_tool", "description": "test tool", "schema": {}}],
        "is_rag": True,
        "policies": {"max_tool_calls": 5},
        "tool_registry": registry,
    }

    state = await agent_node(state)
    assert len(state["tool_calls"]) == 5
    assert state["wants_retrieve"] is True

    state = await tool_executor_node(state)
    assert len(state["tool_results"]) == 5
    assert len(state["tool_messages"]) == 5
    assert calls_count == 5

    ids = {call["id"] for call in state["tool_calls"]}
    msg_ids = {m["tool_call_id"] for m in state["tool_messages"]}
    assert ids == msg_ids


@pytest.mark.asyncio
async def test_max_tool_concurrency_per_request_limits_parallelism() -> None:
    counter = InflightCounter()
    registry = ToolRegistry(max_concurrency_global=50)

    async def slow_tool(args, state):
        _ = args
        await state["counter"].enter()
        try:
            await asyncio.sleep(0.05)
            return {"ok": True}
        finally:
            await state["counter"].exit()

    registry.register(
        ToolSpec(
            name="slow_tool",
            description="test tool",
            schema={"type": "object", "properties": {"query": {"type": "string"}}},
            handler=slow_tool,
        )
    )

    state = {
        "tool_calls": _make_tool_calls(7),
        "policies": {"max_tool_concurrency_per_request": 3},
        "tool_registry": registry,
        "counter": counter,
    }

    out = await tool_executor_node(state)
    assert counter.max_inflight <= 3
    assert len(out["tool_results"]) == 7
    assert len(out["tool_messages"]) == 7

    ids = {call["id"] for call in state["tool_calls"]}
    msg_ids = {m["tool_call_id"] for m in out["tool_messages"]}
    assert ids == msg_ids


@pytest.mark.asyncio
async def test_max_tool_concurrency_global_limits_parallelism_across_requests() -> None:
    global_counter = InflightCounter()
    registry = ToolRegistry(max_concurrency_global=4)

    async def slow_tool(args, state):
        _ = args
        await state["global_counter"].enter()
        try:
            await asyncio.sleep(0.05)
            return {"ok": True}
        finally:
            await state["global_counter"].exit()

    registry.register(
        ToolSpec(
            name="slow_tool",
            description="test tool",
            schema={"type": "object", "properties": {"query": {"type": "string"}}},
            handler=slow_tool,
        )
    )

    state_a = {
        "tool_calls": _make_tool_calls(6),
        "policies": {"max_tool_concurrency_per_request": 10},
        "tool_registry": registry,
        "global_counter": global_counter,
    }
    state_b = {
        "tool_calls": _make_tool_calls(6),
        "policies": {"max_tool_concurrency_per_request": 10},
        "tool_registry": registry,
        "global_counter": global_counter,
    }

    out_a, out_b = await asyncio.gather(
        tool_executor_node(state_a),
        tool_executor_node(state_b),
    )

    assert global_counter.max_inflight <= 4
    assert len(out_a["tool_results"]) == 6
    assert len(out_b["tool_results"]) == 6


@pytest.mark.asyncio
async def test_per_request_and_global_limits_together() -> None:
    global_counter = InflightCounter()
    registry = ToolRegistry(max_concurrency_global=4)

    async def slow_tool(args, state):
        _ = args
        await state["global_counter"].enter()
        await state["request_counter"].enter()
        try:
            await asyncio.sleep(0.05)
            return {"ok": True}
        finally:
            await state["request_counter"].exit()
            await state["global_counter"].exit()

    registry.register(
        ToolSpec(
            name="slow_tool",
            description="test tool",
            schema={"type": "object", "properties": {"query": {"type": "string"}}},
            handler=slow_tool,
        )
    )

    state_a = {
        "tool_calls": _make_tool_calls(6),
        "policies": {"max_tool_concurrency_per_request": 3},
        "tool_registry": registry,
        "global_counter": global_counter,
        "request_counter": InflightCounter(),
    }
    state_b = {
        "tool_calls": _make_tool_calls(6),
        "policies": {"max_tool_concurrency_per_request": 3},
        "tool_registry": registry,
        "global_counter": global_counter,
        "request_counter": InflightCounter(),
    }

    out_a, out_b = await asyncio.gather(
        tool_executor_node(state_a),
        tool_executor_node(state_b),
    )

    assert global_counter.max_inflight <= 4
    assert state_a["request_counter"].max_inflight <= 3
    assert state_b["request_counter"].max_inflight <= 3
    assert len(out_a["tool_results"]) == 6
    assert len(out_b["tool_results"]) == 6


@pytest.mark.asyncio
async def test_tool_executor_handles_errors() -> None:
    registry = ToolRegistry(max_concurrency_global=10)

    async def ok_tool(args, state):
        _ = args, state
        return {"value": "ok"}

    async def bad_tool(args, state):
        _ = args, state
        raise RuntimeError("boom")

    registry.register(
        ToolSpec(
            name="ok_tool",
            description="ok",
            schema={"type": "object"},
            handler=ok_tool,
        )
    )
    registry.register(
        ToolSpec(
            name="bad_tool",
            description="bad",
            schema={"type": "object"},
            handler=bad_tool,
        )
    )

    state = {
        "tool_calls": [
            {"id": "c1", "name": "ok_tool", "args": {}},
            {"id": "c2", "name": "bad_tool", "args": {}},
            {"id": "c3", "name": "ok_tool", "args": {}},
        ],
        "policies": {"max_tool_concurrency_per_request": 5},
        "tool_registry": registry,
    }

    out = await tool_executor_node(state)
    assert len(out["tool_results"]) == 3
    assert len(out["tool_messages"]) == 3

    failures = [r for r in out["tool_results"] if not r.ok]
    assert len(failures) == 1
    assert len([r for r in out["tool_results"] if r.ok]) == 2

    error_msgs = []
    for msg in out["tool_messages"]:
        payload = json.loads(msg["content"])
        if payload.get("ok") is False:
            error_msgs.append(payload)
    assert len(error_msgs) == 1
    assert error_msgs[0]["error"] is not None

    bad_msg = next(m for m in out["tool_messages"] if m["tool_call_id"] == "c2")
    bad_payload = json.loads(bad_msg["content"])
    assert bad_payload["ok"] is False
