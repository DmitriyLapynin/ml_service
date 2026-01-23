import json
import logging
import time
from typing import List

from ai_domain.llm.client import LLMConfig
from ai_domain.llm.types import LLMCallContext
from ai_domain.tools.registry import ToolSpec, default_registry
from ai_domain.utils.memory import select_memory_messages

from .prompts import create_agent_prompt, create_agent_prompt_short
from ai_domain.llm.metrics import StateMetricsWriter
from .utils import ensure_lists, log_node, mark_runtime_error, step_begin, step_end


def _format_tools(tools: List[dict], is_rag: bool) -> List[dict]:
    return [tool for tool in tools if is_rag or tool.get("name") != "knowledge_search"]


def _normalize_tools(tools: List[object]) -> List[dict]:
    normalized: List[dict] = []
    for tool in tools:
        if isinstance(tool, ToolSpec):
            normalized.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "schema": tool.schema,
                }
            )
        else:
            normalized.append(tool)
    return normalized


async def agent_node(state: dict) -> dict:
    ensure_lists(state)
    log_node(state, "agent")
    step_index = step_begin(state, "agent")
    start = time.perf_counter()
    state["executed"].append("agent")
    tools = state.get("tools")
    if tools is None:
        tools = default_registry().list()
    tools = _normalize_tools(list(tools))
    is_rag = bool(state.get("is_rag", True))
    logging.info(
        json.dumps(
            {
                "event": "agent_node",
                "trace_id": state.get("trace_id"),
                "node": "agent",
                "rag": is_rag,
                "tools_count": len(tools),
            },
            ensure_ascii=False,
        )
    )
    llm = state.get("llm")
    use_tool_calls = hasattr(llm, "invoke_tool_calls")
    if not use_tool_calls:
        mark_runtime_error(
            state,
            code="tool_calls_unsupported",
            message="LLM does not support native tool calling",
            node="agent_node",
            retryable=False,
        )
        state["tool_calls"] = []
        state["wants_retrieve"] = False
        latency_ms = int((time.perf_counter() - start) * 1000)
        step_end(
            state,
            index=step_index,
            latency_ms=latency_ms,
            status="error",
            reason="tool_calls_unsupported",
        )
        return state

    state["agent_prompt"] = create_agent_prompt_short(is_rag=is_rag)
    state["filtered_tools"] = _format_tools(tools, is_rag)
    state["tool_calls"] = []
    state["wants_retrieve"] = False
    state.setdefault("trace", {}).setdefault("agent", {})

    if use_tool_calls and state["filtered_tools"]:
        history = select_memory_messages(state)
        trace_id = state.get("trace_id")
        call_context = (
            LLMCallContext(
                trace_id=trace_id,
                graph=state.get("graph") or state.get("graph_name"),
                node="agent_node",
                task="agent_decision",
                channel=state.get("channel"),
                tenant_id=state.get("tenant_id"),
                request_id=state.get("request_id"),
                metrics=state.get("metrics_writer") or StateMetricsWriter(state),
            )
            if trace_id
            else None
        )
        metadata = {}
        tool_choice = state.get("tool_choice")
        if tool_choice:
            metadata["tool_choice"] = tool_choice
        response = await llm.invoke_tool_calls(
            [{"role": "system", "content": state["agent_prompt"]}, *history],
            tools=state["filtered_tools"],
            config=LLMConfig(
                model="gpt-4.1-mini",
                max_tokens=1024,
                temperature=0.2,
                metadata=metadata,
            ),
            context=call_context,
        )
        tool_calls = list((response or {}).get("tool_calls") or [])
        max_calls = int((state.get("policies") or {}).get("max_tool_calls", 5))
        if max_calls > 0:
            tool_calls = tool_calls[:max_calls]
        state["tool_calls"] = tool_calls
        state["wants_retrieve"] = bool(tool_calls)
        if tool_calls:
            state["messages"] = [
                *state.get("messages", []),
                {
                    "role": "assistant",
                    "content": (response or {}).get("content") or "",
                    "tool_calls": tool_calls,
                },
            ]
    state["trace"]["agent"]["decision"] = "tools" if state.get("tool_calls") else "no_tools"
    latency_ms = int((time.perf_counter() - start) * 1000)
    step_end(state, index=step_index, latency_ms=latency_ms, status="ok")
    return state
