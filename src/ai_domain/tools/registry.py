from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
import inspect
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional
from uuid import uuid4


ToolHandler = Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any] | Awaitable[Dict[str, Any]]]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    schema: Dict[str, Any] = field(default_factory=dict)
    handler: Optional[ToolHandler] = None


@dataclass(frozen=True)
class ToolResult:
    tool_name: str
    call_id: str
    ok: bool
    result: Dict[str, Any] | None
    error: Dict[str, Any] | None
    latency_ms: int


class ToolRegistry:
    def __init__(self, *, max_concurrency_global: int = 20):
        self._tools: Dict[str, ToolSpec] = {}
        self._global_limiter = asyncio.Semaphore(max_concurrency_global) if max_concurrency_global > 0 else None

    def register(self, tool: ToolSpec) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._tools.get(name)

    def list(self) -> List[ToolSpec]:
        return list(self._tools.values())

    async def execute(
        self,
        name: str,
        args: Dict[str, Any],
        *,
        state: Dict[str, Any] | None = None,
        trace_id: str | None = None,
        call_id: str | None = None,
    ) -> ToolResult:
        call_id = call_id or uuid4().hex[:12]
        if state is None:
            state = {}
        tool = self.get(name)
        if not tool:
            return ToolResult(
                tool_name=name,
                call_id=call_id,
                ok=False,
                result=None,
                error={"code": "tool_not_found", "message": "Tool not registered"},
                latency_ms=0,
            )
        if not tool.handler:
            return ToolResult(
                tool_name=name,
                call_id=call_id,
                ok=False,
                result=None,
                error={"code": "tool_not_implemented", "message": "Tool handler is missing"},
                latency_ms=0,
            )

        validation_error = _validate_args(tool.schema, args)
        if validation_error:
            return ToolResult(
                tool_name=name,
                call_id=call_id,
                ok=False,
                result=None,
                error={"code": "invalid_args", "message": validation_error},
                latency_ms=0,
            )

        logging.info(
            "tool_call_start",
            extra={"tool": name, "call_id": call_id, "trace_id": trace_id},
        )
        start = time.perf_counter()
        try:
            if self._global_limiter is None:
                result = tool.handler(args, state)
                if inspect.isawaitable(result):
                    result = await result
            else:
                async with self._global_limiter:
                    result = tool.handler(args, state)
                    if inspect.isawaitable(result):
                        result = await result
        except Exception as exc:
            latency_ms = int((time.perf_counter() - start) * 1000)
            logging.error(
                "tool_call_error",
                extra={"tool": name, "call_id": call_id, "trace_id": trace_id},
            )
            return ToolResult(
                tool_name=name,
                call_id=call_id,
                ok=False,
                result=None,
                error={"code": "tool_error", "message": str(exc)},
                latency_ms=latency_ms,
            )

        latency_ms = int((time.perf_counter() - start) * 1000)
        logging.info(
            "tool_call_success",
            extra={"tool": name, "call_id": call_id, "trace_id": trace_id},
        )
        return ToolResult(
            tool_name=name,
            call_id=call_id,
            ok=True,
            result=result,
            error=None,
            latency_ms=latency_ms,
        )


def _validate_args(schema: Dict[str, Any], args: Dict[str, Any]) -> str | None:
    if not schema:
        return None
    if schema.get("type") and schema.get("type") != "object":
        return "schema_type_not_object"
    if not isinstance(args, dict):
        return "args_not_object"

    required = schema.get("required") or []
    for key in required:
        if key not in args:
            return f"missing_required:{key}"

    properties = schema.get("properties") or {}
    for key, spec in properties.items():
        if key not in args:
            continue
        expected = spec.get("type")
        if expected and not _matches_type(args[key], expected):
            return f"invalid_type:{key}"
    return None


def _matches_type(value: Any, expected: str) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    return True


async def knowledge_search_handler(args: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    query = (args.get("query") or "").strip()
    top_k = int(args.get("top_k") or 5)
    top_k_per_doc = int(args.get("top_k_per_doc") or top_k)
    funnel_id = state.get("funnel_id")
    resolver = state.get("kb_resolver")
    if resolver and funnel_id:
        results = await resolver.search(
            funnel_id=funnel_id,
            query=query,
            top_k=top_k,
            top_k_per_doc=top_k_per_doc,
        )
        return {"status": "ok", "documents": results}

    kb = state.get("kb_client")
    if kb is None:
        return {"status": "error", "code": "kb_missing", "message": "kb_client is not configured"}
    results = await kb.search(query=query, top_k=top_k)
    return {"status": "ok", "documents": results}


def _schema_type_to_python(expected: str) -> type:
    if expected == "string":
        return str
    if expected == "number":
        return float
    if expected == "integer":
        return int
    if expected == "boolean":
        return bool
    if expected == "array":
        return list
    if expected == "object":
        return dict
    return Any


def to_langchain_tool(tool: ToolSpec) -> object:
    try:
        from langchain.tools import tool as lc_tool  # type: ignore
        from pydantic import Field, create_model
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("langchain is required for tool calling") from exc

    properties = tool.schema.get("properties") or {}
    required = set(tool.schema.get("required") or [])
    fields: Dict[str, tuple[type, Any]] = {}
    for key, spec in properties.items():
        py_type = _schema_type_to_python(spec.get("type", "string"))
        default = ... if key in required else None
        fields[key] = (py_type, Field(default=default, description=spec.get("description") or ""))

    args_schema = create_model(f"{tool.name}_args", **fields)

    @lc_tool(tool.name, args_schema=args_schema, description=tool.description)
    async def _tool_stub(**kwargs):  # type: ignore[no-redef]
        """Tool stub for tool-calling."""
        _ = kwargs
        return ""
    return _tool_stub


def default_registry(*, max_concurrency_global: int = 20) -> ToolRegistry:
    registry = ToolRegistry(max_concurrency_global=max_concurrency_global)
    registry.register(
        ToolSpec(
            name="knowledge_search",
            description="Вызови этот инструмент всегда, когда пользователь задает вопрос",
            schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer"},
                    "top_k_per_doc": {"type": "integer"},
                },
                "required": ["query"],
            },
            handler=knowledge_search_handler,
        )
    )
    return registry
