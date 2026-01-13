from __future__ import annotations

import logging
from dataclasses import dataclass, field
import inspect
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional
from uuid import uuid4


ToolHandler = Callable[[Dict[str, Any]], Dict[str, Any] | Awaitable[Dict[str, Any]]]


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
    def __init__(self):
        self._tools: Dict[str, ToolSpec] = {}

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
        trace_id: str | None = None,
        call_id: str | None = None,
    ) -> ToolResult:
        call_id = call_id or uuid4().hex[:12]
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

        logging.info(f"tool_call_start name={name} call_id={call_id} trace_id={trace_id}")
        start = time.perf_counter()
        try:
            result = tool.handler(args)
            if inspect.isawaitable(result):
                result = await result
        except Exception as exc:
            latency_ms = int((time.perf_counter() - start) * 1000)
            logging.info(f"tool_call_error name={name} call_id={call_id} trace_id={trace_id}")
            return ToolResult(
                tool_name=name,
                call_id=call_id,
                ok=False,
                result=None,
                error={"code": "tool_error", "message": str(exc)},
                latency_ms=latency_ms,
            )

        latency_ms = int((time.perf_counter() - start) * 1000)
        logging.info(f"tool_call_success name={name} call_id={call_id} trace_id={trace_id}")
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


def knowledge_search_stub(args: Dict[str, Any]) -> Dict[str, Any]:
    logging.info(f"knowledge_search called with args={list(args.keys())}")
    return {"status": "ok", "documents": []}


def default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="knowledge_search",
            description="Поиск фактов и документов в базе знаний (заглушка).",
            schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            handler=knowledge_search_stub,
        )
    )
    return registry
