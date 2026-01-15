# src/ai_domain/orchestrator/context_builder.py
from typing import Dict, Any, List
from uuid import uuid4

from ai_domain.graphs.state import GraphState


def normalize_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    normalized = []
    for m in messages:
        if "role" not in m or "content" not in m:
            raise ValueError("Each message must have role and content")
        normalized.append(
            {
                "role": m["role"],
                "content": m["content"] or "",
            }
        )
    return normalized


def build_graph_state(
    *,
    tenant_id: str,
    conversation_id: str,
    channel: str,
    messages: List[Dict[str, str]],
    versions: Dict[str, str],
    policies: Dict[str, Any],
    credentials: Dict[str, Any] | None,
    task_configs: Dict[str, Any] | None = None,
    prompt: str | None = None,
    role_instruction: str | None = None,
    is_rag: bool | None = None,
    tools: List[Dict[str, Any]] | None = None,
    funnel_id: str | None = None,
    request_id: str | None = None,
    memory_strategy: str | None = None,
    memory_params: Dict[str, Any] | None = None,
    model_params: Dict[str, Any] | None = None,
    trace_id: str | None = None,
    graph_name: str | None = None,
) -> GraphState:
    return GraphState(
        trace_id=trace_id or str(uuid4()),
        tenant_id=tenant_id,
        conversation_id=conversation_id,
        channel=channel,
        messages=normalize_messages(messages),
        route=None,
        versions=versions,
        policies=policies,
        credentials=dict(credentials) if credentials else {},
        task_configs=task_configs or {},
        prompt=prompt,
        role_instruction=role_instruction,
        is_rag=bool(is_rag) if is_rag is not None else bool(policies.get("rag_enabled", False)),
        tools=tools or [],
        funnel_id=funnel_id,
        request_id=request_id,
        memory_strategy=memory_strategy,
        memory_params=memory_params or {},
        model_params=model_params or {},
        graph_name=graph_name,
        runtime={
            "degraded": False,
            "errors": [],
        },
    )
