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
    trace_id: str | None = None,
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
        credentials=credentials or {},
        runtime={
            "degraded": False,
            "errors": [],
        },
    )
