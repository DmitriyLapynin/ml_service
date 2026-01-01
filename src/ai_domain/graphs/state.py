# src/ai_domain/graphs/state.py
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class GraphState:
    trace_id: str
    tenant_id: str
    conversation_id: str
    channel: str

    messages: List[Dict[str, str]]
    route: str | None

    versions: Dict[str, str]
    policies: Dict[str, Any]
    credentials: Dict[str, Any]

    runtime: Dict[str, Any]

    # output
    answer: Dict[str, Any] | None = None
    stage: Dict[str, Any] | None = None