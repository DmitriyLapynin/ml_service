# src/ai_domain/graphs/state.py
from dataclasses import dataclass
from dataclasses import field
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

    # request hints (optional, from API layer)
    prompt: str | None = None
    role_instruction: str | None = None
    is_rag: bool | None = None
    tools: List[Dict[str, Any]] | None = None
    funnel_id: str | None = None
    memory_strategy: str | None = None
    memory_params: Dict[str, Any] = field(default_factory=dict)
    model_params: Dict[str, Any] = field(default_factory=dict)

    # output
    answer: Dict[str, Any] | None = None
    stage: Dict[str, Any] | None = None
