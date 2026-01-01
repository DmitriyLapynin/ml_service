from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RAGAgentState:
    trace_id: str = "rag-agent"
    messages: List[Dict[str, Any]] = field(default_factory=list)
    blocked: bool = False
    safety_reason: Optional[str] = None
    plan: Optional[str] = None
    wants_retrieve: bool = False
    context: Optional[str] = None
    answer: Optional[Dict[str, Any]] = None
    executed: List[str] = field(default_factory=list)
