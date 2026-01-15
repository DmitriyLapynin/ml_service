from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, Literal, Optional, Sequence, TypeVar

Role = Literal["system", "user", "assistant", "tool"]
FinishReason = Optional[str]

T = TypeVar("T")

@dataclass(frozen=True)
class LLMCapabilities:
    supports_structured: bool = True
    supports_tool_calls: bool = True
    supports_seed: bool = True

@dataclass(frozen=True)
class LLMMessage:
    role: Role
    content: str

@dataclass(frozen=True)
class LLMUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated: bool = False

@dataclass(frozen=True)
class LLMResponse:
    content: str
    model: str
    provider: str
    usage: LLMUsage
    latency_ms: int
    finish_reason: FinishReason = None
    raw: Optional[Dict[str, Any]] = None  # НО: raw не должен содержать секреты


@dataclass
class LLMCredentials:
    openai_api_key: Optional[str] = None  # BYOK


@dataclass(frozen=True)
class LLMRequest:
    messages: Sequence[LLMMessage]
    model: str
    temperature: Optional[float] = 0.2
    max_output_tokens: int = 2048
    top_p: Optional[float] = None
    seed: Optional[int] = None
    stop: Optional[Sequence[str]] = None
    credentials: Optional[LLMCredentials] = None
    metadata: Dict[str, Any] = field(default_factory=dict)  # trace_id, task_name, etc.


@dataclass(frozen=True)
class StructuredResult(Generic[T]):
    parsed: T | None
    raw: Any | None
    parsing_error: str | None


@dataclass(frozen=True)
class LLMCallContext:
    trace_id: str
    graph: str | None = None
    node: str | None = None
    task: str | None = None
    channel: str | None = None
    tenant_id: str | None = None
    request_id: str | None = None
    metrics: list[dict] | None = None
