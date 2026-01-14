from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


Role = Literal["system", "user", "assistant", "tool"]


class ChatMessage(BaseModel):
    role: Role
    content: str


class ToolDescription(BaseModel):
    name: str = Field(..., description="Tool name / identifier")
    description: str = Field(default="", description="Human readable description")
    schema: Dict[str, Any] = Field(default_factory=dict, description="JSON schema / args schema")


class ModelConfig(BaseModel):
    name: str = Field(default="gpt-4.1-mini", description="Model name (e.g. gpt-4.1-mini)")
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Model/runtime params (temperature, top_p, max_output_tokens, provider, seed, etc.)",
    )


class ChatRequest(BaseModel):
    """
    Модель запроса для чата. Содержит полную историю сообщений.
    Последнее сообщение в списке — текущее от пользователя.
    """

    messages: List[ChatMessage] = Field(
        ...,
        description="Полный список сообщений диалога. Последнее сообщение всегда от 'user'.",
    )
    prompt: Optional[str] = Field(
        default=None,
        description="Инструкция для ассистента",
        examples=["Пожалуйста, ответь на вопрос пользователя максимально подробно."],
    )
    role_instruction: Optional[str] = Field(
        default=None,
        description="Роль для ассистента",
    )
    is_rag: bool = Field(
        default=False,
        description="Использовать ли RAG (Retrieval-Augmented Generation) для ответа.",
        examples=[True],
    )
    tools: Optional[List[ToolDescription]] = Field(
        default=None,
        description="Список описаний инструментов для их создания.",
        examples=[[]],
    )
    crypted_api_key: Optional[str] = Field(
        default=None,
        description="Зашифрованный API ключ (опционально).",
    )
    funnel_id: Optional[str] = Field(
        default="1",
        description="Id воронки, если используется RAG.",
        examples=["1"],
    )

    memory_strategy: Optional[Literal["buffer", "summary"]] = Field(
        default="buffer",
        description="Стратегия управления памятью диалога.",
        examples=["buffer"],
    )
    model: ModelConfig = Field(
        default_factory=lambda: ModelConfig(name="gpt-4.1-mini"),
        description="Конфигурация языковой модели для обработки запроса.",
    )
    memory_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Словарь с параметрами для настройки памяти.",
        examples=[{"k": 3}],
    )


class ChatResponse(BaseModel):
    status: Literal["ok", "degraded", "error"]
    answer: Dict[str, Any] | None = None
    stage: Dict[str, Any] | None = None
    trace_id: str
    versions: Dict[str, Any] | None = None


class ErrorResponse(BaseModel):
    error: str
    trace_id: Optional[str] = None


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


def chat_request_to_orchestrator_request(
    *,
    chat: ChatRequest,
    tenant_id: str,
    conversation_id: str,
    channel: str = "chat",
    idempotency_key: str | None = None,
    credentials: dict | None = None,
    funnel_id: str | None = None,
    trace_id: str | None = None,
) -> Dict[str, Any]:
    """
    Адаптер: внешний ChatRequest -> внутренний request для Orchestrator.run().
    """
    return {
        "tenant_id": tenant_id,
        "conversation_id": conversation_id,
        "channel": channel,
        "messages": [m.model_dump() for m in chat.messages],
        "idempotency_key": idempotency_key,
        "credentials": credentials,
        # extra fields, understood by orchestrator/context_builder/nodes
        "prompt": chat.prompt,
        "role_instruction": chat.role_instruction,
        "is_rag": chat.is_rag,
        "tools": [t.model_dump() for t in chat.tools] if chat.tools else None,
        "funnel_id": funnel_id or chat.funnel_id,
        "memory_strategy": chat.memory_strategy,
        "memory_params": chat.memory_params,
        "model": chat.model.name,
        "model_params": chat.model.params,
        "trace_id": trace_id,
    }
