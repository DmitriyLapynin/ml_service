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
    tenant_id: Optional[str] = Field(
        default=None,
        description="Tenant identifier (optional, defaults to default).",
        examples=["default"],
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
    meta: Dict[str, Any] | None = None


class SystemAnalysisRequest(BaseModel):
    """Модель запроса для анализа диалога."""

    messages: List[ChatMessage] = Field(
        ...,
        description="Полная история сообщений. Последнее сообщение должно быть от пользователя.",
        min_length=1,
    )
    current_stage_number: int = Field(
        ...,
        description="Номер текущего этапа",
    )
    stages_info: List[Dict[str, Any]] = Field(
        ...,
        description="Информация об этапах",
    )
    model_name: Optional[Literal["openai", "google"]] = Field(
        default="openai",
        description="Провайдер языковой модели для использования в сессии.",
        examples=["openai"],
    )
    memory_strategy: Optional[Literal["buffer", "summary"]] = Field(
        default="buffer",
        description="Стратегия управления памятью диалога.",
        examples=["buffer"],
    )
    model_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Словарь с параметрами для настройки модели.",
        examples=[{"temperature": 0.5, "model_name": "gpt-4o-mini"}],
    )
    memory_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Словарь с параметрами для настройки памяти.",
        examples=[{"k": 3}],
    )


class ClientInfo(BaseModel):
    """Информация, извлеченная из сообщения клиента."""

    name: str = Field(description="Полное имя (ФИО) клиента, если указано. Иначе пустая строка.")
    phone: str = Field(description="Контактный номер телефона клиента. Иначе пустая строка.")


class StageConfidence(BaseModel):
    """Информация по уверенности этапа."""

    stage_name: str = Field(description="Название этапа")
    confidence: float = Field(description="Значение уверенности этапа: от 0.00 до 1.00.")


class SalesFunnel(BaseModel):
    """Анализ положения клиента в воронке продаж."""

    use_rag: bool = Field(description="True, если для ответа нужна информация из базы знаний.")
    stage: int | str = Field(description="Этап воронки продаж.")
    stage_confidences: List[StageConfidence] = Field(
        description="Список с уверенностью для КАЖДОГО этапа воронки."
    )
    argument_stage: str = Field(description="Аргументация, почему выбрали именно этот этап")


class ClientSignals(BaseModel):
    """Ключевые сигналы и намерения, выраженные клиентом."""

    target_yes: bool = Field(description="True, если клиент подтвердил запись/встречу/покупку.")
    dont: bool = Field(description="True, если клиент просит не писать ему или не беспокоить.")


class FastAnalytics(BaseModel):
    client_info: ClientInfo
    sales_funnel: SalesFunnel
    client_signals: ClientSignals


class SystemAnalysisResponse(BaseModel):
    """Полный результат анализа сообщения."""

    analysis: FastAnalytics
    meta: Dict[str, Any] | None = None


class KBUploadResponse(BaseModel):
    status: Literal["created", "already_exists", "error"]
    trace_id: str
    funnel_id: str
    kb_id: str
    file_id: str
    source_name: str
    source_type: str
    bytes_size: int
    content_hash: str
    is_duplicate: bool
    message: str


class KBDeleteResponse(BaseModel):
    status: Literal["deleted", "not_found", "conflict"]
    trace_id: str
    funnel_id: str
    kb_id: str
    manifest_updated: bool
    local_index_deleted: bool
    supabase_file_found: bool
    deleted_files: int
    deleted_chunks: int
    deleted_embeddings: int


class ErrorResponse(BaseModel):
    error: str
    trace_id: Optional[str] = None


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


def chat_request_to_orchestrator_request(
    *,
    chat: ChatRequest,
    tenant_id: str,
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
