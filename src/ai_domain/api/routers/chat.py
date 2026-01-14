from fastapi import APIRouter, Depends, Request

from ai_domain.api.deps import get_orchestrator, get_settings
from ai_domain.api.errors import APIError
from ai_domain.api.schemas import ChatRequest, ChatResponse, chat_request_to_orchestrator_request
from ai_domain.orchestrator.service import Orchestrator

router = APIRouter()


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Chat endpoint",
    description=(
        "Example request:\n\n"
        "```\n"
        "curl -X POST http://localhost:8000/v1/chat \\\n"
        "  -H 'Content-Type: application/json' \\\n"
        "  -H 'x-tenant-id: demo' \\\n"
        "  -H 'x-conversation-id: demo' \\\n"
        "  -d '{\"messages\":[{\"role\":\"user\",\"content\":\"Привет!\"}],\"is_rag\":false}'\n"
        "```\n"
    ),
)
async def chat(
    payload: ChatRequest,
    request: Request,
    orchestrator: Orchestrator = Depends(get_orchestrator),
) -> ChatResponse:
    settings = get_settings()
    tenant_id = (
        getattr(request.state, "tenant_id", None)
        or request.headers.get("x-tenant-id")
        or settings.default_tenant_id
    )
    if not tenant_id:
        raise APIError("x-tenant-id is required", status_code=400, code="missing_tenant_id")
    conversation_id = request.headers.get("x-conversation-id") or settings.default_conversation_id
    channel = getattr(request.state, "channel", None) or request.headers.get("x-channel") or "chat"
    idempotency_key = request.headers.get("x-idempotency-key")
    funnel_id = getattr(request.state, "funnel_id", None) or request.headers.get("x-funnel-id")
    trace_id = getattr(request.state, "trace_id", None)

    credentials = getattr(request.state, "credentials", None)
    req = chat_request_to_orchestrator_request(
        chat=payload,
        tenant_id=tenant_id,
        conversation_id=conversation_id,
        channel=channel,
        idempotency_key=idempotency_key,
        credentials=credentials,
        funnel_id=funnel_id,
        trace_id=trace_id,
    )
    result = await orchestrator.run(req)
    return ChatResponse(**result)
