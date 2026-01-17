import json
import logging
import time

from fastapi import APIRouter, Depends, Request

from ai_domain.api.config import get_settings
from ai_domain.api.deps import get_orchestrator
from ai_domain.api.errors import APIError
from ai_domain.api.schemas import (
    ChatRequest,
    ChatResponse,
    chat_request_to_orchestrator_request,
)
from ai_domain.orchestrator.service import Orchestrator
from ai_domain.utils.hashing import messages_fingerprint

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
        "  -d '{\"messages\":[{\"role\":\"user\",\"content\":\"Привет!\"}],\"is_rag\":false,\"tenant_id\":\"default\"}'\n"
        "```\n"
    ),
)
async def chat(
    payload: ChatRequest,
    request: Request,
    orchestrator: Orchestrator = Depends(get_orchestrator),
) -> ChatResponse:
    logger = logging.getLogger(__name__)
    start = time.perf_counter()
    settings = get_settings()
    tenant_id = payload.tenant_id or "default"
    if not tenant_id:
        raise APIError("tenant_id is required", status_code=400, code="missing_tenant_id")
    channel = getattr(request.state, "channel", None) or request.headers.get("x-channel") or "chat"
    idempotency_key = request.headers.get("x-idempotency-key")
    funnel_id = getattr(request.state, "funnel_id", None) or request.headers.get("x-funnel-id")
    trace_id = getattr(request.state, "trace_id", None)

    credentials = getattr(request.state, "credentials", None)
    req = chat_request_to_orchestrator_request(
        chat=payload,
        tenant_id=tenant_id,
        channel=channel,
        idempotency_key=idempotency_key,
        credentials=credentials,
        funnel_id=funnel_id,
        trace_id=trace_id,
    )
    fp = messages_fingerprint([m.model_dump() for m in payload.messages])
    model_params = payload.model.params or {}
    memory_params = payload.memory_params or {}
    status_code = 500
    result: dict | None = None
    try:
        result = await orchestrator.run(req)
        status_code = 200
        return ChatResponse(**result)
    finally:
        latency_ms_total = int((time.perf_counter() - start) * 1000)
        status = (result or {}).get("status") or ("error" if status_code >= 400 else "ok")
        answer = (result or {}).get("answer") or {}
        answer_text = answer.get("text") if isinstance(answer, dict) else None
        answer_chars = len(answer_text or "")
        meta = (result or {}).get("meta") if isinstance(result, dict) else None
        errors_count = len((meta or {}).get("errors") or []) if isinstance(meta, dict) else 0
        llm_calls = (meta or {}).get("llm_calls") if isinstance(meta, dict) else None
        tokens_total = None
        tokens_total_request = None
        tokens_total_last_call = None
        model_used = payload.model.name
        if isinstance(llm_calls, list) and llm_calls:
            last_call = llm_calls[-1]
            usage = last_call.get("usage") if isinstance(last_call, dict) else None
            if isinstance(usage, dict):
                tokens_total = usage.get("total_tokens")
                tokens_total_last_call = usage.get("total_tokens")
            model_used = last_call.get("model") or model_used
            total_sum = 0
            for call in llm_calls:
                usage = call.get("usage") if isinstance(call, dict) else None
                if isinstance(usage, dict):
                    total_sum += int(usage.get("total_tokens") or 0)
            tokens_total_request = total_sum
        log_payload = {
            "event": "api_request",
            "trace_id": trace_id,
            "request_id": getattr(request.state, "request_id", None),
            "tenant_id": tenant_id,
            "channel": channel,
            "graph": getattr(orchestrator, "graph_name", None),
            "path": request.url.path,
            "method": request.method,
            "status_code": status_code,
            "messages_count": fp.get("count"),
            "messages_chars_total": fp.get("total_chars"),
            "input_fingerprint": f"sha256:{fp.get('digest')}",
            "prompt": payload.prompt,
            "role_instruction": payload.role_instruction,
            "is_rag": payload.is_rag,
            "funnel_id": funnel_id or payload.funnel_id,
            "tools_count": len(payload.tools or []),
            "tools": [t.name for t in (payload.tools or [])],
            "has_crypted_api_key": bool(payload.crypted_api_key),
            "memory_strategy": payload.memory_strategy,
            "memory_k": memory_params.get("k"),
            "memory_params": memory_params,
            "model": payload.model.name,
            "model_params": model_params,
            "idempotency_key": idempotency_key,
            "return_trace_meta": bool(req.get("debug") or model_params.get("return_trace_meta")),
        }
        logger.info(json.dumps(log_payload, ensure_ascii=False))
        if settings.debug_logging:
            debug_payload = {
                "event": "api_debug",
                "trace_id": trace_id,
                "messages": [m.model_dump() for m in payload.messages],
                "final_answer": answer_text,
                "tool_args": meta.get("tool_calls") if isinstance(meta, dict) else None,
                "tool_result_preview": meta.get("tool_results_preview") if isinstance(meta, dict) else None,
            }
            logger.info(json.dumps(debug_payload, ensure_ascii=False))
        logger.info(
            json.dumps(
                {
                    "event": "api_response",
                    "trace_id": trace_id,
                    "status_code": status_code,
                    "status": status,
                    "latency_ms_total": latency_ms_total,
                    "answer_chars": answer_chars,
                    "errors_count": errors_count,
                    "model_used": model_used,
                    "tokens_total": tokens_total,
                    "tokens_total_request": tokens_total_request,
                    "tokens_total_last_call": tokens_total_last_call,
                    "llm_calls_count": len(llm_calls) if isinstance(llm_calls, list) else None,
                    "tools_calls_count": len((meta or {}).get("tool_calls") or [])
                    if isinstance(meta, dict)
                    else None,
                },
                ensure_ascii=False,
            )
        )
