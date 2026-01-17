import json
import logging
import time
from uuid import uuid4

from fastapi import APIRouter, Depends, Request

from ai_domain.api.config import get_settings
from ai_domain.api.adapters import system_analysis_request_to_orchestrator_request
from ai_domain.api.schemas import SystemAnalysisRequest, SystemAnalysisResponse
from ai_domain.api.deps import get_llm_client
from ai_domain.utils.hashing import messages_fingerprint
from ai_domain.system_analysis.service import run_system_analysis

router = APIRouter()


@router.post(
    "/system-analysis",
    response_model=SystemAnalysisResponse,
    summary="System analysis endpoint",
)
async def system_analysis(
    payload: SystemAnalysisRequest,
    request: Request,
    llm=Depends(get_llm_client),
) -> SystemAnalysisResponse:
    logger = logging.getLogger(__name__)
    start = time.perf_counter()
    settings = get_settings()
    trace_id = getattr(request.state, "trace_id", None) or uuid4().hex
    tenant_id = getattr(request.state, "tenant_id", None) or "default"
    fp = messages_fingerprint([m.model_dump() for m in payload.messages])
    model_params = payload.model_params or {}
    model_name = model_params.get("model_name") or "gpt-4.1-mini"
    log_payload = {
        "event": "api_request",
        "endpoint": "system_analyze",
        "trace_id": trace_id,
        "request_id": getattr(request.state, "request_id", None),
        "tenant_id": tenant_id,
        "channel": "system_analyze",
        "messages_count": fp.get("count"),
        "messages_chars_total": fp.get("total_chars"),
        "input_fingerprint": f"sha256:{fp.get('digest')}",
        "memory_strategy": payload.memory_strategy,
        "memory_params": payload.memory_params,
        "model_provider": payload.model_name,
        "model_params": model_params,
        "model": model_name,
        "current_stage_number": payload.current_stage_number,
        "stages_count": len(payload.stages_info or []),
        "debug": settings.debug_logging,
    }
    logger.info(json.dumps(log_payload, ensure_ascii=False))
    if settings.debug_logging:
        logger.info(
            json.dumps(
                {
                    "event": "api_debug",
                    "endpoint": "system_analyze",
                    "trace_id": trace_id,
                    "messages": [m.model_dump() for m in payload.messages],
                    "stages_info": payload.stages_info,
                },
                ensure_ascii=False,
            )
        )
    orchestrator_req = system_analysis_request_to_orchestrator_request(
        req=payload,
        trace_id=trace_id,
        tenant_id=tenant_id,
    )
    status_code = 500
    result: dict | None = None
    try:
        result = await run_system_analysis(orchestrator_req=orchestrator_req, llm=llm)
        status_code = 200
        return SystemAnalysisResponse(**result)
    finally:
        latency_ms_total = int((time.perf_counter() - start) * 1000)
        meta = (result or {}).get("meta") if isinstance(result, dict) else None
        llm_calls = (meta or {}).get("llm_calls") if isinstance(meta, dict) else None
        tokens_total = None
        if isinstance(llm_calls, list):
            total_sum = 0
            for call in llm_calls:
                usage = call.get("usage") if isinstance(call, dict) else None
                if isinstance(usage, dict):
                    total_sum += int(usage.get("total_tokens") or 0)
            tokens_total = total_sum
        errors_count = len((meta or {}).get("errors") or []) if isinstance(meta, dict) else 0
        model_used = model_name
        if isinstance(llm_calls, list) and llm_calls:
            last_call = llm_calls[-1]
            model_used = last_call.get("model") or model_used
        status = (result or {}).get("status") or ("error" if status_code >= 400 else "ok")
        logger.info(
            json.dumps(
                {
                    "event": "api_response",
                    "endpoint": "system_analyze",
                    "trace_id": trace_id,
                    "status_code": status_code,
                    "status": status,
                    "latency_ms_total": latency_ms_total,
                    "errors_count": errors_count,
                    "model_used": model_used,
                    "tokens_total": tokens_total,
                },
                ensure_ascii=False,
            )
        )
        if settings.debug_logging:
            logger.info(
                json.dumps(
                    {
                        "event": "api_debug",
                        "endpoint": "system_analyze",
                        "trace_id": trace_id,
                        "analysis": (result or {}).get("analysis"),
                        "meta": meta,
                    },
                    ensure_ascii=False,
                    default=str,
                )
            )
