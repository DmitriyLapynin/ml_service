from __future__ import annotations

from typing import Any, Dict

from ai_domain.api.schemas import SystemAnalysisRequest


def system_analysis_request_to_orchestrator_request(
    *,
    req: SystemAnalysisRequest,
    trace_id: str,
    tenant_id: str = "default",
) -> Dict[str, Any]:
    return {
        "trace_id": trace_id,
        "tenant_id": tenant_id,
        "channel": "system_analysis",
        "messages": [m.model_dump() for m in req.messages],
        "memory_strategy": req.memory_strategy,
        "memory_params": req.memory_params,
        "model_provider": req.model_name,
        "model_params": req.model_params,
        "current_stage_number": req.current_stage_number,
        "stages_info": req.stages_info,
        "graph_name": "system_analysis",
    }
