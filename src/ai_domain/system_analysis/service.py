from __future__ import annotations

import json
import time
from typing import Any, Dict

import logging

from ai_domain.api.schemas import FastAnalytics
from ai_domain.api.config import get_settings
from ai_domain.llm.client import LLMClient, LLMConfig
from ai_domain.llm.metrics import CompositeMetricsWriter, LangSmithWriter, StateMetricsWriter
from ai_domain.llm.types import LLMCallContext
from ai_domain.secrets import get_secret
from ai_domain.utils.hashing import messages_fingerprint
from ai_domain.telemetry.meta import build_meta_from_state
from ai_domain.system_analysis.prompts import (
    SYSTEM_ANALYSIS_PROMPT,
    format_analysis_user_prompt,
)
from ai_domain.agent.nodes.utils import step_begin, step_end


def _build_prompt(*, stages_info: list[dict], current_stage_number: int) -> str:
    return format_analysis_user_prompt(
        stages_info=stages_info,
        current_stage_number=current_stage_number,
    )


async def run_system_analysis(
    *,
    orchestrator_req: Dict[str, Any],
    llm: LLMClient,
) -> Dict[str, Any]:
    start = time.perf_counter()
    trace_id = orchestrator_req.get("trace_id")
    state = {
        "trace_id": trace_id,
        "tenant_id": orchestrator_req.get("tenant_id"),
        "channel": "system_analysis",
        "messages": orchestrator_req.get("messages") or [],
        "memory_strategy": orchestrator_req.get("memory_strategy"),
        "memory_params": orchestrator_req.get("memory_params") or {},
        "trace": {"llm_calls": [], "errors": [], "steps": []},
        "runtime": {"degraded": False, "errors": []},
        "graph_name": "system_analysis",
        "route": "system_analysis",
    }
    settings = get_settings()
    langsmith_api_key = get_secret("LANGSMITH_API_KEY") or get_secret("LANGCHAIN_API_KEY")
    langsmith_writer = LangSmithWriter(
        trace_id=trace_id or "",
        project_name=settings.langsmith_project,
        enabled=True,
        api_key=langsmith_api_key,
        endpoint=settings.langsmith_endpoint,
        inputs={
            "trace_id": trace_id,
            "channel": "system_analysis",
        },
    )
    metrics_writer = CompositeMetricsWriter([StateMetricsWriter(state), langsmith_writer])
    state["metrics_writer"] = metrics_writer

    msg_fp = messages_fingerprint(orchestrator_req.get("messages") or [])
    metrics_writer.begin_trace(
        {
            "name": "system_analysis",
            "metadata": {
                "trace_id": trace_id,
                "tenant_id": orchestrator_req.get("tenant_id"),
                "channel": "system_analysis",
                "graph_name": "system_analysis",
            },
            "inputs": {
                "input_fingerprint": msg_fp.get("digest"),
                "messages_count": msg_fp.get("count"),
                "total_chars": msg_fp.get("total_chars"),
            },
        }
    )

    full_messages = state.get("messages") or []
    messages = full_messages[-10:] if len(full_messages) > 10 else full_messages
    if len(messages) != len(full_messages):
        before_chars = sum(len(m.get("content") or "") for m in full_messages)
        after_chars = sum(len(m.get("content") or "") for m in messages)
        logging.info(
            json.dumps(
                {
                    "event": "memory_trim",
                    "trace_id": trace_id,
                    "strategy": "fixed_last_n",
                    "before_messages_count": len(full_messages),
                    "after_messages_count": len(messages),
                    "before_chars_total": before_chars,
                    "after_chars_total": after_chars,
                    "summary_used": False,
                    "k": 10,
                },
                ensure_ascii=False,
            )
        )
    step_index = step_begin(state, "system_analysis")
    user_prompt = _build_prompt(
        stages_info=orchestrator_req.get("stages_info") or [],
        current_stage_number=int(orchestrator_req.get("current_stage_number") or 0),
    )
    system_content = f"{SYSTEM_ANALYSIS_PROMPT.strip()}\n\nКОНТЕКСТ:\n{user_prompt}"
    llm_messages = [
        {"role": "system", "content": system_content},
        *messages,
    ]

    model_provider = orchestrator_req.get("model_provider") or "openai"
    if model_provider != "openai":
        raise ValueError("unsupported_provider")
    model_params = orchestrator_req.get("model_params") or {}
    model_name = model_params.get("model_name") or "gpt-4.1-mini"
    temperature = model_params.get("temperature", 0.2)
    top_p = model_params.get("top_p")
    if top_p is not None:
        top_p = float(top_p)

    context = (
        LLMCallContext(
            trace_id=trace_id,
            graph="system_analysis",
            node="system_analysis",
            task="system_analyze",
            channel="system_analysis",
            tenant_id=orchestrator_req.get("tenant_id"),
            request_id=None,
            metrics=state.get("metrics_writer"),
        )
        if trace_id
        else None
    )
    analysis = None
    error: Exception | None = None
    try:
        analysis = await llm.invoke_structured(
            FastAnalytics,
            llm_messages,
            config=LLMConfig(
                provider="openai",
                model=model_name,
                temperature=float(temperature) if temperature is not None else 0.2,
                top_p=top_p,
                max_tokens=int(model_params.get("max_tokens", 512)),
            ),
            include_raw=False,
            context=context,
        )
    except Exception as exc:
        error = exc
        state.setdefault("runtime", {}).setdefault("errors", []).append(
            {"code": "system_analysis_failed", "message": str(exc), "node": "system_analysis"}
        )
        raise
    finally:
        latency_ms = int((time.perf_counter() - start) * 1000)
        end_ts = int(time.time() * 1000)
        start_ts = end_ts - latency_ms
        step_end(
            state,
            index=step_index,
            latency_ms=latency_ms,
            status="error" if error else "ok",
        )
        if metrics_writer and hasattr(metrics_writer, "finalize"):
            metrics_writer.finalize(
                {
                    "trace_id": trace_id,
                    "status": "error" if error else "ok",
                    "route": "system_analysis",
                    "degraded": bool(state.get("runtime", {}).get("degraded")),
                }
            )

    if analysis is not None:
        current_stage_number = int(orchestrator_req.get("current_stage_number") or 0)
        sales_funnel = getattr(analysis, "sales_funnel", None)
        if sales_funnel is not None:
            stage_value = getattr(sales_funnel, "stage", None)
            try:
                stage_number = int(stage_value) if stage_value is not None else None
            except (TypeError, ValueError):
                stage_number = None
            if stage_number is not None and current_stage_number and stage_number < current_stage_number:
                setattr(sales_funnel, "stage", current_stage_number)

        client_info = getattr(analysis, "client_info", None)
        client_signals = getattr(analysis, "client_signals", None)
        stage_confidences = getattr(sales_funnel, "stage_confidences", []) if sales_funnel else []
        top_conf = None
        if stage_confidences:
            top_conf = max((sc.confidence for sc in stage_confidences), default=None)
        logging.info(
            json.dumps(
                {
                    "event": "analysis_summary",
                    "trace_id": trace_id,
                    "use_rag": getattr(sales_funnel, "use_rag", None) if sales_funnel else None,
                    "stage": getattr(sales_funnel, "stage", None) if sales_funnel else None,
                    "stage_top_confidence": top_conf,
                    "target_yes": getattr(client_signals, "target_yes", None) if client_signals else None,
                    "dont": getattr(client_signals, "dont", None) if client_signals else None,
                    "client_name_present": bool(getattr(client_info, "name", "")) if client_info else False,
                    "client_phone_present": bool(getattr(client_info, "phone", "")) if client_info else False,
                },
                ensure_ascii=False,
            )
        )
    meta = build_meta_from_state(
        state,
        route="system_analysis",
        start_ts=start_ts,
        end_ts=end_ts,
        total_latency_ms=latency_ms,
        degraded=False,
        default_graph_name="system_analysis",
        rag_defaults={
            "enabled": False,
            "config_id": None,
            "top_k": None,
            "query": None,
            "retrieval_latency_ms": 0,
            "documents": [],
            "deduped_count": 0,
            "final_context_chars": 0,
            "context_truncated": False,
            "truncate_reason": None,
            "rerank_used": False,
        },
    )

    return {
        "analysis": analysis,
        "meta": meta,
    }
