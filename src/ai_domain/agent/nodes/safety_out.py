import time

from ai_domain.llm.types import LLMCallContext

from ai_domain.llm.metrics import StateMetricsWriter
from .safety_in import _check_unsafe_and_injection
from .utils import ensure_lists, log_node, step_begin, step_end


async def safety_out_node(state: dict) -> dict:
    ensure_lists(state)
    log_node(state, "safety_out")
    step_index = step_begin(state, "safety_out")
    start = time.perf_counter()
    state["executed"].append("safety_out")
    answer = state.get("answer") or {}
    text = (answer.get("text") or "").strip()
    trace_id = state.get("trace_id")
    context = (
        LLMCallContext(
            trace_id=trace_id,
            graph=state.get("graph"),
            node="safety_out",
            task="safety_classifier",
            channel=state.get("channel"),
            tenant_id=state.get("tenant_id"),
            request_id=state.get("request_id"),
            metrics=state.get("metrics_writer") or StateMetricsWriter(state),
        )
        if trace_id
        else None
    )
    flags = await _check_unsafe_and_injection(
        text,
        llm=state.get("safety_llm"),
        model=state.get("safety_model"),
        call_context=context,
        state=state,
    )
    if flags["unsafe"]:
        state["answer"] = {"text": "Ответ скрыт политикой безопасности.", "format": "plain"}
    latency_ms = int((time.perf_counter() - start) * 1000)
    step_end(state, index=step_index, latency_ms=latency_ms, status="ok")
    return state
