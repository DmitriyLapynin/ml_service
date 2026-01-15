import logging
import time
from typing import Any, Dict

from ai_domain.agent.safety_prompt import (
    SAFETY_CLASSIFIER_PROMPT,
    SafetyClassifierOutput,
    normalize_classifier_output,
    rule_based_flags,
)
from ai_domain.llm.client import LLMConfig
from ai_domain.llm.types import LLMCallContext

from .utils import ensure_lists, log_node, mark_runtime_error, step_begin, step_end


def _extract_last_user_content(state: dict) -> str:
    for msg in reversed(state.get("messages", [])):
        if msg.get("role") == "user":
            return (msg.get("content") or "").strip()
    return ""


async def _check_unsafe_and_injection(
    text: str,
    *,
    llm: Any | None = None,
    model: str | None = None,
    call_context: LLMCallContext | None = None,
    state: dict | None = None,
) -> Dict[str, bool]:
    """
    Комбинированный чек:
    1) rule-based (ключевые слова),
    2) LLM-классификатор, который возвращает unsafe / injection (strict JSON).

    LLM передаётся извне (через state['safety_llm']) и должен поддерживать:
      await llm.invoke_structured(schema, messages, config, include_raw=...)
    """
    if not text:
        return {"unsafe": False, "injection_suspected": False}

    rules = rule_based_flags(text)
    llm_unsafe = False
    llm_injection = False

    if llm is not None:
        try:
            messages = [
                {"role": "system", "content": SAFETY_CLASSIFIER_PROMPT},
                {"role": "user", "content": text},
            ]
            config = LLMConfig(
                model=model,
                max_tokens=64,
                temperature=0.0,
                top_p=1.0,
                metadata={"required_capabilities": {"supports_structured": True}},
            )
            res = await llm.invoke_structured(
                SafetyClassifierOutput,
                messages,
                config=config,
                include_raw=True,
                context=call_context,
            )
            payload = res.parsed
            if payload is not None:
                parsed = normalize_classifier_output(payload.model_dump())
                llm_unsafe = bool(parsed["unsafe"])
                llm_injection = bool(parsed["injection_suspected"])
        except Exception as e:
            logging.error(f"Ошибка при LLM-классификации безопасности: {e}")
            if state is not None:
                mark_runtime_error(
                    state,
                    code="safety_llm_error",
                    message=str(e),
                    node="safety_classifier",
                    retryable=True,
                )

    unsafe = rules["unsafe"] or llm_unsafe
    injection_suspected = rules["injection_suspected"] or llm_injection
    return {"unsafe": unsafe, "injection_suspected": injection_suspected}


async def safety_in_node(state: dict) -> dict:
    ensure_lists(state)
    log_node(state, "safety_in")
    step_index = step_begin(state, "safety_in")
    start = time.perf_counter()
    state["executed"].append("safety_in")
    last_user = _extract_last_user_content(state)
    trace_id = state.get("trace_id")
    context = (
        LLMCallContext(
            trace_id=trace_id,
            graph=state.get("graph"),
            node="safety_in",
            task="safety_classifier",
            channel=state.get("channel"),
            tenant_id=state.get("tenant_id"),
            request_id=state.get("request_id"),
            metrics=state.get("llm_metrics"),
        )
        if trace_id
        else None
    )
    flags = await _check_unsafe_and_injection(
        last_user,
        llm=state.get("safety_llm"),
        model=state.get("safety_model"),
        call_context=context,
        state=state,
    )
    state["unsafe"] = flags["unsafe"]
    state["injection_suspected"] = flags["injection_suspected"]
    state["blocked"] = flags["unsafe"] or flags["injection_suspected"]
    if state["blocked"]:
        state["safety_reason"] = "unsafe_request"
    latency_ms = int((time.perf_counter() - start) * 1000)
    step_end(state, index=step_index, latency_ms=latency_ms, status="ok")
    return state


def safety_block_node(state: dict) -> dict:
    ensure_lists(state)
    log_node(state, "safety_block")
    step_index = step_begin(state, "safety_block")
    start = time.perf_counter()
    state["executed"].append("safety_block")
    text = "Я не могу ответить на этот запрос."
    if state.get("unsafe"):
        text = (
            "Запрос нарушает политику безопасности. "
            "Уточните вопрос про услуги или запись."
        )
    elif state.get("injection_suspected"):
        text = (
            "Нельзя обходить правила. "
            "Я могу ответить только в рамках политики безопасности."
        )
    state["answer"] = {"text": text, "format": "plain"}
    latency_ms = int((time.perf_counter() - start) * 1000)
    step_end(state, index=step_index, latency_ms=latency_ms, status="ok")
    return state
