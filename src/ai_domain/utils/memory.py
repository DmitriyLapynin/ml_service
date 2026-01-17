from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Sequence

from ai_domain.graphs.state import GraphState


def estimate_tokens(text: str) -> int:
    # Rough heuristic: 4 chars per token (good enough for thresholds).
    return max(1, int(len(text) / 4))


def estimate_message_tokens(messages: Sequence[Dict[str, str]]) -> int:
    total = 0
    for msg in messages:
        total += estimate_tokens(msg.get("content") or "")
    return total


def _get_state_value(state: object, key: str, default: Any) -> Any:
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


def _get_memory_summary(state: object) -> str | None:
    if isinstance(state, dict):
        memory = state.get("memory") or {}
        if isinstance(memory, dict):
            return memory.get("summary")
        return None
    memory = getattr(state, "memory", None)
    return getattr(memory, "summary", None)


def select_memory_messages(state: GraphState | dict) -> List[Dict[str, str]]:
    """
    Срезает сообщения по стратегии/параметрам.
    Поддерживается:
    - buffer / k (последние k сообщений)
    - summary (summary + последние k сообщений)
    """
    logger = logging.getLogger(__name__)
    messages = _get_state_value(state, "messages", []) or []
    strategy = _get_state_value(state, "memory_strategy", "buffer") or "buffer"
    params = _get_state_value(state, "memory_params", {}) or {}
    trace_id = _get_state_value(state, "trace_id", None)
    before_count = len(messages)
    before_chars = sum(len(m.get("content") or "") for m in messages)

    if strategy == "summary":
        k = params.get("k")
        if not isinstance(k, int) or k <= 0:
            k = 6
        summary = _get_memory_summary(state)
        recent = messages[-k:] if messages else []
        if summary:
            trimmed_messages = [{"role": "system", "content": f"Summary of previous conversation:\n{summary}"}] + recent
            _log_memory_trim(
                logger,
                trace_id=trace_id,
                strategy="summary",
                before_messages=messages,
                after_messages=trimmed_messages,
                summary_used=True,
                trim_reason="summary_used",
            )
            return trimmed_messages
        _log_memory_trim(
            logger,
            trace_id=trace_id,
            strategy="summary",
            before_messages=messages,
            after_messages=recent,
            summary_used=False,
            trim_reason="k_limit" if len(recent) < before_count else "no_change",
        )
        return recent

    if strategy == "buffer":
        k = params.get("k")
        if isinstance(k, int) and k > 0:
            trimmed_messages = messages[-k:]
            _log_memory_trim(
                logger,
                trace_id=trace_id,
                strategy="buffer",
                before_messages=messages,
                after_messages=trimmed_messages,
                summary_used=False,
                trim_reason="k_limit" if len(trimmed_messages) < before_count else "no_change",
            )
            return trimmed_messages
    # fallback на полную историю
    _log_memory_trim(
        logger,
        trace_id=trace_id,
        strategy=strategy,
        before_messages=messages,
        after_messages=messages,
        summary_used=False,
        trim_reason="no_change",
    )
    return messages.copy()


def _log_memory_trim(
    logger: logging.Logger,
    *,
    trace_id: str | None,
    strategy: str,
    before_messages: Sequence[Dict[str, str]],
    after_messages: Sequence[Dict[str, str]],
    summary_used: bool,
    trim_reason: str,
) -> None:
    before_count = len(before_messages)
    after_count = len(after_messages)
    before_chars = sum(len(m.get("content") or "") for m in before_messages)
    after_chars = sum(len(m.get("content") or "") for m in after_messages)
    if before_count == after_count and not summary_used:
        return
    logger.info(
        json.dumps(
            {
                "event": "memory_trim",
                "trace_id": trace_id,
                "strategy": strategy,
                "before_messages_count": before_count,
                "after_messages_count": after_count,
                "before_chars_total": before_chars,
                "after_chars_total": after_chars,
                "summary_used": summary_used,
                "trim_reason": trim_reason,
            },
            ensure_ascii=False,
        )
    )
