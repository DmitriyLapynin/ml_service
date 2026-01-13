from __future__ import annotations

from typing import Dict, List, Sequence

from ai_domain.graphs.state import GraphState


def estimate_tokens(text: str) -> int:
    # Rough heuristic: 4 chars per token (good enough for thresholds).
    return max(1, int(len(text) / 4))


def estimate_message_tokens(messages: Sequence[Dict[str, str]]) -> int:
    total = 0
    for msg in messages:
        total += estimate_tokens(msg.get("content") or "")
    return total


def select_memory_messages(state: GraphState) -> List[Dict[str, str]]:
    """
    Срезает сообщения по стратегии/параметрам.
    Поддерживается:
    - buffer / k (последние k сообщений)
    - summary (summary + последние k сообщений)
    """
    messages = getattr(state, "messages", []) or []
    strategy = getattr(state, "memory_strategy", "buffer") or "buffer"
    params = getattr(state, "memory_params", {}) or {}

    if strategy == "summary":
        k = params.get("k")
        if not isinstance(k, int) or k <= 0:
            k = 6
        summary = getattr(getattr(state, "memory", None), "summary", None)
        recent = messages[-k:] if messages else []
        if summary:
            return [{"role": "system", "content": f"Summary of previous conversation:\n{summary}"}] + recent
        return recent

    if strategy == "buffer":
        k = params.get("k")
        if isinstance(k, int) and k > 0:
            return messages[-k:]
    # fallback на полную историю
    return messages.copy()
