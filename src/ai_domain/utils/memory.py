from __future__ import annotations

from typing import Dict, List

from ai_domain.graphs.state import GraphState


def select_memory_messages(state: GraphState) -> List[Dict[str, str]]:
    """
    Срезает сообщения по стратегии/параметрам.
    Поддерживается:
    - buffer / k (последние k сообщений)
    """
    messages = getattr(state, "messages", []) or []
    strategy = getattr(state, "memory_strategy", "buffer") or "buffer"
    params = getattr(state, "memory_params", {}) or {}

    if strategy == "buffer":
        k = params.get("k")
        if isinstance(k, int) and k > 0:
            return messages[-k:]
    # fallback на полную историю
    return messages.copy()
