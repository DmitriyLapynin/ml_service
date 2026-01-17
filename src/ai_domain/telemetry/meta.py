from __future__ import annotations

from typing import Any, Dict


def build_meta_from_state(
    state: Any,
    *,
    route: str | None,
    start_ts: int,
    end_ts: int,
    total_latency_ms: int,
    degraded: bool,
    default_graph_name: str | None = None,
    rag_defaults: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    def _get(attr, default=None):
        if hasattr(state, attr):
            return getattr(state, attr)
        if isinstance(state, dict):
            return state.get(attr, default)
        return default

    runtime = _get("runtime") or {}
    trace = _get("trace") or {}
    graph_name = _get("graph_name") or default_graph_name
    graph = _get("graph") or graph_name

    meta = {
        "graph": graph,
        "graph_name": graph_name,
        "graph_version": _get("graph_version"),
        "route": _get("route") or route,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "total_latency_ms": total_latency_ms,
        "degraded": degraded,
        "errors": runtime.get("errors") or [],
    }
    if isinstance(trace, dict):
        meta.update(trace)

    if meta.get("rag") is None:
        meta["rag"] = rag_defaults or {
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
        }
    return meta
