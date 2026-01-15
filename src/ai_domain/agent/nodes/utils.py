import logging
import time


def ensure_lists(state: dict) -> None:
    state.setdefault("messages", [])
    state.setdefault("executed", [])


def log_node(state: dict, event: str) -> None:
    trace_id = state.get("trace_id")
    graph = state.get("graph") or state.get("graph_name")
    logging.info(
        "agent_node_event node=%s trace_id=%s graph=%s",
        event,
        trace_id,
        graph,
    )


def mark_runtime_error(
    state: dict,
    *,
    code: str,
    message: str,
    node: str,
    retryable: bool = False,
) -> None:
    runtime = state.setdefault("runtime", {})
    runtime.setdefault("errors", [])
    runtime["degraded"] = True
    runtime["errors"].append(
        {
            "code": code,
            "message": message,
            "node": node,
            "retryable": retryable,
        }
    )


def step_begin(state: dict, node: str) -> int:
    runtime = state.setdefault("runtime", {})
    steps = runtime.setdefault("steps", [])
    started_at = int(time.time() * 1000)
    entry = {
        "node": node,
        "started_at": started_at,
        "status": "running",
    }
    steps.append(entry)
    return len(steps) - 1


def step_end(
    state: dict,
    *,
    index: int,
    latency_ms: int,
    status: str = "ok",
    reason: str | None = None,
) -> None:
    steps = state.setdefault("runtime", {}).setdefault("steps", [])
    if 0 <= index < len(steps):
        steps[index]["latency_ms"] = latency_ms
        steps[index]["status"] = status
        if reason:
            steps[index]["reason"] = reason
