import json
import logging
import time


def ensure_lists(state: dict) -> None:
    state.setdefault("messages", [])
    state.setdefault("executed", [])


def log_node(state: dict, event: str) -> None:
    trace_id = state.get("trace_id")
    graph = state.get("graph") or state.get("graph_name")
    logging.info(
        json.dumps(
            {
                "event": "node_event",
                "trace_id": trace_id,
                "node": event,
                "graph": graph,
            },
            ensure_ascii=False,
        )
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
    trace = state.setdefault("trace", {})
    steps = trace.setdefault("steps", [])
    started_at = int(time.time() * 1000)
    entry = {
        "node": node,
        "started_at": started_at,
        "status": "running",
    }
    steps.append(entry)
    logging.info(
        json.dumps(
            {
                "event": "node_start",
                "trace_id": state.get("trace_id"),
                "node": node,
            },
            ensure_ascii=False,
        )
    )
    writer = state.get("metrics_writer")
    if writer and hasattr(writer, "begin_step"):
        writer.begin_step({"index": len(steps) - 1, "node": node, "started_at": started_at})
    return len(steps) - 1


def step_end(
    state: dict,
    *,
    index: int,
    latency_ms: int,
    status: str = "ok",
    reason: str | None = None,
) -> None:
    steps = state.setdefault("trace", {}).setdefault("steps", [])
    if 0 <= index < len(steps):
        steps[index]["latency_ms"] = latency_ms
        steps[index]["status"] = status
        if reason:
            steps[index]["reason"] = reason
    logging.info(
        json.dumps(
            {
                "event": "node_end",
                "trace_id": state.get("trace_id"),
                "node": steps[index]["node"] if 0 <= index < len(steps) else None,
                "latency_ms": latency_ms,
                "status": status,
                "error_code": reason,
            },
            ensure_ascii=False,
        )
    )
    writer = state.get("metrics_writer")
    if writer and hasattr(writer, "end_step"):
        writer.end_step(
            {
                "index": index,
                "status": status,
                "reason": reason,
                "latency_ms": latency_ms,
            }
        )
