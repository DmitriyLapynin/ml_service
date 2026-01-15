import asyncio
import json
import logging
import time

from ai_domain.tools.registry import ToolRegistry, default_registry

from .utils import ensure_lists, log_node, step_begin, step_end


def _redact_args(args: dict) -> dict:
    redacted = {}
    for key, value in args.items():
        key_lower = str(key).lower()
        if any(token in key_lower for token in ("phone", "email", "name", "address")):
            redacted[key] = "***"
        else:
            redacted[key] = value
    return redacted


def _payload_size(payload: dict | None) -> int:
    if not payload:
        return 0
    try:
        return len(json.dumps(payload, ensure_ascii=False))
    except Exception:
        return 0


def _text_preview(text: str, limit: int = 160) -> str:
    if not text:
        return ""
    text = text.strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit].rstrip()}..."


async def tool_executor_node(state: dict) -> dict:
    ensure_lists(state)
    log_node(state, "retrieve")
    step_index = step_begin(state, "retrieve")
    start = time.perf_counter()
    state["executed"].append("retrieve")
    registry: ToolRegistry = state.get("tool_registry") or default_registry()
    tool_calls = state.get("tool_calls") or []
    tool_results = []
    tool_messages = []
    tool_call_meta = []
    rag_meta = {
        "enabled": bool(state.get("is_rag", False)),
        "config_id": state.get("funnel_id"),
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
    if not tool_calls:
        state["context"] = ""
        state["rag_meta"] = rag_meta
        latency_ms = int((time.perf_counter() - start) * 1000)
        step_end(state, index=step_index, latency_ms=latency_ms, status="skip", reason="no_tool_calls")
        return state
    per_request_limit = int((state.get("policies") or {}).get("max_tool_concurrency_per_request", 3))
    limiter = asyncio.Semaphore(per_request_limit) if per_request_limit > 0 else None

    async def _run_call(call: dict):
        name = call.get("name")
        args = call.get("args") or call.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}
        call_id = call.get("id")
        if limiter is None:
            result = await registry.execute(
                name,
                args,
                state=state,
                trace_id=state.get("trace_id"),
                call_id=call_id,
            )
        else:
            async with limiter:
                result = await registry.execute(
                    name,
                    args,
                    state=state,
                    trace_id=state.get("trace_id"),
                    call_id=call_id,
                )
        payload = {
            "ok": result.ok,
            "data": result.result if result.ok else None,
            "error": result.error if not result.ok else None,
        }
        tool_message = {
            "role": "tool",
            "content": json.dumps(payload, ensure_ascii=False),
            "tool_call_id": call_id or result.call_id,
        }
        meta = {
            "call_id": call_id or result.call_id,
            "name": name,
            "args": _redact_args(args if isinstance(args, dict) else {}),
            "latency_ms": result.latency_ms,
            "ok": result.ok,
            "error_code": (result.error or {}).get("code") if result.error else None,
            "error_message": (result.error or {}).get("message") if result.error else None,
            "result_size": _payload_size(result.result if result.ok else result.error),
            "provider": (result.result or {}).get("provider") if isinstance(result.result, dict) else None,
        }
        return result, tool_message, meta

    tasks = [_run_call(call) for call in tool_calls]
    results = await asyncio.gather(*tasks)
    for result, tool_message, meta in results:
        tool_results.append(result)
        tool_messages.append(tool_message)
        tool_call_meta.append(meta)
        if meta.get("name") == "knowledge_search":
            rag_meta["retrieval_latency_ms"] += int(result.latency_ms)
            args = meta.get("args") or {}
            rag_meta["query"] = args.get("query") or rag_meta["query"]
            rag_meta["top_k"] = args.get("top_k") or rag_meta["top_k"]
            docs = []
            if result.ok and isinstance(result.result, dict):
                docs = result.result.get("documents") or []
            if isinstance(docs, list):
                for doc in docs:
                    if not isinstance(doc, dict):
                        continue
                    content = doc.get("content") or ""
                    rag_meta["documents"].append(
                        {
                            "doc_id": doc.get("id"),
                            "source_id": (doc.get("metadata") or {}).get("source"),
                            "chunk_id": doc.get("id"),
                            "score": doc.get("score"),
                            "offset_start": (doc.get("metadata") or {}).get("offset_start"),
                            "offset_end": (doc.get("metadata") or {}).get("offset_end"),
                            "text_preview": _text_preview(content),
                            "metadata": doc.get("metadata") or {},
                        }
                    )
                    rag_meta["final_context_chars"] += len(content)

    state["tool_results"] = tool_results
    state["tool_messages"] = tool_messages
    state["tool_call_meta"] = tool_call_meta
    if rag_meta["documents"]:
        seen = set()
        deduped = []
        for doc in rag_meta["documents"]:
            key = (doc.get("doc_id"), doc.get("chunk_id"))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(doc)
        rag_meta["deduped_count"] = len(rag_meta["documents"]) - len(deduped)
        rag_meta["documents"] = deduped
    state["rag_meta"] = rag_meta
    if tool_results:
        logging.info("tool_results_count=%s", len(tool_results))
    latency_ms = int((time.perf_counter() - start) * 1000)
    step_end(state, index=step_index, latency_ms=latency_ms, status="ok")
    return state
