import asyncio
import json
import logging
import os
import time

from ai_domain.tools.registry import ToolRegistry, default_registry
from ai_domain.utils.hashing import hash_text_short

from .utils import ensure_lists, log_node, step_begin, step_end


def _redact_args(args: dict) -> dict:
    redacted = {}
    for key, value in args.items():
        key_lower = str(key).lower()
        if any(
            token in key_lower
            for token in (
                "phone",
                "email",
                "name",
                "address",
                "token",
                "secret",
                "api_key",
                "apikey",
                "password",
                "key",
            )
        ):
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
    metrics_writer = state.get("metrics_writer")
    tool_calls = state.get("tool_calls") or []
    tool_results = []
    tool_messages = []
    trace = state.setdefault("trace", {})
    tool_call_meta = trace.setdefault("tool_calls", [])
    max_tool_calls_meta = int((state.get("policies") or {}).get("max_tool_calls_meta", 20))
    max_preview_chars = int((state.get("policies") or {}).get("max_preview_chars", 160))
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
        state["tool_messages"] = []
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
        args_preview = _redact_args(args if isinstance(args, dict) else {})
        args_fingerprint = hash_text_short(json.dumps(args_preview, ensure_ascii=False))
        result_payload = result.result if result.ok else result.error
        result_preview = _text_preview(
            json.dumps(result_payload or {}, ensure_ascii=False), limit=max_preview_chars
        )
        result_fingerprint = hash_text_short(json.dumps(result_payload or {}, ensure_ascii=False))
        meta = {
            "call_id": call_id or result.call_id,
            "name": name,
            "args": args_preview,
            "latency_ms": result.latency_ms,
            "ok": result.ok,
            "error_code": (result.error or {}).get("code") if result.error else None,
            "error_message": (result.error or {}).get("message") if result.error else None,
            "result_size": _payload_size(result.result if result.ok else result.error),
            "provider": (result.result or {}).get("provider") if isinstance(result.result, dict) else None,
        }
        if os.getenv("AI_DOMAIN_DEBUG_LOGGING", "false").lower() in {"1", "true", "yes"}:
            logging.info(
                json.dumps(
                    {
                        "event": "tool_debug",
                        "trace_id": state.get("trace_id"),
                        "tool_name": name,
                        "call_id": call_id or result.call_id,
                        "tool_args": args,
                        "tool_result_preview": result_preview,
                    },
                    ensure_ascii=False,
                    default=str,
                )
            )
        if metrics_writer and hasattr(metrics_writer, "add_tool_call"):
            metrics_writer.add_tool_call(
                {
                    "node": "retrieve",
                    "call_id": call_id or result.call_id,
                    "tool_name": name,
                    "ok": result.ok,
                    "latency_ms": result.latency_ms,
                    "args_fingerprint": args_fingerprint,
                    "args_preview": args_preview if state.get("debug") else None,
                    "result_fingerprint": result_fingerprint,
                    "result_preview": result_preview if state.get("debug") else None,
                }
            )
        return result, tool_message, meta

    tasks = [_run_call(call) for call in tool_calls]
    results = await asyncio.gather(*tasks)
    for result, tool_message, meta in results:
        tool_results.append(result)
        tool_messages.append(tool_message)
        if max_tool_calls_meta <= 0 or len(tool_call_meta) < max_tool_calls_meta:
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
                            "text_preview": _text_preview(content, limit=max_preview_chars),
                            "metadata": doc.get("metadata") or {},
                        }
                    )
                    rag_meta["final_context_chars"] += len(content)

    state["tool_results"] = tool_results
    state["tool_messages"] = tool_messages
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
    trace["rag"] = rag_meta
    if metrics_writer and hasattr(metrics_writer, "add_rag_call"):
        doc_ids = [doc.get("doc_id") for doc in rag_meta.get("documents") or []]
        docs_fingerprint = hash_text_short(json.dumps(doc_ids, ensure_ascii=False))
        query = rag_meta.get("query") or ""
        metrics_writer.add_rag_call(
            {
                "node": "retrieve",
                "query_fingerprint": hash_text_short(query),
                "query_preview": _text_preview(query, limit=max_preview_chars) if state.get("debug") else None,
                "docs_fingerprint": docs_fingerprint,
                "docs_count": len(rag_meta.get("documents") or []),
                "metadata": {
                    "top_k": rag_meta.get("top_k"),
                    "config_id": rag_meta.get("config_id"),
                    "scores": [doc.get("score") for doc in rag_meta.get("documents") or []],
                    "doc_ids": doc_ids,
                    "chunk_offsets": [
                        {
                            "start": doc.get("offset_start"),
                            "end": doc.get("offset_end"),
                        }
                        for doc in rag_meta.get("documents") or []
                    ],
                    "retrieval_latency_ms": rag_meta.get("retrieval_latency_ms"),
                },
            }
        )
    if tool_results:
        logging.info(
            json.dumps(
                {
                    "event": "tool_results_count",
                    "trace_id": state.get("trace_id"),
                    "count": len(tool_results),
                },
                ensure_ascii=False,
            )
        )
    latency_ms = int((time.perf_counter() - start) * 1000)
    step_end(state, index=step_index, latency_ms=latency_ms, status="ok")
    return state
