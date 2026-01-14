from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class RagRetrieveNode:
    rag_client: Any
    prompt_repo: Any | None = None
    telemetry: Any | None = None

    # политики и ключи
    rag_policy_key: str = "rag_enabled"
    rag_config_key: str = "funnel_id"  # funnel_id используется как selector KB
    default_top_k: int = 5
    max_context_chars: int = 6000  # защита от огромного контекста

    async def __call__(self, state) -> Any:
        runtime = getattr(state, "runtime", None) or {}
        runtime.setdefault("executed", [])
        runtime.setdefault("errors", [])
        runtime.setdefault("degraded", False)
        runtime.setdefault("rag", {"used": False})

        runtime["executed"].append("rag_retrieve")

        policies: Dict[str, Any] = getattr(state, "policies", {}) or {}
        rag_enabled = bool(policies.get(self.rag_policy_key, False))

        # Если RAG выключен — сразу выходим
        if not rag_enabled:
            runtime["rag"] = {"used": False, "reason": "rag_disabled"}
            setattr(state, "runtime", runtime)
            return state

        # Выбираем query:
        query = self._pick_query(state)
        if not query:
            runtime["rag"] = {"used": False, "reason": "no_query"}
            setattr(state, "runtime", runtime)
            return state

        versions: Dict[str, Any] = getattr(state, "versions", {}) or {}
        rag_config_id = (
            policies.get(self.rag_config_key)
            or versions.get(self.rag_config_key)
            or policies.get("rag_config_id")
            or versions.get("rag_config_id")
        )

        top_k = int(policies.get("rag_top_k", self.default_top_k))

        try:
            docs = await self.rag_client.search(query=query, top_k=top_k, rag_config_id=rag_config_id)
        except Exception as e:
            runtime["degraded"] = True
            runtime["errors"].append({"node": "rag_retrieve", "type": "rag_error", "msg": str(e)})
            runtime["rag"] = {"used": False, "reason": "rag_error"}
            setattr(state, "runtime", runtime)
            return state

        # Нормализуем docs
        # ожидаем список dict: {id, score, content}
        doc_ids: List[str] = []
        scores: List[float] = []
        chunks: List[str] = []

        for d in docs or []:
            doc_id = str(d.get("id") or d.get("doc_id") or "")
            if doc_id:
                doc_ids.append(doc_id)
            scores.append(float(d.get("score") or 0.0))
            content = (d.get("content") or d.get("text") or "").strip()
            if content:
                chunks.append(f"[{doc_id}] {content}")

        context = "\n\n".join(chunks)
        if len(context) > self.max_context_chars:
            context = context[: self.max_context_chars]

        runtime["rag"] = {
            "used": True if doc_ids else False,
            "query": query,
            "doc_ids": doc_ids,
            "scores": scores,
            "context": context,
            "rag_config_id": rag_config_id,
            "top_k": top_k,
        }

        setattr(state, "runtime", runtime)

        if self.telemetry:
            try:
                self.telemetry.log_step(
                    trace_id=getattr(state, "trace_id", None),
                    node="rag_retrieve",
                    meta={"used": runtime["rag"]["used"], "doc_ids": doc_ids, "rag_config_id": rag_config_id},
                )
            except Exception:
                pass

        return state

    def _pick_query(self, state) -> Optional[str]:
        runtime = getattr(state, "runtime", {}) or {}
        analysis = runtime.get("analysis") or {}
        if isinstance(analysis, dict):
            q = analysis.get("rag_query")
            if isinstance(q, str) and q.strip():
                return q.strip()

        # иначе — последний user message
        messages = getattr(state, "messages", []) or []
        for m in reversed(messages):
            if isinstance(m, dict) and m.get("role") == "user":
                c = (m.get("content") or "").strip()
                if c:
                    return c
        return None
