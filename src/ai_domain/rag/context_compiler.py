from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List


@dataclass
class ContextCompiler:
    max_chars: int = 6000

    def compile(self, docs: Iterable[Dict[str, Any]]) -> str:
        chunks: List[str] = []
        for doc in docs:
            doc_id = str(doc.get("id") or doc.get("doc_id") or "")
            content = (doc.get("content") or doc.get("text") or "").strip()
            if not content:
                continue
            if doc_id:
                chunks.append(f"[{doc_id}] {content}")
            else:
                chunks.append(content)
        context = "\n\n".join(chunks)
        if len(context) > self.max_chars:
            context = context[: self.max_chars]
        return context
