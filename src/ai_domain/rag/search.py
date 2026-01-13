from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from ai_domain.rag.embedder import LocalEmbedder
from ai_domain.rag.kb_client import FaissKBClient


@dataclass
class FaissSearchService:
    kb_client: FaissKBClient

    @classmethod
    def from_file(
        cls,
        *,
        path: str | Path,
        model_path: str = "embeddings_models/rubert-mini-frida",
        chunk_size: int = 800,
        overlap: int = 100,
    ) -> "FaissSearchService":
        embedder = LocalEmbedder(model_path=model_path)
        kb_client = FaissKBClient.from_path(
            path=path,
            embedder=embedder,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        return cls(kb_client=kb_client)

    async def search(self, query: str, *, top_k: int = 5) -> List[Dict[str, Any]]:
        return await self.kb_client.search(query=query, top_k=top_k)
