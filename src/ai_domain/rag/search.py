from __future__ import annotations

from dataclasses import dataclass
import os
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
        model_path: str = "",
        chunk_size: int = 800,
        overlap: int = 100,
    ) -> "FaissSearchService":
        if not model_path:
            models_dir = os.getenv("AI_DOMAIN_MODELS_DIR", "embeddings_models")
            model_path = str(Path(models_dir) / "rubert-mini-frida")
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
