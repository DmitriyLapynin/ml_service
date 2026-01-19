from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from ai_domain.rag.embedder import LocalEmbedder
from ai_domain.rag.kb_client import FaissKBClient


def _utc_now() -> str:
    return datetime.utcnow().isoformat() + "Z"


@dataclass
class FunnelKBStore:
    base_dir: Path
    embedder: LocalEmbedder

    def __init__(self, *, base_dir: str | Path, embedder: LocalEmbedder):
        self.base_dir = Path(base_dir)
        self.embedder = embedder

    def _funnel_dir(self, funnel_id: str) -> Path:
        return self.base_dir / funnel_id

    def _kb_dir(self, funnel_id: str, kb_id: str) -> Path:
        return self._funnel_dir(funnel_id) / "kb" / kb_id

    def _manifest_path(self, funnel_id: str) -> Path:
        return self._funnel_dir(funnel_id) / "manifest.json"

    def _load_manifest(self, funnel_id: str) -> Dict[str, Any]:
        path = self._manifest_path(funnel_id)
        if not path.exists():
            return {"funnel_id": funnel_id, "documents": []}
        return json.loads(path.read_text(encoding="utf-8"))

    def _save_manifest(self, funnel_id: str, payload: Dict[str, Any]) -> None:
        path = self._manifest_path(funnel_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(path)

    def add_document(
        self,
        *,
        funnel_id: str,
        source_path: str | Path,
        kb_id: str | None = None,
        chunk_size: int = 400,
        overlap: int = 80,
        min_chunk_chars: int = 120,
        store_source: bool = True,
        supabase_client: Any | None = None,
        file_id: str | None = None,
    ) -> str:
        source_path = Path(source_path)
        kb_id = kb_id or uuid4().hex[:12]
        kb_dir = self._kb_dir(funnel_id, kb_id)
        kb_dir.mkdir(parents=True, exist_ok=True)

        if store_source:
            source_dir = kb_dir / "source"
            source_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, source_dir / source_path.name)

        kb = FaissKBClient.from_path(
            path=source_path,
            embedder=self.embedder,
            chunk_size=chunk_size,
            overlap=overlap,
            min_chunk_chars=min_chunk_chars,
            supabase_client=supabase_client,
            file_id=file_id,
            funnel_id=funnel_id,
            kb_id=kb_id,
        )

        index_path = kb_dir / "faiss.index"
        chunks_path = kb_dir / "chunks.json"
        meta_path = kb_dir / "meta.json"
        mapping_path = kb_dir / "mapping.jsonl"
        kb.save(
            index_path=index_path,
            chunks_path=chunks_path,
            meta_path=meta_path,
            mapping_path=mapping_path,
            extra_meta={
                "kb_id": kb_id,
                "original_filename": source_path.name,
                "source_path": str(source_path),
                "created_at": _utc_now(),
                "chunk_size": chunk_size,
                "overlap": overlap,
                "min_chunk_chars": min_chunk_chars,
                "embedder_model": getattr(self.embedder, "model_path", None),
            },
        )
        logging.info(
            "faiss_built",
            extra={
                "event": "faiss_built",
                "vectors_count": len(kb._chunks),
                "path": str(index_path),
                "funnel_id": funnel_id,
                "kb_id": kb_id,
            },
        )

        manifest = self._load_manifest(funnel_id)
        manifest.setdefault("documents", []).append(
            {
                "kb_id": kb_id,
                "status": "active",
                "original_filename": source_path.name,
                "source_path": str(source_path),
                "created_at": _utc_now(),
            }
        )
        self._save_manifest(funnel_id, manifest)
        return kb_id

    def delete_document(
        self,
        *,
        funnel_id: str,
        kb_id: str,
        delete_files: bool = False,
    ) -> None:
        manifest = self._load_manifest(funnel_id)
        docs = manifest.get("documents") or []
        for doc in docs:
            if doc.get("kb_id") == kb_id:
                doc["status"] = "deleted"
                doc["deleted_at"] = _utc_now()
                break
        self._save_manifest(funnel_id, manifest)

        if delete_files:
            kb_dir = self._kb_dir(funnel_id, kb_id)
            if kb_dir.exists():
                shutil.rmtree(kb_dir)


class FunnelKBResolver:
    def __init__(
        self,
        *,
        base_dir: str | Path,
        embedder: LocalEmbedder,
    ):
        self._base_dir = Path(base_dir)
        self._embedder = embedder
        self._cache: Dict[str, Dict[str, Any]] = {}

    def _manifest_path(self, funnel_id: str) -> Path:
        return self._base_dir / funnel_id / "manifest.json"

    def _kb_dir(self, funnel_id: str, kb_id: str) -> Path:
        return self._base_dir / funnel_id / "kb" / kb_id

    def _load_manifest(self, funnel_id: str) -> Dict[str, Any]:
        path = self._manifest_path(funnel_id)
        if not path.exists():
            return {"funnel_id": funnel_id, "documents": []}
        return json.loads(path.read_text(encoding="utf-8"))

    def _get_active_docs(self, funnel_id: str) -> List[Dict[str, Any]]:
        manifest = self._load_manifest(funnel_id)
        docs = manifest.get("documents") or []
        return [d for d in docs if d.get("status") == "active"]

    def _ensure_cache(self, funnel_id: str) -> None:
        manifest_path = self._manifest_path(funnel_id)
        mtime = manifest_path.stat().st_mtime if manifest_path.exists() else 0
        cached = self._cache.get(funnel_id)
        if cached and cached.get("mtime") == mtime:
            return

        docs = self._get_active_docs(funnel_id)
        kb_map: Dict[str, FaissKBClient] = {}
        for doc in docs:
            kb_id = doc.get("kb_id")
            if not kb_id:
                continue
            kb_dir = self._kb_dir(funnel_id, kb_id)
            index_path = kb_dir / "faiss.index"
            chunks_path = kb_dir / "chunks.json"
            if not index_path.exists() or not chunks_path.exists():
                continue
            kb_map[kb_id] = FaissKBClient.load(
                index_path=index_path,
                chunks_path=chunks_path,
                embedder=self._embedder,
            )

        self._cache[funnel_id] = {"mtime": mtime, "clients": kb_map}

    async def search(
        self,
        *,
        funnel_id: str,
        query: str,
        top_k: int = 5,
        top_k_per_doc: int = 5,
    ) -> List[Dict[str, Any]]:
        self._ensure_cache(funnel_id)
        cached = self._cache.get(funnel_id) or {}
        clients: Dict[str, FaissKBClient] = cached.get("clients") or {}
        results: List[Dict[str, Any]] = []
        for kb_id, kb in clients.items():
            docs = await kb.search(query=query, top_k=top_k_per_doc)
            for doc in docs:
                payload = dict(doc)
                payload["kb_id"] = kb_id
                results.append(payload)

        results.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        return results[:top_k]
