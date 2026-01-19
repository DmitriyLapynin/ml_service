import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from ai_domain.rag.embedder import LocalEmbedder
from ai_domain.rag.kb_client import FaissKBClient, KBChunk
from ai_domain.rag.funnel_store import FunnelKBStore
from ai_domain.rag.supabase_store import upsert_kb_file, update_kb_file_status
from ai_domain.registry.supabase_connector import (
    SupabaseConfigError,
    create_supabase_client_from_env,
)


def _save_chunks(chunks, path: Path) -> None:
    payload = [{"id": c.id, "text": c.text, "metadata": c.metadata} for c in chunks]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _load_chunks(path: Path) -> list[KBChunk]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [KBChunk(id=item["id"], text=item["text"], metadata=item.get("metadata") or {}) for item in data]


def _try_create_supabase() -> Any | None:
    try:
        return create_supabase_client_from_env()
    except SupabaseConfigError:
        return None

def build_index(
    *,
    source_path: Path,
    index_path: Path,
    chunks_path: Path,
    mapping_path: Path | None,
    model_path: str,
    chunk_size: int,
    overlap: int,
) -> None:
    try:
        import faiss  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("faiss is required for FAISS index build") from exc

    embedder = LocalEmbedder(model_path=model_path)
    kb = FaissKBClient.from_path(
        path=source_path,
        embedder=embedder,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    index_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_index = index_path.with_suffix(index_path.suffix + ".tmp")
    faiss.write_index(kb._index, str(tmp_index))
    tmp_index.replace(index_path)
    _save_chunks(kb._chunks, chunks_path)
    if mapping_path:
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_mapping = mapping_path.with_suffix(mapping_path.suffix + ".tmp")
        with tmp_mapping.open("w", encoding="utf-8") as handle:
            for idx, chunk in enumerate(kb._chunks):
                payload = {"vector_id": idx, "chunk_id": chunk.metadata.get("chunk_id")}
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        tmp_mapping.replace(mapping_path)


async def search_index(
    *,
    index_path: Path,
    chunks_path: Path,
    model_path: str,
    query: str,
    top_k: int,
) -> None:
    try:
        import faiss  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("faiss is required for FAISS search") from exc

    embedder = LocalEmbedder(model_path=model_path)
    index = faiss.read_index(str(index_path))
    chunks = _load_chunks(chunks_path)
    kb = FaissKBClient(index=index, chunks=chunks, embedder=embedder)
    results = await kb.search(query=query, top_k=top_k)
    print(json.dumps(results, ensure_ascii=False, indent=2))


async def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS index and run search.")
    parser.add_argument("--source", type=str, default="Dental_Clinic_Knowledge_Base.txt")
    data_dir = Path(os.getenv("AI_DOMAIN_DATA_DIR", "data"))
    models_dir = Path(os.getenv("AI_DOMAIN_MODELS_DIR", "embeddings_models"))
    parser.add_argument("--index-path", type=str, default=str(data_dir / "faiss.index"))
    parser.add_argument("--chunks-path", type=str, default=str(data_dir / "faiss_chunks.json"))
    parser.add_argument("--model-path", type=str, default=str(models_dir / "rubert-mini-frida"))
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--query", type=str, default="")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--funnel-id", type=str, default="")
    parser.add_argument("--kb-id", type=str, default="")
    parser.add_argument("--source-type", type=str, default="")
    parser.add_argument("--source-uri", type=str, default="")
    parser.add_argument("--mapping-path", type=str, default="")
    args = parser.parse_args()

    source_path = Path(args.source)
    index_path = Path(args.index_path)
    chunks_path = Path(args.chunks_path)
    mapping_path = Path(args.mapping_path) if args.mapping_path else None
    embedder = LocalEmbedder(model_path=args.model_path)

    sb = _try_create_supabase()
    kb_id = args.kb_id or ""
    if args.funnel_id and not kb_id:
        kb_id = uuid4().hex[:12]
    file_id = None
    if args.funnel_id and kb_id and sb is not None:
        payload = upsert_kb_file(
            sb,
            funnel_id=args.funnel_id,
            kb_id=kb_id,
            source_name=source_path.name,
            payload_bytes=source_path.read_bytes(),
            source_type=args.source_type or None,
            source_uri=args.source_uri or None,
        )
        file_id = payload.get("file_id")
    try:
        if args.funnel_id:
            store = FunnelKBStore(base_dir=data_dir / "funnels", embedder=embedder)
            store.add_document(
                funnel_id=args.funnel_id,
                source_path=source_path,
                kb_id=kb_id or None,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
                supabase_client=sb,
                file_id=file_id,
            )
        elif args.rebuild or not index_path.exists() or not chunks_path.exists():
            build_index(
                source_path=source_path,
                index_path=index_path,
                chunks_path=chunks_path,
                mapping_path=mapping_path,
                model_path=args.model_path,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
            )
        if sb is not None and file_id:
            update_kb_file_status(sb, file_id=file_id, status="ready")
    except Exception:
        if sb is not None and file_id:
            update_kb_file_status(sb, file_id=file_id, status="failed")
        raise

    if args.query:
        await search_index(
            index_path=index_path,
            chunks_path=chunks_path,
            model_path=args.model_path,
            query=args.query,
            top_k=args.top_k,
        )


if __name__ == "__main__":
    asyncio.run(main())
