import argparse
import asyncio
import json
from pathlib import Path

from ai_domain.rag.embedder import LocalEmbedder
from ai_domain.rag.kb_client import FaissKBClient, KBChunk


def _save_chunks(chunks, path: Path) -> None:
    payload = [{"id": c.id, "text": c.text, "metadata": c.metadata} for c in chunks]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_chunks(path: Path) -> list[KBChunk]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [KBChunk(id=item["id"], text=item["text"], metadata=item.get("metadata") or {}) for item in data]


def build_index(
    *,
    source_path: Path,
    index_path: Path,
    chunks_path: Path,
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
    faiss.write_index(kb._index, str(index_path))
    _save_chunks(kb._chunks, chunks_path)


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
    parser.add_argument("--index-path", type=str, default="data/faiss.index")
    parser.add_argument("--chunks-path", type=str, default="data/faiss_chunks.json")
    parser.add_argument("--model-path", type=str, default="embeddings_models/rubert-mini-frida")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--query", type=str, default="")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    source_path = Path(args.source)
    index_path = Path(args.index_path)
    chunks_path = Path(args.chunks_path)

    if args.rebuild or not index_path.exists() or not chunks_path.exists():
        build_index(
            source_path=source_path,
            index_path=index_path,
            chunks_path=chunks_path,
            model_path=args.model_path,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
        )

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
