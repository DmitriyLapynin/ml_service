from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Protocol

import numpy as np

from ai_domain.rag.embedder import LocalEmbedder
from ai_domain.rag.file_loader import load_text_from_file
from ai_domain.rag.loaders import DocumentLike, load_documents


class Chunker(Protocol):
    def __call__(
        self,
        documents: Sequence[DocumentLike],
        *,
        chunk_size: int,
        overlap: int,
        min_chars: int,
    ) -> List[Tuple[str, Dict[str, Any], int, int]]:
        ...


@dataclass(frozen=True)
class KBChunk:
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _extract_metadata_int(meta: Dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = meta.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _extract_metadata_str(meta: Dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = meta.get(key)
        if value is None:
            continue
        return str(value)
    return None


def _insert_kb_chunks(
    sb: Any | None,
    *,
    file_id: str | None,
    funnel_id: str | None,
    kb_id: str | None,
    chunks: Sequence[KBChunk],
    batch_size: int = 200,
) -> bool:
    if sb is None or not file_id or not funnel_id or not kb_id:
        return False
    try:
        existing = sb.table("kb_chunks").select("id").eq("file_id", file_id).limit(1).execute()
        if getattr(existing, "data", None):
            logging.info(
                "kb_chunks_skipped",
                extra={"event": "kb_chunks_skipped", "file_id": file_id},
            )
            return False
    except Exception as exc:
        logging.warning(
            "supabase_kb_chunks_check_failed",
            extra={"event": "supabase_kb_chunks_check_failed", "error": str(exc)},
        )
        return False
    records: List[Dict[str, Any]] = []
    for idx, chunk in enumerate(chunks):
        meta = chunk.metadata or {}
        records.append(
            {
                "file_id": file_id,
                "funnel_id": funnel_id,
                "kb_id": kb_id,
                "chunk_index": idx,
                "text": chunk.text,
                "text_hash": _sha256_text(chunk.text),
                "char_start": _extract_metadata_int(meta, "offset_start", "char_start"),
                "char_end": _extract_metadata_int(meta, "offset_end", "char_end"),
                "page": _extract_metadata_int(meta, "page_number", "page"),
                "sheet": _extract_metadata_str(meta, "sheet_name", "sheet"),
                "metadata": meta,
            }
        )
    for offset in range(0, len(records), batch_size):
        batch = records[offset : offset + batch_size]
        try:
            resp = sb.table("kb_chunks").insert(batch).execute()
            data = resp.data or []
            index_map = {item.get("chunk_index"): item.get("id") for item in data if item.get("chunk_index") is not None}
            for idx, chunk in enumerate(chunks[offset : offset + batch_size]):
                chunk_id = index_map.get(idx + offset)
                if chunk_id and chunk.metadata is not None:
                    chunk.metadata["chunk_id"] = chunk_id
        except Exception as exc:
            logging.warning(
                "supabase_kb_chunks_insert_failed",
                extra={"event": "supabase_kb_chunks_insert_failed", "error": str(exc)},
            )
    logging.info(
        "kb_chunks_written",
        extra={
            "event": "kb_chunks_written",
            "funnel_id": funnel_id,
            "kb_id": kb_id,
            "count": len(records),
        },
    )
    return True


def _insert_kb_embeddings(
    sb: Any | None,
    *,
    chunks: Sequence[KBChunk],
    embeddings: np.ndarray,
    model_name: str,
    model_version: str,
    batch_size: int = 200,
) -> None:
    if sb is None:
        return
    records: List[Dict[str, Any]] = []
    dim = int(embeddings.shape[1]) if embeddings.ndim == 2 else 0
    for idx, chunk in enumerate(chunks):
        chunk_id = (chunk.metadata or {}).get("chunk_id")
        if not chunk_id:
            continue
        vector = embeddings[idx].tolist()
        norm = float(np.linalg.norm(embeddings[idx]))
        records.append(
            {
                "chunk_id": chunk_id,
                "model_name": model_name,
                "model_version": model_version,
                "dim": dim,
                "embedding": vector,
                "norm": norm,
            }
        )
    if not records:
        return
    for offset in range(0, len(records), batch_size):
        batch = records[offset : offset + batch_size]
        try:
            sb.table("kb_embeddings").insert(batch).execute()
        except Exception as exc:
            logging.warning(
                "supabase_kb_embeddings_insert_failed",
                extra={"event": "supabase_kb_embeddings_insert_failed", "error": str(exc)},
            )
    logging.info(
        "kb_embeddings_written",
        extra={
            "event": "kb_embeddings_written",
            "count": len(records),
            "dim": dim,
            "model_version": model_version,
            "model_name": model_name,
        },
    )

def _chunk_text(text: str, *, chunk_size: int, overlap: int) -> List[Tuple[int, int, str]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("langchain-text-splitters is required for recursive splitting") from exc

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
        is_separator_regex=False,
    )
    docs = splitter.create_documents([text])
    spans: List[Tuple[int, int, str]] = []
    for doc in docs:
        start = int(doc.metadata.get("start_index") or 0)
        content = doc.page_content
        end = start + len(content)
        if content.strip():
            spans.append((start, end, content))
    return spans


def _merge_short_chunks(
    items: List[Tuple[str, Dict[str, Any], int, int]],
    *,
    min_chars: int,
) -> List[Tuple[str, Dict[str, Any], int, int]]:
    if min_chars <= 0:
        return items
    merged: List[Tuple[str, Dict[str, Any], int, int]] = []
    idx = 0
    def _normalized_len(text: str) -> int:
        return len(text.strip())

    def _word_count(text: str) -> int:
        return len([w for w in text.strip().split() if w])

    while idx < len(items):
        content, metadata, start, end = items[idx]
        if _normalized_len(content) >= min_chars or idx == len(items) - 1:
            merged.append((content, metadata, start, end))
            idx += 1
            continue
        next_content, next_meta, next_start, next_end = items[idx + 1]
        combined = f"{content}{next_content}"
        combined_start = start
        combined_end = next_end
        idx += 2
        merged.append((combined, metadata, combined_start, combined_end))

        while idx < len(items):
            if _normalized_len(combined) >= min_chars:
                break
            candidate_content, candidate_meta, candidate_start, candidate_end = items[idx]
            if _word_count(candidate_content) < 3:
                current_page = metadata.get("page_number")
                candidate_page = candidate_meta.get("page_number")
                if candidate_page is not None and current_page is not None and candidate_page != current_page:
                    break
            combined = f"{combined}{candidate_content}"
            combined_end = candidate_end
            merged[-1] = (combined, metadata, combined_start, combined_end)
            idx += 1
    return [item for item in merged if _word_count(item[0]) >= 2]


def _chunk_documents(
    documents: Sequence[DocumentLike],
    *,
    chunk_size: int,
    overlap: int,
    min_chars: int,
) -> List[Tuple[str, Dict[str, Any], int, int]]:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("langchain-text-splitters is required for recursive splitting") from exc

    texts = [doc["text"] for doc in documents]
    metadatas = [doc["metadata"] for doc in documents]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
        is_separator_regex=False,
    )
    docs = splitter.create_documents(texts, metadatas=metadatas)
    chunks: List[Tuple[str, Dict[str, Any], int, int]] = []
    for doc in docs:
        meta = dict(getattr(doc, "metadata", {}) or {})
        start = int(meta.pop("start_index", 0))
        content = doc.page_content
        end = start + len(content)
        if content.strip():
            chunks.append((content, meta, start, end))
    return _merge_short_chunks(chunks, min_chars=min_chars)


def default_chunker(
    documents: Sequence[DocumentLike],
    *,
    chunk_size: int,
    overlap: int,
    min_chars: int,
) -> List[Tuple[str, Dict[str, Any], int, int]]:
    return _chunk_documents(
        documents,
        chunk_size=chunk_size,
        overlap=overlap,
        min_chars=min_chars,
    )


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def _atomic_write_text(path: Path, payload: str) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(payload, encoding="utf-8")
    tmp_path.replace(path)


class FaissKBClient:
    def __init__(
        self,
        *,
        index: Any,
        chunks: Sequence[KBChunk],
        embedder: LocalEmbedder,
    ):
        self._index = index
        self._chunks = list(chunks)
        self._embedder = embedder

    @classmethod
    def from_text_file(
        cls,
        *,
        path: str | Path,
        embedder: LocalEmbedder,
        chunk_size: int = 800,
        overlap: int = 100,
        min_chunk_chars: int = 120,
        chunker: Chunker | None = None,
        supabase_client: Any | None = None,
        file_id: str | None = None,
        funnel_id: str | None = None,
        kb_id: str | None = None,
    ) -> "FaissKBClient":
        try:
            import faiss  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("faiss is required for FAISS-backed RAG") from exc

        text = load_text_from_file(path)
        if chunker is None:
            spans = _chunk_text(text, chunk_size=chunk_size, overlap=overlap)
            chunks: List[KBChunk] = []
            raw_items: List[Tuple[str, Dict[str, Any], int, int]] = []
            for start, end, chunk_text in spans:
                raw_items.append((chunk_text, {"source": str(path)}, start, end))
            merged_items = _merge_short_chunks(raw_items, min_chars=min_chunk_chars)
            for idx, (chunk_text, base_meta, start, end) in enumerate(merged_items):
                trimmed = chunk_text.strip()
                if not trimmed:
                    continue
                trim_left = len(chunk_text) - len(chunk_text.lstrip())
                trim_right = len(chunk_text.rstrip())
                final_start = start + trim_left
                final_end = start + trim_right
                chunks.append(
                    KBChunk(
                        id=f"chunk-{idx:04d}",
                        text=trimmed,
                        metadata={
                            **base_meta,
                            "offset_start": final_start,
                            "offset_end": final_end,
                        },
                    )
                )
            chunks_inserted = _insert_kb_chunks(
                supabase_client,
                file_id=file_id,
                funnel_id=funnel_id,
                kb_id=kb_id,
                chunks=chunks,
            )
            return cls._build_index(
                chunks,
                embedder,
                faiss,
                supabase_client=supabase_client,
                write_embeddings=chunks_inserted,
            )

        documents = [{"text": text, "metadata": {"source": str(path)}}]
        return cls.from_documents(
            documents=documents,
            embedder=embedder,
            chunk_size=chunk_size,
            overlap=overlap,
            min_chunk_chars=min_chunk_chars,
            chunker=chunker,
            supabase_client=supabase_client,
            file_id=file_id,
            funnel_id=funnel_id,
            kb_id=kb_id,
        )

    @classmethod
    def from_documents(
        cls,
        *,
        documents: Sequence[DocumentLike],
        embedder: LocalEmbedder,
        chunk_size: int = 800,
        overlap: int = 100,
        min_chunk_chars: int = 120,
        chunker: Chunker | None = None,
        supabase_client: Any | None = None,
        file_id: str | None = None,
        funnel_id: str | None = None,
        kb_id: str | None = None,
    ) -> "FaissKBClient":
        try:
            import faiss  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("faiss is required for FAISS-backed RAG") from exc

        chunks: List[KBChunk] = []
        split_fn = chunker or default_chunker
        split_chunks = split_fn(
            documents,
            chunk_size=chunk_size,
            overlap=overlap,
            min_chars=min_chunk_chars,
        )
        for idx, (content, metadata, start, end) in enumerate(split_chunks):
            chunk_text = content
            trimmed = chunk_text.strip()
            if not trimmed:
                continue
            trim_left = len(chunk_text) - len(chunk_text.lstrip())
            trim_right = len(chunk_text.rstrip())
            final_start = start + trim_left
            final_end = start + trim_right
            chunk_meta = dict(metadata)
            chunk_meta["offset_start"] = final_start
            chunk_meta["offset_end"] = final_end
            chunks.append(
                KBChunk(
                    id=f"chunk-{idx:04d}",
                    text=trimmed,
                    metadata=chunk_meta,
                )
            )
        chunks_inserted = _insert_kb_chunks(
            supabase_client,
            file_id=file_id,
            funnel_id=funnel_id,
            kb_id=kb_id,
            chunks=chunks,
        )
        return cls._build_index(
            chunks,
            embedder,
            faiss,
            supabase_client=supabase_client,
            write_embeddings=chunks_inserted,
        )

    @classmethod
    def from_path(
        cls,
        *,
        path: str | Path,
        embedder: LocalEmbedder,
        chunk_size: int = 800,
        overlap: int = 100,
        min_chunk_chars: int = 120,
        chunker: Chunker | None = None,
        supabase_client: Any | None = None,
        file_id: str | None = None,
        funnel_id: str | None = None,
        kb_id: str | None = None,
    ) -> "FaissKBClient":
        documents = load_documents(path)
        return cls.from_documents(
            documents=documents,
            embedder=embedder,
            chunk_size=chunk_size,
            overlap=overlap,
            min_chunk_chars=min_chunk_chars,
            chunker=chunker,
            supabase_client=supabase_client,
            file_id=file_id,
            funnel_id=funnel_id,
            kb_id=kb_id,
        )

    @classmethod
    def _build_index(
        cls,
        chunks: Sequence[KBChunk],
        embedder: LocalEmbedder,
        faiss_module: Any,
        *,
        supabase_client: Any | None = None,
        model_version: str | None = None,
        write_embeddings: bool = True,
    ) -> "FaissKBClient":
        embeddings = embedder.embed_texts([c.text for c in chunks])
        if embeddings.size == 0:
            raise ValueError("no embeddings produced from input file")
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings, dtype="float32")
        embeddings = embeddings.astype("float32")
        embeddings = _l2_normalize(embeddings)
        if write_embeddings:
            _insert_kb_embeddings(
                supabase_client,
                chunks=chunks,
                embeddings=embeddings,
                model_name=getattr(embedder, "model_path", "unknown"),
                model_version=model_version or "unknown",
            )
        dim = embeddings.shape[1]
        index = faiss_module.IndexFlatIP(dim)
        index.add(embeddings)
        return cls(index=index, chunks=chunks, embedder=embedder)

    async def search(
        self,
        *,
        query: str,
        top_k: int = 5,
        rag_config_id: str | None = None,
    ) -> List[Dict[str, Any]]:
        _ = rag_config_id
        if not query:
            return []
        query_vec = self._embedder.embed_texts([query])
        if query_vec.size == 0:
            return []
        query_vec = _l2_normalize(query_vec.astype("float32"))
        distances, indices = self._index.search(query_vec, int(top_k))
        results: List[Dict[str, Any]] = []
        for score, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self._chunks):
                continue
            chunk = self._chunks[int(idx)]
            results.append(
                {
                    "id": chunk.id,
                    "score": float(score),
                    "content": chunk.text,
                    "metadata": chunk.metadata,
                }
            )
        return results

    def save(
        self,
        *,
        index_path: str | Path,
        chunks_path: str | Path,
        meta_path: str | Path | None = None,
        mapping_path: str | Path | None = None,
        extra_meta: Dict[str, Any] | None = None,
    ) -> None:
        try:
            import faiss  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("faiss is required for FAISS save") from exc

        index_path = Path(index_path)
        chunks_path = Path(chunks_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        chunks_path.parent.mkdir(parents=True, exist_ok=True)

        tmp_index = index_path.with_suffix(index_path.suffix + ".tmp")
        faiss.write_index(self._index, str(tmp_index))
        tmp_index.replace(index_path)
        chunks_payload = [
            {"id": c.id, "text": c.text, "metadata": c.metadata} for c in self._chunks
        ]
        _atomic_write_text(
            chunks_path,
            json.dumps(chunks_payload, ensure_ascii=False, indent=2),
        )
        if mapping_path:
            mapping_path = Path(mapping_path)
            mapping_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_mapping = mapping_path.with_suffix(mapping_path.suffix + ".tmp")
            with tmp_mapping.open("w", encoding="utf-8") as handle:
                for idx, chunk in enumerate(self._chunks):
                    payload = {"vector_id": idx, "chunk_id": chunk.metadata.get("chunk_id")}
                    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            tmp_mapping.replace(mapping_path)

        if meta_path:
            meta_path = Path(meta_path)
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            meta = {
                "chunks": len(self._chunks),
            }
            if extra_meta:
                meta.update(extra_meta)
            _atomic_write_text(meta_path, json.dumps(meta, ensure_ascii=False, indent=2))

    @classmethod
    def load(
        cls,
        *,
        index_path: str | Path,
        chunks_path: str | Path,
        embedder: LocalEmbedder,
    ) -> "FaissKBClient":
        try:
            import faiss  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("faiss is required for FAISS load") from exc

        index = faiss.read_index(str(index_path))
        payload = json.loads(Path(chunks_path).read_text(encoding="utf-8"))
        chunks = [
            KBChunk(
                id=item["id"],
                text=item["text"],
                metadata=item.get("metadata") or {},
            )
            for item in payload
        ]
        return cls(index=index, chunks=chunks, embedder=embedder)
