import numpy as np
import pytest

import json
from pathlib import Path

from ai_domain.rag.kb_client import FaissKBClient
from ai_domain.rag.delete_service import delete_kb_document
from ai_domain.rag.supabase_store import upsert_kb_file, find_kb_file_by_hash, sha256_bytes


class FakeResponse:
    def __init__(self, data):
        self.data = data


class FakeTable:
    def __init__(self, name, store):
        self._name = name
        self._store = store
        self._filters = {}
        self._limit = None
        self._update_payload = None
        self._select_fields = None
        self._on_conflict = None

    def select(self, fields):
        self._select_fields = fields
        return self

    def match(self, payload):
        self._filters.update(payload)
        return self

    def eq(self, key, value):
        self._filters[key] = value
        return self

    def limit(self, value):
        self._limit = value
        return self

    def insert(self, payload):
        self._update_payload = payload
        return self

    def update(self, payload):
        self._update_payload = payload
        return self

    def upsert(self, payload, on_conflict=None):
        self._update_payload = payload
        self._on_conflict = on_conflict
        return self

    def delete(self):
        self._delete = True
        return self

    def in_(self, key, values):
        self._in_filter = (key, set(values))
        return self

    def execute(self):
        table = self._store.setdefault(self._name, [])
        in_filter = getattr(self, "_in_filter", None)
        if self._update_payload is not None and isinstance(self._update_payload, list):
            out = []
            for row in self._update_payload:
                row = dict(row)
                if self._on_conflict:
                    conflict_keys = [k.strip() for k in self._on_conflict.split(",")]
                    existing = next(
                        (r for r in table if all(r.get(k) == row.get(k) for k in conflict_keys)),
                        None,
                    )
                    if existing:
                        existing.update(row)
                        out.append(existing)
                        continue
                row.setdefault("id", f"{self._name}_{len(table)+1}")
                table.append(row)
                out.append(row)
            return FakeResponse(out)
        if self._update_payload is not None and isinstance(self._update_payload, dict) and self._filters:
            for row in table:
                if all(row.get(k) == v for k, v in self._filters.items()):
                    row.update(self._update_payload)
            return FakeResponse([])
        if self._update_payload is not None and isinstance(self._update_payload, dict):
            row = dict(self._update_payload)
            if self._on_conflict:
                conflict_keys = [k.strip() for k in self._on_conflict.split(",")]
                existing = next(
                    (r for r in table if all(r.get(k) == row.get(k) for k in conflict_keys)),
                    None,
                )
                if existing:
                    existing.update(row)
                    return FakeResponse([existing])
            row.setdefault("id", f"{self._name}_{len(table)+1}")
            table.append(row)
            return FakeResponse([row])

        results = [row for row in table if all(row.get(k) == v for k, v in self._filters.items())]
        if in_filter:
            key, values = in_filter
            results = [row for row in results if row.get(key) in values]
        if self._limit is not None:
            results = results[: self._limit]
        if getattr(self, "_delete", False):
            for row in list(table):
                if row in results:
                    table.remove(row)
            return FakeResponse([])
        return FakeResponse(results)


class FakeSupabase:
    def __init__(self):
        self._store = {}

    def table(self, name):
        return FakeTable(name, self._store)

    def count(self, name):
        return len(self._store.get(name, []))

    def rows(self, name):
        return list(self._store.get(name, []))


class DummyEmbedder:
    model_path = "dummy-embedder"

    def embed_texts(self, texts):
        if not texts:
            return np.zeros((0, 0), dtype="float32")
        dim = 3
        data = []
        for idx, _ in enumerate(texts):
            data.append([float(idx + 1), 0.0, 0.5])
        return np.array(data, dtype="float32")


def simple_chunker(documents, *, chunk_size, overlap, min_chars):
    _ = chunk_size, overlap, min_chars
    chunks = []
    for doc in documents:
        text = doc["text"]
        if text.strip():
            chunks.append((text, dict(doc.get("metadata") or {}), 0, len(text)))
    return chunks


def _write_manifest(path: Path, kb_id: str) -> None:
    payload = {
        "funnel_id": path.parent.name,
        "documents": [
            {"kb_id": kb_id, "status": "active", "original_filename": "doc.txt"}
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


@pytest.mark.asyncio
async def test_ingest_txt_writes_supabase_records(tmp_path):
    pytest.importorskip("faiss")
    sb = FakeSupabase()
    funnel_id = "f1"
    kb_id = "k1"
    payload = b"Hello world.\nSecond line."

    record = upsert_kb_file(
        sb,
        funnel_id=funnel_id,
        kb_id=kb_id,
        source_name="doc.txt",
        payload_bytes=payload,
        source_type="txt",
    )
    file_id = record["file_id"]

    document = [{"text": payload.decode("utf-8"), "metadata": {"source": "doc.txt"}}]
    kb = FaissKBClient.from_documents(
        documents=document,
        embedder=DummyEmbedder(),
        chunk_size=10,
        overlap=0,
        min_chunk_chars=1,
        chunker=simple_chunker,
        supabase_client=sb,
        file_id=file_id,
        funnel_id=funnel_id,
        kb_id=kb_id,
    )

    assert sb.count("kb_files") == 1
    assert sb.count("kb_chunks") == len(kb._chunks)
    assert sb.count("kb_embeddings") == len(kb._chunks)
    chunks = sb.rows("kb_chunks")
    embeddings = sb.rows("kb_embeddings")
    assert all(row.get("file_id") == file_id for row in chunks)
    assert all(row.get("funnel_id") == funnel_id for row in chunks)
    assert all(row.get("kb_id") == kb_id for row in chunks)
    chunk_ids = {row.get("id") for row in chunks}
    assert all(row.get("chunk_id") in chunk_ids for row in embeddings)
    assert all(row.get("model_name") == "dummy-embedder" for row in embeddings)
    assert all(row.get("dim") == 3 for row in embeddings)
    chunk_indexes = [row.get("chunk_index") for row in chunks]
    assert len(chunk_indexes) == len(set(chunk_indexes))
    embedding_chunk_ids = [row.get("chunk_id") for row in embeddings]
    assert len(embedding_chunk_ids) == len(set(embedding_chunk_ids))


@pytest.mark.asyncio
async def test_reingest_same_file_does_not_duplicate(tmp_path):
    pytest.importorskip("faiss")
    sb = FakeSupabase()
    funnel_id = "f1"
    kb_id = "k1"
    payload = b"Same content for hashing."

    record = upsert_kb_file(
        sb,
        funnel_id=funnel_id,
        kb_id=kb_id,
        source_name="doc.txt",
        payload_bytes=payload,
        source_type="txt",
    )
    file_id = record["file_id"]

    document = [{"text": payload.decode("utf-8"), "metadata": {"source": "doc.txt"}}]
    FaissKBClient.from_documents(
        documents=document,
        embedder=DummyEmbedder(),
        chunk_size=8,
        overlap=0,
        min_chunk_chars=1,
        chunker=simple_chunker,
        supabase_client=sb,
        file_id=file_id,
        funnel_id=funnel_id,
        kb_id=kb_id,
    )
    chunks_first = sb.count("kb_chunks")
    embeddings_first = sb.count("kb_embeddings")

    record2 = upsert_kb_file(
        sb,
        funnel_id=funnel_id,
        kb_id=kb_id,
        source_name="doc.txt",
        payload_bytes=payload,
        source_type="txt",
    )
    file_id2 = record2["file_id"]
    assert file_id2 == file_id

    FaissKBClient.from_documents(
        documents=document,
        embedder=DummyEmbedder(),
        chunk_size=8,
        overlap=0,
        min_chunk_chars=1,
        chunker=simple_chunker,
        supabase_client=sb,
        file_id=file_id2,
        funnel_id=funnel_id,
        kb_id=kb_id,
    )

    assert sb.count("kb_files") == 1
    assert sb.count("kb_chunks") == chunks_first
    assert sb.count("kb_embeddings") == embeddings_first


@pytest.mark.asyncio
async def test_find_kb_file_by_hash_returns_existing():
    sb = FakeSupabase()
    funnel_id = "f1"
    kb_id = "k1"
    payload = b"Repeat content."
    record = upsert_kb_file(
        sb,
        funnel_id=funnel_id,
        kb_id=kb_id,
        source_name="doc.txt",
        payload_bytes=payload,
        source_type="txt",
    )
    content_hash = sha256_bytes(payload)
    found = find_kb_file_by_hash(
        sb,
        funnel_id=funnel_id,
        content_hash=content_hash,
    )
    assert found is not None
    assert found.get("id") == record["file_id"]
    assert found.get("kb_id") == kb_id


def test_delete_kb_document_removes_everything(tmp_path):
    base_dir = tmp_path / "funnels"
    funnel_id = "f1"
    kb_id = "k1"
    kb_dir = base_dir / funnel_id / "kb" / kb_id
    kb_dir.mkdir(parents=True, exist_ok=True)
    (kb_dir / "faiss.index").write_text("x", encoding="utf-8")
    (kb_dir / "chunks.json").write_text("x", encoding="utf-8")

    manifest_path = base_dir / funnel_id / "manifest.json"
    _write_manifest(manifest_path, kb_id)

    sb = FakeSupabase()
    file_id = "file_1"
    sb._store["kb_files"] = [{"id": file_id, "funnel_id": funnel_id, "kb_id": kb_id}]
    sb._store["kb_chunks"] = [
        {"id": "c1", "file_id": file_id},
        {"id": "c2", "file_id": file_id},
    ]
    sb._store["kb_embeddings"] = [
        {"id": "e1", "chunk_id": "c1"},
        {"id": "e2", "chunk_id": "c2"},
    ]

    result = delete_kb_document(
        base_dir=base_dir,
        funnel_id=funnel_id,
        kb_id=kb_id,
        supabase_client=sb,
    )
    assert result.status == "deleted"
    assert result.manifest_updated is True
    assert result.local_index_deleted is True
    assert result.supabase_file_found is True
    assert not kb_dir.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    doc = manifest["documents"][0]
    assert doc["status"] == "deleted"
    assert "deleted_at" in doc

    assert sb.rows("kb_files") == []
    assert sb.rows("kb_chunks") == []
    assert sb.rows("kb_embeddings") == []


def test_delete_kb_document_idempotent(tmp_path):
    base_dir = tmp_path / "funnels"
    funnel_id = "f1"
    kb_id = "k1"
    sb = FakeSupabase()

    result = delete_kb_document(
        base_dir=base_dir,
        funnel_id=funnel_id,
        kb_id=kb_id,
        supabase_client=sb,
    )
    assert result.status == "not_found"
    assert result.supabase_file_found is False
