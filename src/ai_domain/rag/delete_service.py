from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def _utc_now() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


@dataclass
class DeleteResult:
    status: str
    manifest_updated: bool
    local_index_deleted: bool
    supabase_file_found: bool
    deleted_files: int
    deleted_chunks: int
    deleted_embeddings: int


def delete_kb_document(
    *,
    base_dir: Path,
    funnel_id: str,
    kb_id: str,
    supabase_client: Any | None,
) -> DeleteResult:
    manifest_updated = False
    local_index_deleted = False
    supabase_file_found = False
    deleted_files = 0
    deleted_chunks = 0
    deleted_embeddings = 0

    funnel_dir = base_dir / funnel_id
    kb_dir = funnel_dir / "kb" / kb_id
    uploads_dir = funnel_dir / "_uploads" / kb_id
    manifest_path = funnel_dir / "manifest.json"

    lock_path = funnel_dir / f".lock.{kb_id}"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.exists():
        return DeleteResult(
            status="conflict",
            manifest_updated=False,
            local_index_deleted=False,
            supabase_file_found=False,
            deleted_files=0,
            deleted_chunks=0,
            deleted_embeddings=0,
        )
    lock_path.write_text(_utc_now(), encoding="utf-8")

    try:
        logging.info(
            json.dumps(
                {
                    "event": "kb_delete_start",
                    "funnel_id": funnel_id,
                    "kb_id": kb_id,
                },
                ensure_ascii=False,
            )
        )
        # Step 4: manifest first
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            docs = manifest.get("documents") or []
            found = False
            for doc in docs:
                if doc.get("kb_id") == kb_id:
                    found = True
                    doc["status"] = "deleted"
                    doc["deleted_at"] = _utc_now()
                    break
            if found:
                _atomic_write_json(manifest_path, manifest)
                manifest_updated = True

        # Step 5: local files
        if kb_dir.exists():
            shutil.rmtree(kb_dir)
            local_index_deleted = True
        else:
            local_index_deleted = False
        if uploads_dir.exists():
            shutil.rmtree(uploads_dir)
        logging.info(
            json.dumps(
                {
                    "event": "kb_delete_local_done",
                    "funnel_id": funnel_id,
                    "kb_id": kb_id,
                    "ok": local_index_deleted,
                },
                ensure_ascii=False,
            )
        )

        # Step 6: supabase cleanup
        file_id = None
        if supabase_client is not None:
            resp = (
                supabase_client.table("kb_files")
                .select("id")
                .eq("funnel_id", funnel_id)
                .eq("kb_id", kb_id)
                .limit(1)
                .execute()
            )
            data = resp.data or []
            if data:
                file_id = data[0]["id"]
            else:
                supabase_file_found = False

        if supabase_client is not None and file_id:
            supabase_file_found = True
            chunks_resp = (
                supabase_client.table("kb_chunks")
                .select("id")
                .eq("file_id", file_id)
                .execute()
            )
            chunk_rows = chunks_resp.data or []
            chunk_ids = [row.get("id") for row in chunk_rows if row.get("id")]
            if chunk_ids:
                supabase_client.table("kb_embeddings").delete().in_("chunk_id", chunk_ids).execute()
                deleted_embeddings = len(chunk_ids)
            supabase_client.table("kb_chunks").delete().eq("file_id", file_id).execute()
            deleted_chunks = len(chunk_rows)
            supabase_client.table("kb_files").delete().eq("id", file_id).execute()
            deleted_files = 1
        logging.info(
            json.dumps(
                {
                    "event": "kb_delete_supabase_done",
                    "funnel_id": funnel_id,
                    "kb_id": kb_id,
                    "file_found": supabase_file_found,
                    "deleted_files": deleted_files,
                    "deleted_chunks": deleted_chunks,
                    "deleted_embeddings": deleted_embeddings,
                },
                ensure_ascii=False,
            )
        )

        status = "deleted"
        if not manifest_updated and not local_index_deleted and not supabase_file_found:
            status = "not_found"
        logging.info(
            json.dumps(
                {
                    "event": "kb_delete_done",
                    "funnel_id": funnel_id,
                    "kb_id": kb_id,
                    "status": status,
                },
                ensure_ascii=False,
            )
        )
        return DeleteResult(
            status=status,
            manifest_updated=manifest_updated,
            local_index_deleted=local_index_deleted,
            supabase_file_found=supabase_file_found,
            deleted_files=deleted_files,
            deleted_chunks=deleted_chunks,
            deleted_embeddings=deleted_embeddings,
        )
    finally:
        try:
            lock_path.unlink()
        except Exception:
            pass
