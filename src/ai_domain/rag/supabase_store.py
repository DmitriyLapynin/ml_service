from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def infer_source_type(name: str) -> str:
    ext = Path(name).suffix.lower().lstrip(".")
    if ext in {"txt", "pdf", "xlsx", "xls"}:
        return ext
    return "raw"


def find_kb_file_by_hash(
    sb: Any,
    *,
    funnel_id: str,
    content_hash: str,
) -> Dict[str, Any] | None:
    try:
        resp = (
            sb.table("kb_files")
            .select("id,kb_id,content_hash,bytes_size,source_name,source_type")
            .eq("funnel_id", funnel_id)
            .eq("content_hash", content_hash)
            .limit(1)
            .execute()
        )
        if resp.data:
            return resp.data[0]
    except Exception as exc:
        logging.warning(
            "supabase_kb_files_lookup_failed",
            extra={"event": "supabase_kb_files_lookup_failed", "error": str(exc)},
        )
    return None


def upsert_kb_file(
    sb: Any,
    *,
    funnel_id: str,
    kb_id: str,
    source_name: str,
    payload_bytes: bytes,
    source_type: str | None = None,
    source_uri: str | None = None,
) -> Dict[str, Any]:
    content_hash = sha256_bytes(payload_bytes)
    bytes_size = len(payload_bytes)
    source_type = source_type or infer_source_type(source_name)
    match = {
        "funnel_id": funnel_id,
        "kb_id": kb_id,
        "content_hash": content_hash,
    }
    try:
        existing = sb.table("kb_files").select("id").match(match).limit(1).execute()
        if existing.data:
            return {
                "file_id": existing.data[0]["id"],
                "content_hash": content_hash,
                "bytes_size": bytes_size,
                "source_type": source_type,
                "source_name": source_name,
                "is_duplicate": True,
            }
        insert_payload = {
            **match,
            "source_type": source_type,
            "source_name": source_name,
            "source_uri": source_uri,
            "bytes_size": bytes_size,
            "status": "ingesting",
        }
        resp = sb.table("kb_files").insert(insert_payload).execute()
        if resp.data:
            return {
                "file_id": resp.data[0]["id"],
                "content_hash": content_hash,
                "bytes_size": bytes_size,
                "source_type": source_type,
                "source_name": source_name,
                "is_duplicate": False,
            }
    except Exception as exc:
        logging.warning(
            "supabase_kb_files_upsert_failed",
            extra={"event": "supabase_kb_files_upsert_failed", "error": str(exc)},
        )
    raise RuntimeError("kb_files_upsert_failed")


def update_kb_file_status(sb: Any, *, file_id: str, status: str) -> None:
    try:
        sb.table("kb_files").update({"status": status}).eq("id", file_id).execute()
    except Exception as exc:
        logging.warning(
            "supabase_kb_files_status_failed",
            extra={"event": "supabase_kb_files_status_failed", "error": str(exc)},
        )
