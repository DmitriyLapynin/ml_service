from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from ai_domain.api.deps import get_kb_store, get_settings, get_supabase_client
from ai_domain.api.schemas import KBUploadResponse
from ai_domain.rag.supabase_store import (
    find_kb_file_by_hash,
    infer_source_type,
    sha256_bytes,
    upsert_kb_file,
    update_kb_file_status,
)


router = APIRouter()


@router.post(
    "/kb/upload",
    response_model=KBUploadResponse,
    summary="Upload KB file and build index",
)
async def upload_kb_file(
    request: Request,
    file: UploadFile = File(...),
    funnel_id: str = Form(...),
    supabase_client=Depends(get_supabase_client),
    kb_store=Depends(get_kb_store),
) -> KBUploadResponse:
    trace_id = getattr(request.state, "trace_id", None)
    start = time.perf_counter()
    if supabase_client is None:
        raise HTTPException(status_code=500, detail="Supabase is not configured")

    payload_bytes = await file.read()
    if not payload_bytes:
        raise HTTPException(status_code=400, detail="File is empty")

    content_hash = sha256_bytes(payload_bytes)
    existing = find_kb_file_by_hash(
        supabase_client,
        funnel_id=funnel_id,
        content_hash=content_hash,
    )
    if existing:
        return KBUploadResponse(
            status="already_exists",
            trace_id=trace_id or "",
            funnel_id=funnel_id,
            kb_id=str(existing.get("kb_id")),
            file_id=str(existing.get("id")),
            source_name=str(existing.get("source_name")),
            source_type=str(existing.get("source_type")),
            bytes_size=int(existing.get("bytes_size") or 0),
            content_hash=str(existing.get("content_hash")),
            is_duplicate=True,
            message="File already exists, ingestion skipped.",
        )

    kb_id = uuid4().hex[:12]
    source_name = file.filename or f"{kb_id}.bin"
    source_type = infer_source_type(source_name)

    record = upsert_kb_file(
        supabase_client,
        funnel_id=funnel_id,
        kb_id=kb_id,
        source_name=source_name,
        payload_bytes=payload_bytes,
        source_type=source_type,
    )
    file_id = record["file_id"]
    is_duplicate = bool(record.get("is_duplicate"))
    if is_duplicate:
        return KBUploadResponse(
            status="already_exists",
            trace_id=trace_id or "",
            funnel_id=funnel_id,
            kb_id=kb_id,
            file_id=file_id,
            source_name=record["source_name"],
            source_type=record["source_type"],
            bytes_size=record["bytes_size"],
            content_hash=record["content_hash"],
            is_duplicate=True,
            message="File already exists, ingestion skipped.",
        )

    settings = get_settings()
    upload_dir = Path(settings.rag_base_dir) / "_uploads" / funnel_id / kb_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = upload_dir / source_name
    tmp_path.write_bytes(payload_bytes)

    status = "ok"
    try:
        kb_store.add_document(
            funnel_id=funnel_id,
            source_path=tmp_path,
            kb_id=kb_id,
            chunk_size=400,
            overlap=80,
            min_chunk_chars=120,
            supabase_client=supabase_client,
            file_id=file_id,
        )
        update_kb_file_status(supabase_client, file_id=file_id, status="ready")
    except Exception as exc:
        status = "error"
        update_kb_file_status(supabase_client, file_id=file_id, status="failed")
        logging.error(
            json.dumps(
                {
                    "event": "kb_upload_failed",
                    "trace_id": trace_id,
                    "funnel_id": funnel_id,
                    "kb_id": kb_id,
                    "error": str(exc),
                },
                ensure_ascii=False,
            )
        )
        raise
    finally:
        latency_ms = int((time.perf_counter() - start) * 1000)
        logging.info(
            "kb_ingest_done",
            extra={
                "event": "kb_ingest_done",
                "trace_id": trace_id,
                "funnel_id": funnel_id,
                "kb_id": kb_id,
                "latency_ms": latency_ms,
                "status": status,
            },
        )
        try:
            tmp_path.unlink()
        except Exception:
            pass

    return KBUploadResponse(
        status="created" if status == "ok" else "error",
        trace_id=trace_id or "",
        funnel_id=funnel_id,
        kb_id=kb_id,
        file_id=file_id,
        source_name=record["source_name"],
        source_type=record["source_type"],
        bytes_size=record["bytes_size"],
        content_hash=record["content_hash"],
        is_duplicate=False,
        message="File ingested successfully." if status == "ok" else "Ingestion failed.",
    )
    logging.info(
        "kb_ingest_start",
        extra={
            "event": "kb_ingest_start",
            "trace_id": trace_id,
            "funnel_id": funnel_id,
            "kb_id": kb_id,
            "file_hash": record["content_hash"],
            "file_name": record["source_name"],
        },
    )
