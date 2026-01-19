from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request

from ai_domain.api.deps import get_settings, get_supabase_client
from ai_domain.api.schemas import KBDeleteResponse
from ai_domain.rag.delete_service import delete_kb_document


router = APIRouter()


@router.delete(
    "/kb/{funnel_id}/{kb_id}",
    response_model=KBDeleteResponse,
    summary="Delete KB document",
)
async def delete_kb(
    request: Request,
    funnel_id: str,
    kb_id: str,
    supabase_client=Depends(get_supabase_client),
) -> KBDeleteResponse:
    trace_id = getattr(request.state, "trace_id", None) or ""
    result = delete_kb_document(
        base_dir=Path(get_settings().rag_base_dir),
        funnel_id=funnel_id,
        kb_id=kb_id,
        supabase_client=supabase_client,
    )
    if result.status == "conflict":
        raise HTTPException(status_code=409, detail="Delete lock exists")
    return KBDeleteResponse(
        status=result.status,
        trace_id=trace_id,
        funnel_id=funnel_id,
        kb_id=kb_id,
        manifest_updated=result.manifest_updated,
        local_index_deleted=result.local_index_deleted,
        supabase_file_found=result.supabase_file_found,
        deleted_files=result.deleted_files,
        deleted_chunks=result.deleted_chunks,
        deleted_embeddings=result.deleted_embeddings,
    )
