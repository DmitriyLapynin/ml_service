from __future__ import annotations

import logging
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from ai_domain.api.errors import APIError
from ai_domain.orchestrator.service import OrchestratorError


def register_exception_handlers(app) -> None:
    logger = logging.getLogger(__name__)

    @app.exception_handler(RequestValidationError)
    async def handle_validation(request: Request, exc: RequestValidationError):  # noqa: ARG001
        trace_id = getattr(request.state, "trace_id", None)
        response = JSONResponse(
            status_code=422,
            content={"error": "validation_error", "details": exc.errors(), "trace_id": trace_id},
        )
        if trace_id:
            response.headers["X-Trace-Id"] = trace_id
        return response

    @app.exception_handler(OrchestratorError)
    async def handle_orchestrator(request: Request, exc: OrchestratorError):  # noqa: ARG001
        trace_id = getattr(request.state, "trace_id", None)
        code = getattr(exc, "code", "orchestrator_error")
        status_code = getattr(exc, "status_code", 500)
        response = JSONResponse(
            status_code=status_code,
            content={"error": code, "message": str(exc), "trace_id": trace_id},
        )
        if trace_id:
            response.headers["X-Trace-Id"] = trace_id
        return response

    @app.exception_handler(APIError)
    async def handle_api_error(request: Request, exc: APIError):  # noqa: ARG001
        trace_id = getattr(request.state, "trace_id", None)
        response = JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.code, "message": str(exc), "trace_id": trace_id},
        )
        if trace_id:
            response.headers["X-Trace-Id"] = trace_id
        return response

    @app.exception_handler(Exception)
    async def handle_unknown(request: Request, exc: Exception):  # noqa: ARG001
        trace_id = getattr(request.state, "trace_id", None)
        logger.exception("Unhandled error", extra={"trace_id": trace_id, "path": request.url.path})
        response = JSONResponse(
            status_code=500,
            content={"error": "internal_error", "trace_id": trace_id},
        )
        if trace_id:
            response.headers["X-Trace-Id"] = trace_id
        return response
