from __future__ import annotations

import base64
import json
import logging
import time
from uuid import uuid4

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from ai_domain.api.errors import APIError
from functools import lru_cache

from ai_domain.secrets import get_secret


def _decrypt_api_key(value: str) -> str:
    try:
        raw = base64.b64decode(value.encode("utf-8"), validate=True)
        decoded = raw.decode("utf-8").strip()
    except Exception as exc:
        raise APIError("Invalid crypted_api_key", status_code=400, code="invalid_crypted_api_key") from exc
    if not decoded:
        raise APIError("Invalid crypted_api_key", status_code=400, code="invalid_crypted_api_key")
    return decoded


def _resolve_credentials(crypted_api_key: str | None) -> dict:
    if crypted_api_key is not None:
        return {"openai_api_key": _decrypt_api_key(str(crypted_api_key)), "source": "crypted_api_key"}
    fallback = _cached_openai_key()
    if not fallback:
        raise APIError("OPENAI_API_KEY is not configured", status_code=500, code="missing_openai_api_key")
    return {"openai_api_key": fallback, "source": "env"}


@lru_cache
def _cached_openai_key() -> str | None:
    return get_secret("OPENAI_API_KEY")


class TraceLoggingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self._logger = logging.getLogger(__name__)

    async def dispatch(self, request: Request, call_next):
        trace_id = request.headers.get("x-trace-id") or uuid4().hex
        tenant_id = request.headers.get("x-tenant-id")
        funnel_id = request.headers.get("x-funnel-id")
        channel = request.headers.get("x-channel")
        request_id = request.headers.get("x-request-id")
        request.state.trace_id = trace_id
        request.state.tenant_id = tenant_id
        request.state.funnel_id = funnel_id
        request.state.channel = channel
        request.state.request_id = request_id
        client_host = request.client.host if request.client else None
        user_agent = request.headers.get("user-agent")
        start = time.perf_counter()
        response: Response | None = None
        error: Exception | None = None
        try:
            if request.method.upper() in {"POST", "PUT", "PATCH"}:
                path = request.url.path.rstrip("/")
                if path.endswith("/chat"):
                    crypted = request.headers.get("x-crypted-api-key")
                    request.state.credentials = _resolve_credentials(crypted)
            response = await call_next(request)
            return response
        except Exception as exc:
            error = exc
            raise
        finally:
            latency_ms = int((time.perf_counter() - start) * 1000)
            status_code = response.status_code if response else getattr(error, "status_code", 500)
            event = "api_request"
            extra = {
                "event": event,
                "trace_id": trace_id,
                "tenant_id": tenant_id,
                "funnel_id": funnel_id,
                "channel": channel,
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": status_code,
                "latency_ms": latency_ms,
                "client_ip": client_host,
                "user_agent": user_agent,
                "outcome": "error" if error else "success",
            }
            if error:
                self._logger.error(event, extra=extra)
            else:
                self._logger.info(event, extra=extra)
            if response:
                response.headers["X-Trace-Id"] = trace_id


def setup_middlewares(app) -> None:
    app.add_middleware(TraceLoggingMiddleware)
