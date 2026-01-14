from __future__ import annotations

from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI

from ai_domain.api.exception_handlers import register_exception_handlers
from ai_domain.api.middleware import setup_middlewares
from ai_domain.api.routes import router
from ai_domain.api.deps import get_orchestrator


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    # Warm up heavy dependencies early to fail fast on bad config.
    logging.info("Warming up orchestrator...")
    _ = get_orchestrator()
    logging.info("Orchestrator warmed up")
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="ai-domain",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )
    setup_middlewares(app)
    register_exception_handlers(app)
    app.include_router(router)
    return app


app = create_app()
