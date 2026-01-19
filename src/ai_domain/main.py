from __future__ import annotations

from contextlib import asynccontextmanager
import json
import logging
import os
from pathlib import Path
from fastapi import FastAPI

from ai_domain.api.exception_handlers import register_exception_handlers
from ai_domain.api.middleware import setup_middlewares
from ai_domain.api.routes import router
from ai_domain.api.deps import get_orchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)



@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    # Warm up heavy dependencies early to fail fast on bad config.
    logging.info(
        json.dumps(
            {"event": "startup", "message": "Warming up orchestrator..."},
            ensure_ascii=False,
        )
    )
    _ = get_orchestrator()
    logging.info(
        json.dumps(
            {"event": "startup", "message": "Orchestrator warmed up"},
            ensure_ascii=False,
        )
    )
    data_dir = Path(os.getenv("AI_DOMAIN_DATA_DIR", "data"))
    data_dir.mkdir(parents=True, exist_ok=True)
    probe_path = data_dir / ".rw_check"
    probe_path.write_text("ok", encoding="utf-8")
    probe_path.unlink()
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
