from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass
from functools import lru_cache
from typing import List

from dotenv import load_dotenv

def _split_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(frozen=True)
class APISettings:
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"
    debug: bool = False
    allowed_origins: List[str] = None
    rag_base_dir: str = str(Path(os.getenv("AI_DOMAIN_DATA_DIR", "data")) / "funnels")
    default_tenant_id: str | None = None
    langsmith_tracing: bool = False
    langsmith_project: str | None = None
    langsmith_endpoint: str | None = None
    debug_logging: bool = False

    @classmethod
    def from_env(cls) -> "APISettings":
        load_dotenv()
        allowed = os.getenv("API_ALLOWED_ORIGINS", "")
        data_dir = Path(os.getenv("AI_DOMAIN_DATA_DIR", "data"))
        return cls(
            host=os.getenv("API_HOST", "0.0.0.0"),
            port=int(os.getenv("API_PORT", "8000")),
            log_level=os.getenv("API_LOG_LEVEL", "info"),
            debug=os.getenv("API_DEBUG", "false").lower() in {"1", "true", "yes"},
            allowed_origins=_split_csv(allowed) if allowed else [],
            rag_base_dir=os.getenv("RAG_BASE_DIR", str(data_dir / "funnels")),
            default_tenant_id=os.getenv("API_DEFAULT_TENANT_ID") or None,
            langsmith_tracing=os.getenv("LANGSMITH_TRACING", "false").lower() in {"1", "true", "yes"},
            langsmith_project=os.getenv("LANGSMITH_PROJECT"),
            langsmith_endpoint=os.getenv("LANGSMITH_ENDPOINT"),
            debug_logging=os.getenv("AI_DOMAIN_DEBUG_LOGGING", "false").lower() in {"1", "true", "yes"},
        )


@lru_cache
def get_settings() -> APISettings:
    return APISettings.from_env()
