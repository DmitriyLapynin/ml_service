import sys
from pathlib import Path

# чтобы видеть src/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
import asyncio
from dataclasses import dataclass, field
from typing import Any, List

from ai_domain.llm.base import LLMProvider
from ai_domain.llm.types import LLMMessage, LLMRequest, LLMResponse, LLMUsage, LLMCapabilities


@pytest.fixture
def base_request():
    return LLMRequest(
        messages=[
            LLMMessage(role="system", content="You are helpful."),
            LLMMessage(role="user", content="Hello"),
        ],
        model="gpt-test",
        temperature=0.2,
        max_output_tokens=32,
        metadata={"trace_id": "t-123", "task": "unit_test"},
    )


@dataclass
class FakeLLMProvider(LLMProvider):
    name: str = "fake"
    script: List[Any] = field(default_factory=lambda: ["ok"])
    latency_ms: int = 1
    capabilities: LLMCapabilities = field(default_factory=LLMCapabilities)

    async def generate(self, req: LLMRequest) -> LLMResponse:
        await asyncio.sleep(0)
        item = self.script.pop(0) if self.script else "ok"
        if isinstance(item, Exception):
            raise item
        return LLMResponse(
            content=str(item),
            model=req.model,
            provider=self.name,
            usage=LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2, estimated=False),
            latency_ms=self.latency_ms,
            finish_reason="stop",
            raw=None,
        )


class CountingLimiter:
    def __init__(self, max_inflight: int):
        self.max_inflight = max_inflight
        self.current = 0
        self.peak = 0
        self._lock = asyncio.Lock()
        self._sem = asyncio.Semaphore(max_inflight)

    async def __aenter__(self):
        await self._sem.acquire()
        async with self._lock:
            self.current += 1
            self.peak = max(self.peak, self.current)
        return self

    async def __aexit__(self, exc_type, exc, tb):
        async with self._lock:
            self.current -= 1
        self._sem.release()
        return False


@pytest.fixture
def fake_provider():
    return FakeLLMProvider()


@pytest.fixture
def tmp_data_dir(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("AI_DOMAIN_DATA_DIR", str(data_dir))
    return data_dir


@pytest.fixture
def tmp_models_dir(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    monkeypatch.setenv("AI_DOMAIN_MODELS_DIR", str(models_dir))
    return models_dir


@pytest.fixture
def tmp_secrets_dir(tmp_path, monkeypatch):
    secrets_dir = tmp_path / "secrets"
    secrets_dir.mkdir()
    monkeypatch.setenv("AI_DOMAIN_SECRETS_DIR", str(secrets_dir))
    return secrets_dir
