import os

import pytest

from ai_domain.llm.openai_provider import OpenAIProvider
from ai_domain.llm.routing import LLMRouter, ModelRoute
from ai_domain.llm.rate_limit import ConcurrencyLimiter
from ai_domain.llm.circuit_breaker import CircuitBreaker
from ai_domain.llm.retry import RetryPolicy
from ai_domain.llm.types import LLMMessage, LLMRequest


pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="LIVE: требуется OPENAI_API_KEY",
)


@pytest.mark.asyncio
async def test_openai_live_smoke():
    """
    Смоук-тест, вызывающий реальный OpenAI.
    Запускается только если установлен OPENAI_API_KEY.
    """
    # Берём OpenAI ключ из окружения
    provider = OpenAIProvider(platform_api_key=os.environ["OPENAI_API_KEY"])

    router = LLMRouter(
        providers={"openai": provider},
        route=ModelRoute(
            primary_provider="openai",
            primary_model="gpt-4.1-mini",
            retry_policy=RetryPolicy(max_attempts=2, base_delay_s=0.2, max_delay_s=1.0),
        ),
        limiter=ConcurrencyLimiter(max_inflight=2),
        breaker=CircuitBreaker(failure_threshold=3, reset_timeout_s=5),
    )

    req = LLMRequest(
        messages=[
            LLMMessage(role="system", content="Ты лаконичный ассистент. Ответь коротко."),
            LLMMessage(role="user", content="Скажи привет"),
        ],
        model="gpt-4.1-mini",
        max_output_tokens=32,
        metadata={"trace_id": "live-smoke"},
    )

    resp = await router.generate(req)

    assert resp.content
    assert "прив" in resp.content.lower()  # привет/приветствую и т.п.
    assert resp.usage.total_tokens >= 1
