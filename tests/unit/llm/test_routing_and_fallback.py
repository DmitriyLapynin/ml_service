import pytest

from ai_domain.llm.circuit_breaker import CircuitBreaker
from ai_domain.llm.errors import LLMTimeout
from ai_domain.llm.rate_limit import ConcurrencyLimiter
from ai_domain.llm.retry import RetryPolicy
from ai_domain.llm.routing import LLMRouter, ModelRoute
from ai_domain.llm.types import LLMCapabilities


@pytest.mark.asyncio
async def test_router_uses_primary_when_ok(base_request):
    from tests.conftest import FakeLLMProvider

    primary = FakeLLMProvider(name="openai", script=["primary_ok"])
    fallback = FakeLLMProvider(name="local", script=["fallback_ok"])

    router = LLMRouter(
        providers={"openai": primary, "local": fallback},
        route=ModelRoute(
            primary_provider="openai",
            primary_model="m1",
            fallback_provider="local",
            fallback_model="m2",
            retry_policy=RetryPolicy(max_attempts=1, base_delay_s=0.0, max_delay_s=0.0),
        ),
        limiter=ConcurrencyLimiter(max_inflight=10),
        breaker=CircuitBreaker(failure_threshold=5, reset_timeout_s=1),
    )

    resp = await router.generate(base_request)
    assert resp.content == "primary_ok"


@pytest.mark.asyncio
async def test_router_skips_fallback_when_capabilities_missing(base_request):
    from tests.conftest import FakeLLMProvider

    primary = FakeLLMProvider(name="openai", script=[LLMTimeout("boom")])
    fallback = FakeLLMProvider(
        name="openrouter",
        script=["fallback_ok"],
        capabilities=LLMCapabilities(supports_structured=False, supports_tool_calls=True, supports_seed=True),
    )

    base_request.metadata["required_capabilities"] = {"supports_structured": True}

    router = LLMRouter(
        providers={"openai": primary, "openrouter": fallback},
        route=ModelRoute(
            primary_provider="openai",
            primary_model="m1",
            fallback_provider="openrouter",
            fallback_model="m2",
            retry_policy=RetryPolicy(max_attempts=1, base_delay_s=0.0, max_delay_s=0.0),
        ),
        limiter=ConcurrencyLimiter(max_inflight=10),
        breaker=CircuitBreaker(failure_threshold=5, reset_timeout_s=1),
    )

    with pytest.raises(LLMTimeout):
        await router.generate(base_request)
