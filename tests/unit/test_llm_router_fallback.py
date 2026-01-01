import pytest
from ai_domain.llm.routing import LLMRouter, ModelRoute
from ai_domain.llm.retry import RetryPolicy
from ai_domain.llm.circuit_breaker import CircuitBreaker
from ai_domain.llm.rate_limit import ConcurrencyLimiter
from ai_domain.llm.errors import LLMUnavailable, LLMTimeout

@pytest.mark.asyncio
async def test_router_uses_primary_when_ok(base_request):
    from conftest import FakeLLMProvider

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