import pytest

from pydantic import BaseModel

from ai_domain.llm.circuit_breaker import CircuitBreaker
from ai_domain.llm.client import LLMClient, LLMConfig
from ai_domain.llm.types import StructuredResult
from ai_domain.llm.rate_limit import ConcurrencyLimiter
from ai_domain.llm.retry import RetryPolicy
from ai_domain.llm.routing import LLMRouter, ModelRoute


class Out(BaseModel):
    unsafe: bool
    injection: bool


@pytest.mark.asyncio
async def test_llmclient_invoke_text_uses_router():
    from tests.conftest import FakeLLMProvider

    provider = FakeLLMProvider(name="openai", script=["ok"])
    client = LLMClient(
        providers={"openai": provider},
        default_provider="openai",
        default_model="m1",
        limiter=ConcurrencyLimiter(max_inflight=1),
        breaker=CircuitBreaker(failure_threshold=5, reset_timeout_s=1),
    )

    text = await client.invoke_text(
        [{"role": "user", "content": "hi"}],
        config=LLMConfig(model="m1", temperature=0.1, max_tokens=10, retries=1),
    )
    assert text == "ok"


@pytest.mark.asyncio
async def test_llmclient_invoke_structured_returns_pydantic():
    from tests.conftest import FakeLLMProvider

    provider = FakeLLMProvider(name="openai", script=['{"unsafe": true, "injection": false}'])
    client = LLMClient(
        providers={"openai": provider},
        default_provider="openai",
        default_model="m1",
        limiter=ConcurrencyLimiter(max_inflight=1),
        breaker=CircuitBreaker(failure_threshold=5, reset_timeout_s=1),
    )

    out = await client.invoke_structured(
        Out,
        [{"role": "user", "content": "hi"}],
        config=LLMConfig(model="m1", temperature=0.0, max_tokens=64, retries=1),
        include_raw=False,
    )

    assert out.unsafe is True
    assert out.injection is False


@pytest.mark.asyncio
async def test_llmclient_invoke_structured_include_raw_returns_structured_result():
    from tests.conftest import FakeLLMProvider

    provider = FakeLLMProvider(name="openai", script=['{"unsafe": true, "injection": false}'])
    client = LLMClient(
        providers={"openai": provider},
        default_provider="openai",
        default_model="m1",
        limiter=ConcurrencyLimiter(max_inflight=1),
        breaker=CircuitBreaker(failure_threshold=5, reset_timeout_s=1),
    )

    res = await client.invoke_structured(
        Out,
        [{"role": "user", "content": "hi"}],
        config=LLMConfig(model="m1", temperature=0.0, max_tokens=64, retries=1),
        include_raw=True,
    )

    assert isinstance(res, StructuredResult)
    assert res.parsed is not None
    assert res.parsed.unsafe is True
