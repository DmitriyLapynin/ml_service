import pytest

from ai_domain.llm.circuit_breaker import CircuitBreaker
from ai_domain.llm.errors import LLMTimeout
from ai_domain.llm.langchain_adapter import LangChainLLMAdapter
from ai_domain.llm.rate_limit import ConcurrencyLimiter
from ai_domain.llm.retry import RetryPolicy
from ai_domain.llm.routing import LLMRouter, ModelRoute
from ai_domain.agent.safety_prompt import get_safety_classifier_langchain_prompt


@pytest.mark.asyncio
async def test_langchain_chain_can_call_llmrouter_generate():
    from tests.conftest import FakeLLMProvider

    router = LLMRouter(
        providers={"fake": FakeLLMProvider(name="fake", script=['{"unsafe": true, "injection": false}'])},
        route=ModelRoute(
            primary_provider="fake",
            primary_model="m1",
            retry_policy=RetryPolicy(max_attempts=1, base_delay_s=0.0, max_delay_s=0.0),
        ),
        limiter=ConcurrencyLimiter(max_inflight=1),
        breaker=CircuitBreaker(failure_threshold=5, reset_timeout_s=1),
    )

    prompt = get_safety_classifier_langchain_prompt()
    from ai_domain.agent.safety_prompt import SafetyClassifierOutput

    adapter = LangChainLLMAdapter(llm=router, model="m1")
    structured = adapter.with_structured_output(SafetyClassifierOutput, include_raw=True)
    out = await (prompt | structured).ainvoke({"text": "hello"})
    assert out["parsed"].unsafe is True


@pytest.mark.asyncio
async def test_langchain_chain_uses_router_retry_policy():
    from tests.conftest import FakeLLMProvider

    router = LLMRouter(
        providers={"fake": FakeLLMProvider(name="fake", script=[LLMTimeout("t"), '{"unsafe": false, "injection": false}'])},
        route=ModelRoute(
            primary_provider="fake",
            primary_model="m1",
            retry_policy=RetryPolicy(max_attempts=2, base_delay_s=0.0, max_delay_s=0.0),
        ),
        limiter=ConcurrencyLimiter(max_inflight=1),
        breaker=CircuitBreaker(failure_threshold=5, reset_timeout_s=1),
    )

    prompt = get_safety_classifier_langchain_prompt()
    from ai_domain.agent.safety_prompt import SafetyClassifierOutput

    adapter = LangChainLLMAdapter(llm=router, model="m1")
    structured = adapter.with_structured_output(SafetyClassifierOutput, include_raw=True)
    out = await (prompt | structured).ainvoke({"text": "hello"})
    assert out["parsed"].unsafe is False
