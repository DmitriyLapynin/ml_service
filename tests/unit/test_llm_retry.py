import pytest
from ai_domain.llm.retry import with_retries, RetryPolicy
from ai_domain.llm.errors import LLMTimeout, LLMInvalidRequest

@pytest.mark.asyncio
async def test_with_retries_succeeds_after_failures():
    calls = {"n": 0}

    async def fn():
        calls["n"] += 1
        if calls["n"] < 3:
            raise LLMTimeout("timeout")
        return "ok"

    res = await with_retries(fn, policy=RetryPolicy(max_attempts=3, base_delay_s=0.0, max_delay_s=0.0))
    assert res == "ok"
    assert calls["n"] == 3

@pytest.mark.asyncio
async def test_with_retries_does_not_retry_non_retryable():
    calls = {"n": 0}

    async def fn():
        calls["n"] += 1
        raise LLMInvalidRequest("bad input")

    with pytest.raises(LLMInvalidRequest):
        await with_retries(fn, policy=RetryPolicy(max_attempts=5, base_delay_s=0.0, max_delay_s=0.0))

    assert calls["n"] == 1
