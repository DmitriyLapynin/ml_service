import pytest
import asyncio

@pytest.mark.asyncio
async def test_concurrency_limiter_caps_parallelism():
    from tests.conftest import CountingLimiter  # берем тестовый семафор из conftest

    limiter = CountingLimiter(max_inflight=2)

    async def work():
        async with limiter:
            await asyncio.sleep(0.01)

    await asyncio.gather(*(work() for _ in range(10)))
    assert limiter.peak <= 2
