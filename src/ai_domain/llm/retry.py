from __future__ import annotations
import random, asyncio, time
from dataclasses import dataclass
from typing import Callable, Awaitable, TypeVar
from .errors import LLMError

T = TypeVar("T")

@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay_s: float = 0.2
    max_delay_s: float = 2.0
    jitter: float = 0.2  # 20%

def _sleep_for(attempt: int, policy: RetryPolicy) -> float:
    # exponential backoff with jitter
    delay = min(policy.max_delay_s, policy.base_delay_s * (2 ** (attempt - 1)))
    jitter = delay * policy.jitter * (random.random() * 2 - 1)
    return max(0.0, delay + jitter)

async def with_retries(fn: Callable[[], Awaitable[T]], *, policy: RetryPolicy) -> T:
    last_exc: Exception | None = None
    for attempt in range(1, policy.max_attempts + 1):
        try:
            return await fn()
        except LLMError as e:
            last_exc = e
            if not e.retryable or attempt == policy.max_attempts:
                raise
            await asyncio.sleep(_sleep_for(attempt, policy))
        except Exception as e:
            # неизвестные ошибки считаем retryable 1-2 раза
            last_exc = e
            if attempt == policy.max_attempts:
                raise
            await asyncio.sleep(_sleep_for(attempt, policy))
    assert last_exc is not None
    raise last_exc