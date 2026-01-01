from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
from .base import LLMProvider
from .types import LLMRequest, LLMResponse
from .errors import LLMError, LLMUnavailable
from .retry import RetryPolicy, with_retries
from .rate_limit import ConcurrencyLimiter
from .circuit_breaker import CircuitBreaker

@dataclass
class ModelRoute:
    primary_provider: str
    primary_model: str
    fallback_provider: Optional[str] = None
    fallback_model: Optional[str] = None
    retry_policy: RetryPolicy = RetryPolicy(max_attempts=3)

class LLMRouter:
    def __init__(
        self,
        *,
        providers: Dict[str, LLMProvider],
        route: ModelRoute,
        limiter: ConcurrencyLimiter,
        breaker: CircuitBreaker,
    ):
        self._providers = providers
        self._route = route
        self._limiter = limiter
        self._breaker = breaker

    async def generate(self, req: LLMRequest) -> LLMResponse:
        # выбираем модель по умолчанию из route, если не задана
        req_primary = LLMRequest(
            messages=req.messages,
            model=req.model or self._route.primary_model,
            temperature=req.temperature,
            max_output_tokens=req.max_output_tokens,
            top_p=req.top_p,
            seed=req.seed,
            stop=req.stop,
            credentials=req.credentials,
            metadata=req.metadata,
        )

        async def _call_primary():
            if not self._breaker.allow():
                raise LLMUnavailable("Circuit breaker open")
            async with self._limiter:
                return await self._providers[self._route.primary_provider].generate(req_primary)

        try:
            resp = await with_retries(_call_primary, policy=self._route.retry_policy)
            self._breaker.record_success()
            return resp
        except Exception as e:
            self._breaker.record_failure(e)

            # fallback
            if self._route.fallback_provider and self._route.fallback_model:
                req_fb = LLMRequest(
                    messages=req.messages,
                    model=self._route.fallback_model,
                    temperature=req.temperature,
                    max_output_tokens=req.max_output_tokens,
                    top_p=req.top_p,
                    seed=req.seed,
                    stop=req.stop,
                    credentials=req.credentials,
                    metadata=req.metadata,
                )

                async def _call_fb():
                    async with self._limiter:
                        return await self._providers[self._route.fallback_provider].generate(req_fb)

                return await with_retries(_call_fb, policy=RetryPolicy(max_attempts=2))
            raise
