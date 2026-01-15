from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional
from .base import LLMProvider
import time

from .types import LLMRequest, LLMResponse, LLMCapabilities
from ai_domain.utils.hashing import hash_message_contents
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
        telemetry: Optional[Any] = None,
    ):
        self._providers = providers
        self._route = route
        self._limiter = limiter
        self._breaker = breaker
        self._telemetry = telemetry

    def _required_capabilities(self, req: LLMRequest) -> LLMCapabilities | None:
        required = req.metadata.get("required_capabilities") if req.metadata else None
        if isinstance(required, LLMCapabilities):
            return required
        if isinstance(required, dict):
            return LLMCapabilities(
                supports_structured=bool(required.get("supports_structured", False)),
                supports_tool_calls=bool(required.get("supports_tool_calls", False)),
                supports_seed=bool(required.get("supports_seed", False)),
            )
        if req.seed is not None:
            return LLMCapabilities(supports_structured=False, supports_tool_calls=False, supports_seed=True)
        return None

    def _supports(self, provider: LLMProvider, required: LLMCapabilities | None) -> bool:
        if required is None:
            return True
        caps = getattr(provider, "capabilities", LLMCapabilities())
        if required.supports_structured and not caps.supports_structured:
            return False
        if required.supports_tool_calls and not caps.supports_tool_calls:
            return False
        if required.supports_seed and not caps.supports_seed:
            return False
        return True

    def _log_event(
        self,
        *,
        req: LLMRequest,
        provider_name: str,
        latency_ms: int,
        outcome: str,
        resp: LLMResponse | None = None,
        error: Exception | None = None,
    ) -> None:
        if not self._telemetry or not hasattr(self._telemetry, "event"):
            return
        meta = req.metadata or {}
        payload = {
            "trace_id": meta.get("trace_id"),
            "graph_name": meta.get("graph_name"),
            "node_name": meta.get("node_name"),
            "task": meta.get("task"),
            "model": req.model,
            "provider": provider_name,
            "latency_ms": latency_ms,
            "outcome": outcome,
            "error_type": type(error).__name__ if error else None,
            "prompt_key": meta.get("prompt_key"),
            "prompt_version": meta.get("prompt_version"),
            "prompt_vars": meta.get("prompt_vars"),
            "message_count": len(req.messages),
            "message_roles": [m.role for m in req.messages],
            "messages_hash": hash_message_contents([m.content for m in req.messages]),
        }
        if resp:
            payload["tokens"] = {
                "prompt": resp.usage.prompt_tokens,
                "completion": resp.usage.completion_tokens,
                "total": resp.usage.total_tokens,
                "estimated": resp.usage.estimated,
            }
            payload["finish_reason"] = resp.finish_reason
        self._telemetry.event("llm_call", payload)

    def _with_latency(self, resp: LLMResponse, latency_ms: int) -> LLMResponse:
        if resp.latency_ms and resp.latency_ms > 0:
            return resp
        return LLMResponse(
            content=resp.content,
            model=resp.model,
            provider=resp.provider,
            usage=resp.usage,
            latency_ms=latency_ms,
            finish_reason=resp.finish_reason,
            raw=resp.raw,
        )

    def _with_raw(self, resp: LLMResponse, raw_extra: dict) -> LLMResponse:
        merged = dict(resp.raw or {})
        merged.update(raw_extra or {})
        return LLMResponse(
            content=resp.content,
            model=resp.model,
            provider=resp.provider,
            usage=resp.usage,
            latency_ms=resp.latency_ms,
            finish_reason=resp.finish_reason,
            raw=merged,
        )

    async def generate(self, req: LLMRequest) -> LLMResponse:
        required_caps = self._required_capabilities(req)
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

        attempts_primary = 0

        async def _call_primary():
            nonlocal attempts_primary
            attempts_primary += 1
            if not self._breaker.allow():
                raise LLMUnavailable("Circuit breaker open")
            if not self._supports(self._providers[self._route.primary_provider], required_caps):
                raise LLMUnavailable("Primary provider lacks required capabilities")
            async with self._limiter:
                start = time.perf_counter()
                try:
                    resp = await self._providers[self._route.primary_provider].generate(req_primary)
                    latency_ms = int((time.perf_counter() - start) * 1000)
                    resp = self._with_latency(resp, latency_ms)
                    self._log_event(
                        req=req_primary,
                        provider_name=self._route.primary_provider,
                        latency_ms=latency_ms,
                        outcome="ok",
                        resp=resp,
                    )
                    return resp
                except Exception as e:
                    latency_ms = int((time.perf_counter() - start) * 1000)
                    self._log_event(
                        req=req_primary,
                        provider_name=self._route.primary_provider,
                        latency_ms=latency_ms,
                        outcome="error",
                        error=e,
                    )
                    raise

        try:
            resp = await with_retries(_call_primary, policy=self._route.retry_policy)
            resp = self._with_raw(
                resp,
                {"retry_count": max(0, attempts_primary - 1), "fallback_used": False},
            )
            self._breaker.record_success()
            return resp
        except Exception as e:
            self._breaker.record_failure(e)

            # fallback
            if self._route.fallback_provider and self._route.fallback_model:
                if not self._supports(self._providers[self._route.fallback_provider], required_caps):
                    raise
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

                attempts_fb = 0

                async def _call_fb():
                    nonlocal attempts_fb
                    attempts_fb += 1
                    async with self._limiter:
                        start = time.perf_counter()
                        try:
                            resp = await self._providers[self._route.fallback_provider].generate(req_fb)
                            latency_ms = int((time.perf_counter() - start) * 1000)
                            resp = self._with_latency(resp, latency_ms)
                            self._log_event(
                                req=req_fb,
                                provider_name=self._route.fallback_provider,
                                latency_ms=latency_ms,
                                outcome="ok",
                                resp=resp,
                            )
                            return resp
                        except Exception as e:
                            latency_ms = int((time.perf_counter() - start) * 1000)
                            self._log_event(
                                req=req_fb,
                                provider_name=self._route.fallback_provider,
                                latency_ms=latency_ms,
                                outcome="error",
                                error=e,
                            )
                            raise

                resp = await with_retries(_call_fb, policy=RetryPolicy(max_attempts=2))
                resp = self._with_raw(
                    resp,
                    {"retry_count": max(0, attempts_fb - 1), "fallback_used": True},
                )
                return resp
            raise
