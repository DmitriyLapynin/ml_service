from __future__ import annotations

from dataclasses import dataclass, field, replace
import os
import json
import logging
import time
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Type
from uuid import uuid4

from pydantic import BaseModel

from ai_domain.llm.circuit_breaker import CircuitBreaker
from ai_domain.llm.errors import (
    LLMError,
    LLMTimeout,
    LLMRateLimited,
    LLMUnavailable,
    LLMInvalidRequest,
    LLMAuthError,
    LLMProviderError,
)
from ai_domain.llm.rate_limit import ConcurrencyLimiter
from ai_domain.llm.retry import RetryPolicy
from ai_domain.llm.routing import LLMRouter, ModelRoute
from ai_domain.llm.types import (
    LLMCredentials,
    LLMCallContext,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    LLMCapabilities,
    StructuredResult,
)
from ai_domain.llm.model_caps import get_model_capabilities
from ai_domain.utils.hashing import messages_fingerprint, hash_text_short
from ai_domain.secrets import get_secret
from ai_domain.tools.registry import ToolSpec, to_langchain_tool

ProviderName = Literal["openai", "openrouter"]


@dataclass(frozen=True)
class LLMConfig:
    provider: ProviderName = "openai"
    model: Optional[str] = None
    temperature: Optional[float] = 0.2
    top_p: Optional[float] = None
    max_tokens: int = 2048
    timeout_s: Optional[float] = None  # best-effort (provider-dependent)
    retries: int = 2
    seed: Optional[int] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResolvedLLMConfig:
    provider: ProviderName
    model: str
    temperature: Optional[float] = 0.2
    top_p: Optional[float] = None
    max_tokens: int = 2048
    timeout_s: Optional[float] = None
    retries: int = 2
    seed: Optional[int] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


def _normalize_messages(messages: Iterable[Dict[str, str]]) -> List[LLMMessage]:
    out: List[LLMMessage] = []
    for m in messages:
        out.append(LLMMessage(role=m["role"], content=m.get("content") or ""))
    return out


class LLMClient:
    """
    Single entry-point for LLM calls in the codebase.

    - text: `invoke_text(messages, config) -> str`
    - structured: `invoke_structured(schema, messages, config) -> BaseModel | StructuredResult`
    - compatibility: provides `.generate(...)` in both styles used in the repo
      (kwargs-style and LLMRequest-style), returning `LLMResponse`.
    """

    def __init__(
        self,
        *,
        providers: Dict[str, Any],
        default_provider: ProviderName = "openai",
        default_model: str = "gpt-4.1-mini",
        fallback_provider: Optional[ProviderName] = None,
        fallback_model: Optional[str] = None,
        limiter: Optional[ConcurrencyLimiter] = None,
        breaker: Optional[CircuitBreaker] = None,
        telemetry: Any | None = None,
    ):
        self._providers = providers
        self._default_provider = default_provider
        self._default_model = default_model
        self._fallback_provider = fallback_provider
        self._fallback_model = fallback_model
        self._limiter = limiter or ConcurrencyLimiter(max_inflight=5)
        self._breaker = breaker or CircuitBreaker(failure_threshold=5, reset_timeout_s=5)
        self._telemetry = telemetry

    def _build_router(self, *, config: LLMConfig | ResolvedLLMConfig) -> LLMRouter:
        provider = getattr(config, "provider", None) or self._default_provider
        model = getattr(config, "model", None) or self._default_model
        route = ModelRoute(
            primary_provider=provider,
            primary_model=model,
            fallback_provider=self._fallback_provider,
            fallback_model=self._fallback_model,
            retry_policy=RetryPolicy(max_attempts=max(1, int(config.retries))),
        )
        return LLMRouter(
            providers=self._providers,
            route=route,
            limiter=self._limiter,
            breaker=self._breaker,
            telemetry=self._telemetry,
        )

    def _sanitize_config(
        self,
        config: LLMConfig | ResolvedLLMConfig,
        *,
        provider: str,
        model: str,
        context: LLMCallContext | None,
    ) -> tuple[LLMConfig | ResolvedLLMConfig, list[str]]:
        caps = get_model_capabilities(model)
        removed: list[str] = []
        updated = config
        if not caps.supports_top_p and getattr(config, "top_p", None) is not None:
            removed.append("top_p")
            updated = replace(updated, top_p=None)
        if not caps.supports_seed and getattr(config, "seed", None) is not None:
            removed.append("seed")
            updated = replace(updated, seed=None)
        if not caps.supports_temperature and getattr(config, "temperature", None) is not None:
            removed.append("temperature")
            updated = replace(updated, temperature=None)
        if removed:
            logger = logging.getLogger(__name__)
            logger.info(
                "llm_config_sanitized",
                extra={
                    "event": "llm_config_sanitized",
                    "provider": provider,
                    "model": model,
                    "removed": removed,
                    "trace_id": context.trace_id if context else None,
                    "node": context.node if context else None,
                    "task": context.task if context else None,
                },
            )
        return updated, removed

    def _attach_required_capabilities(self, metadata: Dict[str, Any], required: LLMCapabilities) -> Dict[str, Any]:
        out = dict(metadata or {})
        existing = out.get("required_capabilities") or {}
        out["required_capabilities"] = {
            "supports_structured": bool(existing.get("supports_structured")) or required.supports_structured,
            "supports_tool_calls": bool(existing.get("supports_tool_calls")) or required.supports_tool_calls,
            "supports_seed": bool(existing.get("supports_seed")) or required.supports_seed,
        }
        return out

    def _classify_error(self, err: Exception) -> str:
        if isinstance(err, LLMTimeout):
            return "timeout"
        if isinstance(err, LLMRateLimited):
            return "rate_limit"
        if isinstance(err, LLMInvalidRequest):
            return "invalid_request"
        if isinstance(err, LLMAuthError):
            return "auth_error"
        if isinstance(err, LLMProviderError):
            return "provider_error"
        if isinstance(err, LLMUnavailable):
            msg = str(err).lower()
            if "capabilit" in msg:
                return "capability_error"
            return "unavailable"
        if isinstance(err, ValueError):
            return "parse_error"
        if isinstance(err, LLMError):
            return "llm_error"
        return "unknown"

    def _build_payload(
        self,
        *,
        call_id: str,
        context: LLMCallContext | None,
        provider: str,
        model: str,
        structured: bool,
        schema_name: str | None,
        messages: List[Dict[str, str]],
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "call_id": call_id,
            "trace_id": context.trace_id if context else None,
            "graph": context.graph if context else None,
            "node": context.node if context else None,
            "task": context.task if context else None,
            "channel": context.channel if context else None,
            "tenant_id": context.tenant_id if context else None,
            "request_id": context.request_id if context else None,
            "provider": provider,
            "model": model,
            "structured": structured,
            "schema": schema_name,
            "messages": messages_fingerprint(messages),
            "prompt_key": metadata.get("prompt_key"),
            "prompt_version": metadata.get("prompt_version"),
        }

    def _estimate_tokens(self, text: str, *, model: str | None) -> int:
        if not text:
            return 0
        try:
            import tiktoken  # type: ignore

            try:
                enc = tiktoken.encoding_for_model(model or "")
            except Exception:
                enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            return max(1, len(text) // 4)

    def _estimate_tokens_for_messages(
        self,
        messages: List[Dict[str, str]],
        *,
        model: str | None,
    ) -> int:
        joined = "\n".join(f"{m.get('role','')}:{m.get('content','')}" for m in messages)
        return self._estimate_tokens(joined, model=model)

    def _output_meta(self, text: str) -> Dict[str, Any]:
        text = text or ""
        return {
            "output_chars": len(text),
            "output_fingerprint": hash_text_short(text) if text else None,
        }

    def _record_metrics(
        self,
        *,
        context: LLMCallContext | None,
        payload: Dict[str, Any],
        latency_ms: int,
        usage: Dict[str, Any],
        **extra: Any,
    ) -> None:
        self._log_llm_call_end(payload=payload, latency_ms=latency_ms, usage=usage, **extra)
        if not context or context.metrics is None:
            return
        entry = {
            "call_id": payload.get("call_id"),
            "provider": payload.get("provider"),
            "model": payload.get("model"),
            "structured": payload.get("structured"),
            "node": payload.get("node"),
            "task": payload.get("task"),
            "latency_ms": latency_ms,
            "usage": usage,
        }
        entry.update(extra)
        add_llm_call = getattr(context.metrics, "add_llm_call", None)
        if callable(add_llm_call):
            add_llm_call(entry)
            return
        try:
            context.metrics.append(entry)
        except Exception:
            return

    def _log_llm_call_end(
        self,
        *,
        payload: Dict[str, Any],
        latency_ms: int,
        usage: Dict[str, Any] | None,
        **extra: Any,
    ) -> None:
        logger = logging.getLogger(__name__)
        usage = usage or {}
        logger.info(
            json.dumps(
                {
                    "event": "llm_call_end",
                    "op": extra.get("op") or payload.get("task"),
                    "trace_id": payload.get("trace_id"),
                    "node": payload.get("node"),
                    "provider": payload.get("provider"),
                    "model": payload.get("model"),
                    "latency_ms": latency_ms,
                    "usage_prompt_tokens": usage.get("prompt_tokens"),
                    "usage_completion_tokens": usage.get("completion_tokens"),
                    "usage_total_tokens": usage.get("total_tokens"),
                    "retry_count": extra.get("retry_count"),
                    "fallback_used": extra.get("fallback_used"),
                    "circuit_breaker_open": extra.get("circuit_breaker_open"),
                    "finish_reason": extra.get("finish_reason"),
                    "response_format": extra.get("response_format"),
                    "parse_ok": extra.get("parse_ok"),
                    "schema_name": extra.get("schema_name") or payload.get("schema"),
                },
                ensure_ascii=False,
            )
        )

    def _debug_enabled(self) -> bool:
        return os.getenv("AI_DOMAIN_DEBUG_LOGGING", "false").lower() in {"1", "true", "yes"}

    def _log_llm_debug_messages(
        self,
        *,
        payload: Dict[str, Any],
        messages: List[Dict[str, Any]],
        response: Any,
    ) -> None:
        if not self._debug_enabled():
            return
        logger = logging.getLogger(__name__)
        logger.info(
            json.dumps(
                {
                    "event": "llm_debug_messages",
                    "trace_id": payload.get("trace_id"),
                    "node": payload.get("node"),
                    "messages": messages,
                    "response": response,
                },
                ensure_ascii=False,
                default=str,
            )
        )

    def _extract_usage_from_langchain_response(self, response: Any) -> Dict[str, Any] | None:
        meta = getattr(response, "response_metadata", None) or {}
        usage = meta.get("token_usage") or meta.get("usage")
        if not isinstance(usage, dict):
            return None
        prompt = usage.get("prompt_tokens") or usage.get("input_tokens")
        completion = usage.get("completion_tokens") or usage.get("output_tokens")
        total = usage.get("total_tokens")
        return {
            "prompt_tokens": int(prompt) if prompt is not None else None,
            "completion_tokens": int(completion) if completion is not None else None,
            "total_tokens": int(total) if total is not None else None,
            "estimated": False,
            "completion_unknown": completion is None,
        }

    def _extract_finish_reason_from_langchain_response(self, response: Any) -> str | None:
        meta = getattr(response, "response_metadata", None) or {}
        return meta.get("finish_reason")

    def _resolve_tool_api_key(
        self,
        provider: ProviderName,
        credentials: Optional[LLMCredentials],
    ) -> str:
        if provider == "openai" and credentials and credentials.openai_api_key:
            return credentials.openai_api_key
        provider_obj = self._providers.get(provider)
        platform_key = getattr(provider_obj, "platform_api_key", None)
        if platform_key:
            return platform_key
        if provider == "openai":
            return get_secret("OPENAI_API_KEY", required=True)
        if provider == "openrouter":
            return get_secret("OPENROUTER_API_KEY", required=True)
        raise LLMInvalidRequest(f"Unsupported tool provider: {provider}")

    def _build_chat_openai(
        self,
        provider: ProviderName,
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: Optional[float],
        timeout_s: Optional[float],
        credentials: Optional[LLMCredentials],
    ):
        try:
            from langchain_openai import ChatOpenAI  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("langchain-openai is required for tool calling") from exc

        api_key = self._resolve_tool_api_key(provider, credentials)
        kwargs: Dict[str, Any] = {
            "api_key": api_key,
            "model": model,
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p
        if timeout_s is not None:
            kwargs["timeout"] = timeout_s
        if provider == "openrouter":
            kwargs["base_url"] = "https://openrouter.ai/api/v1"
        return ChatOpenAI(**kwargs)

    def _normalize_tool_specs(self, tools: Sequence[ToolSpec | Dict[str, Any]]) -> List[ToolSpec]:
        normalized: List[ToolSpec] = []
        for tool in tools:
            if isinstance(tool, ToolSpec):
                normalized.append(tool)
            else:
                normalized.append(ToolSpec(**tool))
        return normalized

    def _to_langchain_messages(self, messages: Sequence[Dict[str, Any]]):
        try:
            from langchain_core.messages import (  # type: ignore
                AIMessage,
                HumanMessage,
                SystemMessage,
                ToolMessage,
            )
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("langchain-core is required for tool calling") from exc

        lc_messages = []
        for m in messages:
            role = m.get("role")
            content = m.get("content") or ""
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "assistant":
                tool_calls = m.get("tool_calls")
                if tool_calls is not None:
                    lc_messages.append(AIMessage(content=content, tool_calls=tool_calls))
                else:
                    lc_messages.append(AIMessage(content=content))
            elif role == "tool":
                lc_messages.append(ToolMessage(content=content, tool_call_id=m.get("tool_call_id") or "tool"))
            else:
                if role != "user":
                    raise ValueError(f"Unknown role: {role}")
                lc_messages.append(HumanMessage(content=content))
        return lc_messages

    async def invoke_text(
        self,
        messages: List[Dict[str, str]],
        *,
        config: LLMConfig | ResolvedLLMConfig,
        credentials: Optional[LLMCredentials] = None,
        context: LLMCallContext | None = None,
    ) -> str:
        logger = logging.getLogger(__name__)
        call_id = uuid4().hex[:12]
        provider = getattr(config, "provider", None) or self._default_provider
        model = getattr(config, "model", None) or self._default_model
        config, _ = self._sanitize_config(
            config,
            provider=provider,
            model=model,
            context=context,
        )
        metadata = dict(config.metadata or {})
        if config.seed is not None:
            metadata = self._attach_required_capabilities(
                metadata,
                LLMCapabilities(supports_seed=True, supports_structured=False, supports_tool_calls=False),
            )
        payload = self._build_payload(
            call_id=call_id,
            context=context,
            provider=provider,
            model=model,
            structured=False,
            schema_name=None,
            messages=messages,
            metadata=metadata,
        )
        logger.info(json.dumps({"event": "llm_call_start", **payload}, ensure_ascii=False))
        start = time.perf_counter()
        req = LLMRequest(
            messages=_normalize_messages(messages),
            model=model,
            temperature=config.temperature,
            max_output_tokens=config.max_tokens,
            top_p=config.top_p,
            seed=config.seed,
            credentials=credentials,
            metadata={"tags": config.tags, **metadata},
        )
        router = self._build_router(config=config)
        try:
            resp = await router.generate(req)
            print(resp)
        except Exception as e:
            latency_ms = int((time.perf_counter() - start) * 1000)
            self._log_llm_call_end(
                payload=payload,
                latency_ms=latency_ms,
                usage=None,
                response_format="text",
            )
            logger.exception(
                json.dumps(
                    {
                        "event": "llm_call_error",
                        **payload,
                        "latency_ms": latency_ms,
                        "outcome": "error",
                        "error_kind": self._classify_error(e),
                    },
                    ensure_ascii=False,
                )
            )
            raise
        latency_ms = int((time.perf_counter() - start) * 1000)
        usage = {
            "prompt_tokens": resp.usage.prompt_tokens if resp.usage else None,
            "completion_tokens": resp.usage.completion_tokens if resp.usage else None,
            "total_tokens": resp.usage.total_tokens if resp.usage else None,
            "estimated": bool(resp.usage.estimated) if resp.usage else False,
            "completion_unknown": False,
        }
        if not resp.usage or resp.usage.total_tokens == 0:
            prompt_tokens = self._estimate_tokens_for_messages(messages, model=model)
            completion_tokens = self._estimate_tokens(resp.content or "", model=model)
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "estimated": True,
                "completion_unknown": False,
            }
        self._record_metrics(
            context=context,
            payload=payload,
            latency_ms=latency_ms,
            usage=usage,
            **self._output_meta(resp.content or ""),
            finish_reason=resp.finish_reason,
            response_format="text",
            retry_count=(resp.raw or {}).get("retry_count") if isinstance(resp.raw, dict) else None,
            fallback_used=(resp.raw or {}).get("fallback_used") if isinstance(resp.raw, dict) else None,
            usage_source="estimated" if usage.get("estimated") else "provider",
            request_tokens_estimated=bool(usage.get("estimated")),
            usage_from_provider=not bool(usage.get("estimated")),
        )
        logger.info(
            json.dumps(
                {
                    "event": "llm_call_success",
                    **payload,
                    "latency_ms": latency_ms,
                    "outcome": "success",
                },
                ensure_ascii=False,
            )
        )
        self._log_llm_debug_messages(
            payload=payload,
            messages=messages,
            response=resp.content,
        )
        return resp.content

    async def invoke_structured(
        self,
        schema: Type[BaseModel],
        messages: List[Dict[str, str]],
        config: LLMConfig | ResolvedLLMConfig,
        credentials: Optional[LLMCredentials] = None,
        include_raw: bool = False,
        context: LLMCallContext | None = None,
    ):
        # Uses LangChain structured output as the single standard mechanism.
        from ai_domain.llm.langchain_adapter import LangChainLLMAdapter

        logger = logging.getLogger(__name__)
        call_id = uuid4().hex[:12]
        provider = getattr(config, "provider", None) or self._default_provider
        model = getattr(config, "model", None) or self._default_model
        config, _ = self._sanitize_config(
            config,
            provider=provider,
            model=model,
            context=context,
        )
        metadata = self._attach_required_capabilities(
            dict(config.metadata or {}),
            LLMCapabilities(supports_structured=True, supports_tool_calls=False, supports_seed=config.seed is not None),
        )
        payload = self._build_payload(
            call_id=call_id,
            context=context,
            provider=provider,
            model=model,
            structured=True,
            schema_name=schema.__name__,
            messages=messages,
            metadata=metadata,
        )
        logger.info(json.dumps({"event": "llm_call_start", **payload}, ensure_ascii=False))
        start = time.perf_counter()

        router = self._build_router(config=config)
        adapter = LangChainLLMAdapter(
            llm=router,
            model=model,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            metadata=metadata,
        )

        structured = adapter.with_structured_output(schema, include_raw=include_raw)

        # Convert our messages into LangChain BaseMessages with strict role mapping.
        try:
            from langchain_core.messages import (  # type: ignore
                AIMessage,
                HumanMessage,
                SystemMessage,
                ToolMessage,
            )
        except Exception as e:  # pragma: no cover
            raise RuntimeError("langchain-core is required for invoke_structured") from e

        lc_messages = []
        has_system = False
        for m in messages:
            role = m["role"]
            content = m.get("content") or ""
            if role == "system":
                has_system = True
                lc_messages.append(SystemMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
            elif role == "tool":
                lc_messages.append(ToolMessage(content=content, tool_call_id=m.get("tool_call_id") or "tool"))
            else:
                if role != "user":
                    raise ValueError(f"Unknown role: {role}")
                lc_messages.append(HumanMessage(content=content))

        # Ensure we have a system message to inject format instructions into.
        if not has_system:
            lc_messages.insert(0, SystemMessage(content=""))

        try:
            result = await structured.ainvoke(lc_messages)
        except Exception as e:
            latency_ms = int((time.perf_counter() - start) * 1000)
            self._log_llm_call_end(
                payload=payload,
                latency_ms=latency_ms,
                usage=None,
                response_format="structured",
                parse_ok=False,
                schema_name=schema.__name__,
            )
            logger.exception(
                json.dumps(
                    {
                        "event": "llm_call_error",
                        **payload,
                        "latency_ms": latency_ms,
                        "outcome": "error",
                        "error_kind": self._classify_error(e),
                    },
                    ensure_ascii=False,
                )
            )
            raise

        latency_ms = int((time.perf_counter() - start) * 1000)
        if include_raw:
            output_text = ""
            if isinstance(result, dict) and "raw" in result:
                output_text = str(result.get("raw") or "")
        else:
            output_text = str(result) if result is not None else ""
        prompt_tokens = self._estimate_tokens_for_messages(messages, model=model)
        completion_tokens = self._estimate_tokens(output_text, model=model)
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "estimated": True,
            "completion_unknown": False,
        }
        self._record_metrics(
            context=context,
            payload=payload,
            latency_ms=latency_ms,
            usage=usage,
            **self._output_meta(output_text),
            finish_reason=None,
            response_format="structured",
            retry_count=None,
            fallback_used=False,
            usage_source="estimated",
            request_tokens_estimated=True,
            usage_from_provider=False,
            parse_ok=True,
            schema_name=schema.__name__,
        )
        logger.info(
            json.dumps(
                {
                    "event": "llm_call_success",
                    **payload,
                    "latency_ms": latency_ms,
                    "outcome": "success",
                    "include_raw": include_raw,
                },
                ensure_ascii=False,
            )
        )
        self._log_llm_debug_messages(
            payload=payload,
            messages=messages,
            response=result,
        )

        if include_raw:
            if isinstance(result, StructuredResult):
                return result
            raw = None
            parsing_error = None
            parsed = result
            if isinstance(result, dict):
                parsed = result.get("parsed")
                raw = result.get("raw")
                parsing_error = result.get("parsing_error")
            if isinstance(parsed, dict):
                parsed = schema(**parsed)
            return StructuredResult(parsed=parsed, raw=raw, parsing_error=parsing_error)

        if isinstance(result, dict):
            parsed = result.get("parsed")
            if isinstance(parsed, dict):
                return schema(**parsed)
            if isinstance(parsed, BaseModel):
                return parsed
            return schema(**result)
        if isinstance(result, BaseModel):
            return result
        return schema(**result)

        # Credentials are currently only supported by the router/provider stack.
        # For BYOK, pass credentials to the provider via our LLMRequest path.
        # (Kept as argument for forward compatibility.)
        _ = credentials

    async def invoke_tool_calls(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Sequence[ToolSpec | Dict[str, Any]],
        config: LLMConfig | ResolvedLLMConfig,
        credentials: Optional[LLMCredentials] = None,
        context: LLMCallContext | None = None,
    ):
        logger = logging.getLogger(__name__)
        call_id = uuid4().hex[:12]
        provider = getattr(config, "provider", None) or self._default_provider
        model = getattr(config, "model", None) or self._default_model
        config, _ = self._sanitize_config(
            config,
            provider=provider,
            model=model,
            context=context,
        )
        metadata = self._attach_required_capabilities(
            dict(config.metadata or {}),
            LLMCapabilities(supports_tool_calls=True, supports_structured=False, supports_seed=config.seed is not None),
        )
        payload = self._build_payload(
            call_id=call_id,
            context=context,
            provider=provider,
            model=model,
            structured=False,
            schema_name=None,
            messages=messages,
            metadata=metadata,
        )
        payload["tool_calling"] = True
        payload["tools_count"] = len(tools)
        logger.info(json.dumps({"event": "llm_call_start", **payload}, ensure_ascii=False))
        start = time.perf_counter()
        fallback_used = False

        tool_specs = self._normalize_tool_specs(tools)
        lc_tools = [to_langchain_tool(tool) for tool in tool_specs]
        lc_messages = self._to_langchain_messages(messages)

        def _call_with_provider(provider_name: ProviderName):
            llm = self._build_chat_openai(
                provider_name,
                model=model,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                timeout_s=config.timeout_s,
                credentials=credentials,
            )
            tool_choice = metadata.get("tool_choice")
            if tool_choice:
                llm = llm.bind_tools(lc_tools, tool_choice=tool_choice)
            else:
                llm = llm.bind_tools(lc_tools)
            return llm

        try:
            if not self._breaker.allow():
                raise LLMUnavailable("Circuit breaker open")
            llm = _call_with_provider(provider)
            async with self._limiter:
                response = await llm.ainvoke(lc_messages)
            self._breaker.record_success()
        except Exception as e:
            self._breaker.record_failure(e)
            latency_ms = int((time.perf_counter() - start) * 1000)
            self._log_llm_call_end(
                payload=payload,
                latency_ms=latency_ms,
                usage=None,
                response_format="tool_calls",
            )
            logger.exception(
                json.dumps(
                    {
                        "event": "llm_call_error",
                        **payload,
                        "latency_ms": latency_ms,
                        "outcome": "error",
                        "error_kind": self._classify_error(e),
                    },
                    ensure_ascii=False,
                )
            )
            if self._fallback_provider and self._fallback_provider != provider:
                fallback_provider = self._fallback_provider
                payload["provider"] = fallback_provider
                logger.info(json.dumps({"event": "llm_call_start", **payload}, ensure_ascii=False))
                start = time.perf_counter()
                try:
                    if not self._breaker.allow():
                        raise LLMUnavailable("Circuit breaker open")
                    llm = _call_with_provider(fallback_provider)
                    async with self._limiter:
                        response = await llm.ainvoke(lc_messages)
                    self._breaker.record_success()
                    fallback_used = True
                except Exception as fallback_error:
                    self._breaker.record_failure(fallback_error)
                    latency_ms = int((time.perf_counter() - start) * 1000)
                    self._log_llm_call_end(
                        payload=payload,
                        latency_ms=latency_ms,
                        usage=None,
                        response_format="tool_calls",
                    )
                    logger.exception(
                        json.dumps(
                            {
                                "event": "llm_call_error",
                                **payload,
                                "latency_ms": latency_ms,
                                "outcome": "error",
                                "error_kind": self._classify_error(fallback_error),
                            },
                            ensure_ascii=False,
                        )
                    )
                    raise
            else:
                raise

        latency_ms = int((time.perf_counter() - start) * 1000)
        usage = self._extract_usage_from_langchain_response(response)
        if usage is None:
            prompt_tokens = self._estimate_tokens_for_messages(messages, model=model)
            output_text = getattr(response, "content", "") or ""
            completion_tokens = self._estimate_tokens(output_text, model=model)
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "estimated": True,
                "completion_unknown": False,
            }
        response_format = "tool_calls"
        tool_call_list = getattr(response, "tool_calls", None) or []
        if not tool_call_list:
            response_format = "text"
        self._record_metrics(
            context=context,
            payload=payload,
            latency_ms=latency_ms,
            usage=usage,
            **self._output_meta(getattr(response, "content", "") or ""),
            finish_reason=self._extract_finish_reason_from_langchain_response(response),
            response_format=response_format,
            retry_count=0,
            fallback_used=fallback_used,
            usage_source="estimated" if usage.get("estimated") else "provider",
            request_tokens_estimated=bool(usage.get("estimated")),
            usage_from_provider=not bool(usage.get("estimated")),
        )
        logger.info(
            json.dumps(
                {
                    "event": "llm_call_success",
                    **payload,
                    "latency_ms": latency_ms,
                    "outcome": "success",
                },
                ensure_ascii=False,
            )
        )
        self._log_llm_debug_messages(
            payload=payload,
            messages=messages,
            response={
                "content": getattr(response, "content", "") or "",
                "tool_calls": getattr(response, "tool_calls", []) or [],
            },
        )
        return {
            "content": getattr(response, "content", "") or "",
            "tool_calls": getattr(response, "tool_calls", []) or [],
        }

    async def invoke_tool_response(
        self,
        messages: List[Dict[str, Any]],
        *,
        config: LLMConfig | ResolvedLLMConfig,
        credentials: Optional[LLMCredentials] = None,
        context: LLMCallContext | None = None,
    ) -> str:
        logger = logging.getLogger(__name__)
        call_id = uuid4().hex[:12]
        provider = getattr(config, "provider", None) or self._default_provider
        model = getattr(config, "model", None) or self._default_model
        config, _ = self._sanitize_config(
            config,
            provider=provider,
            model=model,
            context=context,
        )
        metadata = self._attach_required_capabilities(
            dict(config.metadata or {}),
            LLMCapabilities(supports_tool_calls=True, supports_structured=False, supports_seed=config.seed is not None),
        )
        payload = self._build_payload(
            call_id=call_id,
            context=context,
            provider=provider,
            model=model,
            structured=False,
            schema_name=None,
            messages=messages,
            metadata=metadata,
        )
        payload["tool_calling"] = True
        payload["tools_count"] = 0
        logger.info(json.dumps({"event": "llm_call_start", **payload}, ensure_ascii=False))
        start = time.perf_counter()

        lc_messages = self._to_langchain_messages(messages)
        try:
            llm = self._build_chat_openai(
                provider,
                model=model,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                timeout_s=config.timeout_s,
                credentials=credentials,
            )
            response = await llm.ainvoke(lc_messages)
        except Exception as e:
            latency_ms = int((time.perf_counter() - start) * 1000)
            self._log_llm_call_end(
                payload=payload,
                latency_ms=latency_ms,
                usage=None,
                response_format="tool_response",
            )
            logger.exception(
                json.dumps(
                    {
                        "event": "llm_call_error",
                        **payload,
                        "latency_ms": latency_ms,
                        "outcome": "error",
                        "error_kind": self._classify_error(e),
                    },
                    ensure_ascii=False,
                )
            )
            raise

        latency_ms = int((time.perf_counter() - start) * 1000)
        usage = self._extract_usage_from_langchain_response(response)
        if usage is None:
            prompt_tokens = self._estimate_tokens_for_messages(messages, model=model)
            output_text = getattr(response, "content", "") or ""
            completion_tokens = self._estimate_tokens(output_text, model=model)
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "estimated": True,
                "completion_unknown": False,
            }
        self._record_metrics(
            context=context,
            payload=payload,
            latency_ms=latency_ms,
            usage=usage,
            **self._output_meta(getattr(response, "content", "") or ""),
            finish_reason=self._extract_finish_reason_from_langchain_response(response),
            response_format="tool_response",
            retry_count=0,
            fallback_used=False,
            usage_source="estimated" if usage.get("estimated") else "provider",
            request_tokens_estimated=bool(usage.get("estimated")),
            usage_from_provider=not bool(usage.get("estimated")),
        )
        logger.info(
            json.dumps(
                {
                    "event": "llm_call_success",
                    **payload,
                    "latency_ms": latency_ms,
                    "outcome": "success",
                },
                ensure_ascii=False,
            )
        )
        self._log_llm_debug_messages(
            payload=payload,
            messages=messages,
            response=getattr(response, "content", "") or "",
        )
        return getattr(response, "content", "") or ""

    async def stream_text(self, *args: Any, **kwargs: Any):  # pragma: no cover
        raise NotImplementedError("stream_text is not implemented yet")

    async def decide_tool(self, req: Any) -> None:  # pragma: no cover
        _ = req
        return None

    async def generate(self, *args: Any, **kwargs: Any) -> LLMResponse:
        """
        Compatibility adapter for existing code:
        - generate(req: LLMRequest) -> LLMResponse
        - generate(messages=[...], model=..., max_output_tokens=..., temperature=..., top_p=...) -> LLMResponse
        """
        context = kwargs.pop("context", None)
        logger = logging.getLogger(__name__)
        call_id = uuid4().hex[:12]
        if args and len(args) == 1 and not kwargs:
            req = args[0]
            if not isinstance(req, LLMRequest):
                raise TypeError("generate(req) expects LLMRequest")
            config = LLMConfig(
                provider=self._default_provider,
                model=req.model or None,
                temperature=req.temperature,
                top_p=req.top_p,
                max_tokens=req.max_output_tokens,
                seed=req.seed,
                metadata=req.metadata or {},
            )
            provider = getattr(config, "provider", None) or self._default_provider
            model = getattr(config, "model", None) or self._default_model
            config, _ = self._sanitize_config(
                config,
                provider=provider,
                model=model,
                context=context,
            )
            req = LLMRequest(
                messages=req.messages,
                model=config.model or model,
                temperature=config.temperature,
                top_p=config.top_p,
                max_output_tokens=config.max_tokens,
                seed=config.seed,
                credentials=req.credentials,
                metadata=req.metadata or {},
            )
            messages = [{"role": m.role, "content": m.content} for m in req.messages]
            payload = self._build_payload(
                call_id=call_id,
                context=context,
                provider=provider,
                model=model,
                structured=False,
                schema_name=None,
                messages=messages,
                metadata=req.metadata or {},
            )
            logger.info(json.dumps({"event": "llm_call_start", **payload}, ensure_ascii=False))
            start = time.perf_counter()
            router = self._build_router(config=config)
            try:
                resp = await router.generate(req)
            except Exception as e:
                latency_ms = int((time.perf_counter() - start) * 1000)
                self._log_llm_call_end(
                    payload=payload,
                    latency_ms=latency_ms,
                    usage=None,
                    response_format="text",
                )
                logger.exception(
                    json.dumps(
                        {
                            "event": "llm_call_error",
                            **payload,
                            "latency_ms": latency_ms,
                            "outcome": "error",
                            "error_kind": self._classify_error(e),
                        },
                        ensure_ascii=False,
                    )
                )
                raise
            latency_ms = int((time.perf_counter() - start) * 1000)
            usage = {
                "prompt_tokens": resp.usage.prompt_tokens if resp.usage else None,
                "completion_tokens": resp.usage.completion_tokens if resp.usage else None,
                "total_tokens": resp.usage.total_tokens if resp.usage else None,
                "estimated": bool(resp.usage.estimated) if resp.usage else False,
                "completion_unknown": False,
            }
            if not resp.usage or resp.usage.total_tokens == 0:
                prompt_tokens = self._estimate_tokens_for_messages(messages, model=model)
                completion_tokens = self._estimate_tokens(resp.content or "", model=model)
                usage = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                    "estimated": True,
                    "completion_unknown": False,
                }
            self._record_metrics(
                context=context,
                payload=payload,
                latency_ms=latency_ms,
                usage=usage,
                finish_reason=resp.finish_reason,
                response_format="text",
                retry_count=(resp.raw or {}).get("retry_count") if isinstance(resp.raw, dict) else None,
                fallback_used=(resp.raw or {}).get("fallback_used") if isinstance(resp.raw, dict) else None,
                usage_source="estimated" if usage.get("estimated") else "provider",
                request_tokens_estimated=bool(usage.get("estimated")),
                usage_from_provider=not bool(usage.get("estimated")),
            )
            logger.info(
                json.dumps(
                    {
                        "event": "llm_call_success",
                        **payload,
                        "latency_ms": latency_ms,
                        "outcome": "success",
                    },
                    ensure_ascii=False,
                )
            )
            return resp

        messages = kwargs["messages"]
        model = kwargs.get("model") or self._default_model
        temperature = kwargs.get("temperature", 0.2)
        top_p = kwargs.get("top_p")
        max_output_tokens = kwargs.get("max_output_tokens", 2048)

        metadata = kwargs.get("metadata") or {}
        config = LLMConfig(
            provider=self._default_provider,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_output_tokens,
            seed=kwargs.get("seed"),
        )
        config, _ = self._sanitize_config(
            config,
            provider=self._default_provider,
            model=model,
            context=context,
        )
        if kwargs.get("seed") is not None:
            metadata = self._attach_required_capabilities(
                metadata,
                LLMCapabilities(supports_seed=True, supports_structured=False, supports_tool_calls=False),
            )
        payload = self._build_payload(
            call_id=call_id,
            context=context,
            provider=self._default_provider,
            model=model,
            structured=False,
            schema_name=None,
            messages=messages,
            metadata=metadata,
        )
        logger.info(json.dumps({"event": "llm_call_start", **payload}, ensure_ascii=False))
        start = time.perf_counter()
        req = LLMRequest(
            messages=_normalize_messages(messages),
            model=config.model or model,
            temperature=config.temperature,
            top_p=config.top_p,
            max_output_tokens=config.max_tokens,
            seed=config.seed,
            metadata=metadata,
        )
        router = self._build_router(config=config)
        try:
            resp = await router.generate(req)
        except Exception as e:
            latency_ms = int((time.perf_counter() - start) * 1000)
            self._log_llm_call_end(
                payload=payload,
                latency_ms=latency_ms,
                usage=None,
                response_format="text",
            )
            logger.exception(
                json.dumps(
                    {
                        "event": "llm_call_error",
                        **payload,
                        "latency_ms": latency_ms,
                        "outcome": "error",
                        "error_kind": self._classify_error(e),
                    },
                    ensure_ascii=False,
                )
            )
            raise
        latency_ms = int((time.perf_counter() - start) * 1000)
        usage = {
            "prompt_tokens": resp.usage.prompt_tokens if resp.usage else None,
            "completion_tokens": resp.usage.completion_tokens if resp.usage else None,
            "total_tokens": resp.usage.total_tokens if resp.usage else None,
            "estimated": bool(resp.usage.estimated) if resp.usage else False,
            "completion_unknown": False,
        }
        if not resp.usage or resp.usage.total_tokens == 0:
            prompt_tokens = self._estimate_tokens_for_messages(messages, model=model)
            completion_tokens = self._estimate_tokens(resp.content or "", model=model)
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "estimated": True,
                "completion_unknown": False,
            }
        self._record_metrics(
            context=context,
            payload=payload,
            latency_ms=latency_ms,
            usage=usage,
            **self._output_meta(resp.content or ""),
            finish_reason=resp.finish_reason,
            response_format="text",
            retry_count=(resp.raw or {}).get("retry_count") if isinstance(resp.raw, dict) else None,
            fallback_used=(resp.raw or {}).get("fallback_used") if isinstance(resp.raw, dict) else None,
            usage_source="estimated" if usage.get("estimated") else "provider",
            request_tokens_estimated=bool(usage.get("estimated")),
            usage_from_provider=not bool(usage.get("estimated")),
        )
        logger.info(
            json.dumps(
                {
                    "event": "llm_call_success",
                    **payload,
                    "latency_ms": latency_ms,
                    "outcome": "success",
                },
                ensure_ascii=False,
            )
        )
        return resp
