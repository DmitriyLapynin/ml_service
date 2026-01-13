from __future__ import annotations

from dataclasses import dataclass, field
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
from ai_domain.utils.hashing import messages_fingerprint
from ai_domain.secrets import get_secret
from ai_domain.tools.registry import ToolSpec, to_langchain_tool

ProviderName = Literal["openai", "openrouter"]


@dataclass(frozen=True)
class LLMConfig:
    provider: ProviderName = "openai"
    model: Optional[str] = None
    temperature: float = 0.2
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
    temperature: float = 0.2
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
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
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
        logger.info("llm_call_start", extra={"event": "llm_call_start", **payload})
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
        except Exception as e:
            latency_ms = int((time.perf_counter() - start) * 1000)
            logger.exception(
                "llm_call_error",
                extra={
                    "event": "llm_call_error",
                    **payload,
                    "latency_ms": latency_ms,
                    "outcome": "error",
                    "error_kind": self._classify_error(e),
                },
            )
            raise
        latency_ms = int((time.perf_counter() - start) * 1000)
        logger.info(
            "llm_call_success",
            extra={
                "event": "llm_call_success",
                **payload,
                "latency_ms": latency_ms,
                "outcome": "success",
            },
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
        logger.info("llm_call_start", extra={"event": "llm_call_start", **payload})
        start = time.perf_counter()

        router = self._build_router(config=config)
        adapter = LangChainLLMAdapter(
            llm=router,
            model=model,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p if config.top_p is not None else 1.0,
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
            logger.exception(
                "llm_call_error",
                extra={
                    "event": "llm_call_error",
                    **payload,
                    "latency_ms": latency_ms,
                    "outcome": "error",
                    "error_kind": self._classify_error(e),
                },
            )
            raise

        latency_ms = int((time.perf_counter() - start) * 1000)
        logger.info(
            "llm_call_success",
            extra={
                "event": "llm_call_success",
                **payload,
                "latency_ms": latency_ms,
                "outcome": "success",
                "include_raw": include_raw,
            },
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
        logger.info("llm_call_start", extra={"event": "llm_call_start", **payload})
        start = time.perf_counter()

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
            llm = _call_with_provider(provider)
            response = await llm.ainvoke(lc_messages)
        except Exception as e:
            latency_ms = int((time.perf_counter() - start) * 1000)
            logger.exception(
                "llm_call_error",
                extra={
                    "event": "llm_call_error",
                    **payload,
                    "latency_ms": latency_ms,
                    "outcome": "error",
                    "error_kind": self._classify_error(e),
                },
            )
            if self._fallback_provider and self._fallback_provider != provider:
                fallback_provider = self._fallback_provider
                payload["provider"] = fallback_provider
                logger.info("llm_call_start", extra={"event": "llm_call_start", **payload})
                start = time.perf_counter()
                try:
                    llm = _call_with_provider(fallback_provider)
                    response = await llm.ainvoke(lc_messages)
                except Exception as fallback_error:
                    latency_ms = int((time.perf_counter() - start) * 1000)
                    logger.exception(
                        "llm_call_error",
                        extra={
                            "event": "llm_call_error",
                            **payload,
                            "latency_ms": latency_ms,
                            "outcome": "error",
                            "error_kind": self._classify_error(fallback_error),
                        },
                    )
                    raise
            else:
                raise

        latency_ms = int((time.perf_counter() - start) * 1000)
        logger.info(
            "llm_call_success",
            extra={
                "event": "llm_call_success",
                **payload,
                "latency_ms": latency_ms,
                "outcome": "success",
            },
        )
        return response

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
        logger.info("llm_call_start", extra={"event": "llm_call_start", **payload})
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
            logger.exception(
                "llm_call_error",
                extra={
                    "event": "llm_call_error",
                    **payload,
                    "latency_ms": latency_ms,
                    "outcome": "error",
                    "error_kind": self._classify_error(e),
                },
            )
            raise

        latency_ms = int((time.perf_counter() - start) * 1000)
        logger.info(
            "llm_call_success",
            extra={
                "event": "llm_call_success",
                **payload,
                "latency_ms": latency_ms,
                "outcome": "success",
            },
        )
        return getattr(response, "content", "") or ""

    async def stream_text(self, *args: Any, **kwargs: Any):  # pragma: no cover
        raise NotImplementedError("stream_text is not implemented yet")

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
            logger.info("llm_call_start", extra={"event": "llm_call_start", **payload})
            start = time.perf_counter()
            router = self._build_router(config=config)
            try:
                resp = await router.generate(req)
            except Exception as e:
                latency_ms = int((time.perf_counter() - start) * 1000)
                logger.exception(
                    "llm_call_error",
                    extra={
                        "event": "llm_call_error",
                        **payload,
                        "latency_ms": latency_ms,
                        "outcome": "error",
                        "error_kind": self._classify_error(e),
                    },
                )
                raise
            latency_ms = int((time.perf_counter() - start) * 1000)
            logger.info(
                "llm_call_success",
                extra={
                    "event": "llm_call_success",
                    **payload,
                    "latency_ms": latency_ms,
                    "outcome": "success",
                },
            )
            return resp

        messages = kwargs["messages"]
        model = kwargs.get("model") or self._default_model
        temperature = kwargs.get("temperature", 0.2)
        top_p = kwargs.get("top_p")
        max_output_tokens = kwargs.get("max_output_tokens", 2048)

        metadata = kwargs.get("metadata") or {}
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
        logger.info("llm_call_start", extra={"event": "llm_call_start", **payload})
        start = time.perf_counter()
        req = LLMRequest(
            messages=_normalize_messages(messages),
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
            seed=kwargs.get("seed"),
            metadata=metadata,
        )
        config = LLMConfig(
            provider=self._default_provider,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_output_tokens,
        )
        router = self._build_router(config=config)
        try:
            resp = await router.generate(req)
        except Exception as e:
            latency_ms = int((time.perf_counter() - start) * 1000)
            logger.exception(
                "llm_call_error",
                extra={
                    "event": "llm_call_error",
                    **payload,
                    "latency_ms": latency_ms,
                    "outcome": "error",
                    "error_kind": self._classify_error(e),
                },
            )
            raise
        latency_ms = int((time.perf_counter() - start) * 1000)
        logger.info(
            "llm_call_success",
            extra={
                "event": "llm_call_success",
                **payload,
                "latency_ms": latency_ms,
                "outcome": "success",
            },
        )
        return resp
