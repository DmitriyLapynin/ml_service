from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any

from ai_domain.llm.base import LLMProvider
from ai_domain.llm.types import LLMRequest, LLMResponse, LLMUsage, LLMCapabilities
from ai_domain.llm.errors import (
    LLMTimeout, LLMRateLimited, LLMUnavailable, LLMInvalidRequest, LLMProviderError
)
from ai_domain.llm.client_cache import TTLRUClientCache


@dataclass
class OpenAIProvider(LLMProvider):
    """
    Provider, который умеет:
    - platform key (один клиент на сервис)
    - BYOK key (ключ на кампанию) через TTL/LRU in-memory cache
    """
    name: str = "openai"
    capabilities: LLMCapabilities = LLMCapabilities(
        supports_structured=True,
        supports_tool_calls=True,
        supports_seed=True,
    )
    platform_api_key: Optional[str] = None
    byok_cache: Optional[TTLRUClientCache] = None
    timeout_s: float = 20.0

    _platform_client: Optional[Any] = None

    def __post_init__(self):
        if self.platform_api_key:
            self._platform_client = self._make_client(self.platform_api_key)
        if self.byok_cache is None:
            # можно оставить None, если BYOK не используешь
            pass

    def _make_client(self, api_key: str) -> Any:
        # Здесь твой реальный OpenAI SDK клиент.
        # Важно: нигде не логируем api_key.
        from openai import OpenAI
        return OpenAI(api_key=api_key)

    def _get_client(self, req: LLMRequest) -> Any:
        """
        Выбор клиента:
        1) если есть req.credentials.openai_api_key (BYOK) -> cache.get_or_create
        2) иначе -> platform client
        """
        api_key = None
        if req.credentials and req.credentials.openai_api_key:
            api_key = req.credentials.openai_api_key

        if api_key:
            if not self.byok_cache:
                # BYOK ключ пришёл, но кэш не настроен — лучше явно упасть
                raise LLMInvalidRequest("BYOK key provided but cache is not configured", code="BYOK_CACHE_NOT_CONFIGURED")
            return self.byok_cache.get_or_create(api_key=api_key, factory=self._make_client)

        if self._platform_client:
            return self._platform_client

        raise LLMInvalidRequest("No OpenAI key configured (platform or BYOK)", code="NO_OPENAI_KEY")

    async def generate(self, req: LLMRequest) -> LLMResponse:
        client = self._get_client(req)

        try:
            # пример вызова; подстрой под свой SDK и параметры
            # Важно: req.messages должны быть нормализованы в LLMMessage,
            # а здесь мы формируем формат, нужный OpenAI.
            messages = [{"role": m.role, "content": m.content} for m in req.messages]

            # Синхронный SDK можно вызывать в threadpool, но если у тебя async SDK — используй его.
            # Для краткости оставлю прямой вызов. В проде лучше уйти в anyio.to_thread.run_sync
            resp = client.chat.completions.create(
                model=req.model,
                messages=messages,
                temperature=req.temperature,
                max_tokens=req.max_output_tokens,
                timeout=self.timeout_s,
            )

            content = (resp.choices[0].message.content or "").strip()
            usage = getattr(resp, "usage", None)

            return LLMResponse(
                content=content,
                model=req.model,
                provider=self.name,
                usage=LLMUsage(
                    prompt_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
                    completion_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
                    total_tokens=getattr(usage, "total_tokens", 0) if usage else 0,
                    estimated=False,
                ),
                latency_ms=0,  # проставляй из таймера снаружи/внутри
                finish_reason=getattr(resp.choices[0], "finish_reason", None),
                raw=None,
            )

        except Exception as e:
            # Важно: маппинг ошибок под retry/breaker.
            # Здесь подставь свой детектор типов ошибок SDK.
            msg = str(e).lower()
            if "rate limit" in msg or "429" in msg:
                raise LLMRateLimited(str(e))
            if "timeout" in msg:
                raise LLMTimeout(str(e))
            if "invalid" in msg or "bad request" in msg or "400" in msg:
                raise LLMInvalidRequest(str(e))
            raise LLMProviderError(str(e))

    def invalidate_byok_key(self, api_key: str) -> None:
        if self.byok_cache:
            self.byok_cache.invalidate(api_key)
