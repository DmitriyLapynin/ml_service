from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class FakeLLMResponse:
    content: str


class FakeLLM:
    """Configurable fake LLM for tests."""
    def __init__(self, *, content: str = "", raise_exc: Exception | None = None):
        self._content = content
        self._raise = raise_exc
        self.calls: List[Dict[str, Any]] = []

    async def generate(self, *, messages, model=None, max_output_tokens=256, temperature=0.2):
        self.calls.append(
            {"messages": messages, "model": model, "max_output_tokens": max_output_tokens, "temperature": temperature}
        )
        if self._raise:
            raise self._raise
        return FakeLLMResponse(content=self._content)


class FakePromptRepo:
    """Minimal prompt repo that also counts calls (for caching tests you may replace)."""
    def __init__(self, prompts: Optional[Dict[str, str]] = None):
        self.prompts = prompts or {}
        self.calls: List[Dict[str, Any]] = []

    def get_prompt(self, prompt_key: str, version: str, channel: str = "any") -> str:
        self.calls.append({"prompt_key": prompt_key, "version": version, "channel": channel})
        # fallback
        return self.prompts.get(prompt_key) or f"PROMPT[{prompt_key}] v={version} ch={channel}"


class FakeRagClient:
    def __init__(self, *, docs: Optional[List[Dict[str, Any]]] = None, raise_exc: Exception | None = None):
        self.docs = docs or []
        self._raise = raise_exc
        self.calls: List[Dict[str, Any]] = []

    async def search(self, *, query: str, top_k: int = 5, rag_config_id: str | None = None):
        self.calls.append({"query": query, "top_k": top_k, "rag_config_id": rag_config_id})
        if self._raise:
            raise self._raise
        return self.docs


class FakeTelemetry:
    def __init__(self):
        self.steps: List[Dict[str, Any]] = []

    def log_step(self, *, trace_id: str | None, node: str, meta: Dict[str, Any]):
        self.steps.append({"trace_id": trace_id, "node": node, "meta": meta})