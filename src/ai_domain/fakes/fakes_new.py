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

    async def generate(self, *args, **kwargs):
        """
        Поддерживает оба стиля вызова:
        - generate(req)
        - generate(messages=..., model=..., ...)
        """
        if args and len(args) == 1 and not kwargs:
            req = args[0]
            # LLMRequest-подобный объект
            payload = {
                "messages": getattr(req, "messages", None),
                "model": getattr(req, "model", None),
                "max_output_tokens": getattr(req, "max_output_tokens", None),
                "temperature": getattr(req, "temperature", None),
            }
        else:
            payload = {
                "messages": kwargs.get("messages"),
                "model": kwargs.get("model"),
                "max_output_tokens": kwargs.get("max_output_tokens"),
                "temperature": kwargs.get("temperature"),
            }

        self.calls.append(payload)
        if self._raise:
            raise self._raise
        return FakeLLMResponse(content=self._content)

    async def decide_tool(self, req) -> None:
        """
        В тестах по умолчанию никакой tool не вызываем.
        Реальные решения покрываются отдельными тестами tools_loop.
        """
        self.calls.append({"decide_tool": req})
        return None


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


class FakeToolExecutor:
    def __init__(self):
        self.calls: List[Dict[str, Any]] = []

    async def execute(self, tool_name: str, args: Dict[str, Any], state) -> Dict[str, Any]:
        self.calls.append({"tool_name": tool_name, "args": args, "state_trace": getattr(state, 'trace_id', None)})
        return {"status": "ok", "tool": tool_name}
