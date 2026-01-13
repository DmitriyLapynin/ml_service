from __future__ import annotations
import json
import time
from typing import Dict, Tuple, Any, Mapping, Optional

class PromptNotFound(Exception):
    pass


class _SafeFormatDict(dict):
    def __missing__(self, key: str) -> str:  # pragma: no cover
        return "{" + key + "}"


class PromptRepository:
    def __init__(self, supabase_client, *, ttl_seconds: int = 120):
        self.sb = supabase_client
        self.ttl = ttl_seconds
        self._cache: Dict[Tuple[str, str], tuple[str, float]] = {}

    def get_prompt(self, *args, **kwargs) -> str:
        """
        Compatibility wrapper for different call styles used across nodes:
        - get_prompt(prompt_key, version, channel="chat")
        - get_prompt(prompt_key="...", version="...", channel="chat")
        Template support:
        - Plain string: returned as-is.
        - Format-string template: "Hello {name}" rendered if variables are provided.
        - JSON: {"template":"...{x}...", "defaults":{...}} rendered with provided variables overriding defaults.
        """
        if args:
            prompt_key = args[0]
            version = args[1] if len(args) > 1 else None
        else:
            prompt_key = kwargs.get("prompt_key")
            version = kwargs.get("version")

        if not prompt_key or not version:
            raise ValueError("prompt_key and version are required")

        variables: Optional[Mapping[str, Any]] = kwargs.get("variables")

        cache_key = (prompt_key, version)
        now = time.time()

        # cache hit
        if cache_key in self._cache:
            value, expires_at = self._cache[cache_key]
            if expires_at > now:
                return value
            del self._cache[cache_key]

        # Supabase fetch
        res = (
            self.sb.table("prompts")
            .select("content")
            .eq("key", prompt_key)
            .eq("version", version)
            .limit(1)
            .execute()
        )

        if not res.data:
            raise PromptNotFound(f"Prompt {prompt_key}:{version} not found")

        content = res.data[0]["content"]
        self._cache[cache_key] = (content, now + self.ttl)
        return self._render(content, variables=variables)

    def _render(self, content: str, *, variables: Optional[Mapping[str, Any]] = None) -> str:
        if not isinstance(content, str):
            return str(content)

        text = content.strip()
        if text.startswith("{") and text.endswith("}"):
            try:
                data = json.loads(text)
                if isinstance(data, dict) and "template" in data:
                    template = str(data.get("template", ""))
                    defaults = data.get("defaults") or {}
                    merged = dict(defaults) if isinstance(defaults, dict) else {}
                    if variables:
                        merged.update(dict(variables))
                    return template.format_map(_SafeFormatDict(merged))
            except Exception:
                pass

        if variables:
            try:
                return text.format_map(_SafeFormatDict(dict(variables)))
            except Exception:
                return text
        return text
