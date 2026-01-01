from __future__ import annotations
import time
from typing import Dict, Tuple

class PromptNotFound(Exception):
    pass


class PromptRepository:
    def __init__(self, supabase_client, *, ttl_seconds: int = 120):
        self.sb = supabase_client
        self.ttl = ttl_seconds
        self._cache: Dict[Tuple[str, str], tuple[str, float]] = {}

    def get_prompt(self, *, prompt_key: str, version: str) -> str:
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
        return content