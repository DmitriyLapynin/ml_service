from __future__ import annotations
from .types import LLMUsage

def usage_from_openai(raw_usage: dict | None) -> LLMUsage:
    if not raw_usage:
        return LLMUsage(estimated=True)
    pt = int(raw_usage.get("prompt_tokens", 0) or 0)
    ct = int(raw_usage.get("completion_tokens", 0) or 0)
    tt = int(raw_usage.get("total_tokens", pt + ct) or (pt + ct))
    return LLMUsage(prompt_tokens=pt, completion_tokens=ct, total_tokens=tt, estimated=False)