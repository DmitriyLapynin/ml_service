from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class StaticPromptRepo:
    """
    Lightweight prompt repository for local/dev usage.

    Supports the call patterns used by nodes:
    - get_prompt(prompt_key, version, channel="...")
    - get_prompt(prompt_key="...", version="...")
    """

    prompts: Mapping[str, str]

    def get_prompt(self, prompt_key: str, version: str, channel: str = "any") -> str:  # noqa: ARG002
        return self.prompts[prompt_key]

