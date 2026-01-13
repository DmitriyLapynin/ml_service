from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ai_domain.llm.types import LLMCapabilities

if TYPE_CHECKING:
    from .types import LLMRequest, LLMResponse

class LLMProvider(ABC):
    name: str
    capabilities: LLMCapabilities = LLMCapabilities()

    @abstractmethod
    async def generate(self, req: LLMRequest) -> LLMResponse:
        raise NotImplementedError
