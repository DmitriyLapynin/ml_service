from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .types import LLMRequest, LLMResponse

class LLMProvider(ABC):
    name: str

    @abstractmethod
    async def generate(self, req: LLMRequest) -> LLMResponse:
        raise NotImplementedError