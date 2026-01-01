from __future__ import annotations
import asyncio
from dataclasses import dataclass

@dataclass
class ConcurrencyLimiter:
    max_inflight: int

    def __post_init__(self):
        self._sem = asyncio.Semaphore(self.max_inflight)

    async def __aenter__(self):
        await self._sem.acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self._sem.release()
        return False