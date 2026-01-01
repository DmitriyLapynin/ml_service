from __future__ import annotations
import time
from dataclasses import dataclass
from .errors import LLMUnavailable, LLMRateLimited, LLMTimeout, LLMProviderError

@dataclass
class CircuitBreaker:
    failure_threshold: int = 5
    reset_timeout_s: int = 30

    def __post_init__(self):
        self._failures = 0
        self._opened_at: float | None = None

    def allow(self) -> bool:
        if self._opened_at is None:
            return True
        # half-open after timeout
        if (time.time() - self._opened_at) >= self.reset_timeout_s:
            return True
        return False

    def record_success(self):
        self._failures = 0
        self._opened_at = None

    def record_failure(self, exc: Exception):
        # считаем только “внешние” сбои
        if isinstance(exc, (LLMUnavailable, LLMRateLimited, LLMTimeout, LLMProviderError)):
            self._failures += 1
            if self._failures >= self.failure_threshold:
                self._opened_at = time.time()