from __future__ import annotations

class LLMError(Exception):
    """Базовая ошибка LLM слоя."""
    code: str = "LLM_ERROR"
    retryable: bool = False

    def __init__(self, message: str, *, code: str | None = None, retryable: bool | None = None):
        super().__init__(message)
        if code is not None:
            self.code = code
        if retryable is not None:
            self.retryable = retryable

class LLMTimeout(LLMError):
    code = "LLM_TIMEOUT"
    retryable = True

class LLMRateLimited(LLMError):
    code = "LLM_RATE_LIMIT"
    retryable = True

class LLMUnavailable(LLMError):
    code = "LLM_UNAVAILABLE"
    retryable = True

class LLMAuthError(LLMError):
    code = "LLM_AUTH"
    retryable = False

class LLMInvalidRequest(LLMError):
    code = "LLM_INVALID_REQUEST"
    retryable = False

class LLMProviderError(LLMError):
    code = "LLM_PROVIDER_ERROR"
    retryable = True