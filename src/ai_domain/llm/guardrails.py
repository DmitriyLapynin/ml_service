from __future__ import annotations
from .types import LLMRequest
from .errors import LLMInvalidRequest

def validate_request(req: LLMRequest, *, max_input_chars: int = 1_000_000) -> None:
    if not req.messages:
        raise LLMInvalidRequest("No messages", code="EMPTY_MESSAGES", retryable=False)
    if req.max_output_tokens <= 0:
        raise LLMInvalidRequest("max_output_tokens must be positive", code="BAD_MAX_TOKENS", retryable=False)
    total_chars = sum(len(m.content or "") for m in req.messages)
    if total_chars > max_input_chars:
        raise LLMInvalidRequest("Input too large", code="INPUT_TOO_LARGE", retryable=False)