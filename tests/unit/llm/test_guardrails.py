import pytest
from ai_domain.llm.guardrails import validate_request
from ai_domain.llm.types import LLMRequest, LLMMessage
from ai_domain.llm.errors import LLMInvalidRequest

def test_guardrails_empty_messages_rejected():
    req = LLMRequest(messages=[], model="x", max_output_tokens=10)
    with pytest.raises(LLMInvalidRequest) as e:
        validate_request(req)
    assert e.value.code in ("EMPTY_MESSAGES", "LLM_INVALID_REQUEST")

def test_guardrails_bad_max_tokens_rejected():
    req = LLMRequest(messages=[{"role":"user","content":"hi"}], model="x", max_output_tokens=0)  # type: ignore
    with pytest.raises(LLMInvalidRequest):
        validate_request(req)

def test_guardrails_input_too_large_rejected():
    huge = "a" * 10_000
    req = LLMRequest(
        messages=[LLMMessage(role="user", content=huge)],
        model="x",
        max_output_tokens=10,
    )
    with pytest.raises(LLMInvalidRequest):
        validate_request(req, max_input_chars=100)