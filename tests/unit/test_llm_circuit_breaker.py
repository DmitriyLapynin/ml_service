import pytest
from ai_domain.llm.circuit_breaker import CircuitBreaker
from ai_domain.llm.errors import LLMTimeout, LLMInvalidRequest

def test_circuit_breaker_opens_after_threshold():
    br = CircuitBreaker(failure_threshold=3, reset_timeout_s=9999)
    assert br.allow() is True

    br.record_failure(LLMTimeout("t1"))
    br.record_failure(LLMTimeout("t2"))
    assert br.allow() is True

    br.record_failure(LLMTimeout("t3"))
    assert br.allow() is False  # opened

def test_circuit_breaker_does_not_count_invalid_request():
    br = CircuitBreaker(failure_threshold=1, reset_timeout_s=9999)
    br.record_failure(LLMInvalidRequest("bad"))
    assert br.allow() is True  # should not open on invalid request