import pytest

from ai_domain.orchestrator.service import Orchestrator
from ai_domain.graphs.state import GraphState


class DegradedGraph:
    async def invoke(self, state: GraphState) -> GraphState:
        state.runtime["degraded"] = True
        state.runtime["errors"].append("llm_timeout")
        state.answer = {"text": "partial answer"}
        state.stage = {"current": "final"}
        return state


class FakeDeps:
    async def get(self, k): return None
    async def mark_in_progress(self, k): pass
    async def save(self, k, v): pass
    async def clear(self, k): pass


class FakeVersionResolver:
    async def resolve(self, *a, **k):
        return {"system_prompt": "v1", "model": "gpt-test"}


class FakePolicyResolver:
    def resolve(self, *a, **k):
        return {"rag_enabled": False, "max_output_tokens": 64}


class FakeTelemetry:
    def error(self, *a, **k): pass


@pytest.mark.asyncio
async def test_orchestrator_degraded_response():
    orch = Orchestrator(
        graph=DegradedGraph(),
        idempotency=FakeDeps(),
        version_resolver=FakeVersionResolver(),
        policy_resolver=FakePolicyResolver(),
        telemetry=FakeTelemetry(),
    )

    req = {
        "tenant_id": "t1",
        "conversation_id": "c1",
        "channel": "chat",
        "messages": [{"role": "user", "content": "hi"}],
    }

    resp = await orch.run(req)

    assert resp["status"] == "degraded"
    assert resp["answer"]["text"] == "partial answer"