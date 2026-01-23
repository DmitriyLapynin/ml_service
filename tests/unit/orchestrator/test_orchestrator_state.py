import pytest

from ai_domain.orchestrator.service import Orchestrator
from ai_domain.graphs.state import GraphState


class ContractGraph:
    async def invoke(self, state: GraphState) -> GraphState:
        # Проверяем, что orchestrator всё передал
        assert state.versions["system_prompt"] == "sys_v1"
        assert state.policies["rag_enabled"] is True
        assert state.credentials["openai_api_key"] == "decrypted-key"

        state.answer = {"text": "ok"}
        state.stage = {"current": "final"}
        return state


class FakeIdempotency:
    async def get(self, key): return None
    async def mark_in_progress(self, key): pass
    async def save(self, key, value): pass
    async def clear(self, key): pass


class FakeVersionResolver:
    async def resolve(self, tenant_id, channel):
        return {
            "system_prompt": "sys_v1",
            "model": "gpt-test",
        }


class FakePolicyResolver:
    def resolve(self, channel):
        return {
            "rag_enabled": True,
            "max_output_tokens": 128,
        }


class FakeTelemetry:
    def error(self, trace_id, exc): pass


@pytest.mark.asyncio
async def test_orchestrator_graph_contract():
    orch = Orchestrator(
        graph=ContractGraph(),
        idempotency=FakeIdempotency(),
        version_resolver=FakeVersionResolver(),
        policy_resolver=FakePolicyResolver(),
        telemetry=FakeTelemetry(),
    )

    req = {
        "tenant_id": "t1",
        "conversation_id": "c1",
        "channel": "chat",
        "messages": [{"role": "user", "content": "hi"}],
        "credentials": {
            "openai": "encrypted",
        },
    }

    # подменяем decrypt
    orch._decrypt_credentials = lambda _: {"openai_api_key": "decrypted-key"}

    resp = await orch.run(req)

    assert resp["answer"]["text"] == "ok"