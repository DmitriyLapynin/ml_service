import pytest

from ai_domain.orchestrator.service import Orchestrator
from ai_domain.graphs.main_graph import build_graph
from ai_domain.orchestrator.context_builder import normalize_messages
from ai_domain.fakes.fakes_new import (
    FakeLLM,
    FakePromptRepo,
    FakeRagClient,
    FakeTelemetry,
    FakeToolExecutor,
)


class InlineIdempotency:
    def __init__(self):
        self.store = {}

    async def get(self, key):
        return self.store.get(key)

    async def mark_in_progress(self, key):
        self.store[key] = None

    async def save(self, key, value):
        self.store[key] = value

    async def clear(self, key):
        self.store.pop(key, None)


class InlineVersionResolver:
    async def resolve(self, tenant_id: str, channel: str):
        return {
            "system_prompt": "v1",
            "analysis_prompt": "v1",
            "tool_prompt": "v1",
            "rag_config_id": "rc1",
            "model": "fake",
        }


class InlinePolicyResolver:
    def resolve(self, channel: str):
        return {
            "rag_enabled": True,
            "rag_top_k": 2,
            "max_output_tokens": 64,
            "voice_ssml": False,
        }


class InlineTelemetry:
    def error(self, trace_id, exc):
        pass


@pytest.mark.asyncio
async def test_orchestrator_full_stack_chat():
    # Фейки для зависимостей графа
    deps = type(
        "Deps",
        (),
        {
            "llm": FakeLLM(content='{"stage":"ok","rag_suggested": false}'),
            "prompt_repo": FakePromptRepo(
                {
                    "analysis_prompt": "Верни JSON со stage и rag_suggested=false.",
                    "system_prompt": "Ты помощник. Отвечай коротко.",
                    "tool_prompt": "Реши, нужно ли вызывать инструмент.",
                }
            ),
            "rag_client": FakeRagClient(docs=[{"id": "d1", "score": 0.9, "content": "doc"}]),
            "telemetry": FakeTelemetry(),
            "tool_executor": FakeToolExecutor(),
        },
    )()

    graph = build_graph(deps=deps)

    class GraphWrapper:
        async def invoke(self, state):
            res = await graph.ainvoke(state)
            if not isinstance(res, dict):
                return res
            # Оборачиваем dict в простой объект с нужными атрибутами,
            # чтобы соответствовать ожиданиям orchestrator.
            class Wrapper:
                def __init__(self, data):
                    self.__dict__.update(data)
            return Wrapper(res)

    orchestrator = Orchestrator(
        graph=GraphWrapper(),
        idempotency=InlineIdempotency(),
        version_resolver=InlineVersionResolver(),
        policy_resolver=InlinePolicyResolver(),
        telemetry=InlineTelemetry(),
    )

    request = {
        "tenant_id": "t1",
        "conversation_id": "c1",
        "channel": "chat",
        "messages": normalize_messages([{"role": "user", "content": "Привет, расскажи что-нибудь"}]),
        "idempotency_key": "req1",
    }

    resp = await orchestrator.run(request)

    assert resp["status"] == "ok"
    assert resp["answer"] is not None
    assert isinstance(resp["answer"].get("text"), str) and resp["answer"]["text"]
