import os

import pytest

from ai_domain.llm.circuit_breaker import CircuitBreaker
from ai_domain.llm.openai_provider import OpenAIProvider
from ai_domain.llm.rate_limit import ConcurrencyLimiter
from ai_domain.llm.retry import RetryPolicy
from ai_domain.llm.routing import LLMRouter, ModelRoute
from ai_domain.llm.types import LLMMessage, LLMRequest
from ai_domain.orchestrator.context_builder import normalize_messages
from ai_domain.orchestrator.service import Orchestrator
from ai_domain.registry.static_prompt_repo import StaticPromptRepo
from ai_domain.telemetry.noop import NoOpTelemetry
from tests.deps import TestDeps


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
            "model": "gpt-4.1-mini",
        }


class InlinePolicyResolver:
    def resolve(self, channel: str):
        return {
            # В этом LIVE-тесте выключаем RAG, чтобы не зависеть от внешнего RAG сервиса.
            "rag_enabled": False,
            "rag_top_k": 2,
            "max_output_tokens": 64,
            "voice_ssml": False,
        }


class InlineTelemetry:
    def error(self, trace_id, exc):
        pass


class RouterLLMAdapter:
    def __init__(self, router: LLMRouter):
        self.router = router

    async def generate(self, *args, **kwargs):
        _ = kwargs.pop("context", None)
        if args and len(args) == 1:
            req = args[0]
        else:
            req = LLMRequest(
                messages=[LLMMessage(**m) for m in kwargs["messages"]],
                model=kwargs["model"],
                max_output_tokens=kwargs.get("max_output_tokens", 256),
                temperature=kwargs.get("temperature", 0.2),
            )
        return await self.router.generate(req)

    async def decide_tool(self, _):
        return None



class GraphWrapper:
    def __init__(self, graph):
        self.graph = graph

    async def invoke(self, state):
        res = await self.graph.ainvoke(state)
        if isinstance(res, dict):
            class Wrapper:
                def __init__(self, data):
                    self.__dict__.update(data)

            return Wrapper(res)
        return res


def make_deps(llm):
    return TestDeps(
        llm=llm,
        prompt_repo=StaticPromptRepo(
            {
                "analysis_prompt": (
                    "Верни СТРОГО JSON без пояснений: "
                    '{"stage":"ok","rag_suggested":false}.'
                ),
                "system_prompt": "Ты вежливый ассистент. Ответь коротко.",
                "tool_prompt": "Если нужен инструмент — опиши его, иначе верни null.",
            }
        ),
        rag_client=None,
        telemetry=NoOpTelemetry(),
        tool_executor=None,
    )


def build_live_router_llm():
    provider = OpenAIProvider(platform_api_key=os.environ["OPENAI_API_KEY"])
    router = LLMRouter(
        providers={"openai": provider},
        route=ModelRoute(
            primary_provider="openai",
            primary_model="gpt-4.1-mini",
            retry_policy=RetryPolicy(max_attempts=2, base_delay_s=0.3, max_delay_s=1.0),
        ),
        limiter=ConcurrencyLimiter(max_inflight=1),
        breaker=CircuitBreaker(failure_threshold=3, reset_timeout_s=5),
    )
    return RouterLLMAdapter(router)


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="LIVE: requires OPENAI_API_KEY",
)
@pytest.mark.asyncio
async def test_orchestrator_via_router_and_graph():
    try:
        from ai_domain.graphs.main_graph import build_graph
    except ModuleNotFoundError as e:
        # Позволяет собирать тесты даже если запущены не из poetry env.
        if "langgraph" in str(e):
            pytest.skip("langgraph is not installed; run via `poetry run pytest`")
        raise

    llm = build_live_router_llm()

    deps = make_deps(llm)
    graph = build_graph(deps=deps)

    orchestrator = Orchestrator(
        graph=GraphWrapper(graph),
        idempotency=InlineIdempotency(),
        version_resolver=InlineVersionResolver(),
        policy_resolver=InlinePolicyResolver(),
        telemetry=InlineTelemetry(),
    )

    request = {
        "tenant_id": "tenant",
        "conversation_id": "conv",
        "channel": "chat",
        "messages": normalize_messages([{"role": "user", "content": "Привет!"}]),
        "idempotency_key": "router-test",
    }

    resp = await orchestrator.run(request)

    assert resp["status"] == "ok"
    assert resp["stage"] == {"current": "final"}
    assert isinstance(resp["answer"].get("text"), str) and resp["answer"]["text"]
