import pytest

from ai_domain.fakes.fakes_new import (
    FakeLLM,
    FakePromptRepo,
    FakeRagClient,
    FakeTelemetry,
    FakeToolExecutor,
)
from ai_domain.graphs.main_graph import build_graph
from ai_domain.graphs.state import GraphState
from ai_domain.orchestrator.policy_resolver import build_task_configs
from tests.deps import TestDeps


def make_state(channel: str) -> GraphState:
    versions = {
        "model": "fake",
        "analysis_prompt": "v1",
        "rag_config_id": "rc1",
        "system_prompt": "v1",
        "tool_prompt": "v1",
        "memory_summary_prompt": "v1",
    }
    policies = {
        "rag_enabled": True,
        "rag_top_k": 2,
        "max_output_tokens": 64,
        "voice_ssml": False,
    }
    task_configs = build_task_configs(
        versions=versions,
        policies=policies,
        model_override=None,
        model_params={},
    )
    return GraphState(
        trace_id="t1",
        tenant_id="tenant",
        conversation_id="conv",
        channel=channel,
        route=None,
        messages=[{"role": "user", "content": "hi"}],
        versions=versions,
        policies=policies,
        credentials={"openai_api_key": "x"},
        task_configs=task_configs,
        runtime={"executed": [], "errors": [], "degraded": False, "prompts_used": []},
        answer={"text": "", "format": "plain", "meta": {}},
        stage=None,
    )


def make_deps(llm: FakeLLM, prompt_repo: FakePromptRepo, rag_client: FakeRagClient, telemetry: FakeTelemetry):
    return TestDeps(
        llm=llm,
        prompt_repo=prompt_repo,
        rag_client=rag_client,
        telemetry=telemetry,
        tool_executor=FakeToolExecutor(),
    )


@pytest.mark.asyncio
async def test_e2e_chat():
    deps = make_deps(
        llm=FakeLLM(content='{"stage":"closing","rag_suggested": true, "signals":{"target_yes":true}}'),
        prompt_repo=FakePromptRepo({"analysis_prompt": "Return JSON stage/rag_suggested."}),
        rag_client=FakeRagClient(docs=[{"id": "d1", "score": 0.9, "content": "doc"}]),
        telemetry=FakeTelemetry(),
    )

    graph = build_graph(deps=deps)
    out = await graph.ainvoke(make_state("chat"))

    assert out["route"] == "chat"
    assert out["runtime"]["degraded"] is False


@pytest.mark.asyncio
async def test_e2e_email():
    deps = make_deps(
        llm=FakeLLM(content='{"stage":"email_stage","rag_suggested": false}'),
        prompt_repo=FakePromptRepo({"analysis_prompt": "Return JSON stage/rag_suggested."}),
        rag_client=FakeRagClient(docs=[]),
        telemetry=FakeTelemetry(),
    )

    graph = build_graph(deps=deps)
    st = make_state("email")
    st.policies["rag_enabled"] = False  # email: rag off
    out = await graph.ainvoke(st)

    assert out["route"] == "email"
    assert out["runtime"]["degraded"] is False


@pytest.mark.asyncio
async def test_e2e_voice():
    deps = make_deps(
        llm=FakeLLM(content='{"stage":"voice_stage","rag_suggested": false}'),
        prompt_repo=FakePromptRepo({"analysis_prompt": "Return JSON stage/rag_suggested."}),
        rag_client=FakeRagClient(docs=[]),
        telemetry=FakeTelemetry(),
    )

    graph = build_graph(deps=deps)
    out = await graph.ainvoke(make_state("voice"))

    assert out["route"] == "voice"
    assert out["runtime"]["degraded"] is False
