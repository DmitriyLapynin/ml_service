import pytest

from ai_domain.fakes.fakes_new import FakeLLM, FakePromptRepo, FakeTelemetry
from ai_domain.graphs.state import GraphState
from ai_domain.nodes.final_answer import FinalAnswerNode
from ai_domain.nodes.stage_analysis import StageAnalysisNode
from ai_domain.orchestrator.policy_resolver import build_task_configs


def _make_state():
    versions = {
        "system_prompt": "v1",
        "analysis_prompt": "v1",
        "model": "gpt",
    }
    policies = {"max_output_tokens": 64, "temperature": 0.2}
    task_configs = build_task_configs(
        versions=versions,
        policies=policies,
        model_override=None,
        model_params={},
    )
    return GraphState(
        trace_id="trace",
        tenant_id="tenant",
        conversation_id="conv",
        channel="chat",
        messages=[
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "last"},
        ],
        route=None,
        versions=versions,
        policies=policies,
        credentials={},
        task_configs=task_configs,
        runtime={"degraded": False, "errors": [], "executed": []},
    )


@pytest.mark.asyncio
async def test_nodes_respect_model_params():
    state = _make_state()
    state.task_configs = build_task_configs(
        versions=state.versions,
        policies=state.policies,
        model_override=None,
        model_params={"temperature": 0.42, "top_p": 0.55},
    )

    final_llm = FakeLLM(content="ok")
    final_node = FinalAnswerNode(final_llm, FakePromptRepo({"system_prompt": "prompt"}), FakeTelemetry())
    await final_node(state)
    final_payload = final_llm.calls[-1]
    assert final_payload["temperature"] == 0.42
    assert final_payload["top_p"] == 0.55

    state.task_configs = build_task_configs(
        versions=state.versions,
        policies=state.policies,
        model_override=None,
        model_params={"temperature": 0.18, "top_p": 0.66},
    )
    stage_llm = FakeLLM(content='{"stage":"ok"}')
    stage_node = StageAnalysisNode(stage_llm, FakePromptRepo({"analysis_prompt": "prompt"}), FakeTelemetry())
    await stage_node(state)
    stage_payload = stage_llm.calls[-1]
    assert stage_payload["temperature"] == 0.18
    assert stage_payload["top_p"] == 0.66
