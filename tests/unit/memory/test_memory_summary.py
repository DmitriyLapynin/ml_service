import pytest

from ai_domain.fakes.fakes_new import FakeLLM, FakePromptRepo, FakeTelemetry
from ai_domain.graphs.state import GraphState
from ai_domain.nodes.summarize_memory import SummarizeMemoryNode
from ai_domain.orchestrator.policy_resolver import build_task_configs


def _make_state():
    versions = {
        "memory_summary_prompt": "v1",
        "model": "gpt-test",
    }
    policies = {}
    task_configs = build_task_configs(
        versions=versions,
        policies=policies,
        model_override=None,
        model_params={"max_output_tokens": 64},
    )
    return GraphState(
        trace_id="trace",
        tenant_id="tenant",
        conversation_id="conv",
        channel="chat",
        messages=[
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "latest"},
        ],
        route=None,
        versions=versions,
        policies=policies,
        credentials={},
        task_configs=task_configs,
        runtime={"degraded": False, "errors": [], "executed": []},
        memory_strategy="summary",
        memory_params={"summary_trigger_tokens": 1},
    )


@pytest.mark.asyncio
async def test_memory_summary_node_sets_summary():
    state = _make_state()
    llm = FakeLLM(content="short summary")
    node = SummarizeMemoryNode(llm, FakePromptRepo({"memory_summary_prompt": "summarize"}), FakeTelemetry())

    await node(state)

    assert state.memory.summary == "short summary"
    assert state.memory.summary_meta["prompt_key"] == "memory_summary_prompt"
