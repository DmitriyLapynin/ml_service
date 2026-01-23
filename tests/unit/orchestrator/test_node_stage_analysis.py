import pytest

from ai_domain.nodes.stage_analysis import StageAnalysisNode
from ai_domain.orchestrator.policy_resolver import build_task_configs


class FakePromptRepo:
    def get_prompt(self, prompt_key, version, channel="any"):
        return "Return JSON with stage and rag_suggested."


class FakeLLM:
    async def generate(self, *args, **kwargs):
        class R:
            content = '{"stage":"closing","rag_suggested": true, "signals":{"target_yes":true}}'
        return R()
        class R:
            content = '{"stage":"closing","rag_suggested": true, "signals":{"target_yes":true}}'
        return R()


class State:
    def __init__(self):
        self.channel = "chat"
        self.messages = [{"role": "user", "content": "hi"}]
        self.versions = {"analysis_prompt": "1.0", "model": "fake"}
        self.policies = {"max_output_tokens": 100}
        self.task_configs = build_task_configs(
            versions=self.versions,
            policies=self.policies,
            model_override=None,
            model_params={},
        )
        self.runtime = {}
        self.stage = None
        self.trace_id = "t1"


@pytest.mark.asyncio
async def test_stage_analysis_parses_json_and_sets_stage():
    state = State()
    node = StageAnalysisNode(llm=FakeLLM(), prompt_repo=FakePromptRepo(), telemetry=None)

    out = await node(state)

    assert out.stage == "closing"
    assert out.policies["rag_enabled"] is True
    assert out.runtime["analysis"]["signals"]["target_yes"] is True
    assert out.runtime["prompts_used"][0]["prompt_key"] == "analysis_prompt"


@pytest.mark.asyncio
async def test_stage_analysis_degrades_if_prompt_repo_fails():
    class BadPromptRepo:
        def get_prompt(self, *args, **kwargs):
            raise RuntimeError("no prompt")

    state = State()
    node = StageAnalysisNode(llm=FakeLLM(), prompt_repo=BadPromptRepo(), telemetry=None)

    out = await node(state)

    assert out.runtime["degraded"] is True
    assert any(e["type"] == "prompt_load_error" for e in out.runtime["errors"])
