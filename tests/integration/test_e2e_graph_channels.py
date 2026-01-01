import pytest

from ai_domain.fakes.fakes_new import FakeLLM, FakePromptRepo, FakeRagClient, FakeTelemetry
from tests.deps import TestDeps
from ai_domain.nodes.stage_analysis import StageAnalysisNode
from ai_domain.nodes.rag_retrieve import RagRetrieveNode
from ai_domain.nodes.postprocess import PostprocessNode

# ВАЖНО: импортируй свой билд графа так, как у тебя в проекте.
# Обычно это:
# from ai_domain.graphs.main_graph import build_graph
from ai_domain.graphs.main_graph import build_graph


def make_state(channel: str):
    # state для langgraph у тебя теперь dict — так и делаем
    return {
        "trace_id": "t1",
        "tenant_id": "tenant",
        "conversation_id": "conv",
        "channel": channel,
        "route": None,
        "messages": [{"role": "user", "content": "hi"}],
        "versions": {
            "model": "fake",
            "analysis_prompt": "v1",
            "rag_config_id": "rc1",
        },
        "policies": {
            "rag_enabled": True,
            "rag_top_k": 2,
            "max_output_tokens": 64,
            "voice_ssml": False,
        },
        "credentials": {"openai_api_key": "x"},
        "runtime": {"executed": [], "errors": [], "degraded": False, "prompts_used": []},
        "answer": {"text": "", "format": "plain", "meta": {}},
        "stage": None,
    }


def patch_graph_nodes(monkeypatch, *, llm, prompt_repo, rag_client, telemetry):
    """
    Патчим узлы внутри graphs/main_graph.py так же, как ты делал ранее в routing тестах.
    Предполагаем, что main_graph.build_graph(deps) создаёт узлы через deps или импортирует узлы.
    Ниже мы monkeypatch'им factory/конструкторы узлов (если у тебя другое имя — подстрой).
    """
    # Импортируем модуль, чтобы патчить атрибуты внутри него
    import ai_domain.graphs.main_graph as mg

    # Реальные узлы, которые мы хотим тестить
    stage_node = StageAnalysisNode(llm=llm, prompt_repo=prompt_repo, telemetry=telemetry)
    rag_node = RagRetrieveNode(rag_client=rag_client, telemetry=telemetry)
    post_node = PostprocessNode(telemetry=telemetry)

    async def router(state):
        state["runtime"]["executed"].append("router")
        ch = state["channel"]
        state["route"] = ch
        return state

    async def final_answer(state):
        # простой “финальный” узел для теста сквозняка
        # формирует ответ и учитывает rag context если есть
        state["runtime"]["executed"].append("final_answer")
        rag = (state.get("runtime") or {}).get("rag") or {}
        ctx = rag.get("context") or ""
        base = "OK"
        if ctx:
            base += " +RAG"
        state["answer"] = {"text": base, "format": "plain", "meta": {}}
        return state

    async def voice_stage_light(state):
        state["runtime"]["executed"].append("voice_stage_light")
        return state

    async def voice_answer(state):
        state["runtime"]["executed"].append("voice_answer")
        state["answer"] = {"text": "VOICE OK", "format": "voice", "meta": {}}
        return state

    # Патчим фабрики/узлы, которые main_graph использует
    # Если у тебя в main_graph имена другие — замени тут.
    monkeypatch.setattr(mg, "router_node", router, raising=False)
    monkeypatch.setattr(mg, "stage_analysis_node", stage_node, raising=False)
    monkeypatch.setattr(mg, "rag_retrieve_node", rag_node, raising=False)
    monkeypatch.setattr(mg, "final_answer_node", final_answer, raising=False)
    monkeypatch.setattr(mg, "postprocess_node", post_node, raising=False)

    # voice ветка (если есть отдельные узлы)
    monkeypatch.setattr(mg, "voice_stage_light_node", voice_stage_light, raising=False)
    monkeypatch.setattr(mg, "voice_answer_node", voice_answer, raising=False)





@pytest.mark.asyncio
async def test_e2e_chat(monkeypatch):
    llm = FakeLLM(content='{"stage":"closing","rag_suggested": true, "signals":{"target_yes":true}}')
    prompt_repo = FakePromptRepo({"analysis_prompt": "Return JSON stage/rag_suggested."})
    rag_client = FakeRagClient(docs=[{"id": "d1", "score": 0.9, "content": "doc"}])
    telemetry = FakeTelemetry()

    patch_graph_nodes(monkeypatch, llm=llm, prompt_repo=prompt_repo, rag_client=rag_client, telemetry=telemetry)

    deps = TestDeps(
    llm=FakeLLM(content='{"stage":"voice_stage","rag_suggested": false}'),
    prompt_repo=FakePromptRepo({"analysis_prompt": "Return JSON"}),
    rag_client=FakeRagClient(docs=[]),
    telemetry=FakeTelemetry(),
)

    graph = build_graph(deps=deps)
    out = await graph.ainvoke(make_state("chat"))

    assert out["route"] == "chat"
    assert out["answer"]["text"] in ("OK +RAG", "OK")  # в зависимости от того, как ветка подключена
    assert "router" in out["runtime"]["executed"]
    assert out["runtime"]["degraded"] is False


@pytest.mark.asyncio
async def test_e2e_email(monkeypatch):
    llm = FakeLLM(content='{"stage":"email_stage","rag_suggested": false}')
    prompt_repo = FakePromptRepo({"analysis_prompt": "Return JSON stage/rag_suggested."})
    rag_client = FakeRagClient(docs=[{"id": "d1", "score": 0.9, "content": "doc"}])
    telemetry = FakeTelemetry()

    patch_graph_nodes(monkeypatch, llm=llm, prompt_repo=prompt_repo, rag_client=rag_client, telemetry=telemetry)

    deps = TestDeps(
    llm=FakeLLM(content='{"stage":"voice_stage","rag_suggested": false}'),
    prompt_repo=FakePromptRepo({"analysis_prompt": "Return JSON"}),
    rag_client=FakeRagClient(docs=[]),
    telemetry=FakeTelemetry(),
    )

    graph = build_graph(deps=deps)
    st = make_state("email")
    st["policies"]["rag_enabled"] = False  # email: rag off
    out = await graph.ainvoke(st)

    assert out["route"] == "email"
    assert "router" in out["runtime"]["executed"]
    assert out["runtime"]["degraded"] is False


@pytest.mark.asyncio
async def test_e2e_voice(monkeypatch):
    llm = FakeLLM(content='{"stage":"voice_stage","rag_suggested": false}')
    prompt_repo = FakePromptRepo({"analysis_prompt": "Return JSON stage/rag_suggested."})
    rag_client = FakeRagClient(docs=[])
    telemetry = FakeTelemetry()

    patch_graph_nodes(monkeypatch, llm=llm, prompt_repo=prompt_repo, rag_client=rag_client, telemetry=telemetry)

    deps = TestDeps(
    llm=FakeLLM(content='{"stage":"voice_stage","rag_suggested": false}'),
    prompt_repo=FakePromptRepo({"analysis_prompt": "Return JSON"}),
    rag_client=FakeRagClient(docs=[]),
    telemetry=FakeTelemetry(),
    )

    graph = build_graph(deps=deps)
    out = await graph.ainvoke(make_state("voice"))

    assert out["route"] == "voice"
    assert "router" in out["runtime"]["executed"]
    # В voice ветке у тебя может быть свои узлы — мы патчим voice_stage_light/voice_answer
    assert "voice_answer" in out["runtime"]["executed"]