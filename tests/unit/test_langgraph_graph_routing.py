import pytest

from ai_domain.graphs.main_graph import build_graph
from ai_domain.graphs.state import GraphState


class Deps:
    """Пустой контейнер зависимостей: нам не нужны реальные llm/rag/tools в этих тестах."""
    llm = object()
    prompt_repo = object()
    tool_executor = object()
    rag_client = object()
    telemetry = object()


def make_state(channel: str) -> GraphState:
    # Подстрой под свой GraphState, если у тебя другие поля обязательны.
    # Важно: runtime должен быть dict, где мы собираем executed[].
    return GraphState(
        trace_id="t1",
        # request_id="r1",
        tenant_id="tenant",
        conversation_id="conv",
        channel=channel,
        route=None,
        messages=[{"role": "user", "content": "hi"}],
        versions={"system_prompt": "v1", "model": "fake", "analysis_prompt": "v1", "tool_prompt": "v1"},
        policies={"rag_enabled": True, "max_output_tokens": 64},
        credentials={"openai_api_key": "x"},
        runtime={"executed": [], "errors": [], "degraded": False},
        answer=None,
        stage=None,
    )


def patch_graph_nodes(monkeypatch):
    """
    Меняем реальные узлы на RecordingNodes.
    Важно: патчим там, где они ИСПОЛЬЗУЮТСЯ (импортированы),
    а не где они объявлены.
    """

    class RecordingNode:
        def __init__(self, name: str):
            self.name = name

        async def __call__(self, state: GraphState) -> GraphState:
            state.runtime.setdefault("executed", []).append(self.name)
            return state

    class RecordingRouter:
        async def __call__(self, state: GraphState) -> GraphState:
            state.runtime.setdefault("executed", []).append("router")
            state.route = state.channel  # ключевое: route_selector использует route/channel
            return state

    # --- patch main router import in graphs.main_graph ---
    import ai_domain.graphs.main_graph as main_graph
    monkeypatch.setattr(main_graph, "RouterNode", lambda: RecordingRouter())

    # --- patch chat_flow node classes ---
    import ai_domain.graphs.chat_flow as chat_flow
    monkeypatch.setattr(chat_flow, "StageAnalysisNode", lambda **kwargs: RecordingNode("chat_stage_analysis"))
    monkeypatch.setattr(chat_flow, "RagRetrieveNode", lambda **kwargs: RecordingNode("chat_rag_retrieve"))
    monkeypatch.setattr(chat_flow, "ToolsLoopNode", lambda **kwargs: RecordingNode("chat_tools_loop"))
    monkeypatch.setattr(chat_flow, "FinalAnswerNode", lambda **kwargs: RecordingNode("chat_final_answer"))
    monkeypatch.setattr(chat_flow, "PostprocessNode", lambda **kwargs: RecordingNode("chat_postprocess"))

    # --- patch email_flow node classes ---
    import ai_domain.graphs.email_flow as email_flow
    monkeypatch.setattr(email_flow, "FinalAnswerNode", lambda **kwargs: RecordingNode("email_compose"))
    monkeypatch.setattr(email_flow, "PostprocessNode", lambda **kwargs: RecordingNode("email_postprocess"))

    # --- patch voice_flow node classes ---
    import ai_domain.graphs.voice_flow as voice_flow
    monkeypatch.setattr(voice_flow, "StageAnalysisNode", lambda **kwargs: RecordingNode("voice_stage_light"))
    monkeypatch.setattr(voice_flow, "FinalAnswerNode", lambda **kwargs: RecordingNode("voice_answer"))
    monkeypatch.setattr(voice_flow, "PostprocessNode", lambda **kwargs: RecordingNode("voice_postprocess"))


@pytest.mark.asyncio
async def test_graph_routes_to_chat_flow(monkeypatch):
    patch_graph_nodes(monkeypatch)

    graph = build_graph(deps=Deps())
    state = make_state("chat")

    # LangGraph compiled graph: async invoke обычно ainvoke
    out = await graph.ainvoke(state)

    assert out["runtime"]["executed"] == [
        "router",
        "chat_stage_analysis",
        "chat_rag_retrieve",
        "chat_tools_loop",
        "chat_final_answer",
        "chat_postprocess",
    ]


@pytest.mark.asyncio
async def test_graph_routes_to_email_flow(monkeypatch):
    patch_graph_nodes(monkeypatch)

    graph = build_graph(deps=Deps())
    state = make_state("email")

    out = await graph.ainvoke(state)

    assert out["runtime"]["executed"]== [
        "router",
        "email_compose",
        "email_postprocess",
    ]


@pytest.mark.asyncio
async def test_graph_routes_to_voice_flow(monkeypatch):
    patch_graph_nodes(monkeypatch)

    graph = build_graph(deps=Deps())
    state = make_state("voice")

    out = await graph.ainvoke(state)

    assert out["runtime"]["executed"] == [
        "router",
        "voice_stage_light",
        "voice_answer",
        "voice_postprocess",
    ]