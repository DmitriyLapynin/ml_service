from langgraph.graph import StateGraph, END

from ai_domain.graphs.state import GraphState
from ai_domain.graphs.chat_flow import add_chat_flow
from ai_domain.graphs.email_flow import add_email_flow
from ai_domain.graphs.voice_flow import add_voice_flow

from ai_domain.nodes.router import RouterNode  # <- из nodes/


def build_graph(*, deps) -> object:
    """
    deps: контейнер зависимостей (llm, prompt_repo, tools_executor, telemetry, rag_client, ...)
    Внутри flows мы достанем нужные зависимости и создадим nodes из nodes/
    """
    g = StateGraph(GraphState)

    # 1) Router node (из nodes/)
    g.add_node("router", RouterNode())

    g.set_entry_point("router")

    # 2) Подключаем ветки
    add_chat_flow(g, deps)
    add_email_flow(g, deps)
    add_voice_flow(g, deps)

    # 3) Роутинг по state.route, который выставляет RouterNode
    def route_selector(state: GraphState) -> str:
        return state.route or state.channel

    g.add_conditional_edges(
        "router",
        route_selector,
        {
            "chat": "chat_stage_analysis",
            "email": "email_compose",
            "voice": "voice_stage_light",
        },
    )

    return g.compile()