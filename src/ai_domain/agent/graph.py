import json
import logging
from langgraph.graph import StateGraph, START, END

from .models import RAGAgentState


def build_agent_graph(agent_nodes) -> StateGraph:
    """
    Строит граф RAG-агента с guardrails (вход/выход).
    Узел agent пока заглушка, но место под него выделено.
    """
    logging.info(
        json.dumps(
            {
                "event": "graph_build",
                "graph": "agent_graph",
                "message": "Построение графа RAG-агента с guardrails...",
            },
            ensure_ascii=False,
        )
    )

    workflow = StateGraph(dict)

    # guardrails
    workflow.add_node("safety_in", agent_nodes.safety_in_node)
    workflow.add_node("safety_block", agent_nodes.safety_block_node)

    # основной поток
    workflow.add_node("agent", agent_nodes.agent_node)
    workflow.add_node("retrieve", agent_nodes.tool_executor_node)
    workflow.add_node("generate", agent_nodes.generate_node)

    workflow.add_edge(START, "safety_in")

    workflow.add_conditional_edges(
        "safety_in",
        agent_nodes.safety_router_condition,
        {"block": "safety_block", "ok": "agent"},
    )

    workflow.add_edge("safety_block", END)

    workflow.add_conditional_edges(
        "agent",
        agent_nodes.tool_router_condition,
        {"retrieve": "retrieve", "skip": "generate"},
    )

    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()
