from langgraph.graph import END, StateGraph

from ai_domain.nodes.final_answer import FinalAnswerNode
from ai_domain.nodes.postprocess import PostprocessNode


def add_email_flow(g: StateGraph, deps) -> None:
    # В email обычно отдельный prompt_key/логика в FinalAnswerNode
    g.add_node("email_compose", FinalAnswerNode(
        llm=deps.llm,
        prompt_repo=deps.prompt_repo,
        telemetry=deps.telemetry,
    ))

    g.add_node("email_postprocess", PostprocessNode(
        telemetry=deps.telemetry,
    ))

    g.add_edge("email_compose", "email_postprocess")
    g.add_edge("email_postprocess", END)