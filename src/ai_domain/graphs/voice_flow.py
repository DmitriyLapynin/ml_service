from langgraph.graph import END, StateGraph

from ai_domain.nodes.stage_analysis import StageAnalysisNode
from ai_domain.nodes.final_answer import FinalAnswerNode
from ai_domain.nodes.postprocess import PostprocessNode


def add_voice_flow(g: StateGraph, deps) -> None:
    g.add_node("voice_stage_light", StageAnalysisNode(
        llm=deps.llm,
        prompt_repo=deps.prompt_repo,
        telemetry=deps.telemetry,
    ))

    g.add_node("voice_answer", FinalAnswerNode(
        llm=deps.llm,
        prompt_repo=deps.prompt_repo,
        telemetry=deps.telemetry,
    ))

    g.add_node("voice_postprocess", PostprocessNode(
        telemetry=deps.telemetry,
    ))

    g.add_edge("voice_stage_light", "voice_answer")
    g.add_edge("voice_answer", "voice_postprocess")
    g.add_edge("voice_postprocess", END)