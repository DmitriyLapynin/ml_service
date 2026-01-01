from langgraph.graph import END, StateGraph
from ai_domain.graphs.state import GraphState

from ai_domain.nodes.stage_analysis import StageAnalysisNode
from ai_domain.nodes.rag_retrieve import RagRetrieveNode
from ai_domain.nodes.tools_loop import ToolsLoopNode
from ai_domain.nodes.final_answer import FinalAnswerNode
from ai_domain.nodes.postprocess import PostprocessNode


def add_chat_flow(g: StateGraph, deps) -> None:
    # nodes (из nodes/) создаются здесь, но логика внутри nodes/
    g.add_node("chat_stage_analysis", StageAnalysisNode(
        llm=deps.llm,
        prompt_repo=deps.prompt_repo,
        telemetry=deps.telemetry,
    ))

    g.add_node("chat_rag_retrieve", RagRetrieveNode(
        rag_client=deps.rag_client,
        prompt_repo=deps.prompt_repo,
        telemetry=deps.telemetry,
    ))

    g.add_node("chat_tools_loop", ToolsLoopNode(
        llm=deps.llm,
        prompt_repo=deps.prompt_repo,
        tool_executor=deps.tool_executor,
        telemetry=deps.telemetry,
    ))

    g.add_node("chat_final_answer", FinalAnswerNode(
        llm=deps.llm,
        prompt_repo=deps.prompt_repo,
        telemetry=deps.telemetry,
    ))

    g.add_node("chat_postprocess", PostprocessNode(
        telemetry=deps.telemetry,
    ))

    # edges
    g.add_edge("chat_stage_analysis", "chat_rag_retrieve")

    # conditional: RAG включён?
    def rag_needed(state: GraphState) -> str:
        return "yes" if state.policies.get("rag_enabled") else "no"

    g.add_conditional_edges(
        "chat_rag_retrieve",
        rag_needed,
        {
            "yes": "chat_tools_loop",
            "no": "chat_tools_loop",  # можно в будущем направить сразу в final
        }
    )

    g.add_edge("chat_tools_loop", "chat_final_answer")
    g.add_edge("chat_final_answer", "chat_postprocess")
    g.add_edge("chat_postprocess", END)