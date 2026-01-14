import argparse
import asyncio
import json
import logging

from ai_domain.agent.graph import build_agent_graph
from ai_domain.agent.nodes import AgentNodes
from ai_domain.llm.circuit_breaker import CircuitBreaker
from ai_domain.llm.client import LLMClient
from ai_domain.llm.openai_provider import OpenAIProvider
from ai_domain.llm.rate_limit import ConcurrencyLimiter
from ai_domain.secrets import get_secret
from ai_domain.rag.embedder import LocalEmbedder
from ai_domain.rag.funnel_store import FunnelKBResolver
from ai_domain.tools.registry import default_registry


def build_live_llm() -> LLMClient:
    provider = OpenAIProvider(platform_api_key=get_secret("OPENAI_API_KEY", required=True))
    return LLMClient(
        providers={"openai": provider},
        default_provider="openai",
        default_model="gpt-4.1-mini",
        limiter=ConcurrencyLimiter(max_inflight=1),
        breaker=CircuitBreaker(failure_threshold=3, reset_timeout_s=5),
    )


def make_state(text: str, *, llm, model: str, kb_resolver, funnel_id: str, tool_choice: str) -> dict:
    return {
        "trace_id": "agent-graph-live",
        "graph": "rag_agent",
        "messages": [{"role": "user", "content": text}],
        "tools": default_registry().list(),
        "tool_registry": default_registry(),
        "is_rag": True,
        "model": model,
        "llm": llm,
        "kb_resolver": kb_resolver,
        "funnel_id": funnel_id,
        "tool_choice": tool_choice or None,
    }


async def run_case(label: str, text: str, *, model: str, llm: LLMClient, kb_resolver, funnel_id: str, tool_choice: str):
    graph = build_agent_graph(AgentNodes())
    out = await graph.ainvoke(
        make_state(
            text,
            llm=llm,
            model=model,
            kb_resolver=kb_resolver,
            funnel_id=funnel_id,
            tool_choice=tool_choice,
        )
    )
    result = {
        "label": label,
        "executed": out.get("executed"),
        "wants_retrieve": out.get("wants_retrieve"),
        "tool_calls": out.get("tool_calls"),
        "tool_results": [tr.__dict__ for tr in out.get("tool_results", [])],
        "answer": out.get("answer"),
    }
    print(json.dumps(result, ensure_ascii=False))


async def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Run agent graph end-to-end with real LLM.")
    parser.add_argument("--model", type=str, default="gpt-4.1-mini")
    parser.add_argument("--text", type=str, default="Привет! Что дальше?")
    parser.add_argument("--tool-choice", type=str, default="")
    args = parser.parse_args()

    llm = build_live_llm()
    embedder = LocalEmbedder(model_path="embeddings_models/rubert-mini-frida")
    resolver = FunnelKBResolver(base_dir="data/funnels", embedder=embedder)
    funnel_id = "10"

    await run_case(
        "with_tool_call",
        args.text,
        model=args.model,
        llm=llm,
        kb_resolver=resolver,
        funnel_id=funnel_id,
        tool_choice=args.tool_choice,
    )
    # await run_case("with_tool_call", "Сколько стоят брекеты и какой ваш адрес?", model=args.model, llm=llm)


if __name__ == "__main__":
    asyncio.run(main())
