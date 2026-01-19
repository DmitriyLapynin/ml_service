import argparse
import asyncio
import os
import json
import logging
from pathlib import Path

from ai_domain.agent.nodes import agent_node
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


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )
    parser = argparse.ArgumentParser(description="Run agent_node with a real OpenAI call.")
    parser.add_argument("--text", type=str, required=True, help="User message to classify/plan")
    parser.add_argument("--rag", action="store_true", help="Enable RAG tools")
    parser.add_argument("--model", type=str, default="gpt-4.1-mini", help="Model name")
    args = parser.parse_args()

    llm = build_live_llm()
    models_dir = os.getenv("AI_DOMAIN_MODELS_DIR", "embeddings_models")
    data_dir = os.getenv("AI_DOMAIN_DATA_DIR", "data")
    embedder = LocalEmbedder(model_path=str(Path(models_dir) / "rubert-mini-frida"))
    resolver = FunnelKBResolver(base_dir=str(Path(data_dir) / "funnels"), embedder=embedder)
    state = {
        "trace_id": "agent-node-local",
        "graph": "rag_agent",
        "messages": [{"role": "user", "content": args.text}],
        "is_rag": bool(args.rag),
        "tools": default_registry().list(),
        "model": args.model,
        "model_params": {"temperature": 0.2, "max_tokens": 128},
        "llm": llm,
        "kb_resolver": resolver,
        "funnel_id": "default",
    }

    out = await agent_node(state)
    tool_result = []
    if out.get("tool_calls"):
        registry = default_registry()
        for call in out["tool_calls"]:
            name = call.get("name")
            args = call.get("args") or call.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            tool_result.append(
                await registry.execute(
                    name,
                    args,
                    state=state,
                    trace_id=state.get("trace_id"),
                    call_id=call.get("id"),
                )
            )
    result = {
        "messages": out.get("messages", []),
        "wants_retrieve": out.get("wants_retrieve"),
        "filtered_tools": out.get("filtered_tools"),
        "tool_result": [tr.__dict__ for tr in tool_result] if tool_result else None,
    }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
