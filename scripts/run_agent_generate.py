import argparse
import asyncio
import json
import logging

from ai_domain.agent.nodes import generate_node
from ai_domain.llm.circuit_breaker import CircuitBreaker
from ai_domain.llm.client import LLMClient
from ai_domain.llm.openai_provider import OpenAIProvider
from ai_domain.llm.rate_limit import ConcurrencyLimiter
from ai_domain.secrets import get_secret


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
    parser = argparse.ArgumentParser(description="Run agent generate_node with a real OpenAI call.")
    parser.add_argument("--text", type=str, required=True, help="User message to answer")
    parser.add_argument("--rag", action="store_true", help="Enable RAG context usage")
    parser.add_argument("--model", type=str, default="gpt-4.1-mini", help="Model name")
    args = parser.parse_args()

    llm = build_live_llm()

    state = {
        "trace_id": "agent-generate-local",
        "graph": "rag_agent",
        "messages": [{"role": "user", "content": args.text}],
        "is_rag": bool(args.rag),
        "sub_query_results": (
            [
                {
                    "sub_question": "ценовые варианты",
                    "documents": ["Вариант A: от 40 000 руб.", "Вариант B: от 65 000 руб."],
                }
            ]
            if args.rag
            else []
        ),
        "user_instruction": None,
        "role_instruction": None,
        "model": args.model,
        "model_params": {"temperature": 0.2, "max_tokens": 256},
        "llm": llm,
    }

    out = await generate_node(state)
    result = {
        "answer": out.get("answer", {}).get("text"),
        "format": out.get("answer", {}).get("format"),
    }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
