import argparse
import asyncio
import json
import logging

from ai_domain.agent.nodes import safety_in_node
from ai_domain.llm.circuit_breaker import CircuitBreaker
from ai_domain.llm.client import LLMClient
from ai_domain.llm.openai_provider import OpenAIProvider
from ai_domain.llm.rate_limit import ConcurrencyLimiter
from ai_domain.secrets import get_secret


def build_live_safety_llm() -> LLMClient:
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
    parser = argparse.ArgumentParser(description="Run agent safety_in_node with a real OpenAI call.")
    parser.add_argument("--text", type=str, required=True, help="User message to classify")
    parser.add_argument("--model", type=str, default="gpt-4.1-mini", help="OpenAI model for safety classifier")
    args = parser.parse_args()

    router = build_live_safety_llm()

    state = {
        "trace_id": "safety-local",
        "messages": [{"role": "user", "content": args.text}],
    "safety_llm": router,
        "safety_model": args.model,
        "executed": [],
    }

    out = await safety_in_node(state)

    result = {
        "unsafe": bool(out.get("unsafe", False)),
        "injection": bool(out.get("injection_suspected", False)),
    }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
