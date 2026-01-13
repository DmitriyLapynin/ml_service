import argparse
import asyncio
import json
import logging

from ai_domain.agent.graph import build_agent_graph
from ai_domain.agent.nodes import AgentNodes
from ai_domain.tools.registry import default_registry


class FakeToolResponse:
    def __init__(self, tool_calls):
        self.content = ""
        self.tool_calls = tool_calls


class FakeLLMClient:
    def __init__(self, *, should_call: bool):
        self._should_call = should_call

    async def invoke_tool_calls(self, messages, tools, config, context=None):  # noqa: ARG002
        _ = (messages, tools, config, context)
        if not self._should_call:
            return FakeToolResponse([])
        return FakeToolResponse(
            [
                {
                    "name": "knowledge_search",
                    "args": {"query": "стоимость брекетов"},
                    "id": "call-ks-1",
                }
            ]
        )


def make_state(text: str, *, should_call: bool) -> dict:
    return {
        "trace_id": "agent-graph-local",
        "graph": "rag_agent",
        "messages": [{"role": "user", "content": text}],
        "tools": default_registry().list(),
        "tool_registry": default_registry(),
        "is_rag": True,
        "llm": FakeLLMClient(should_call=should_call),
    }


async def run_case(label: str, text: str, *, should_call: bool):
    graph = build_agent_graph(AgentNodes())
    out = await graph.ainvoke(make_state(text, should_call=should_call))
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
    parser = argparse.ArgumentParser(description="Run agent graph end-to-end.")
    parser.add_argument("--text", type=str, default="Привет! Что дальше?")
    args = parser.parse_args()

    await run_case("no_tool_call", args.text, should_call=False)
    await run_case("with_tool_call", "Сколько стоят брекеты?", should_call=True)


if __name__ == "__main__":
    asyncio.run(main())
