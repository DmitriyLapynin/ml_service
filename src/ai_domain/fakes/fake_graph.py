# tests/fakes/fake_graph.py
from ai_domain.graphs.state import GraphState


class FakeGraph:
    async def invoke(self, state: GraphState) -> GraphState:
        state.route = state.channel
        state.stage = {
            "current": "final",
            "confidence": 0.9,
        }
        state.answer = {
            "text": f"fake answer for {state.channel}",
        }
        return state