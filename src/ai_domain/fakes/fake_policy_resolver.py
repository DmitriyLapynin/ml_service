# tests/fakes/fake_policy_resolver.py
class FakePolicyResolver:
    def resolve(self, channel: str):
        return {
            "rag_enabled": channel == "chat",
            "allowed_tools": [],
            "max_tool_calls": 5,
            "max_tool_concurrency_per_request": 3,
            "max_tool_concurrency_global": 20,
        }
