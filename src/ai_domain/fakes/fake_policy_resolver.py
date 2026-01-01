# tests/fakes/fake_policy_resolver.py
class FakePolicyResolver:
    def resolve(self, channel: str):
        return {
            "rag_enabled": channel == "chat",
            "allowed_tools": [],
        }