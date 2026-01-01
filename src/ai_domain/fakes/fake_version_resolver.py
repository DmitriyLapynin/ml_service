# tests/fakes/fake_version_resolver.py
class FakeVersionResolver:
    async def resolve(self, tenant_id: str, channel: str):
        return {
            "system_prompt": "v1",
            "stage_prompt": "v1",
            "graph": "v1",
        }