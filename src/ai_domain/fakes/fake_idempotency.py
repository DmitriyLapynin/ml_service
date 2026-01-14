# tests/fakes/fake_idempotency.py
class FakeIdempotency:
    def __init__(self):
        self.store = {}

    async def get(self, key):
        return self.store.get(key)

    async def mark_in_progress(self, key, ttl_seconds: int = 60):  # noqa: ARG002
        self.store[key] = {"status": "in_progress"}

    async def save(self, key, value):
        self.store[key] = value

    async def clear(self, key):
        self.store.pop(key, None)
