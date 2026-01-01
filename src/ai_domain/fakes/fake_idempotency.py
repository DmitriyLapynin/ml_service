# tests/fakes/fake_idempotency.py
class FakeIdempotency:
    def __init__(self):
        self.store = {}

    async def get(self, key):
        return self.store.get(key)

    async def mark_in_progress(self, key):
        self.store[key] = None

    async def save(self, key, value):
        self.store[key] = value

    async def clear(self, key):
        self.store.pop(key, None)