import pytest

from ai_domain.nodes.rag_retrieve import RagRetrieveNode


class FakeRagClient:
    async def search(self, query, top_k=5, rag_config_id=None):
        return [
            {"id": "doc1", "score": 0.9, "content": "First chunk"},
            {"id": "doc2", "score": 0.8, "content": "Second chunk"},
        ]


class State:
    def __init__(self):
        self.channel = "chat"
        self.messages = [{"role": "user", "content": "price?"}]
        self.versions = {"rag_config_id": "rc1"}
        self.policies = {"rag_enabled": True, "rag_top_k": 2}
        self.runtime = {}
        self.trace_id = "t1"


@pytest.mark.asyncio
async def test_rag_retrieve_fetches_docs_and_builds_context():
    state = State()
    node = RagRetrieveNode(rag_client=FakeRagClient())

    out = await node(state)

    assert out.runtime["rag"]["used"] is True
    assert out.runtime["rag"]["doc_ids"] == ["doc1", "doc2"]
    assert "First chunk" in out.runtime["rag"]["context"]


@pytest.mark.asyncio
async def test_rag_retrieve_skips_when_disabled():
    state = State()
    state.policies["rag_enabled"] = False

    node = RagRetrieveNode(rag_client=FakeRagClient())
    out = await node(state)

    assert out.runtime["rag"]["used"] is False
    assert out.runtime["rag"]["reason"] == "rag_disabled"


@pytest.mark.asyncio
async def test_rag_retrieve_degrades_on_error():
    class BadRagClient:
        async def search(self, *args, **kwargs):
            raise RuntimeError("db down")

    state = State()
    node = RagRetrieveNode(rag_client=BadRagClient())

    out = await node(state)

    assert out.runtime["degraded"] is True
    assert out.runtime["rag"]["used"] is False
    assert any(e["type"] == "rag_error" for e in out.runtime["errors"])