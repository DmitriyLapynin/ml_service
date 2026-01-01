import time
from ai_domain.registry.prompts import PromptRepository


class FakeSupabase:
    def __init__(self):
        self.calls = 0

    def table(self, name):
        return self

    def select(self, *_):
        return self

    def eq(self, *_):
        return self

    def limit(self, *_):
        return self

    def execute(self):
        self.calls += 1
        return type(
            "Resp",
            (),
            {"data": [{"content": "hello"}]},
        )()


def test_prompt_repo_cache():
    sb = FakeSupabase()
    repo = PromptRepository(sb, ttl_seconds=1)

    p1 = repo.get_prompt(prompt_key="system", version="v1")
    p2 = repo.get_prompt(prompt_key="system", version="v1")

    assert p1 == "hello"
    assert p2 == "hello"
    assert sb.calls == 1  # кеш сработал

    time.sleep(1.1)

    repo.get_prompt(prompt_key="system", version="v1")
    assert sb.calls == 2  # TTL истёк