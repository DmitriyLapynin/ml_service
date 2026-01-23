from ai_domain.registry.prompts import PromptRepository


class _SBRes:
    def __init__(self, data):
        self.data = data


class _SBQuery:
    def __init__(self, data):
        self._data = data

    def select(self, _):
        return self

    def eq(self, *_args, **_kwargs):
        return self

    def limit(self, _):
        return self

    def execute(self):
        return _SBRes(self._data)


class _SBClient:
    def __init__(self, data):
        self._data = data

    def table(self, _):
        return _SBQuery(self._data)


def test_prompt_repo_accepts_positional_call():
    repo = PromptRepository(_SBClient(data=[{"content": "hello"}]))
    assert repo.get_prompt("system_prompt", "v1", channel="chat") == "hello"


def test_prompt_repo_accepts_keyword_call():
    repo = PromptRepository(_SBClient(data=[{"content": "hello"}]))
    assert repo.get_prompt(prompt_key="system_prompt", version="v1", channel="chat") == "hello"

