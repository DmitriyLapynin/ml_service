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


def test_supabase_prompt_plain_string():
    repo = PromptRepository(_SBClient(data=[{"content": "hello"}]))
    assert repo.get_prompt(prompt_key="system_prompt", version="v1") == "hello"


def test_supabase_prompt_format_template_with_variables():
    repo = PromptRepository(_SBClient(data=[{"content": "hello {name}"}]))
    assert repo.get_prompt("system_prompt", "v1", variables={"name": "Bob"}) == "hello Bob"


def test_supabase_prompt_format_template_without_variables_returns_raw():
    repo = PromptRepository(_SBClient(data=[{"content": "hello {name}"}]))
    assert repo.get_prompt("system_prompt", "v1") == "hello {name}"


def test_supabase_prompt_json_template_with_defaults_and_override():
    repo = PromptRepository(
        _SBClient(
            data=[
                {
                    "content": '{"template":"Price for {service}: {price}","defaults":{"price":"N/A"}}'
                }
            ]
        )
    )
    out = repo.get_prompt("system_prompt", "v1", variables={"service": "cleaning"})
    assert out == "Price for cleaning: N/A"


def test_supabase_prompt_accepts_channel_param_for_compat():
    repo = PromptRepository(_SBClient(data=[{"content": "hello"}]))
    assert repo.get_prompt("system_prompt", "v1", channel="chat") == "hello"

