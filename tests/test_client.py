from types import SimpleNamespace

from pydantic import BaseModel

from llm_client import LLM


def fake_response(text: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=7),
    )


class FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        item = self.replies.pop(0)
        if isinstance(item, Exception):
            raise item
        return fake_response(item)


class Out(BaseModel):
    a: int


def test_chat_returns_text_and_usage(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_LOG_PATH", str(tmp_path / "log.jsonl"))
    llm = LLM("local", _client=FakeClient(["hi"]))
    r = llm.chat([{"role": "user", "content": "x"}])
    assert r.text == "hi" and r.prompt_tokens == 5 and r.completion_tokens == 7
    assert (tmp_path / "log.jsonl").read_text().count("\n") == 1


def test_chat_json_validates_and_repairs(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_LOG_PATH", str(tmp_path / "log.jsonl"))
    llm = LLM("local", _client=FakeClient(['{"a": "bad"}', '```json\n{"a": 3}\n```']))
    out = llm.chat_json([{"role": "user", "content": "x"}], Out)
    assert out.a == 3


def test_fallback_used_when_primary_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_LOG_PATH", str(tmp_path / "log.jsonl"))
    monkeypatch.setattr("time.sleep", lambda *_: None)
    primary = LLM("local", max_retries=0, _client=FakeClient([ValueError("boom")]))
    backup = LLM("local", _client=FakeClient(["from-backup"]))
    primary.fallback = backup
    assert primary.chat([{"role": "user", "content": "x"}]).text == "from-backup"
