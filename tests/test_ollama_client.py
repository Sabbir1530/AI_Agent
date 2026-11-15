import pytest
import requests
from shared import ollama_client
from shared.ollama_client import OllamaClient


class Resp:
    def __init__(self, status=200, data=None, text=""):
        self.status_code, self._data, self.text = status, data or {}, text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError("bad")

    def json(self):
        return self._data


def test_payload_has_fixed_options_and_format(monkeypatch):
    seen = {}

    def fake_post(url, json, timeout):
        seen.update(url=url, payload=json)
        return Resp(data={"message": {"content": "hi"}})
    monkeypatch.setattr(ollama_client.requests, "post", fake_post)
    OllamaClient().generate("p", system="s", keep_alive=0, format="json", num_predict=7)
    p = seen["payload"]
    assert p["model"] == "qwen2.5:3b" and p["stream"] is False and p["keep_alive"] == 0
    assert p["options"] == {"temperature": 0, "seed": 42, "num_predict": 7}
    assert p["format"] == "json" and p["messages"][0]["role"] == "system"


def test_http_error_has_detail(monkeypatch):
    monkeypatch.setattr(ollama_client.requests, "post",
                        lambda *a, **k: Resp(status=500, text="model not found"))
    with pytest.raises(requests.HTTPError, match="model not found"):
        OllamaClient().chat([{"role": "user", "content": "x"}])


def test_ensure_ready_messages(monkeypatch):
    def down(*a, **k):
        raise requests.ConnectionError("x")
    monkeypatch.setattr(ollama_client.requests, "get", down)
    with pytest.raises(RuntimeError, match="ollama serve"):
        OllamaClient().ensure_ready()
    monkeypatch.setattr(ollama_client.requests, "get",
                        lambda *a, **k: Resp(data={"models": [{"name": "other:1b"}]}))
    with pytest.raises(RuntimeError, match="ollama pull qwen2.5:3b"):
        OllamaClient().ensure_ready()
    monkeypatch.setattr(ollama_client.requests, "get",
                        lambda *a, **k: Resp(data={"models": [{"name": "qwen2.5:3b"}]}))
    OllamaClient().ensure_ready()
