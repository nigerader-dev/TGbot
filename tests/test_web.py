import asyncio
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from service_bot.telegram import TelegramBot, TelegramStatus
from service_bot.web import create_app


@pytest.fixture
def client(knowledge):
    with TestClient(create_app(knowledge=knowledge, telegram_token="")) as client:
        yield client


def test_live_pages_and_headers(client):
    assert client.get("/").status_code == 200
    for path in ["/static/app.css", "/static/app.js", "/static/favicon.svg"]:
        assert client.get(path).status_code == 200
    home = client.get("/")
    assert "Без догадок" in home.text
    assert home.headers["x-content-type-options"] == "nosniff"
    assert "frame-ancestors" not in home.headers["content-security-policy"]
    assert "x-frame-options" not in home.headers
    assert client.get("/health").json()["status"] == "ok"


def test_demo_status_is_honest_and_knowledge_is_public(client, knowledge):
    status = client.get("/api/status").json()
    assert status["mode"] == "demo"
    assert status["telegram"]["state"] == "not_configured"
    assert status["entry_count"] == len(knowledge.entries)
    assert status["policy"] == "verbatim_knowledge_only"
    assert len(client.get("/api/knowledge").json()["entries"]) == len(knowledge.entries)
    assert client.get("/api/status").headers["cache-control"] == "no-store"


@pytest.mark.parametrize(
    "question,entry",
    [
        ("Как почистить КАН Ультра?", "kan_ultra_maintenance"),
        ("Как часто обслуживать КИТ?", "kit_frequency"),
        ("У меня переполнена станция", "station_overflow"),
        ("Как почистить станцию Тверь?", None),
    ],
)
def test_api_acceptance(client, knowledge, question, entry):
    response = client.post("/api/chat", json={"text": question, "session_id": str(uuid4())})
    assert response.status_code == 200
    reply = response.json()
    assert reply["entry_id"] == entry
    if entry:
        assert reply["text"] == knowledge.entries[entry].answer
    else:
        assert reply["status"] == "missing"
        assert reply["source"] is None


def test_context_and_reset(client):
    sid = str(uuid4())
    client.post("/api/chat", json={"text": "КАН Ультра", "session_id": sid})
    assert (
        client.post("/api/chat", json={"text": "Как почистить?", "session_id": sid}).json()[
            "status"
        ]
        == "answer"
    )
    assert client.post("/api/reset", json={"session_id": sid}).json()["reason"] == "reset"
    assert (
        client.post("/api/chat", json={"text": "Как почистить?", "session_id": sid}).json()[
            "status"
        ]
        == "clarify"
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"text": "", "session_id": str(uuid4())},
        {"text": "  ", "session_id": str(uuid4())},
        {"text": "A" * 1501, "session_id": str(uuid4())},
        {"text": "hello", "session_id": "not-a-uuid"},
        {"text": "hello", "session_id": str(uuid4()), "answer": "injected"},
        {"session_id": str(uuid4())},
    ],
)
def test_validation(client, payload):
    assert client.post("/api/chat", json=payload).status_code == 422


def test_oversized_body_and_bad_content_length(client):
    assert client.post("/api/chat", content="A" * 8193).status_code == 413
    assert (
        client.post("/api/chat", content="{}", headers={"content-length": "abc"}).status_code == 413
    )


def test_bad_json(client):
    assert (
        client.post(
            "/api/chat", content="{", headers={"content-type": "application/json"}
        ).status_code
        == 422
    )


def test_html_is_not_treated_as_a_command(client):
    reply = client.post(
        "/api/chat",
        json={
            "text": "Как почистить станцию <img src=x onerror=alert(1)>?",
            "session_id": str(uuid4()),
        },
    ).json()
    assert reply["status"] == "missing"
    assert "<img" not in reply["text"]
    js = client.get("/static/app.js").text
    assert "innerHTML" not in js
    assert "textContent" in js
    assert "localhost" not in js
    assert "127.0.0.1" not in js


def test_configured_telegram_status_does_not_leak_token(knowledge, monkeypatch):
    async def fake_run(self):
        self.status = TelegramStatus("polling", "service_test_bot", "Telegram-бот подключён.")
        await asyncio.Event().wait()

    monkeypatch.setattr(TelegramBot, "run", fake_run)
    token = "123456:fake_token_for_tests_not_a_real_token"
    with TestClient(create_app(knowledge=knowledge, telegram_token=token)) as client:
        status = client.get("/api/status")
        assert status.json()["telegram"]["state"] == "polling"
        assert status.json()["telegram"]["username"] == "service_test_bot"
        assert token not in status.text
        assert token not in client.get("/api/knowledge").text


def test_loading_defaults_from_env(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "")
    monkeypatch.setenv("KNOWLEDGE_BASE_PATH", "knowledge/knowledge_base.json")
    with TestClient(create_app()) as client:
        assert client.get("/health").status_code == 200
