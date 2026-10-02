import asyncio
import json

import httpx
import pytest

from service_bot.deployment import probe_telegram, run_telegram
from service_bot.telegram import TelegramFailure

FAKE_TOKEN = "123456:fake_test_token_not_for_real_telegram"


def response_handler(request):
    method = request.url.path.split("/")[-1]
    result = {
        "getMe": {"username": "service_test_bot"},
        "getWebhookInfo": {"url": ""},
        "getUpdates": [{"update_id": 1, "message": {"text": "private user text"}}],
    }.get(method, True)
    return httpx.Response(200, json={"ok": True, "result": result})


def test_probe_outputs_only_public_metadata(engine):
    requests = []

    def handler(request):
        requests.append(request)
        return response_handler(request)

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            data = await probe_telegram(FAKE_TOKEN, engine, client=client)
            assert data["bot_url"] == "https://t.me/service_test_bot"
            assert data["status"] == "api_checked"
            assert data["entry_count"] == 3
            assert FAKE_TOKEN not in json.dumps(data)
            assert "private user text" not in json.dumps(data)
            assert not client.is_closed

    asyncio.run(scenario())
    update_request = next(r for r in requests if r.url.path.endswith("/getUpdates"))
    assert json.loads(update_request.content)["offset"] == 0
    assert json.loads(update_request.content)["timeout"] == 0
    assert not any(r.url.path.endswith("/sendMessage") for r in requests)


def test_probe_closes_its_client(engine, monkeypatch):
    original = httpx.AsyncClient
    clients = []

    def factory(**kwargs):
        client = original(transport=httpx.MockTransport(response_handler), **kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(httpx, "AsyncClient", factory)
    asyncio.run(probe_telegram(FAKE_TOKEN, engine))
    assert clients[0].is_closed


def test_probe_rejects_bad_updates(engine):
    def handler(request):
        if request.url.path.endswith("/getUpdates"):
            return httpx.Response(200, json={"ok": True, "result": {"unexpected": "shape"}})
        return response_handler(request)

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(TelegramFailure, match="получение сообщений"):
                await probe_telegram(FAKE_TOKEN, engine, client=client)

    asyncio.run(scenario())


def test_probe_authentication_failure_is_safe(engine):
    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(401, json={"ok": False, "error_code": 401})
            )
        ) as client:
            with pytest.raises(TelegramFailure) as error:
                await probe_telegram(FAKE_TOKEN, engine, client=client)
            assert FAKE_TOKEN not in str(error.value)

    asyncio.run(scenario())


class FakeBot:
    def __init__(self):
        self.started = False
        self.closed = False

    async def run(self):
        self.started = True
        try:
            await asyncio.Event().wait()
        finally:
            self.closed = True


def test_duration_cancels_and_awaits_polling():
    bot = FakeBot()
    asyncio.run(run_telegram(bot, duration=0.01))
    assert bot.started and bot.closed


def test_no_duration_uses_normal_polling():
    class FinishedBot:
        ran = False

        async def run(self):
            self.ran = True

    bot = FinishedBot()
    asyncio.run(run_telegram(bot))
    assert bot.ran


def test_terminal_error_is_not_hidden_by_duration():
    class FailedBot:
        async def run(self):
            raise RuntimeError("test failure")

    with pytest.raises(RuntimeError, match="test failure"):
        asyncio.run(run_telegram(FailedBot(), duration=1))
