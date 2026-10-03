import asyncio
import json

import httpx
import pytest

from service_bot.telegram import TelegramBot, TelegramFailure

FAKE_TOKEN = "123456:fake_test_token_not_for_real_telegram"


def message(text="Как почистить КАН Ультра?", *, uid=1, chat_id=123, kind="private"):
    return {
        "update_id": uid,
        "message": {
            "message_id": uid,
            "chat": {"id": chat_id, "type": kind},
            "from": {"id": chat_id, "is_bot": False},
            "text": text,
        },
    }


def callback(data, *, uid=2):
    return {
        "update_id": uid,
        "callback_query": {
            "id": "callback-1",
            "from": {"id": 123, "is_bot": False},
            "data": data,
            "message": {"chat": {"id": 123, "type": "private"}, "message_id": 1},
        },
    }


def success(result=True):
    return httpx.Response(200, json={"ok": True, "result": result})


def base_handler(request):
    method = request.url.path.split("/")[-1]
    if method == "getMe":
        return success({"username": "test_service_bot", "is_bot": True})
    if method == "getWebhookInfo":
        return success({"url": ""})
    return success()


def test_initialization(assistant):
    calls = []

    def handler(request):
        calls.append(request.url.path.split("/")[-1])
        return base_handler(request)

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            bot = TelegramBot(FAKE_TOKEN, assistant, client=client)
            await bot.initialize()
            assert bot.status.state == "polling"
            assert bot.status.username == "test_service_bot"
            assert FAKE_TOKEN not in str(bot.status.public_view())
            assert calls == ["getMe", "getWebhookInfo", "setMyCommands"]

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "query,entry_id",
    [
        ("Как почистить КАН Ультра?", "kan_ultra_maintenance"),
        ("Как часто обслуживать КИТ?", "kit_frequency"),
        ("У меня переполнена станция", "station_overflow"),
        ("Как почистить станцию Тверь?", None),
    ],
)
def test_telegram_acceptance(knowledge, query, entry_id, assistant):
    sent = []

    def handler(request):
        if request.url.path.endswith("/sendMessage"):
            sent.append(json.loads(request.content))
        return base_handler(request)

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            bot = TelegramBot(FAKE_TOKEN, assistant, client=client)
            await bot.handle_update(message(query))

    asyncio.run(scenario())
    assert sent[0]["chat_id"] == 123
    assert "parse_mode" not in sent[0]
    if entry_id:
        assert sent[0]["text"] == knowledge.entries[entry_id].answer
    else:
        assert "нет информации" in sent[0]["text"]
        assert "сервисный отдел" in sent[0]["text"]


def test_private_only_and_nontext(assistant):
    sent = []

    def handler(request):
        if request.url.path.endswith("/sendMessage"):
            sent.append(json.loads(request.content))
        return success()

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            bot = TelegramBot(FAKE_TOKEN, assistant, client=client)
            await bot.handle_update(message(kind="group"))
            bot_message = message()
            bot_message["message"]["from"]["is_bot"] = True
            await bot.handle_update(bot_message)
            nontext = message()
            del nontext["message"]["text"]
            await bot.handle_update(nontext)

    asyncio.run(scenario())
    assert len(sent) == 1
    assert "только текстовые вопросы" in sent[0]["text"]


def test_callbacks_and_clarification(knowledge, assistant):
    methods = []
    sent = []

    def handler(request):
        methods.append(request.url.path.split("/")[-1])
        if request.url.path.endswith("/sendMessage"):
            sent.append(json.loads(request.content))
        return success()

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            bot = TelegramBot(FAKE_TOKEN, assistant, client=client)
            await bot.handle_update(message("Как самому обслужить станцию?"))
            assert sent[-1]["reply_markup"]["inline_keyboard"][0][0]["callback_data"] == (
                "station:kan_ultra"
            )
            await bot.handle_update(callback("station:kan_ultra"))
            assert sent[-1]["text"] == knowledge.entries["kan_ultra_maintenance"].answer
            await bot.handle_update(callback("faq:kit_frequency", uid=3))
            assert sent[-1]["text"] == knowledge.entries["kit_frequency"].answer
            await bot.handle_update(callback("reset", uid=4))
            assert "Контекст сброшен" in sent[-1]["text"]
            size = len(sent)
            for data in ["station:unknown", "faq:unknown", "inject:answer"]:
                await bot.handle_update(callback(data, uid=5))
            assert len(sent) == size

    asyncio.run(scenario())
    assert "answerCallbackQuery" in methods


@pytest.mark.parametrize("token", ["", "no-token", "123:x", "secret\nwith\nnewlines"])
def test_invalid_token_format_never_calls_network(token, assistant):
    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: pytest.fail("Unexpected API request"))
        ) as client:
            bot = TelegramBot(token, assistant, client=client)
            with pytest.raises(TelegramFailure, match="Неверный формат"):
                await bot.initialize()

    asyncio.run(scenario())


def test_active_webhook_is_not_deleted(assistant):
    calls = []

    def handler(request):
        method = request.url.path.split("/")[-1]
        calls.append(method)
        if method == "getWebhookInfo":
            return success({"url": "https://existing-service.example/webhook"})
        return base_handler(request)

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            bot = TelegramBot(FAKE_TOKEN, assistant, client=client)
            with pytest.raises(TelegramFailure, match="активен webhook"):
                await bot.initialize()

    asyncio.run(scenario())
    assert "deleteWebhook" not in calls


def test_invalid_username(assistant):
    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: success({"username": "<injected>"}))
        ) as client:
            with pytest.raises(TelegramFailure, match="имя"):
                await TelegramBot(FAKE_TOKEN, assistant, client=client).initialize()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "code,retryable",
    [(401, False), (403, False), (409, False), (400, False), (429, True), (500, True)],
)
def test_errors_are_sanitized(code, retryable, assistant):
    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    code,
                    json={
                        "ok": False,
                        "error_code": code,
                        "description": f"Error at {FAKE_TOKEN}",
                        "parameters": {"retry_after": 2},
                    },
                )
            )
        ) as client:
            bot = TelegramBot(FAKE_TOKEN, assistant, client=client)
            with pytest.raises(TelegramFailure) as error:
                await bot._call("getMe")
            assert error.value.retryable == retryable
            assert FAKE_TOKEN not in str(error.value)
            if code == 429:
                assert error.value.retry_after == 2

    asyncio.run(scenario())


@pytest.mark.parametrize("body", ["not-json", "[]"])
def test_bad_api_response(body, assistant):
    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, text=body))
        ) as client:
            with pytest.raises(TelegramFailure) as error:
                await TelegramBot(FAKE_TOKEN, assistant, client=client)._call("getMe")
            assert error.value.retryable

    asyncio.run(scenario())


def test_network_error_does_not_leak_token(assistant):
    def handler(request):
        raise httpx.ConnectError(f"Failed URL {request.url}", request=request)

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(TelegramFailure) as error:
                await TelegramBot(FAKE_TOKEN, assistant, client=client)._call("getMe")
            assert error.value.retryable
            assert FAKE_TOKEN not in str(error.value)

    asyncio.run(scenario())


def test_client_must_be_initialized(assistant):
    async def scenario():
        with pytest.raises(RuntimeError):
            await TelegramBot(FAKE_TOKEN, assistant)._call("getMe")

    asyncio.run(scenario())


def test_retry_send_uses_identical_cached_reply(engine, assistant):
    sent = []

    def handler(request):
        if request.url.path.endswith("/sendMessage"):
            sent.append(json.loads(request.content)["text"])
            if len(sent) == 1:
                return httpx.Response(500, json={"ok": False, "error_code": 500})
        return success()

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            bot = TelegramBot(FAKE_TOKEN, assistant, client=client)
            engine.respond("Как самому обслужить станцию?", "tg:123:123")
            update = message("КАН Ультра", uid=9)
            with pytest.raises(TelegramFailure):
                await bot.handle_update(update)
            assert 9 in bot._pending
            await bot.handle_update(update)
            assert not bot._pending

    asyncio.run(scenario())
    assert sent[0] == sent[1]
    assert "youtu.be" in sent[1]


def test_polling_offsets_and_cancellation(assistant):
    polls = []
    sent = []

    def handler(request):
        method = request.url.path.split("/")[-1]
        if method == "getUpdates":
            payload = json.loads(request.content)
            polls.append(payload)
            if len(polls) == 1:
                return success(
                    [
                        None,
                        {"update_id": "bad"},
                        message("/start", uid=4),
                        message("Как почистить станцию Тверь?", uid=5),
                    ]
                )
            raise asyncio.CancelledError
        if method == "sendMessage":
            sent.append(json.loads(request.content))
        return base_handler(request)

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            bot = TelegramBot(FAKE_TOKEN, assistant, client=client)
            with pytest.raises(asyncio.CancelledError):
                await bot.run()
            assert bot.offset == 6
            assert bot.status.state == "stopped"

    asyncio.run(scenario())
    assert polls[0]["offset"] == 0
    assert polls[1]["offset"] == 6
    assert polls[0]["allowed_updates"] == ["message", "callback_query"]
    assert len(sent) == 2


def test_polling_conflict_stops_with_clear_status(assistant):
    def handler(request):
        if request.url.path.endswith("/getUpdates"):
            return httpx.Response(409, json={"ok": False, "error_code": 409})
        return base_handler(request)

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            bot = TelegramBot(FAKE_TOKEN, assistant, client=client)
            await bot.run()
            assert bot.status.state == "error"
            assert "Конфликт" in bot.status.message

    asyncio.run(scenario())


def test_blocked_chat_does_not_stop_other_chats(assistant):
    polls = [0]
    accepted = []

    def handler(request):
        method = request.url.path.split("/")[-1]
        if method == "getUpdates":
            polls[0] += 1
            if polls[0] == 1:
                return success([message(uid=1, chat_id=111), message(uid=2, chat_id=222)])
            raise asyncio.CancelledError
        if method == "sendMessage":
            payload = json.loads(request.content)
            if payload["chat_id"] == 111:
                return httpx.Response(403, json={"ok": False, "error_code": 403})
            accepted.append(payload)
        return base_handler(request)

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            bot = TelegramBot(FAKE_TOKEN, assistant, client=client)
            with pytest.raises(asyncio.CancelledError):
                await bot.run()
            assert bot.offset == 3

    asyncio.run(scenario())
    assert len(accepted) == 1
    assert accepted[0]["chat_id"] == 222


@pytest.mark.parametrize("failure_phase", ["getMe", "getUpdates"])
def test_transient_errors_are_retried(monkeypatch, failure_phase, assistant):
    attempts = [0]
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    def handler(request):
        method = request.url.path.split("/")[-1]
        if method == failure_phase:
            attempts[0] += 1
            if attempts[0] == 1:
                return httpx.Response(
                    429, json={"ok": False, "error_code": 429, "parameters": {"retry_after": 3}}
                )
        if method == "getUpdates":
            if failure_phase == "getMe" or attempts[0] > 1:
                raise asyncio.CancelledError
        return base_handler(request)

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            bot = TelegramBot(FAKE_TOKEN, assistant, client=client)
            with pytest.raises(asyncio.CancelledError):
                await bot.run()

    asyncio.run(scenario())
    assert sleeps == [3]


def test_owned_client_is_closed(assistant, monkeypatch):
    original = httpx.AsyncClient
    clients = []

    def handler(request):
        if request.url.path.endswith("/getUpdates"):
            raise asyncio.CancelledError
        return base_handler(request)

    def factory(**kwargs):
        client = original(transport=httpx.MockTransport(handler), **kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(httpx, "AsyncClient", factory)

    async def scenario():
        with pytest.raises(asyncio.CancelledError):
            await TelegramBot(FAKE_TOKEN, assistant).run()

    asyncio.run(scenario())
    assert clients[0].is_closed
