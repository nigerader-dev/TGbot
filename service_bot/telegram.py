"""Telegram Bot API adapter. Secrets and user text are never logged.

Text questions go through the shared AI assistant; buttons are only a convenience
for clarifications and the welcome message, never a required input method.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import asdict, dataclass

import httpx

from .ai import AIAssistant
from .engine import Reply

TOKEN_FORMAT = re.compile(r"^[0-9]{5,20}:[A-Za-z0-9_-]{20,100}$")


@dataclass
class TelegramStatus:
    state: str = "not_configured"
    username: str | None = None
    message: str = "Telegram-токен не настроен. Сейчас доступно веб-демо."
    bot_id: int | None = None

    def public_view(self) -> dict:
        return asdict(self)


class TelegramFailure(Exception):
    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        retry_after: int = 0,
        code: int | None = None,
    ):
        super().__init__(message)
        self.retryable = retryable
        self.retry_after = retry_after
        self.code = code


class TelegramBot:
    def __init__(
        self, token: str, assistant: AIAssistant, *, client: httpx.AsyncClient | None = None
    ):
        self._token = token.strip()
        self.assistant = assistant
        self.engine = assistant.engine
        self._client = client
        self.status = TelegramStatus()
        self.offset = 0
        # Retrying a failed send must not run the conversational policy twice.
        self._pending: dict[int, tuple[int, Reply]] = {}

    async def _call(self, method: str, payload: dict | None = None):
        if self._client is None:
            raise RuntimeError("Telegram client is not initialised")
        try:
            response = await self._client.post(
                f"https://api.telegram.org/bot{self._token}/{method}", json=payload or {}
            )
        except httpx.HTTPError as exc:
            # Never surface HTTP exception strings: their URL contains the token.
            raise TelegramFailure(
                "Telegram временно недоступен. Повторяем подключение.", retryable=True
            ) from exc
        try:
            data = response.json()
        except ValueError as exc:
            raise TelegramFailure(
                "Telegram вернул некорректный ответ. Повторяем подключение.", retryable=True
            ) from exc
        if not isinstance(data, dict):
            raise TelegramFailure("Некорректный ответ Telegram.", retryable=True)
        if response.is_success and data.get("ok") is True:
            return data.get("result")
        code = data.get("error_code", response.status_code)
        if code == 429:
            wait = data.get("parameters", {}).get("retry_after", 5)
            wait = min(max(wait, 1), 300) if isinstance(wait, int) else 5
            raise TelegramFailure(
                "Telegram ограничил частоту запросов. Ожидаем повтор.",
                retryable=True,
                retry_after=wait,
            )
        if code == 401:
            raise TelegramFailure("Telegram отклонил токен. Проверьте TELEGRAM_BOT_TOKEN.")
        if code == 409:
            raise TelegramFailure(
                "Конфликт Telegram: остановите другой polling-процесс или отключите webhook."
            )
        if code == 403:
            raise TelegramFailure(
                "Отправка запрещена: возможно, пользователь заблокировал бота.", code=403
            )
        raise TelegramFailure(
            "Ошибка Telegram API. Проверьте подключение.",
            retryable=isinstance(code, int) and code >= 500,
        )

    async def initialize(self) -> None:
        if not TOKEN_FORMAT.fullmatch(self._token):
            raise TelegramFailure("Неверный формат TELEGRAM_BOT_TOKEN.")
        self.status.state = "starting"
        self.status.message = "Подключаемся к Telegram."
        me = await self._call("getMe")
        username = me.get("username") if isinstance(me, dict) else None
        if not username or not re.fullmatch(r"[A-Za-z0-9_]{5,32}", username):
            raise TelegramFailure("Не удалось определить имя Telegram-бота.")
        webhook = await self._call("getWebhookInfo")
        if not isinstance(webhook, dict):
            raise TelegramFailure("Не удалось проверить webhook Telegram.")
        if webhook.get("url"):
            raise TelegramFailure(
                "Для этого бота активен webhook. Отключите его перед запуском polling. "
                "Прототип не удаляет чужие настройки автоматически."
            )
        await self._call(
            "setMyCommands",
            {
                "commands": [
                    {"command": "start", "description": "Начать и показать примеры"},
                    {"command": "help", "description": "Что умеет помощник"},
                    {"command": "status", "description": "Режим работы и версия базы"},
                    {"command": "reset", "description": "Сбросить модель станции"},
                ]
            },
        )
        self.status = TelegramStatus(
            "polling",
            username,
            "Telegram-бот подключён.",
            bot_id=me.get("id") if isinstance(me.get("id"), int) else None,
        )

    def _menu(self, reply: Reply) -> dict | None:
        """Buttons are optional shortcuts; every question can be asked as plain text."""
        if reply.options:
            buttons = [
                [{"text": option["label"], "callback_data": f"station:{option['id']}"}]
                for option in reply.options
            ]
        elif reply.status == "info" and reply.reason in {"welcome", "help"}:
            buttons = [
                [{"text": entry.title, "callback_data": f"faq:{entry.id}"}]
                for entry in self.engine.knowledge.document.entries
            ]
        else:
            return None
        buttons.append([{"text": "Сбросить контекст", "callback_data": "reset"}])
        return {"inline_keyboard": buttons}

    async def _send(self, chat_id: int, reply: Reply) -> None:
        payload = {
            "chat_id": chat_id,
            "text": reply.text,
            # Plain text: neither model names nor user messages can inject HTML/Markdown.
            "link_preview_options": {"is_disabled": True},
        }
        menu = self._menu(reply)
        if menu is not None:
            payload["reply_markup"] = menu
        await self._call("sendMessage", payload)

    async def handle_update(self, update: dict) -> None:
        callback = update.get("callback_query")
        message = callback.get("message", {}) if callback else update.get("message", {})
        chat = message.get("chat", {})
        if chat.get("type") != "private" or not isinstance(chat.get("id"), int):
            return
        sender = callback.get("from", {}) if callback else message.get("from", {})
        if sender.get("is_bot"):
            return
        chat_id = chat["id"]
        session_id = f"tg:{chat_id}:{sender.get('id', chat_id)}"
        update_id = update.get("update_id", -1)
        if callback:
            await self._call("answerCallbackQuery", {"callback_query_id": callback["id"]})
        if update_id in self._pending:
            _, reply = self._pending[update_id]
        elif callback:
            data = callback.get("data", "")
            if data == "reset":
                reply = self.engine.reset(session_id)
            elif data.startswith("station:"):
                station = self.engine.knowledge.stations.get(data.removeprefix("station:"))
                if station is None:
                    return
                reply = await self.assistant.respond(station.name, session_id)
            elif data.startswith("faq:"):
                entry = self.engine.knowledge.entries.get(data.removeprefix("faq:"))
                if entry is None:
                    return
                reply = await self.assistant.respond(entry.examples[0], session_id)
            else:
                return
        else:
            text = message.get("text")
            if not isinstance(text, str):
                reply = Reply(
                    "info",
                    "Пока я принимаю только текстовые вопросы. Напишите вопрос сообщением.",
                    "text_only",
                )
            else:
                reply = await self.assistant.respond(text, session_id)
        self._pending[update_id] = (chat_id, reply)
        await self._send(chat_id, reply)
        self._pending.pop(update_id, None)

    async def run(self) -> None:
        owns_client = self._client is None
        if owns_client:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(40, connect=10))
        delay = 1
        initialized = False
        try:
            while True:
                try:
                    if not initialized:
                        await self.initialize()
                        initialized = True
                    updates = await self._call(
                        "getUpdates",
                        {
                            "offset": self.offset,
                            "timeout": 25,
                            "limit": 30,
                            "allowed_updates": ["message", "callback_query"],
                        },
                    )
                    if not isinstance(updates, list):
                        raise TelegramFailure("Некорректные обновления Telegram.", retryable=True)
                    self.status.state = "polling"
                    self.status.message = "Telegram-бот подключён."
                    delay = 1
                    for update in updates:
                        if not isinstance(update, dict):
                            continue
                        update_id = update.get("update_id")
                        if not isinstance(update_id, int) or update_id < self.offset:
                            continue
                        try:
                            await self.handle_update(update)
                        except TelegramFailure as exc:
                            if exc.retryable:
                                raise
                            # One blocked chat must not stop all other conversations.
                            if exc.code != 403:
                                raise
                            self._pending.pop(update_id, None)
                        self.offset = update_id + 1
                except TelegramFailure as exc:
                    if not exc.retryable:
                        self.status.state = "error"
                        self.status.message = str(exc)
                        return
                    self.status.state = "retrying"
                    self.status.message = str(exc)
                    await asyncio.sleep(exc.retry_after or delay)
                    delay = min(delay * 2, 30)
        except asyncio.CancelledError:
            self.status.state = "stopped"
            self.status.message = "Telegram-бот остановлен."
            raise
        finally:
            if owns_client and self._client is not None:
                await self._client.aclose()
