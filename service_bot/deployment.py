"""Safe Telegram connection check and a time-bounded live acceptance session."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import httpx

from .ai import AIAssistant
from .telegram import TelegramBot, TelegramFailure


async def probe_telegram(
    token: str, assistant: AIAssistant, *, client: httpx.AsyncClient | None = None
) -> dict:
    """Check the real API; return only public metadata, never updates or credentials."""
    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(timeout=httpx.Timeout(15, connect=10))
    try:
        bot = TelegramBot(token, assistant, client=client)
        await bot.initialize()
        updates = await bot._call(
            "getUpdates",
            {
                "offset": 0,
                "timeout": 0,
                "limit": 1,
                "allowed_updates": ["message", "callback_query"],
            },
        )
        if not isinstance(updates, list):
            raise TelegramFailure("Не удалось проверить получение сообщений Telegram.")
        return {
            "status": "api_checked",
            "username": bot.status.username,
            "bot_id": bot.status.bot_id,
            "bot_url": f"https://t.me/{bot.status.username}",
            "checked_at": datetime.now(UTC).isoformat(),
            "knowledge_revision": assistant.knowledge.document.revision,
            "entry_count": len(assistant.knowledge.entries),
            "ai_enabled": assistant.enabled,
            "ai_model": assistant.model_label if assistant.enabled else None,
        }
    finally:
        if owns_client:
            await client.aclose()


async def run_telegram(bot: TelegramBot, *, duration: int | None = None) -> None:
    """Normal polling by default. Optional deadline is for manual live CI testing."""
    if duration is None:
        await bot.run()
        return
    try:
        await asyncio.wait_for(bot.run(), timeout=duration)
    except TimeoutError:
        # wait_for has cancelled and awaited the task; the HTTP client is closed.
        return
