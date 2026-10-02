"""Run: python -m service_bot [serve | telegram | validate-kb]."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import uvicorn
from dotenv import load_dotenv

from .deployment import probe_telegram, run_telegram
from .engine import AnswerEngine
from .knowledge import DEFAULT_KNOWLEDGE_PATH, ROOT, KnowledgeStore
from .telegram import TelegramBot, TelegramFailure
from .web import create_app


def main() -> int:
    load_dotenv(ROOT / ".env", override=False)
    parser = argparse.ArgumentParser(description="Сервисный Telegram-бот без выдуманных ответов")
    commands = parser.add_subparsers(dest="command")
    serve = commands.add_parser("serve", help="Веб-демо и Telegram, если настроен токен")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    telegram = commands.add_parser("telegram", help="Только Telegram polling, без веб-сервера")
    telegram.add_argument("--duration", type=int, help="Время ручной проверки в секундах (1–1800)")
    probe = commands.add_parser("probe-telegram", help="Проверить Bot API без публикации токена")
    probe.add_argument("--output", type=Path, help="Записать только публичные данные подключения")
    commands.add_parser("validate-kb", help="Проверить структуру базы знаний")
    args = parser.parse_args()
    duration = getattr(args, "duration", None)
    if duration is not None and not 1 <= duration <= 1800:
        parser.error("--duration должен быть от 1 до 1800 секунд")
    path = Path(os.getenv("KNOWLEDGE_BASE_PATH") or DEFAULT_KNOWLEDGE_PATH)
    try:
        knowledge = KnowledgeStore.load(path)
    except (OSError, ValueError) as exc:
        print(
            f"База знаний не прошла проверку: {type(exc).__name__}. Проверьте {path}.",
            file=sys.stderr,
        )
        return 1
    if args.command == "validate-kb":
        print(
            f"База знаний корректна: {len(knowledge.entries)} записи, "
            f"{len(knowledge.stations)} модели, версия {knowledge.document.revision}."
        )
        return 0
    if args.command in {"telegram", "probe-telegram"}:
        token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        if not token:
            print("Задайте TELEGRAM_BOT_TOKEN в окружении или локальном .env.", file=sys.stderr)
            return 1
        engine = AnswerEngine(knowledge)
        if args.command == "probe-telegram":
            try:
                public = asyncio.run(probe_telegram(token, engine))
            except TelegramFailure as exc:
                print(str(exc), file=sys.stderr)
                return 1
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(public, ensure_ascii=False, indent=2), "utf-8")
            print(f"Bot API проверен: {public['bot_url']}. Токен не публикуется.")
            return 0
        bot = TelegramBot(token, engine)
        try:
            asyncio.run(run_telegram(bot, duration=duration))
        except KeyboardInterrupt:
            return 0
        if bot.status.state == "error":
            print(bot.status.message, file=sys.stderr)
            return 1
        return 0
    app = create_app(knowledge=knowledge)
    uvicorn.run(
        app,
        host=getattr(args, "host", "0.0.0.0"),
        port=getattr(args, "port", int(os.getenv("PORT", "8000"))),
        access_log=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
