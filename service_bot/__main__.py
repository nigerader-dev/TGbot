"""Run: python -m service_bot [serve | telegram | probe-telegram | ai-check | validate-kb]."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import uvicorn
from dotenv import load_dotenv

from .ai import AIAssistant, knowledge_digest
from .deployment import probe_telegram, run_telegram
from .engine import AnswerEngine
from .knowledge import DEFAULT_KNOWLEDGE_PATH, ROOT, KnowledgeStore
from .llm import LLMRouter, resolve_llm_config
from .telegram import TelegramBot, TelegramFailure
from .web import create_app

MAX_AI_CHECK_DURATION = 21600
PROBE_QUESTIONS = (
    "Как почистить КАН Ультра?",
    "как часто обслуживать КИТ",
    "у меня переполнена станция, что делать?",
    "есть инструкция по обслуживанию КАН?",
    "как почистить станцию Тверь?",
)


def build_assistant(knowledge: KnowledgeStore) -> AIAssistant:
    config = resolve_llm_config(os.environ)
    router = LLMRouter(config) if config else None
    return AIAssistant(AnswerEngine(knowledge), router=router, config=config)


async def run_ai_check(knowledge: KnowledgeStore, assistant: AIAssistant) -> int:
    if not assistant.enabled:
        print(
            "ИИ-слой выключен: не задан LLM_API_KEY / GH_MODELS_TOKEN. "
            "Бот отвечает детерминированными правилами базы знаний.",
            file=sys.stderr,
        )
        return 3
    print(f"Провайдер: {assistant.config.provider}, модель: {assistant.config.model}")
    print(f"Записей в базе: {len(knowledge.entries)}, версия: {knowledge.document.revision}")
    answered_by_model = 0
    for question in PROBE_QUESTIONS:
        reply = await assistant.respond(question, "ai-check")
        meta = reply.ai or {}
        layer = meta.get("layer")
        if layer == "llm" and meta.get("action") == "answer":
            answered_by_model += 1
        print(
            f"- {question!r}: status={reply.status}, entry_id={reply.entry_id}, "
            f"layer={layer}, action={meta.get('action')}, latency_ms={meta.get('latency_ms')}"
        )
    if answered_by_model:
        print("Проверка ИИ: успешно, модель вернула решения по смыслу")
        return 0
    router = assistant.router
    print(
        "Проверка ИИ: модель не дала решений; "
        f"state={getattr(router, 'state', 'unknown')}, "
        f"last_error={getattr(router, 'last_error', None) or 'нет'}, "
        f"detail={getattr(router, 'last_error_detail', None) or 'нет'!r}, "
        f"calls={getattr(router, 'calls', 0)}, fallbacks={getattr(router, 'fallbacks', 0)}. "
        "Бот продолжит отвечать на правилах базы знаний."
    )
    return 3


def main() -> int:
    load_dotenv(ROOT / ".env", override=False)
    parser = argparse.ArgumentParser(description="Сервисный Telegram-бот без выдуманных ответов")
    commands = parser.add_subparsers(dest="command")
    serve = commands.add_parser("serve", help="Веб-демо и Telegram, если настроен токен")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    telegram = commands.add_parser("telegram", help="Только Telegram polling, без веб-сервера")
    telegram.add_argument(
        "--duration",
        type=int,
        help=f"Время работы в секундах (1–{MAX_AI_CHECK_DURATION}) для плановых и CI-сессий",
    )
    probe = commands.add_parser("probe-telegram", help="Проверить Bot API без публикации токена")
    probe.add_argument("--output", type=Path, help="Записать только публичные данные подключения")
    ai_check = commands.add_parser(
        "ai-check", help="Проверить ИИ-слой: провайдер, модель и решения по тестовым вопросам"
    )
    ai_check.add_argument(
        "--offline", action="store_true", help="Показать настройку и промпт без сетевых запросов"
    )
    commands.add_parser("validate-kb", help="Проверить структуру базы знаний")
    args = parser.parse_args()
    duration = getattr(args, "duration", None)
    if duration is not None and not 1 <= duration <= MAX_AI_CHECK_DURATION:
        parser.error(f"--duration должен быть от 1 до {MAX_AI_CHECK_DURATION} секунд")
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
    if args.command == "ai-check":
        assistant = build_assistant(knowledge)
        if args.offline:
            config = assistant.config
            print(
                "Провайдер: "
                + (f"{config.provider}, модель: {config.model}" if config else "выключен")
            )
            print(f"Записей в базе: {len(knowledge.entries)}")
            print("Промпт маршрутизации:")
            print(knowledge_digest(knowledge))
            return 0
        try:
            return asyncio.run(run_ai_check(knowledge, assistant))
        finally:
            asyncio.run(assistant.aclose())
    if args.command in {"telegram", "probe-telegram"}:
        token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        if not token:
            print("Задайте TELEGRAM_BOT_TOKEN в окружении или локальном .env.", file=sys.stderr)
            return 1
        assistant = build_assistant(knowledge)
        if assistant.enabled:
            print(f"ИИ-слой: {assistant.model_label} (ответы всё равно берутся из базы знаний).")
        else:
            print("ИИ-слой не настроен: отвечают правила базы знаний.")
        if args.command == "probe-telegram":
            try:
                public = asyncio.run(probe_telegram(token, assistant))
            except TelegramFailure as exc:
                print(str(exc), file=sys.stderr)
                return 1
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(public, ensure_ascii=False, indent=2), "utf-8")
            print(f"Bot API проверен: {public['bot_url']}. Токен не публикуется.")
            return 0
        bot = TelegramBot(token, assistant)
        try:
            asyncio.run(run_telegram(bot, duration=duration))
        except KeyboardInterrupt:
            return 0
        finally:
            asyncio.run(assistant.aclose())
        if bot.status.state == "error":
            print(bot.status.message, file=sys.stderr)
            return 1
        return 0
    app = create_app(knowledge=knowledge, assistant=build_assistant(knowledge))
    uvicorn.run(
        app,
        host=getattr(args, "host", "0.0.0.0"),
        port=getattr(args, "port", int(os.getenv("PORT", "8000"))),
        access_log=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
