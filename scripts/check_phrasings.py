"""Проверка формулировок: понимает ли бот перефразировки, не выходя за смысл базы.

Скрипт прогоняет набор вариантов написания через тот же конвейер, что и бот, и
проверяет три вещи:

1. ожидаемая запись найдена, и текст ответа совпал с базой **дословно**;
2. вопросы вне базы (чужая модель, тема, которой нет) получают честный отказ;
3. диалоговый контекст используется, но не подменяет отсутствующие сведения.

Запуск:  python scripts/check_phrasings.py [--json путь] [--pause секунды]

Без ИИ-модели проверяются детерминированные правила (они работают всегда). Если
LLM_* заданы, проверяется весь конвейер вместе с моделью, а в отчёте виден слой
каждого ответа. Ответы, ушедшие в резерв из-за лимита запросов, повторяются —
иначе отчёт показывал бы лимит провайдера, а не понимание формулировок.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from service_bot.__main__ import build_assistant  # noqa: E402
from service_bot.knowledge import KnowledgeStore  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
from phrasings import CASES, CONTEXT_CASES  # noqa: E402

RATE_LIMIT_WAIT = 20.0
MAX_ATTEMPTS = 3


def _rate_limited(assistant) -> bool:
    return any(getattr(router, "last_error", None) == "http_429" for router in assistant.routers)


async def ask(assistant, question: str, session: str, pause: float):
    """Один вопрос; при лимите провайдера — пауза и повтор, чтобы не портить отчёт."""
    for attempt in range(MAX_ATTEMPTS):
        reply = await assistant.respond(question, session)
        if (reply.ai or {}).get("layer") != "rules_fallback":
            break
        if attempt + 1 >= MAX_ATTEMPTS or not _rate_limited(assistant):
            break
        print(f"       лимит провайдера, повтор через {RATE_LIMIT_WAIT:.0f} с…")
        await asyncio.sleep(RATE_LIMIT_WAIT)
    if pause:
        await asyncio.sleep(pause)
    return reply


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", help="куда сохранить машинный отчёт")
    parser.add_argument(
        "--pause",
        type=float,
        default=None,
        help="пауза между вопросами, секунды (по умолчанию 2 при включённой модели)",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    knowledge = KnowledgeStore.load(root / "knowledge" / "knowledge_base.json")
    assistant = build_assistant(knowledge)
    if assistant.enabled:
        print(f"Конвейер с ИИ-моделью: {assistant.model_label}")
    else:
        print("ИИ-модель не настроена: проверяются детерминированные правила базы.")
    pause = args.pause if args.pause is not None else (2.0 if assistant.enabled else 0.0)

    failures: list[str] = []
    rows: list[dict] = []

    def record(question: str, expected: str | None, reply, note: str) -> None:
        layer = (reply.ai or {}).get("layer")
        ok = reply.entry_id == expected
        if expected is not None:
            entry = knowledge.entries.get(expected)
            ok = ok and entry is not None and reply.text == entry.answer
        else:
            ok = ok and reply.status in {"missing", "clarify"}
        mark = "ok  " if ok else "ОШИБКА"
        print(f"{mark} {question!r}")
        print(
            f"       ожидалось {expected!r}: получено {reply.entry_id!r}, "
            f"status={reply.status}, layer={layer}"
        )
        rows.append(
            {
                "question": question,
                "expected": expected,
                "entry_id": reply.entry_id,
                "status": reply.status,
                "layer": layer,
                "note": note,
                "ok": ok,
            }
        )
        if not ok:
            failures.append(f"{question!r}: ожидалось {expected!r}, получено {reply.entry_id!r}")

    for question, expected, note in CASES:
        reply = asyncio.run(ask(assistant, question, "phrasings", pause))
        record(question, expected, reply, note)

    print("\nКонтекст диалога:")
    for first, second, expected, note in CONTEXT_CASES:
        session = f"ctx-{abs(hash((first, second))) % 10**6}"
        asyncio.run(ask(assistant, first, session, pause))
        reply = asyncio.run(ask(assistant, second, session, pause))
        record(f"{first} → {second}", expected, reply, note)
    asyncio.run(assistant.aclose())

    total = len(rows)
    passed = sum(1 for row in rows if row["ok"])
    llm_rows = sum(1 for row in rows if row["layer"] == "llm")
    guard_rows = sum(1 for row in rows if str(row["layer"]).startswith("rules"))
    print(
        f"\nИтого: {passed}/{total} верно; решений модели: {llm_rows}; "
        f"ответов правил базы: {guard_rows}."
    )
    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {"passed": passed, "total": total, "llm_answers": llm_rows, "rows": rows},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    if failures:
        print("Проблемные формулировки:")
        for item in failures:
            print(f"- {item}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
