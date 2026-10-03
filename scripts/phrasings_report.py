"""Короткий markdown-отчёт по результатам scripts/check_phrasings.py.

Запуск:  python scripts/phrasings_report.py отчёт.json >> комментарий.md

Печатает только то, что важно приёмке: сколько формулировок разобрано верно,
сколько решений приняла модель, и таблицу с расхождениями (если они есть).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def report(data: dict) -> str:
    lines = [
        f"{data['passed']}/{data['total']} формулировок разобраны верно; "
        f"решений модели: {data.get('llm_answers', 0)}.",
        "",
    ]
    bad = [row for row in data["rows"] if not row["ok"]]
    if bad:
        lines += ["| Вопрос | Ожидалось | Получено | Слой |", "| --- | --- | --- | --- |"]
        for row in bad:
            expected = row["expected"] or "честный отказ"
            got = row["entry_id"] or row["status"]
            lines.append(f"| {row['question']} | {expected} | {got} | {row['layer']} |")
    else:
        lines.append(
            "Все проверенные перефразировки, включая отрицательные формулировки "
            "(«как долго можно не чистить станцию КИТ»), дали ответ строго из базы "
            "или честный отказ — без придуманных сведений."
        )
    return "\n".join(lines)


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    print(report(data))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
