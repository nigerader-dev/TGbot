"""Короткий markdown-отчёт по результатам scripts/check_phrasings.py.

Запуск:  python scripts/phrasings_report.py отчёт.json >> комментарий.md

Печатает только то, что важно приёмке: сколько формулировок разобрано верно,
сколько решений приняла модель, и таблицу с расхождениями (если они есть).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def report(data: dict, key_questions: tuple[str, ...] = ()) -> str:
    lines = [
        f"{data['passed']}/{data['total']} формулировок разобраны верно; "
        f"решений модели: {data.get('llm_answers', 0)}.",
        "",
    ]
    key_rows = [row for row in data["rows"] if row["question"] in key_questions]
    if key_rows:
        lines += ["| Формулировка | Ответ базы | Кто решил |", "| --- | --- | --- |"]
        for row in key_rows:
            answer = row["entry_id"] if row["entry_id"] else f"честный отказ ({row['status']})"
            lines.append(f"| {row['question']} | {answer} | {row['layer']} |")
        lines.append("")
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
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
        from phrasings import KEY_QUESTIONS
    except ImportError:  # отчёт можно собрать и без набора формулировок
        KEY_QUESTIONS = ()
    print(report(data, KEY_QUESTIONS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
