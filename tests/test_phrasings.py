"""Понимание формулировок детерминированными правилами базы (без сети)."""

from __future__ import annotations

import pytest
from phrasings import CASES, CONTEXT_CASES

from service_bot.ai import AIAssistant
from service_bot.engine import AnswerEngine


@pytest.fixture
def assistant(knowledge):
    return AIAssistant(AnswerEngine(knowledge))


@pytest.mark.parametrize(
    "question,expected,note",
    [pytest.param(question, expected, note, id=note) for question, expected, note in CASES],
)
def test_phrasing_is_understood(assistant, knowledge, question, expected, note):
    reply = assistant.engine.respond(question, "phrasings")
    assert reply.entry_id == expected, f"{note}: {reply.status}/{reply.reason}"
    if expected is not None:
        # Ответ всегда совпадает с базой дословно.
        assert reply.text == knowledge.entries[expected].answer
    else:
        assert reply.status in {"missing", "clarify"}
        assert "youtu.be" not in reply.text
        assert "один раз в год" not in reply.text


@pytest.mark.parametrize(
    "first,second,expected,note",
    [
        pytest.param(first, second, expected, note, id=note)
        for first, second, expected, note in CONTEXT_CASES
    ],
)
def test_dialogue_context_is_used(assistant, knowledge, first, second, expected, note):
    session = f"ctx-{note}"
    assistant.engine.respond(first, session)
    reply = assistant.engine.respond(second, session)
    assert reply.entry_id == expected, f"{note}: {reply.status}/{reply.reason}"
    if expected is not None:
        assert reply.text == knowledge.entries[expected].answer
    else:
        assert reply.status in {"missing", "clarify"}
        assert "youtu.be" not in reply.text


def test_new_questions_do_not_reuse_a_stale_topic(knowledge):
    """Контекст не должен «прилипать»: неизвестная модель по-прежнему отказ."""
    engine = AnswerEngine(knowledge)
    engine.respond("Как часто обслуживать КИТ?", "s")
    reply = engine.respond("Как почистить станцию Тверь?", "s")
    assert reply.status == "missing"
    assert reply.entry_id is None
