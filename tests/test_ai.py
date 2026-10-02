"""The model may only select an entry; text always comes from the knowledge base."""

import asyncio
import json

import httpx
import pytest

from service_bot.ai import AIAssistant
from service_bot.engine import SERVICE_REFERRAL, AnswerEngine
from service_bot.knowledge import KnowledgeStore
from service_bot.llm import LLMConfig, LLMRouter

FAKE_KEY = "sk-model-key-must-never-leak"
MODEL = "openai/gpt-4o-mini"


def build(engine: AnswerEngine, verdict_for, *, requests=None, retries=0, cooldown=300.0):
    """An assistant whose model answers come from a decision function."""

    def handler(request):
        payload = json.loads(request.content)
        if requests is not None:
            requests.append(payload)
        verdict = verdict_for(payload["messages"][-1]["content"])
        if verdict is None:
            return httpx.Response(503, json={"error": {"message": "unavailable"}})
        content = verdict if isinstance(verdict, str) else json.dumps(verdict, ensure_ascii=False)
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    config = LLMConfig(
        provider="github",
        base_url="https://models.github.ai/inference",
        model=MODEL,
        api_key=FAKE_KEY,
        timeout=5.0,
        retries=retries,
        cooldown=cooldown,
    )
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    router = LLMRouter(config, client=client)
    return AIAssistant(engine, router=router, config=config), router


def answer(entry_id: str, confidence: float = 0.9):
    return {"action": "answer", "entry_id": entry_id, "confidence": confidence, "reason": "тест"}


def by_keyword(user_message: str):
    """A stand-in for a well-behaved small model."""
    text = user_message.casefold()
    if "тверь" in text or "войну" in text:
        return {"action": "no_answer", "reason": "нет в базе"}
    if "переполн" in text:
        return answer("station_overflow")
    if "кит" in text and ("част" in text or "год" in text):
        return answer("kit_frequency")
    if "кан" in text or "инструкц" in text or "обслуж" in text:
        return answer("kan_ultra_maintenance")
    return {"action": "no_answer", "reason": "нет в базе"}


def ask(assistant: AIAssistant, question: str, session: str = "s"):
    reply = asyncio.run(assistant.respond(question, session))
    asyncio.run(assistant.aclose())
    return reply


@pytest.mark.parametrize(
    "question,entry_id",
    [
        ("Как почистить КАН Ультра?", "kan_ultra_maintenance"),
        ("как самому обслужить станцию КАН", "kan_ultra_maintenance"),
        ("есть инструкция по обслуживанию КАН?", "kan_ultra_maintenance"),
        ("как часто нужно обслуживать станцию КИТ?", "kit_frequency"),
        ("сколько раз в год чистить КИТ", "kit_frequency"),
        ("У меня переполнена станция. Что делать?", "station_overflow"),
    ],
)
def test_model_selects_an_entry_and_the_text_stays_verbatim(engine, knowledge, question, entry_id):
    assistant, _ = build(engine, by_keyword)
    reply = ask(assistant, question)
    assert reply.status == "answer"
    assert reply.entry_id == entry_id
    assert reply.text == knowledge.entries[entry_id].answer  # verbatim, no model rewriting
    assert reply.source == knowledge.entries[entry_id].source.model_dump()
    assert reply.ai is not None
    assert reply.ai["layer"] == "llm"
    assert reply.ai["model"] == MODEL
    assert FAKE_KEY not in json.dumps(reply.to_dict(), ensure_ascii=False)


def test_unknown_model_is_blocked_before_the_model_is_asked(engine):
    requests = []
    assistant, router = build(engine, by_keyword, requests=requests)
    reply = ask(assistant, "Как почистить станцию Тверь?")
    assert reply.status == "missing"
    assert reply.text == (
        "В моей базе знаний нет информации по обслуживанию станции Тверь. " + SERVICE_REFERRAL
    )
    assert reply.entry_id is None
    assert reply.source is None
    assert requests == []  # the model was never asked: nothing to hallucinate from
    assert router.calls == 0
    assert reply.ai is not None
    assert reply.ai["layer"] == "rules"


@pytest.mark.parametrize(
    "question",
    [
        "Сколько стоит почистить КАН Ультра?",
        "Какая гарантия на КИТ?",
        "Как отремонтировать насос КАН Ультра?",
        "Как почистить фильтр КАН Ультра?",
    ],
)
def test_out_of_scope_topics_never_reach_the_model(engine, question):
    requests = []
    assistant, _ = build(engine, by_keyword, requests=requests)
    reply = ask(assistant, question)
    assert reply.status == "missing"
    assert reply.entry_id is None
    assert requests == []


def test_an_unrelated_question_never_gets_a_fabricated_contact(engine):
    # "телефон" is not in the knowledge base and is not a blocked pattern: the model is
    # asked, declines, and no contact details are invented.
    requests = []
    assistant, _ = build(engine, by_keyword, requests=requests)
    reply = ask(assistant, "Какой телефон у сервисного отдела?")
    assert requests, "the model may be asked for an unknown topic"
    assert reply.status == "missing"
    assert reply.entry_id is None
    assert "телефон" not in reply.text
    assert reply.text.endswith(SERVICE_REFERRAL)


def test_a_wrong_model_choice_is_rejected_and_the_verified_entry_wins(engine, knowledge):
    # The model tries to hand out the КИТ period for a КАН Ультра question.
    assistant, _ = build(engine, lambda _: answer("kit_frequency"))
    reply = ask(assistant, "Как почистить КАН Ультра?")
    assert reply.entry_id == "kan_ultra_maintenance"
    assert reply.text == knowledge.entries["kan_ultra_maintenance"].answer
    assert reply.ai["layer"] == "rules_guard"


def test_an_unknown_entry_id_is_never_sent_to_the_client(engine):
    assistant, _ = build(engine, lambda _: answer("invented_entry"))
    reply = ask(assistant, "Как почистить КАН Ультра?")
    assert reply.entry_id == "kan_ultra_maintenance"
    assert reply.ai["layer"] == "rules_guard"


def test_a_topic_that_has_no_shared_words_is_rejected(engine):
    # A weak model tries to answer an unrelated question with a cleaning video.
    assistant, _ = build(engine, lambda _: answer("kan_ultra_maintenance"))
    reply = ask(assistant, "Кто написал Войну и мир?")
    assert reply.status == "missing"
    assert reply.entry_id is None
    assert "youtu.be" not in reply.text
    assert reply.text.endswith(SERVICE_REFERRAL)


def test_model_declining_to_answer_produces_an_honest_refusal(engine):
    assistant, _ = build(engine, lambda _: {"action": "no_answer", "reason": "нет темы"})
    reply = ask(assistant, "Как почистить станцию КИТ?")
    assert reply.status == "missing"
    assert reply.reason == "ai_no_answer"
    assert "станции КИТ" in reply.text
    assert reply.text.endswith(SERVICE_REFERRAL)
    assert reply.source is None


def test_model_can_request_a_clarification(engine):
    assistant, _ = build(engine, lambda _: {"action": "clarify", "reason": "нет модели"})
    reply = ask(assistant, "Как самому обслужить станцию?")
    assert reply.status == "clarify"
    assert {option["id"] for option in reply.options} == {"kan_ultra", "kit"}
    assert reply.ai["layer"] == "llm"


def test_unavailable_model_falls_back_to_the_rules(engine, knowledge):
    assistant, _ = build(engine, lambda _: None)
    reply = ask(assistant, "Как часто нужно обслуживать станцию КИТ?")
    assert reply.status == "answer"
    assert reply.text == knowledge.entries["kit_frequency"].answer
    assert reply.ai["layer"] == "rules_fallback"
    assert "модель недоступна" in reply.ai["note"]


def test_generation_stays_broken_after_the_breaker_trips(engine, knowledge):
    requests = []

    def verdict_for(_):
        return None

    assistant, router = build(engine, verdict_for, requests=requests, cooldown=60.0)
    for _ in range(3):
        ask(assistant, "Как часто нужно обслуживать станцию КИТ?")
    assert router.state == "cooling_down"
    # New conversations keep working on rules only.
    assistant2 = AIAssistant(AnswerEngine(knowledge), router=router, config=router.config)
    reply = asyncio.run(assistant2.respond("Как почистить КАН Ультра?", "s"))
    assert reply.status == "answer"
    assert reply.text == knowledge.entries["kan_ultra_maintenance"].answer
    assert reply.ai["layer"] == "rules_fallback"


def test_prompt_contains_the_knowledge_base_and_no_secrets(engine):
    requests = []
    assistant, _ = build(engine, by_keyword, requests=requests)
    ask(assistant, "Как почистить КАН Ультра?")
    payload = requests[0]
    system = payload["messages"][0]["content"]
    user = payload["messages"][1]["content"]
    assert "Тверь" not in system  # nothing outside the knowledge base
    assert "youtu.be/WW1Hqh3_WNk" in system  # the approved fact is given as the only source
    assert "НЕ пишешь инструкций" in system
    assert "Как почистить КАН Ультра?" in user
    assert FAKE_KEY not in json.dumps(payload, ensure_ascii=False)


def test_clarification_context_is_shared_with_the_model(engine):
    requests = []
    assistant, _ = build(
        engine, lambda _: {"action": "clarify", "reason": "модель не названа"}, requests=requests
    )
    first = asyncio.run(assistant.respond("Как самому обслужить станцию?", "s"))
    assert first.status == "clarify"
    assert {option["id"] for option in first.options} == {"kan_ultra", "kit"}
    second = asyncio.run(assistant.respond("КАН Ультра", "s"))
    assert second.entry_id == "kan_ultra_maintenance"
    # A bare model name is understood because the rules share the dialogue context.
    context = requests[1]["messages"][-1]["content"]
    assert "определена модель КАН Ультра" in context
    assert "тема: «Очистка и обслуживание»" in context
    asyncio.run(assistant.aclose())


def test_status_command_reports_the_mode_without_secrets(engine):
    assistant, _ = build(engine, by_keyword)
    reply = ask(assistant, "/status")
    assert reply.status == "info"
    assert MODEL in reply.text
    assert FAKE_KEY not in reply.text
    assert "версия" in reply.text


def test_rules_only_assistant_marks_replies_as_rules(knowledge):
    assistant = AIAssistant(AnswerEngine(knowledge))
    reply = asyncio.run(assistant.respond("Как почистить КАН Ультра?", "s"))
    assert reply.ai["layer"] == "rules"
    assert assistant.status()["mode"] == "rules_only"
    assert assistant.enabled is False
    assert KnowledgeStore.load().entries["kan_ultra_maintenance"].answer == reply.text


def test_the_chain_falls_through_to_the_next_provider(engine, knowledge):
    """The first provider is down, the second answers: the layer is still llm."""

    def failing(request):
        return httpx.Response(503, json={"error": {"message": "unavailable"}})

    def working(request):
        payload = json.loads(request.content)
        verdict = by_keyword(payload["messages"][-1]["content"])
        content = json.dumps(verdict, ensure_ascii=False)
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    primary = LLMConfig(
        provider="github",
        base_url="https://models.github.ai/inference",
        model=MODEL,
        api_key=FAKE_KEY,
        retries=0,
    )
    secondary = LLMConfig(
        provider="llm7",
        base_url="https://api.llm7.io/v1",
        model="GLM-5.3-Flash",
        api_key="",
        retries=0,
        needs_key=False,
    )
    assistant = AIAssistant(
        engine,
        router=LLMRouter(primary, client=httpx.AsyncClient(transport=httpx.MockTransport(failing))),
        config=primary,
        fallback_routers=[
            LLMRouter(secondary, client=httpx.AsyncClient(transport=httpx.MockTransport(working)))
        ],
        fallback_configs=[secondary],
    )
    reply = ask(assistant, "Как почистить КАН Ультра?")
    assert reply.status == "answer"
    assert reply.text == knowledge.entries["kan_ultra_maintenance"].answer
    assert reply.ai["layer"] == "llm"
    assert reply.ai["provider"] == "llm7"
    assert reply.ai["model"] == "GLM-5.3-Flash"


def test_status_describes_the_provider_chain_without_secrets(engine):
    primary = LLMConfig(
        provider="github",
        base_url="https://models.github.ai/inference",
        model=MODEL,
        api_key=FAKE_KEY,
        retries=0,
    )
    secondary = LLMConfig(
        provider="llm7",
        base_url="https://api.llm7.io/v1",
        model="GLM-5.3-Flash",
        api_key="",
        retries=0,
        needs_key=False,
    )
    assistant = AIAssistant(
        engine,
        router=LLMRouter(primary),
        config=primary,
        fallback_routers=[LLMRouter(secondary)],
        fallback_configs=[secondary],
    )
    status = assistant.status()
    assert status["enabled"] is True
    assert status["model_label"] == f"github/{MODEL} → llm7/GLM-5.3-Flash"
    assert [view["provider"] for view in status["routers"]] == ["github", "llm7"]
    assert FAKE_KEY not in json.dumps(status, ensure_ascii=False)
    reply = ask(assistant, "/status")
    assert FAKE_KEY not in reply.text
    assert "llm7/GLM-5.3-Flash" in reply.text
