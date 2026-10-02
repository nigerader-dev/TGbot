"""Shared policy for Telegram and the demo. No LLM, free-form completion or web search."""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Literal

from .knowledge import Entry, KnowledgeStore
from .matcher import Matcher

MAX_QUESTION_LENGTH = 1500
SERVICE_REFERRAL = (
    "Рекомендую обратиться в сервисный отдел компании для получения точной консультации."
)
WELCOME = (
    "Здравствуйте! Я помощник сервисного отдела. Отвечаю только по предоставленной базе знаний.\n\n"
    "Сейчас в ней: обслуживание КАН Ультра, периодичность обслуживания КИТ и действия при "
    "переполнении станции. Если информации нет, направлю в сервисный отдел.\n\n"
    "Напишите вопрос и модель станции или выберите пример."
)
HELP = WELCOME + "\n\n/start — примеры вопросов\n/reset — сброс модели станции\n/help — помощь"


@dataclass(frozen=True)
class Reply:
    status: Literal["answer", "missing", "clarify", "info"]
    text: str
    reason: str
    entry_id: str | None = None
    source: dict | None = None
    station_id: str | None = None
    options: tuple[dict, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Conversation:
    station_id: str | None = None
    pending_intent: str | None = None
    updated_at: float = 0


class AnswerEngine:
    def __init__(
        self,
        knowledge: KnowledgeStore,
        *,
        session_ttl: float = 900,
        max_sessions: int = 2000,
        clock=time.monotonic,
    ):
        self.knowledge = knowledge
        self.matcher = Matcher(knowledge)
        self.session_ttl = session_ttl
        self.max_sessions = max_sessions
        self.clock = clock
        self._sessions: OrderedDict[str, Conversation] = OrderedDict()

    def _conversation(self, session_id: str) -> Conversation:
        now = self.clock()
        expired = [
            key
            for key, value in self._sessions.items()
            if now - value.updated_at >= self.session_ttl
        ]
        for key in expired:
            del self._sessions[key]
        state = self._sessions.pop(session_id, Conversation())
        state.updated_at = now
        self._sessions[session_id] = state
        while len(self._sessions) > self.max_sessions:
            self._sessions.popitem(last=False)
        return state

    def reset(self, session_id: str) -> Reply:
        self._sessions.pop(session_id, None)
        return Reply("info", "Контекст сброшен. Напишите новый вопрос и модель станции.", "reset")

    def _answer(self, entry: Entry) -> Reply:
        # The only constructor for factual replies. The text is copied verbatim.
        return Reply(
            "answer",
            entry.answer,
            "verified_entry",
            entry_id=entry.id,
            source=entry.source.model_dump(),
            station_id=entry.station_id,
        )

    def _missing(self, *, reason: str, station_name: str | None = None) -> Reply:
        if station_name:
            text = f"В моей базе знаний нет информации по обслуживанию станции {station_name}. "
        else:
            text = "В моей базе знаний нет информации по этому вопросу. "
        return Reply("missing", text + SERVICE_REFERRAL, reason)

    def respond(self, text: str, session_id: str) -> Reply:
        text = text.strip()
        if not text or len(text) > MAX_QUESTION_LENGTH:
            return Reply(
                "info",
                f"Напишите вопрос длиной от 1 до {MAX_QUESTION_LENGTH} символов.",
                "invalid_text",
            )
        if text.startswith("/"):
            command = text.split()[0].split("@")[0].casefold()
            if command == "/reset":
                return self.reset(session_id)
            if command == "/start":
                self.reset(session_id)
                return Reply("info", WELCOME, "welcome")
            return Reply("info", HELP, "help")
        state = self._conversation(session_id)
        analysis = self.matcher.analyze(text)

        # A generic emergency entry applies independently of the station model.
        # Unsupported requests (diagnostics/prevention/etc.) still fail closed.
        if (
            "overflow" in analysis.intents
            and not analysis.unsupported_topic
            and len(analysis.intents) == 1
        ):
            entry = self.knowledge.find(None, "overflow")
            if entry:
                if analysis.unknown_terms or len(analysis.stations) > 1:
                    state.station_id = None
                elif analysis.stations:
                    state.station_id = next(iter(analysis.stations))
                state.pending_intent = None
                return self._answer(entry)

        if analysis.unknown_terms or analysis.unsupported_topic:
            # Never carry a previously recognised model into a new unknown-model request.
            state.station_id = None
            state.pending_intent = None
            return self._missing(
                reason="unsupported_topic" if analysis.unsupported_topic else "unrecognised_terms",
                station_name=analysis.unknown_station,
            )

        if len(analysis.stations) > 1:
            state.station_id = None
            state.pending_intent = (
                next(iter(analysis.intents)) if len(analysis.intents) == 1 else None
            )
            return self._clarify("multiple_stations")

        if analysis.stations:
            state.station_id = next(iter(analysis.stations))
        if len(analysis.intents) > 1:
            state.pending_intent = None
            return self._missing(reason="multiple_topics")
        if analysis.has_negated_overflow and not analysis.intents:
            state.pending_intent = None
            return self._missing(reason="negated_overflow")

        intent = next(iter(analysis.intents), None)
        if intent is None and analysis.stations and state.pending_intent:
            intent = state.pending_intent
        if intent is None:
            state.pending_intent = None
            if analysis.stations:
                return Reply(
                    "info",
                    f"Модель: {self.knowledge.stations[state.station_id].name}. "
                    "Напишите ваш вопрос.",
                    "model_selected",
                    station_id=state.station_id,
                )
            return self._missing(reason="no_intent")

        # All other generic entries are supported when added by a KB administrator.
        generic = self.knowledge.find(None, intent)
        if generic:
            state.pending_intent = None
            return self._answer(generic)
        if state.station_id is None:
            state.pending_intent = intent
            return self._clarify("station_required")

        state.pending_intent = None
        entry = self.knowledge.find(state.station_id, intent)
        if entry is None:
            return self._missing(reason="no_entry_for_model_and_intent")
        return self._answer(entry)

    def _clarify(self, reason: str) -> Reply:
        return Reply(
            "clarify",
            "Уточните, пожалуйста, модель станции. В базе есть сведения о моделях: "
            + ", ".join(station.name for station in self.knowledge.document.stations)
            + ". Ответы для разных станций нельзя подменять друг другом. "
            "Выберите модель или напишите её название.",
            reason,
            options=tuple(
                {"id": station.id, "label": station.name}
                for station in self.knowledge.document.stations
            ),
        )
