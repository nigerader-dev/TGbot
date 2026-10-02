"""Reply policy shared by Telegram and the demo.

Two layers produce text for a client:

1. this module — deterministic rules with hard guards (unknown models, out-of-scope
   topics, several models or topics, negations), and
2. :mod:`service_bot.ai` — an optional LLM that may only *select* an existing entry.

Neither layer can invent a fact: the only constructor of a factual reply is
:meth:`AnswerEngine.answer_reply`, which copies ``entry.answer`` verbatim.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Literal

from .knowledge import Entry, KnowledgeStore
from .matcher import Analysis, Matcher

MAX_QUESTION_LENGTH = 1500
SERVICE_REFERRAL = (
    "Рекомендую обратиться в сервисный отдел компании для получения точной консультации."
)
WELCOME = (
    "Здравствуйте! Я помощник сервисного отдела. Отвечаю только по предоставленной базе знаний.\n\n"
    "Сейчас в ней: обслуживание КАН Ультра, периодичность обслуживания КИТ и действия при "
    "переполнении станции. Если информации нет, направлю в сервисный отдел.\n\n"
    "Просто напишите вопрос своими словами — например «как почистить КАН Ультра» или "
    "«как часто обслуживать КИТ»."
)
HELP = (
    WELCOME + "\n\n/start — примеры вопросов\n/reset — сброс модели станции\n"
    "/status — режим работы и версия базы\n/help — помощь"
)


@dataclass(frozen=True)
class Reply:
    status: Literal["answer", "missing", "clarify", "info"]
    text: str
    reason: str
    entry_id: str | None = None
    source: dict | None = None
    station_id: str | None = None
    options: tuple[dict, ...] = ()
    ai: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Conversation:
    station_id: str | None = None
    pending_intent: str | None = None
    updated_at: float = 0


@dataclass(frozen=True)
class Decision:
    """Deterministic outcome plus what the AI layer is allowed to do with it."""

    reply: Reply
    stage: Literal["answer", "missing", "clarify", "info", "guard", "command", "invalid"]
    blocked: bool
    analysis: Analysis | None = None
    context: str | None = None
    station_name: str | None = None
    intent_id: str | None = None


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
        return self.info_reply("Контекст сброшен. Напишите новый вопрос и модель станции.", "reset")

    # -- reply constructors -------------------------------------------------

    def answer_reply(self, entry: Entry, *, ai: dict | None = None) -> Reply:
        # The only constructor for factual replies. The text is copied verbatim.
        return Reply(
            "answer",
            entry.answer,
            "verified_entry",
            entry_id=entry.id,
            source=entry.source.model_dump(),
            station_id=entry.station_id,
            ai=ai,
        )

    def missing_reply(
        self, *, reason: str, station_name: str | None = None, ai: dict | None = None
    ) -> Reply:
        if station_name:
            text = f"В моей базе знаний нет информации по обслуживанию станции {station_name}. "
        else:
            text = "В моей базе знаний нет информации по этому вопросу. "
        return Reply("missing", text + SERVICE_REFERRAL, reason, ai=ai)

    def info_reply(self, text: str, reason: str, **fields) -> Reply:
        return Reply("info", text, reason, **fields)

    # -- policy -------------------------------------------------------------

    def _context_note(self, state: Conversation, intent_id: str | None = None) -> str | None:
        """Dialogue context handed to the AI layer; the rules always stay the source."""
        notes: list[str] = []
        if state.station_id and state.station_id in self.knowledge.stations:
            name = self.knowledge.stations[state.station_id].name
            notes.append(f"в диалоге уже определена модель {name}")
        intent = self.knowledge.intents.get(intent_id or state.pending_intent or "")
        if intent is not None:
            notes.append(f"тема: «{intent.name}»")
        if state.pending_intent:
            notes.append("клиент должен назвать модель станции")
        return "; ".join(notes) if notes else None

    def decision(self, text: str, session_id: str) -> Decision:
        text = text.strip()
        if not text or len(text) > MAX_QUESTION_LENGTH:
            return Decision(
                self.info_reply(
                    f"Напишите вопрос длиной от 1 до {MAX_QUESTION_LENGTH} символов.",
                    "invalid_text",
                ),
                "invalid",
                True,
            )
        if text.startswith("/"):
            command = text.split()[0].split("@")[0].casefold()
            if command == "/reset":
                return Decision(self.reset(session_id), "command", True)
            if command == "/start":
                self.reset(session_id)
                return Decision(self.info_reply(WELCOME, "welcome"), "command", True)
            return Decision(self.info_reply(HELP, "help"), "command", True)

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
                return Decision(self.answer_reply(entry), "answer", False, analysis)

        if analysis.unknown_station or analysis.unsupported_topic or analysis.unknown_terms:
            # Never carry a previously recognised model into a new unknown request.
            state.station_id = None
            state.pending_intent = None
            # An unknown model name or an out-of-scope topic is a hard stop: the AI
            # layer is not allowed to look for a "reasonable" answer there.
            blocked = bool(analysis.unknown_station or analysis.unsupported_topic)
            if analysis.unsupported_topic:
                reason = "unsupported_topic"
            elif analysis.unknown_station:
                reason = "unrecognised_model"
            else:
                reason = "unrecognised_terms"
            return Decision(
                self.missing_reply(reason=reason, station_name=analysis.unknown_station),
                "guard" if blocked else "missing",
                blocked,
                analysis,
            )

        if len(analysis.stations) > 1:
            state.station_id = None
            state.pending_intent = (
                next(iter(analysis.intents)) if len(analysis.intents) == 1 else None
            )
            return Decision(self._clarify("multiple_stations"), "guard", True, analysis)

        if analysis.stations:
            state.station_id = next(iter(analysis.stations))
        if len(analysis.intents) > 1:
            state.pending_intent = None
            return Decision(self.missing_reply(reason="multiple_topics"), "guard", True, analysis)
        if analysis.has_negated_overflow and not analysis.intents:
            state.pending_intent = None
            return Decision(self.missing_reply(reason="negated_overflow"), "guard", True, analysis)

        intent = next(iter(analysis.intents), None)
        if intent is None and analysis.stations and state.pending_intent:
            intent = state.pending_intent
        if intent is None:
            state.pending_intent = None
            if analysis.stations:
                return Decision(
                    self.info_reply(
                        f"Модель: {self.knowledge.stations[state.station_id].name}. "
                        "Напишите ваш вопрос.",
                        "model_selected",
                        station_id=state.station_id,
                    ),
                    "info",
                    True,
                    analysis,
                )
            return Decision(
                self.missing_reply(reason="no_intent"),
                "missing",
                False,
                analysis,
                context=self._context_note(state),
            )

        # All other generic entries are supported when added by a KB administrator.
        generic = self.knowledge.find(None, intent)
        if generic:
            state.pending_intent = None
            return Decision(self.answer_reply(generic), "answer", False, analysis, intent_id=intent)
        if state.station_id is None:
            state.pending_intent = intent
            return Decision(
                self._clarify("station_required"),
                "clarify",
                False,
                analysis,
                context=self._context_note(state),
                intent_id=intent,
            )

        state.pending_intent = None
        entry = self.knowledge.find(state.station_id, intent)
        if entry is None:
            station = self.knowledge.stations[state.station_id]
            return Decision(
                self.missing_reply(reason="no_entry_for_model_and_intent"),
                "missing",
                False,
                analysis,
                context=self._context_note(state),
                station_name=station.name,
                intent_id=intent,
            )
        return Decision(
            self.answer_reply(entry),
            "answer",
            False,
            analysis,
            context=self._context_note(state, intent),
            station_name=self.knowledge.stations[state.station_id].name,
            intent_id=intent,
        )

    def respond(self, text: str, session_id: str) -> Reply:
        """Deterministic policy only: used when the AI layer is disabled or failed."""
        return self.decision(text, session_id).reply

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

    # -- helpers used by the AI layer --------------------------------------

    def session_station_name(self, session_id: str) -> str | None:
        state = self._sessions.get(session_id)
        if state and state.station_id:
            station = self.knowledge.stations.get(state.station_id)
            if station is not None:
                return station.name
        return None

    def intent_title(self, intent_id: str | None) -> str | None:
        intent = self.knowledge.intents.get(intent_id or "")
        return intent.name if intent else None


def ai_metadata(
    *,
    layer: str,
    config=None,
    verdict=None,
    deterministic_entry_id: str | None = None,
    note: str | None = None,
) -> dict:
    """Small, log-safe description of how a reply was produced."""
    data: dict = {"layer": layer}
    if config is not None:
        data["provider"] = config.provider
        data["model"] = config.model
    if verdict is not None:
        data["action"] = verdict.action
        data["confidence"] = verdict.confidence
        data["latency_ms"] = verdict.latency_ms
        if verdict.reason:
            data["reason"] = verdict.reason
    if deterministic_entry_id:
        data["deterministic_entry_id"] = deterministic_entry_id
    if note:
        data["note"] = note
    return data
