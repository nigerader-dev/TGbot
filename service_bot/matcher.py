"""Bounded Russian intent matching; ambiguous/unrecognised text is never guessed."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .knowledge import KnowledgeStore, normalize

WORD = re.compile(r"[^\W_]+")
PROCEDURAL = re.compile(
    r"\bкак\s+(?:(?:мне|нам|самому|самой|самостоятельно|правильно|можно|нужно)\s+){0,3}"
    r"(?:почист|очист|чист|обслуж|промы|помы|вымы)\w*"
)
NEGATION = re.compile(r"\b(?:не|нет)\s+(?:\w+\s+){0,1}$")
MODEL_MARKER = re.compile(
    r"\b(?:станци(?:я|и|ю|ей|ям|ями|ях)|станций|септик(?:а|у|е|ом|и|ов|ам|ами|ах)?|"
    r"модел(?:ь|и|ью|ей|ям|ями|ях)|установ(?:ка|ки|ку|ке|кой|ок|кам|ками|ках))\b"
)
SERVICE_PREFIX = re.compile(
    r"\b(?:почистить|очистить|прочистить|промыть|обслужить|обслуживать|обслуживание|"
    r"обслуживанию|чистить|помыть|вымыть|инструкци(?:я|ю|и|ей)|"
    r"видеоинструкци(?:я|ю|и|ей))\b"
)
COORDINATOR = re.compile(r"\b(?:и|или)\b")
# Do not treat an unknown proper name such as 'Обслуживатор' as an action verb.
MODEL_ACTION_WORD = re.compile(
    r"(?:обслуж(?:ить|ивать|ивают|ивается|ивание|иванию|ивания|ивании)|"
    r"(?:почист|очист|прочист|пром|пом|вым)(?:ить|ыть|ывать)|"
    r"чист(?:ить|ят|ится|ка|ки|ке|ку)|"
    r"(?:переполн|затоп|залит)(?:ена|ен|ено|ены|илась|ился))\Z"
)


@dataclass(frozen=True)
class Analysis:
    stations: frozenset[str]
    intents: frozenset[str]
    unknown_terms: tuple[str, ...]
    unknown_station: str | None
    unsupported_topic: bool
    has_negated_overflow: bool


class Matcher:
    def __init__(self, knowledge: KnowledgeStore):
        self.knowledge = knowledge
        aliases = [
            (normalize(alias), station.id)
            for station in knowledge.document.stations
            for alias in station.aliases
        ]
        self.aliases = [
            (re.compile(r"(?<!\w)" + re.escape(alias).replace(r"\ ", r"\s+") + r"(?!\w)"), sid)
            for alias, sid in sorted(aliases, key=lambda item: len(item[0]), reverse=True)
        ]
        self.intent_patterns = {
            intent.id: [re.compile(pattern) for pattern in intent.patterns]
            for intent in knowledge.document.intents
        }
        self.context_patterns = [
            re.compile(r"(?:" + pattern + r")\Z")
            for pattern in knowledge.document.context_word_patterns
        ]
        self.unsupported_patterns = [
            re.compile(pattern) for pattern in knowledge.document.unsupported_topic_patterns
        ]

    def analyze(self, raw_text: str) -> Analysis:
        text = normalize(raw_text)
        covered = [False] * len(text)
        station_spans: list[tuple[int, int, str]] = []
        for pattern, sid in self.aliases:
            for match in pattern.finditer(text):
                if not any(covered[match.start() : match.end()]):
                    station_spans.append((match.start(), match.end(), sid))
                    covered[match.start() : match.end()] = [True] * (match.end() - match.start())

        intents: set[str] = set()
        negated_overflow = False
        for intent_id, patterns in self.intent_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    covered[match.start() : match.end()] = [True] * (match.end() - match.start())
                    if intent_id == "overflow" and NEGATION.search(text[: match.start()]):
                        negated_overflow = True
                        continue
                    intents.add(intent_id)

        # "Как часто чистить" is frequency, not two independent requests.
        if "frequency" in intents and "maintenance" in intents and not PROCEDURAL.search(text):
            intents.remove("maintenance")

        unknown_tokens = []
        for word in WORD.finditer(text):
            if any(covered[word.start() : word.end()]):
                continue
            if any(pattern.fullmatch(word.group()) for pattern in self.context_patterns):
                continue
            unknown_tokens.append(word)

        unknown_station = None
        # Model recognition is independent of intent recognition. Otherwise an unknown
        # brand sharing an intent stem could be swallowed and inherit an older model.
        candidates = [(match, True) for match in MODEL_MARKER.finditer(text)]
        candidates += [(match, False) for match in SERVICE_PREFIX.finditer(text)]
        if station_spans:
            candidates += [(match, False) for match in COORDINATOR.finditer(text)]
        for marker, is_model_marker in sorted(candidates, key=lambda item: item[0].start()):
            cursor = marker.end()
            for word in WORD.finditer(text, cursor):
                if re.search(r"[.,!?;]", text[cursor : word.start()]):
                    break
                if word.start() - marker.end() > 60:
                    break
                if any(start <= word.start() < end for start, end, _ in station_spans):
                    break
                cursor = word.end()
                context_word = any(
                    pattern.fullmatch(word.group()) for pattern in self.context_patterns
                )
                if (context_word and not word.group().isdigit()) or MODEL_ACTION_WORD.fullmatch(
                    word.group()
                ):
                    continue
                if is_model_marker:
                    unknown_station = word.group().capitalize()
                if not any(item.start() == word.start() for item in unknown_tokens):
                    unknown_tokens.append(word)
                break
        if unknown_tokens:
            # Only echo a bounded model-like name, never an arbitrary user instruction.
            markers = list(MODEL_MARKER.finditer(text))
            for word in unknown_tokens:
                if any(0 <= word.start() - marker.end() < 30 for marker in markers):
                    unknown_station = word.group().capitalize()
                    break

        # A recognised prefix is not the same model: КАН Лайт / КИТ-5 / КАН Ультра 2.
        # Even allowed numeric/context words immediately after an alias must not be ignored.
        for _, end, sid in station_spans:
            suffix = re.match(
                r"\s+(\d+|плюс|лайт|мини|макс|про|plus|light|mini|max|pro)\b", text[end:]
            )
            if suffix:
                unknown_station = self.knowledge.stations[sid].name + " " + suffix.group(1)
                unknown_tokens.append(suffix)

        return Analysis(
            stations=frozenset(span[2] for span in station_spans),
            intents=frozenset(intents),
            unknown_terms=tuple(word.group() for word in unknown_tokens),
            unknown_station=unknown_station,
            unsupported_topic=any(pattern.search(text) for pattern in self.unsupported_patterns),
            has_negated_overflow=negated_overflow,
        )
