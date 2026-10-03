"""Validated, read-only knowledge. User messages cannot change this store."""

from __future__ import annotations

import re
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_KNOWLEDGE_PATH = ROOT / "knowledge" / "knowledge_base.json"
Identifier = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]{0,39}$")]
ShortText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=180)]
Pattern = Annotated[str, StringConstraints(min_length=1, max_length=400)]


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Station(FrozenModel):
    id: Identifier
    name: ShortText
    aliases: tuple[ShortText, ...] = Field(min_length=1, max_length=30)


class Intent(FrozenModel):
    id: Identifier
    name: ShortText
    patterns: tuple[Pattern, ...] = Field(min_length=1, max_length=30)


class Source(FrozenModel):
    label: ShortText
    url: str | None = None

    @model_validator(mode="after")
    def validate_url(self) -> Source:
        if self.url and not re.fullmatch(r"https://[^\s<>]+", self.url):
            raise ValueError("Source URL must be an HTTPS URL")
        return self


class Entry(FrozenModel):
    id: Identifier
    station_id: Identifier | None
    intent_id: Identifier
    title: ShortText
    answer: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=3500)]
    source: Source
    examples: tuple[ShortText, ...] = Field(min_length=1, max_length=20)


class KnowledgeDocument(FrozenModel):
    schema_version: Literal[1]
    revision: ShortText
    stations: tuple[Station, ...] = Field(min_length=1)
    intents: tuple[Intent, ...] = Field(min_length=1)
    context_word_patterns: tuple[Pattern, ...] = Field(min_length=1)
    unsupported_topic_patterns: tuple[Pattern, ...] = Field(min_length=1)
    entries: tuple[Entry, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_integrity(self) -> KnowledgeDocument:
        for collection in (self.stations, self.intents, self.entries):
            ids = [item.id for item in collection]
            if len(ids) != len(set(ids)):
                raise ValueError("Duplicate IDs in knowledge base")
        station_ids = {station.id for station in self.stations}
        intent_ids = {intent.id for intent in self.intents}
        scopes: set[tuple[str | None, str]] = set()
        for entry in self.entries:
            if entry.station_id is not None and entry.station_id not in station_ids:
                raise ValueError(f"Unknown station reference: {entry.station_id}")
            if entry.intent_id not in intent_ids:
                raise ValueError(f"Unknown intent reference: {entry.intent_id}")
            scope = (entry.station_id, entry.intent_id)
            if scope in scopes:
                raise ValueError("Ambiguous station/intent pair")
            scopes.add(scope)
        patterns = list(self.context_word_patterns) + list(self.unsupported_topic_patterns)
        patterns.extend(pattern for intent in self.intents for pattern in intent.patterns)
        for pattern in patterns:
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"Invalid matching pattern: {pattern}") from exc
        # Generic facts must not silently override model-specific facts.
        for station_id, intent_id in scopes:
            if station_id is not None and (None, intent_id) in scopes:
                raise ValueError("An intent cannot have both generic and station-specific entries")
        aliases: dict[str, str] = {}
        for station in self.stations:
            for alias in station.aliases:
                key = normalize(alias)
                if key in aliases and aliases[key] != station.id:
                    raise ValueError("A station alias cannot point to different models")
                aliases[key] = station.id
        return self


def normalize(text: str) -> str:
    import unicodedata

    text = unicodedata.normalize("NFKC", text).casefold().replace("ё", "е")
    text = re.sub(r"[-‐‑–—_\u200b]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class KnowledgeStore:
    def __init__(self, document: KnowledgeDocument):
        self.document = document
        self.stations = MappingProxyType({station.id: station for station in document.stations})
        self.intents = MappingProxyType({intent.id: intent for intent in document.intents})
        self.entries = MappingProxyType({entry.id: entry for entry in document.entries})
        self.by_scope = MappingProxyType(
            {(entry.station_id, entry.intent_id): entry for entry in document.entries}
        )

    @classmethod
    def load(cls, path: Path = DEFAULT_KNOWLEDGE_PATH) -> KnowledgeStore:
        return cls(KnowledgeDocument.model_validate_json(path.read_text(encoding="utf-8")))

    def find(self, station_id: str | None, intent_id: str) -> Entry | None:
        return self.by_scope.get((station_id, intent_id))

    def public_view(self) -> dict:
        return {
            "revision": self.document.revision,
            "stations": [station.model_dump() for station in self.document.stations],
            "intents": [intent.model_dump() for intent in self.document.intents],
            "entries": [entry.model_dump() for entry in self.document.entries],
        }
