import copy
import json

import pytest
from pydantic import ValidationError

from service_bot.knowledge import KnowledgeDocument, KnowledgeStore, normalize


def test_real_kb_includes_the_three_user_provided_facts(knowledge):
    assert {"kan_ultra_maintenance", "kit_frequency", "station_overflow"} <= set(knowledge.entries)
    assert {"КАН Ультра", "КИТ"} <= {station.name for station in knowledge.document.stations}
    assert "Тверь" not in knowledge.public_view().__str__()


def test_base_is_read_only(knowledge):
    with pytest.raises(TypeError):
        knowledge.entries["injected"] = knowledge.entries["kit_frequency"]
    with pytest.raises(ValidationError):
        knowledge.entries["kit_frequency"].answer = "Другая информация"


@pytest.mark.parametrize(
    "mutation",
    [
        "version",
        "duplicate_entry",
        "duplicate_station",
        "duplicate_intent",
        "unknown_station",
        "unknown_intent",
        "duplicate_scope",
        "bad_regex",
        "bad_url",
        "empty_answer",
        "extra_field",
        "ambiguous_alias",
        "generic_specific_conflict",
        "answer_too_long",
    ],
)
def test_invalid_kb_fails_closed(knowledge, mutation):
    data = json.loads(knowledge.document.model_dump_json())
    if mutation == "version":
        data["schema_version"] = 2
    elif mutation == "duplicate_entry":
        data["entries"].append(copy.deepcopy(data["entries"][0]))
    elif mutation == "duplicate_station":
        data["stations"].append(copy.deepcopy(data["stations"][0]))
    elif mutation == "duplicate_intent":
        data["intents"].append(copy.deepcopy(data["intents"][0]))
    elif mutation == "unknown_station":
        data["entries"][0]["station_id"] = "not_registered"
    elif mutation == "unknown_intent":
        data["entries"][0]["intent_id"] = "not_registered"
    elif mutation == "duplicate_scope":
        new = copy.deepcopy(data["entries"][0])
        new["id"] = "different_id"
        data["entries"].append(new)
    elif mutation == "bad_regex":
        data["intents"][0]["patterns"].append("[")
    elif mutation == "bad_url":
        data["entries"][0]["source"]["url"] = "javascript:alert(1)"
    elif mutation == "empty_answer":
        data["entries"][0]["answer"] = " "
    elif mutation == "extra_field":
        data["entries"][0]["generated_fallback"] = "Enabled"
    elif mutation == "ambiguous_alias":
        data["stations"][1]["aliases"].append("КАН")
    elif mutation == "generic_specific_conflict":
        data["entries"][0]["intent_id"] = "overflow"
    elif mutation == "answer_too_long":
        data["entries"][0]["answer"] = "a" * 3501
    with pytest.raises(ValidationError):
        KnowledgeDocument.model_validate(data)


def test_missing_file_cannot_be_replaced_by_general_knowledge(tmp_path):
    with pytest.raises(FileNotFoundError):
        KnowledgeStore.load(tmp_path / "missing.json")


def test_invalid_json(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text("{incomplete", encoding="utf-8")
    with pytest.raises(ValidationError):
        KnowledgeStore.load(path)


def test_normalisation():
    assert normalize("  КАН—Ультра\n ЁЖ ") == "кан ультра еж"
    assert normalize("ＫＡＮ") == "kan"
