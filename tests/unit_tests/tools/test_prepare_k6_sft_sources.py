# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from pathlib import Path

from tools.prepare_k6_sft_sources import (
    MOJIBAKE_MARKERS,
    _alpaca_record,
    _entity_spans,
    _finerweb_record,
    _is_aya_qa,
    _normalize_afriinstruct_record,
    _parse_gemma_chat,
    _record,
    _ShardWriter,
    _translation_pair_records,
    _translation_record,
)


def test_afriinstruct_normalization_is_conservative() -> None:
    base = {
        "messages": [
            {"role": "user", "content": "  Translate this  "},
            {"role": "assistant", "content": "  Fassara wannan  "},
        ],
        "lang": "hau-eng",
        "task": "translation",
        "source": "FLORES",
    }

    record = _normalize_afriinstruct_record(base)

    assert record is not None
    assert record["lang"] == "hau"
    assert record["task"] == "translation"
    assert record["direction"] == "hau-eng"
    assert record["messages"][0]["content"] == "Translate this"
    assert _normalize_afriinstruct_record({**base, "lang": "swa", "task": "Multitask"})["lang"] == "swh"
    assert _normalize_afriinstruct_record({**base, "task": "POS"}) is None

    ner = _normalize_afriinstruct_record(
        {
            **base,
            "task": "NER",
            "messages": [
                base["messages"][0],
                {"role": "assistant", "content": '[("Kano", "B-LOC"), ("Musa", "B-PER")]'},
            ],
        }
    )
    assert ner is not None
    assert ner["entity_types"] == ["LOC", "PER"]


def test_aya_qa_classification_is_language_aware() -> None:
    assert _is_aya_qa("hau", "Menene babban birnin Najeriya")
    assert _is_aya_qa("zul", "Ubani umongameli?")
    assert _is_aya_qa("amh", "የኢትዮጵያ ዋና ከተማ ምንድን ነው፧")
    assert _is_aya_qa("wol", "Kan mooy bind téere bi")
    assert not _is_aya_qa("yor", "Kọ àpilẹ̀kọ nípa èdè Yorùbá")


def test_afriinstruct_normalizes_k10_language_aliases() -> None:
    base = {
        "messages": [
            {"role": "user", "content": "Prompt"},
            {"role": "assistant", "content": "Answer"},
        ],
        "task": "Multitask",
        "source": "xP3",
    }

    for language in ("amh", "xho", "sna", "wol"):
        record = _normalize_afriinstruct_record({**base, "lang": language})
        assert record is not None
        assert record["lang"] == language


def test_translation_record_sets_direction_and_rejects_copies() -> None:
    record = _translation_record("Good morning", "Barka da safiya", "hau", "source", {"domain": "general"})

    assert record is not None
    assert record["direction"] == "eng-hau"
    assert record["task"] == "translation"
    assert record["messages"][1]["content"] == "Barka da safiya"
    assert _translation_record("same text", " Same   text ", "hau", "source", {}) is None
    assert _translation_record("valid", None, "hau", "source", {}) is None


def test_translation_pair_records_include_both_directions() -> None:
    records = _translation_pair_records("Good morning", "Maakye", "twi", "mafand", {"domain": "news"})

    assert [record["direction"] for record in records] == ["eng-twi", "twi-eng"]
    assert records[0]["messages"][-1]["content"] == "Maakye"
    assert records[1]["messages"][-1]["content"] == "Good morning"


def test_k14_instruction_and_crane_chat_normalization() -> None:
    record = _alpaca_record(
        {"instruction": " Kyerɛkyerɛ eyi mu ", "input": "nan", "output": " Mmuae "},
        "twi",
        "alpaca_twi",
    )

    assert record is not None
    assert record["messages"][0]["content"] == "Kyerɛkyerɛ eyi mu"
    assert record["messages"][1]["content"] == "Mmuae"
    assert _parse_gemma_chat(
        "<start_of_turn>user\nEkibuuzo<end_of_turn>\n<start_of_turn>model\nEky'okuddamu<end_of_turn>"
    ) == ("Ekibuuzo", "Eky'okuddamu")


def test_finerweb_normalizes_valid_character_spans() -> None:
    record = _finerweb_record(
        {
            "text": "Xasan wuxuu joogaa Muqdisho.",
            "char_spans": json.dumps(
                [
                    {"start": 0, "end": 5, "label": "person"},
                    {"start": 19, "end": 27, "label": "location / city"},
                    {"start": -1, "end": 500, "label": "invalid"},
                ]
            ),
        },
        "som",
    )

    assert record is not None
    entities = json.loads(record["messages"][-1]["content"])
    assert entities == [
        {"type": "person", "text": "Xasan"},
        {"type": "location / city", "text": "Muqdisho"},
    ]


def test_entity_spans_normalize_person_tags_and_close_boundaries() -> None:
    spans = _entity_spans(
        ["Neo", "Putini", "uvakashele", "eKapa", "namuhla"],
        ["B-PERS", "I-PERS", "O", "B-LOC", "O"],
    )

    assert spans == [{"type": "PER", "text": "Neo Putini"}, {"type": "LOC", "text": "eKapa"}]
    assert any(marker in "DitÃ...Â¡weletÃ...Â¡wa" for marker in MOJIBAKE_MARKERS)


def test_writer_deduplicates_and_materializes_bounded_shards(tmp_path: Path) -> None:
    writer = _ShardWriter(tmp_path, "source", "revision", shard_size=1)
    first = _record(user="Question one", assistant="Answer one", language="hau", task="qa", source="source")
    second = _record(user="Question two", assistant="Answer two", language="ibo", task="qa", source="source")
    assert first is not None and second is not None

    writer.write(first)
    writer.write(first)
    writer.write(second)
    summary = writer.close()

    assert summary["records"] == 2
    assert summary["duplicates"] == 1
    assert summary["by_language"] == {"hau": 1, "ibo": 1}
    manifest = json.loads((tmp_path / "train_meta.json").read_text(encoding="utf-8"))
    assert len(manifest) == 2
    assert all((tmp_path / entry["file_name"]).is_file() for entry in manifest.values())
