# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
from types import SimpleNamespace

from tools.afriinstruct_benchmark import (
    PAPER_PREFIX,
    best_qa_f1,
    build_prompt,
    configure_sdpa,
    extract_topic,
    extract_topic_paper_fuzzy,
    load_benchmarks,
    load_predictions,
    score_records,
    token_f1,
)


def _record(instruction="Question", output="Answer", task="QA"):
    return {
        "instruction": instruction,
        "output": output,
        "lang": "swa",
        "source": "test",
        "task": task,
    }


def test_load_benchmarks_deduplicates_records(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps([_record(), _record("Other")]), encoding="utf-8")
    second.write_text(json.dumps([_record()]), encoding="utf-8")

    records = load_benchmarks([first, second])

    assert len(records) == 2
    assert all(len(record["record_id"]) == 64 for record in records)


def test_build_prompt_preserves_paper_prefix():
    prompt = build_prompt("Translate this")
    assert prompt.startswith(PAPER_PREFIX)
    assert prompt.endswith("Translate this")


def test_token_f1_uses_normalized_token_counts():
    assert token_f1("The quick, quick fox", "quick fox") == 80.0
    assert token_f1("wrong", "answer") == 0.0
    assert best_qa_f1("Moroni", "['Wrong', 'Moroni']") == 100.0
    assert best_qa_f1("No reference exists", "[]") == 0.0


def test_extract_topic_handles_generated_sentence():
    assert extract_topic("The best label is science/technology.") == "science/technology"
    assert extract_topic("unknown") is None


def test_extract_topic_paper_fuzzy_handles_misspelling_and_threshold():
    assert extract_topic_paper_fuzzy("poltics") == "politics"
    assert extract_topic_paper_fuzzy("unknown") is None


def test_configure_sdpa_updates_top_and_text_configs():
    model = SimpleNamespace(
        config=SimpleNamespace(
            _attn_implementation="flash_attention_2",
            text_config=SimpleNamespace(_attn_implementation="flash_attention_2"),
        )
    )

    configure_sdpa(model)

    assert model.config._attn_implementation == "sdpa"
    assert model.config.text_config._attn_implementation == "sdpa"


def test_score_records_reports_qa_and_topic_metrics(tmp_path):
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(
        json.dumps([_record(output="['Moroni']"), _record("Topic", "travel", "topic-classification")]),
        encoding="utf-8",
    )
    records = load_benchmarks([benchmark])
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(
        "\n".join(
            json.dumps({"record_id": record["record_id"], "prediction": prediction})
            for record, prediction in zip(records, ("Moroni", "travel"))
        )
        + "\n",
        encoding="utf-8",
    )

    result = score_records(records, load_predictions(predictions_path))

    assert result["scored"] == 2
    assert result["qa_token_f1"] == 100.0
    assert result["topic_accuracy"] == 100.0
    assert result["topic_macro_f1"] == 100.0 / 7.0


def test_score_records_reproduces_paper_fuzzy_topic_omission(tmp_path):
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(
        json.dumps(
            [
                _record("Topic one", "politics", "topic-classification"),
                _record("Topic two", "sports", "topic-classification"),
            ]
        ),
        encoding="utf-8",
    )
    records = load_benchmarks([benchmark])
    predictions = {
        records[0]["record_id"]: {"prediction": "poltics"},
        records[1]["record_id"]: {"prediction": "unknown"},
    }

    result = score_records(records, predictions, topic_matching="paper-fuzzy")

    assert result["topic_matched"] == 1
    assert result["topic_unmatched"] == 1
    assert result["topic_match_rate"] == 50.0
    assert result["topic_accuracy"] == 100.0
    assert result["topic_macro_f1"] == 100.0


def test_score_records_reports_translation_chrf(tmp_path):
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(json.dumps([_record(output="Habari", task="translation")]), encoding="utf-8")
    records = load_benchmarks([benchmark])
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(
        json.dumps({"record_id": records[0]["record_id"], "prediction": "Habari"}) + "\n",
        encoding="utf-8",
    )

    result = score_records(records, load_predictions(predictions_path))

    assert result["translation_count"] == 1
    assert result["translation_chrf"] == 100.0
