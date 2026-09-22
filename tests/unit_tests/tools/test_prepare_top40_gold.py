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
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tools.prepare_top40_gold import LEGACY_SYSTEM_PROMPT, PreparationConfig, prepare_dataset


class _FakeTokenizer:
    model_max_length = 4_096

    def apply_chat_template(self, conversation, *, tokenize, add_generation_prompt):
        assert tokenize is True
        assert add_generation_prompt is False
        text = "".join(message["content"] for message in conversation)
        input_ids = list(range(5_000 if "overlength" in text else len(text.split()) + 4))
        return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}


def _conversation(user: str, assistant: str) -> str:
    return json.dumps(
        [
            {"role": "system", "content": LEGACY_SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    )


def _write_dataset(root: Path, dataset_id: str, splits: dict[str, list[dict[str, object]]]) -> None:
    shards = []
    split_counts = {}
    for split, records in splits.items():
        split_dir = root / dataset_id / split
        split_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{dataset_id}-{split}-00000.parquet"
        pq.write_table(pa.Table.from_pylist(records), split_dir / filename)
        shards.append({"split": split, "filename": filename, "rows": len(records)})
        split_counts[split] = len(records)
    manifest = {"id": dataset_id, "splits": split_counts, "shards": shards}
    (root / dataset_id / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _read_manifest_records(output_dir: Path, split: str) -> list[dict[str, object]]:
    meta = json.loads((output_dir / f"{split}_meta.json").read_text(encoding="utf-8"))
    records = []
    for entry in meta.values():
        path = output_dir / entry["file_name"]
        records.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
    return records


def test_prepare_dataset_preserves_validation_deduplicates_and_normalizes(tmp_path: Path, monkeypatch) -> None:
    source_root = tmp_path / "source"
    duplicate = {"messages": _conversation("shared", "answer"), "language": "amh", "task": "classification"}
    _write_dataset(
        source_root,
        "afrisenti",
        {
            "train": [
                duplicate,
                {"messages": _conversation("train", "\ufeffneutral"), "language": "amh", "task": "classification"},
            ],
            "validation": [duplicate],
            "test": [{"messages": _conversation("test-only", "negative"), "language": "amh"}],
        },
    )
    monkeypatch.setattr(os, "statvfs", lambda _: type("Stat", (), {"f_flag": os.ST_RDONLY})())
    output_dir = tmp_path / "prepared"

    summary = prepare_dataset(
        PreparationConfig(
            source_root=source_root,
            output_dir=output_dir,
            validation_fraction=0.5,
            shard_size=1,
            dataset_ids=("afrisenti",),
        ),
        tokenizer=_FakeTokenizer(),
    )

    train = _read_manifest_records(output_dir, "train")
    validation = _read_manifest_records(output_dir, "validation")
    assert len(train) == 1
    assert len(validation) == 1
    assert train[0]["messages"] == [
        {"role": "user", "content": "train"},
        {"role": "assistant", "content": "neutral"},
    ]
    assert train[0]["lang"] == "amh"
    assert train[0]["task"] == "classification"
    assert train[0]["source"] == "afrisenti"
    assert train[0]["source_split"] == "train"
    assert 0 < train[0]["_text_tokens"] <= 4_096
    assert validation[0]["messages"][0]["content"] == "shared"
    assert all(record["messages"][0]["content"] != "test-only" for record in train + validation)
    assert summary["stats"]["duplicate_by_split"] == {"train": 1}
    assert summary["stats"]["bom_characters_removed"] == 1
    assert summary["test_splits_materialized"] is False


def test_prepare_dataset_rejects_unapproved_languages_and_unexpected_system_prompts(
    tmp_path: Path, monkeypatch
) -> None:
    source_root = tmp_path / "source"
    unexpected_system = json.dumps(
        [
            {"role": "system", "content": "A task-specific instruction that must not be discarded."},
            {"role": "user", "content": "Prompt"},
            {"role": "assistant", "content": "Answer"},
        ]
    )
    _write_dataset(
        source_root,
        "afrisenti",
        {
            "train": [
                {"messages": _conversation("approved", "positive"), "language": "swa", "task": "classification"},
                {"messages": _conversation("unknown", "neutral"), "language": "unknown", "task": "classification"},
                {"messages": unexpected_system, "language": "amh", "task": "classification"},
                {"messages": _conversation("overlength", "neutral"), "language": "amh", "task": "classification"},
            ],
            "validation": [
                {"messages": _conversation("validation", "positive"), "language": "amh", "task": "classification"}
            ],
        },
    )
    monkeypatch.setattr(os, "statvfs", lambda _: type("Stat", (), {"f_flag": os.ST_RDONLY})())

    summary = prepare_dataset(
        PreparationConfig(source_root=source_root, output_dir=tmp_path / "prepared", dataset_ids=("afrisenti",)),
        tokenizer=_FakeTokenizer(),
    )

    train = _read_manifest_records(tmp_path / "prepared", "train")
    assert len(train) == 1
    assert train[0]["lang"] == "swh"
    assert summary["stats"]["rejected_by_reason"] == {
        "language_outside_report": 1,
        "overlength_sequence": 1,
        "unexpected_system_prompt": 1,
    }


def test_prepare_dataset_excludes_substantive_reserved_benchmark_overlap(tmp_path: Path, monkeypatch) -> None:
    source_root = tmp_path / "source"
    benchmark_text = (
        "This benchmark passage contains enough distinct words to trigger the fragment overlap policy safely."
    )
    _write_dataset(
        source_root,
        "afrisenti",
        {
            "train": [
                {
                    "messages": _conversation(benchmark_text, "positive"),
                    "language": "amh",
                    "task": "classification",
                },
                {
                    "messages": _conversation(
                        "A unique training prompt with sufficient words for the test case.", "neutral"
                    ),
                    "language": "amh",
                    "task": "classification",
                },
            ],
            "validation": [
                {
                    "messages": _conversation(
                        "A unique validation prompt with sufficient words for the test case.", "positive"
                    ),
                    "language": "amh",
                    "task": "classification",
                }
            ],
        },
    )
    _write_dataset(
        source_root,
        "afriqa",
        {"test": [{"messages": _conversation(benchmark_text, "reserved answer")}]},
    )
    monkeypatch.setattr(os, "statvfs", lambda _: type("Stat", (), {"f_flag": os.ST_RDONLY})())

    summary = prepare_dataset(
        PreparationConfig(source_root=source_root, output_dir=tmp_path / "prepared", dataset_ids=("afrisenti",)),
        tokenizer=_FakeTokenizer(),
    )

    train = _read_manifest_records(tmp_path / "prepared", "train")
    assert len(train) == 1
    assert train[0]["messages"][0]["content"].startswith("A unique training prompt")
    assert summary["stats"]["contaminated_by_benchmark"] == {"afriqa:test": 1}


def test_prepare_dataset_refuses_nonempty_output(tmp_path: Path, monkeypatch) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    output_dir = tmp_path / "prepared"
    output_dir.mkdir()
    (output_dir / "existing").write_text("keep", encoding="utf-8")
    monkeypatch.setattr(os, "statvfs", lambda _: type("Stat", (), {"f_flag": os.ST_RDONLY})())

    with pytest.raises(FileExistsError, match="not empty"):
        prepare_dataset(
            PreparationConfig(source_root=source_root, output_dir=output_dir, dataset_ids=("afrisenti",)),
            tokenizer=_FakeTokenizer(),
        )
