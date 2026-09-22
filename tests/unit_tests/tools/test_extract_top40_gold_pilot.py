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

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nemo_automodel.components.datasets.vlm.datasets import make_meta_dataset
from tools.extract_top40_gold_pilot import _extract_dataset, extract_pilot, prepare_pilot


def test_extract_pilot_rejects_writable_source(tmp_path: Path) -> None:
    (tmp_path / "source").mkdir()
    with pytest.raises(RuntimeError, match="read-only filesystem"):
        extract_pilot(tmp_path / "source", tmp_path / "output", sample_count=3)


def test_extract_dataset_caps_records_and_preserves_rows(tmp_path: Path) -> None:
    dataset_root = tmp_path / "source" / "example"
    shard_dir = dataset_root / "train"
    shard_dir.mkdir(parents=True)
    shard_name = "example-train-00000.parquet"
    pq.write_table(
        pa.Table.from_pylist([{"language": "hau", "text": f"row-{index}"} for index in range(5)]),
        shard_dir / shard_name,
    )
    manifest = {"shards": [{"split": "train", "filename": shard_name, "rows": 5}]}
    (dataset_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    result = _extract_dataset(
        tmp_path / "source",
        tmp_path / "output",
        "example",
        "candidate",
        ("train",),
        sample_count=3,
    )

    output_path = tmp_path / "output" / "candidate" / "example.jsonl"
    samples = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert result.samples_written == 3
    assert result.source_shards == [shard_name]
    assert [sample["record"]["text"] for sample in samples] == ["row-0", "row-1", "row-2"]
    assert all(sample["disposition"] == "candidate" for sample in samples)


def _write_pilot_sample(pilot_root: Path, disposition: str, dataset_id: str, messages: list[dict[str, str]]) -> None:
    destination = pilot_root / disposition / f"{dataset_id}.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    sample = {
        "dataset_id": dataset_id,
        "disposition": disposition,
        "source_split": "train",
        "sample_index": 0,
        "record": {
            "messages": json.dumps(messages),
            "language": "hau",
            "task": "classification",
        },
    }
    destination.write_text(json.dumps(sample) + "\n", encoding="utf-8")


def test_prepare_pilot_writes_native_messages_and_separate_manifests(tmp_path: Path) -> None:
    messages = [
        {"role": "system", "content": "Answer helpfully."},
        {"role": "user", "content": "Classify this text."},
        {"role": "assistant", "content": "neutral"},
    ]
    pilot_root = tmp_path / "pilot"
    _write_pilot_sample(pilot_root, "candidate", "candidate_data", messages)
    _write_pilot_sample(pilot_root, "reserved", "benchmark_data", messages)

    results = prepare_pilot(pilot_root, tmp_path / "prepared")

    candidate_path = tmp_path / "prepared" / "candidate" / "candidate_data.jsonl"
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    assert candidate["messages"] == messages[1:]
    assert candidate["language"] == "hau"
    assert candidate["pilot_dataset_id"] == "candidate_data"
    assert [result.records_written for result in results] == [1, 1]
    assert [result.system_messages_removed for result in results] == [1, 1]

    train_meta = json.loads((tmp_path / "prepared" / "train_meta.json").read_text(encoding="utf-8"))
    reserved_meta = json.loads((tmp_path / "prepared" / "reserved_meta.json").read_text(encoding="utf-8"))
    assert set(train_meta) == {"candidate_data"}
    assert set(reserved_meta) == {"benchmark_data"}
    assert train_meta["candidate_data"]["columns"] == {"messages": "messages"}
    assert train_meta["candidate_data"]["file_name"] == "candidate/candidate_data.jsonl"

    dataset = make_meta_dataset(str(tmp_path / "prepared" / "train_meta.json"))
    assert len(dataset) == 1
    assert dataset[0]["conversation"] == [
        {"role": "user", "content": [{"type": "text", "text": "Classify this text."}]},
        {"role": "assistant", "content": [{"type": "text", "text": "neutral"}]},
    ]


def test_prepare_pilot_rejects_invalid_role_sequence(tmp_path: Path) -> None:
    pilot_root = tmp_path / "pilot"
    _write_pilot_sample(
        pilot_root,
        "candidate",
        "invalid",
        [{"role": "assistant", "content": "answer without a prompt"}],
    )
    _write_pilot_sample(
        pilot_root,
        "reserved",
        "benchmark_data",
        [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ],
    )

    with pytest.raises(ValueError, match="must alternate user/assistant"):
        prepare_pilot(pilot_root, tmp_path / "prepared")
