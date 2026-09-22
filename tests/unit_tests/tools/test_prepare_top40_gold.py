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

from tools.prepare_top40_gold import PreparationConfig, prepare_dataset


def _conversation(user: str, assistant: str) -> str:
    return json.dumps(
        [
            {"role": "system", "content": "Answer helpfully."},
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
        )
    )

    train = _read_manifest_records(output_dir, "train")
    validation = _read_manifest_records(output_dir, "validation")
    assert len(train) == 1
    assert len(validation) == 1
    assert train[0]["messages"] == [
        {"role": "user", "content": "train"},
        {"role": "assistant", "content": "neutral"},
    ]
    assert validation[0]["messages"][0]["content"] == "shared"
    assert all(record["messages"][0]["content"] != "test-only" for record in train + validation)
    assert summary["stats"]["duplicate_by_split"] == {"train": 1}
    assert summary["stats"]["bom_characters_removed"] == 1
    assert summary["test_splits_materialized"] is False


def test_prepare_dataset_refuses_nonempty_output(tmp_path: Path, monkeypatch) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    output_dir = tmp_path / "prepared"
    output_dir.mkdir()
    (output_dir / "existing").write_text("keep", encoding="utf-8")
    monkeypatch.setattr(os, "statvfs", lambda _: type("Stat", (), {"f_flag": os.ST_RDONLY})())

    with pytest.raises(FileExistsError, match="not empty"):
        prepare_dataset(PreparationConfig(source_root=source_root, output_dir=output_dir, dataset_ids=("afrisenti",)))
