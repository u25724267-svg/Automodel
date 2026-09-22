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

from tools.audit_top40_gold_contamination import MOUNTED_RESERVED_DATASETS, audit_and_filter
from tools.benchmark_contamination import BenchmarkBlocklist


def _messages(user: str, assistant: str = "answer") -> list[dict[str, str]]:
    return [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]


def _write_reserved_sources(root: Path, contaminated_text: str) -> None:
    for dataset_id in MOUNTED_RESERVED_DATASETS:
        split_dir = root / dataset_id / "test"
        split_dir.mkdir(parents=True)
        filename = f"{dataset_id}-test.parquet"
        records = [{"messages": json.dumps(_messages(contaminated_text))}]
        pq.write_table(pa.Table.from_pylist(records), split_dir / filename)
        manifest = {"shards": [{"split": "test", "filename": filename, "rows": 1}]}
        (root / dataset_id / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _write_prepared_split(root: Path, split: str, records: list[dict[str, object]]) -> None:
    path = root / "processed" / split / "candidate-00000.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record) + "\n")
    manifest = {
        "candidate-00000": {
            "file_name": path.relative_to(root).as_posix(),
            "columns": {"messages": "messages"},
            "sample_ratio": 1.0,
        }
    }
    (root / f"{split}_meta.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_audit_and_filter_removes_mounted_benchmark_matches(tmp_path: Path, monkeypatch) -> None:
    benchmark_text = "This benchmark passage contains enough distinct words for a stable fragment contamination match."
    reserved_root = tmp_path / "reserved"
    _write_reserved_sources(reserved_root, benchmark_text)
    data_root = tmp_path / "prepared"
    contaminated = {"messages": _messages(benchmark_text), "source_dataset": "candidate", "_text_tokens": 20}
    clean = {
        "messages": _messages("Completely unrelated training prompt."),
        "source_dataset": "candidate",
        "_text_tokens": 10,
    }
    _write_prepared_split(data_root, "train", [contaminated, clean])
    _write_prepared_split(data_root, "validation", [clean])
    blocklist_path = tmp_path / "blocklist.sqlite3"
    with BenchmarkBlocklist(blocklist_path):
        pass
    monkeypatch.setattr(os, "statvfs", lambda _: type("Stat", (), {"f_flag": os.ST_RDONLY})())

    report = audit_and_filter(data_root, reserved_root, blocklist_path)

    train_path = data_root / "processed" / "train" / "candidate-00000.jsonl"
    remaining = [json.loads(line) for line in train_path.read_text().splitlines()]
    assert remaining == [clean]
    assert report["removed_by_split"] == {"train": 1}
    assert report["removed_by_source"] == {"candidate": 1}
    assert report["augmentation"]["origins_processed"] == len(MOUNTED_RESERVED_DATASETS)
