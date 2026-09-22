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

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from tools.merge_afrigemma_inventory import merge_inventory_outputs


def _record(source: str, prompt: str) -> dict[str, object]:
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "answer"},
        ],
        "source": source,
        "lang": "eng",
        "task": "instruction",
        "prepared_split": "train",
        "_text_tokens": 10,
    }


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def test_merge_inventory_outputs_uses_rebuilt_validation_and_deduplicates(tmp_path: Path) -> None:
    partial = tmp_path / "partial"
    continuation = tmp_path / "continuation"
    output = tmp_path / "final"
    validation_record = _record("suffix", "shared")
    validation_record["prepared_split"] = "validation"
    _write_jsonl(partial / "processed/train/prefix-00000.jsonl", [_record("prefix", "shared")])
    _write_jsonl(continuation / "processed/train/suffix-00000.jsonl", [_record("suffix", "unique")])
    _write_jsonl(continuation / "processed/validation/suffix-00000.jsonl", [validation_record])
    (continuation / "validation_meta.json").write_text(
        json.dumps(
            {
                "suffix": {
                    "file_name": "processed/validation/suffix-00000.jsonl",
                    "columns": {"messages": "messages"},
                    "sample_ratio": 1.0,
                }
            }
        ),
        encoding="utf-8",
    )
    (continuation / "summary.json").write_text(
        json.dumps(
            {
                "dataset_ids": ["prefix", "suffix"],
                "train_dataset_ids": ["suffix"],
                "resume_after_dataset": "prefix",
            }
        ),
        encoding="utf-8",
    )
    database = sqlite3.connect(continuation / "benchmark_blocklist.sqlite3")
    database.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID")
    database.commit()
    database.close()

    summary = merge_inventory_outputs(partial, continuation, output)

    assert summary["written_by_split"] == {"train": 1, "validation": 1}
    assert summary["duplicates_removed_by_split"] == {"train": 1}
    train_meta = json.loads((output / "train_meta.json").read_text(encoding="utf-8"))
    train_records = [
        json.loads(line)
        for entry in train_meta.values()
        for line in (output / entry["file_name"]).read_text(encoding="utf-8").splitlines()
    ]
    assert [record["messages"][0]["content"] for record in train_records] == ["unique"]
