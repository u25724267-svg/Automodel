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

import pytest

from tools.filter_top40_gold_lengths import filter_precomputed_lengths


def _write_split(root: Path, split: str, token_counts: list[int]) -> None:
    data_path = root / "processed" / split / "example-00000.jsonl"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    with data_path.open("w", encoding="utf-8") as output_file:
        for index, token_count in enumerate(token_counts):
            output_file.write(json.dumps({"id": index, "_text_tokens": token_count}) + "\n")
    manifest = {
        "example-00000": {
            "file_name": data_path.relative_to(root).as_posix(),
            "columns": {"messages": "messages"},
            "sample_ratio": 1.0,
        }
    }
    (root / f"{split}_meta.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_filter_precomputed_lengths_removes_overlong_records(tmp_path: Path) -> None:
    _write_split(tmp_path, "train", [100, 4_080, 4_081])
    _write_split(tmp_path, "validation", [200, 5_000])

    summary = filter_precomputed_lengths(tmp_path, max_text_tokens=4_080)

    train_path = tmp_path / "processed" / "train" / "example-00000.jsonl"
    validation_path = tmp_path / "processed" / "validation" / "example-00000.jsonl"
    assert [json.loads(line)["_text_tokens"] for line in train_path.read_text().splitlines()] == [100, 4_080]
    assert [json.loads(line)["_text_tokens"] for line in validation_path.read_text().splitlines()] == [200]
    assert summary["removed_by_split"] == {"train": 1, "validation": 1}
    assert summary["removed_by_dataset"] == {"example": 2}

    with pytest.raises(FileExistsError, match="already finalized"):
        filter_precomputed_lengths(tmp_path, max_text_tokens=4_080)


def test_filter_precomputed_lengths_rejects_missing_counts_without_replacing_sources(tmp_path: Path) -> None:
    _write_split(tmp_path, "train", [100])
    _write_split(tmp_path, "validation", [200])
    validation_path = tmp_path / "processed" / "validation" / "example-00000.jsonl"
    validation_path.write_text(json.dumps({"id": 0}) + "\n", encoding="utf-8")
    train_path = tmp_path / "processed" / "train" / "example-00000.jsonl"
    original_train = train_path.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid _text_tokens"):
        filter_precomputed_lengths(tmp_path, max_text_tokens=4_080)

    assert train_path.read_text(encoding="utf-8") == original_train
