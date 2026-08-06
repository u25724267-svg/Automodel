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

from tools.benchmark_contamination import BenchmarkBlocklist
from tools.prepare_inkuba import (
    LANGUAGE_SPLITS,
    _cell_token_targets,
    _cell_validation_targets,
    prepare_records,
    profile_records,
)


def _row(index: int, task: str = "mmt", source: str = "wmt22", target: str | None = None) -> dict[str, str]:
    return {
        "instruction": f"Translate item {index}",
        "inputs": f"Input sentence {index}",
        "targets": target or f"Target sentence {index}",
        "task": task,
        "data_source": source,
    }


def test_profile_and_prepare_exclude_benchmarks_sources_and_wrong_scripts(tmp_path: Path) -> None:
    blocklist_path = tmp_path / "benchmarks.sqlite3"
    with BenchmarkBlocklist(blocklist_path) as blocklist:
        blocklist.add_text("Input sentence 2", benchmark="belebele:hau:test", field="passage")

    records = {}
    for language_index, language in enumerate(LANGUAGE_SPLITS):
        offset = language_index * 1_000
        rows = [_row(offset + index) for index in range(20)]
        if language == "hau":
            rows.extend(
                [
                    _row(100, source="afriqa"),
                    _row(101, source="sib-200"),
                    _row(102, target="やはり画像が表示されています"),
                ]
            )
        records[f"{language}_train"] = rows
        records[f"{language}_dev"] = [_row(offset + 200 + index) for index in range(3)]

    token_counter = lambda text: len(text.split())
    profile = profile_records(
        records,
        token_counter=token_counter,
        benchmark_blocklist=blocklist_path,
        token_sample_fraction=1.0,
    )
    output_dir = tmp_path / "prepared"
    summary = prepare_records(
        records,
        profile=profile,
        token_counter=token_counter,
        benchmark_blocklist=blocklist_path,
        output_dir=output_dir,
        train_token_budget=500,
        validation_records_per_language=2,
        shard_size=4,
    )

    assert summary["stats"]["rejections"]["blocked_source"] >= 2
    assert summary["stats"]["rejections"]["unexpected_script"] >= 1
    assert summary["stats"]["rejections"]["benchmark_contamination"] >= 1
    assert summary["stats"]["by_language"].keys() == LANGUAGE_SPLITS.keys()
    assert summary["stats"]["written_tokens"] > 0
    validation_manifest = json.loads((output_dir / "validation_meta.json").read_text(encoding="utf-8"))
    assert validation_manifest
    written = [
        json.loads(line)
        for path in output_dir.glob("processed/*/*.jsonl")
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert all(row["messages"][0]["content"] != "Translate item 2\n\nInput sentence 2" for row in written)
    assert all(row["source"] != "inkuba:afriqa" for row in written)
    assert all("_text_tokens" in row for row in written)


def test_validation_targets_preserve_task_weights_and_equal_sources() -> None:
    profile = {
        "cells": {
            "dev|hau|mmt|wmt22": {"records": 100},
            "dev|hau|mmt|mafand": {"records": 100},
            "dev|hau|sentiment|afrisenti": {"records": 100},
            "dev|hau|topic|masakhanews": {"records": 100},
            "dev|hau|ner|masakhaner": {"records": 100},
            "dev|hau|pos|masakhapos": {"records": 100},
        }
    }

    targets = _cell_validation_targets(profile, 100)

    assert sum(targets.values()) == 100
    assert targets["dev|hau|mmt|wmt22"] == 25
    assert targets["dev|hau|mmt|mafand"] == 25
    assert targets["dev|hau|sentiment|afrisenti"] == 15
    assert targets["dev|hau|topic|masakhanews"] == 15
    assert targets["dev|hau|ner|masakhaner"] == 10
    assert targets["dev|hau|pos|masakhapos"] == 10


def test_train_targets_reallocate_small_source_shortfall() -> None:
    profile = {
        "cells": {
            "train|hau|mmt|small": {"records": 1, "average_tokens": 10.0},
            "train|hau|mmt|large": {"records": 100, "average_tokens": 10.0},
            "train|hau|sentiment|source": {"records": 100, "average_tokens": 10.0},
        }
    }

    targets = _cell_token_targets(profile, 500)

    assert sum(targets.values()) >= 99
    assert targets["train|hau|mmt|small"] == 10
    assert targets["train|hau|mmt|large"] > targets["train|hau|sentiment|source"]
