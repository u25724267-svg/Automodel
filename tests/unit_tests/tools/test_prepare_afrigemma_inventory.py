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
from pathlib import Path

from openpyxl import Workbook

from tools.prepare_afrigemma_inventory import load_inventory_policy


def _manifest(root: Path, dataset_id: str, splits: tuple[str, ...]) -> None:
    dataset_dir = root / dataset_id
    dataset_dir.mkdir(parents=True)
    shards = []
    for split in splits:
        (dataset_dir / split).mkdir()
        filename = f"{dataset_id}-{split}-00000.parquet"
        (dataset_dir / split / filename).touch()
        shards.append({"split": split, "filename": filename})
    (dataset_dir / "manifest.json").write_text(json.dumps({"shards": shards}), encoding="utf-8")


def test_load_inventory_policy_selects_enabled_train_sources_and_validation_splits(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    _manifest(source_root, "train_with_validation", ("train", "validation", "test"))
    _manifest(source_root, "train_with_dev", ("train", "dev"))
    _manifest(source_root, "train_only", ("train",))
    _manifest(source_root, "afriqa", ("train", "test"))
    _manifest(source_root, "disabled", ("train",))

    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "sft_datasets.csv"
    worksheet.append(["Dataset ID", "Task", "Purpose", "Status Flag"])
    worksheet.append(["train_with_validation", "instruction", "train", True])
    worksheet.append(["train_with_dev", "translation", "train", True])
    worksheet.append(["train_only", "classification", "train", True])
    worksheet.append(["afriqa", "qa", "train", True])
    worksheet.append(["disabled", "instruction", "train", False])
    worksheet.append(["evaluation", "qa", "eval", True])
    workbook_path = tmp_path / "inventory.xlsx"
    workbook.save(workbook_path)

    policy = load_inventory_policy(workbook_path, source_root)

    assert policy.dataset_ids == ("train_with_validation", "train_with_dev", "train_only")
    assert dict(policy.dataset_tasks) == {
        "train_with_validation": "instruction",
        "train_with_dev": "translation",
        "train_only": "classification",
    }
    assert dict(policy.validation_source_splits) == {
        "train_with_validation": "validation",
        "train_with_dev": "dev",
        "train_only": None,
    }
    assert dict(policy.skipped) == {
        "afriqa": "reserved_evaluation",
        "disabled": "disabled",
        "evaluation": "purpose:eval",
    }
