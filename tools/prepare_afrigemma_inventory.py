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

"""Prepare a capped training mixture from the complete AfriGemma inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from nemo_automodel.shared.import_utils import safe_import
from tools.prepare_top40_gold import RESERVED_DATASETS, PreparationConfig, prepare_dataset

OPENPYXL_AVAILABLE, openpyxl = safe_import(
    "openpyxl",
    msg="openpyxl is required to read the AfriGemma inventory workbook. Install project dependencies with uv.",
)

logger = logging.getLogger(__name__)

TOP40_LANGUAGES = frozenset(
    {
        "afr",
        "amh",
        "ara",
        "bam",
        "bem",
        "deu",
        "din",
        "eng",
        "fon",
        "fra",
        "fuj",
        "gaa",
        "hau",
        "ibo",
        "kab",
        "kam",
        "kik",
        "kin",
        "lin",
        "lug",
        "nso",
        "nya",
        "pcm",
        "plt",
        "por",
        "run",
        "sna",
        "som",
        "sot",
        "spa",
        "swh",
        "tig",
        "tso",
        "tsn",
        "tum",
        "twi",
        "wol",
        "xho",
        "yor",
        "zul",
    }
)


@dataclass(frozen=True)
class InventoryPolicy:
    """Workbook-derived dataset policy and exclusion evidence."""

    dataset_ids: tuple[str, ...]
    dataset_tasks: tuple[tuple[str, str], ...]
    validation_source_splits: tuple[tuple[str, str | None], ...]
    skipped: tuple[tuple[str, str], ...]


def _manifest(source_root: Path, dataset_id: str) -> dict[str, object]:
    return json.loads((source_root / dataset_id / "manifest.json").read_text(encoding="utf-8"))


def load_inventory_policy(workbook_path: Path, source_root: Path) -> InventoryPolicy:
    """Select mounted and enabled training families from the inventory workbook.

    Args:
        workbook_path: Enriched AfriGemma inventory workbook.
        source_root: Read-only bucket mount containing dataset manifests.

    Returns:
        Ordered dataset IDs, task and validation policies, and skipped reasons.
    """
    if not OPENPYXL_AVAILABLE:
        raise RuntimeError("openpyxl is required to read the AfriGemma inventory workbook")
    workbook = openpyxl.load_workbook(workbook_path, read_only=True, data_only=True)
    worksheet = workbook["sft_datasets.csv"]
    rows = worksheet.iter_rows(values_only=True)
    headers = next(rows)
    selected = []
    tasks = []
    validation_splits = []
    skipped = []
    reserved = set(RESERVED_DATASETS)

    for row in rows:
        if not row or not row[0] or str(row[0]).strip().casefold() in {"id", "dataset_id", "dataset id"}:
            continue
        record = dict(zip(headers, row))
        dataset_id = str(record["Dataset ID"]).strip()
        purpose = str(record.get("Purpose") or "").strip().casefold()
        task = str(record.get("Task") or "").strip().casefold()
        if purpose != "train":
            skipped.append((dataset_id, f"purpose:{purpose or 'missing'}"))
            continue
        if record.get("Status Flag") is not True:
            skipped.append((dataset_id, "disabled"))
            continue
        if dataset_id in reserved:
            skipped.append((dataset_id, "reserved_evaluation"))
            continue
        manifest_path = source_root / dataset_id / "manifest.json"
        if not manifest_path.is_file():
            skipped.append((dataset_id, "not_mounted"))
            continue
        manifest = _manifest(source_root, dataset_id)
        available_splits = {
            shard.get("split")
            for shard in manifest.get("shards", [])
            if isinstance(shard, dict) and isinstance(shard.get("split"), str)
        }
        if "train" not in available_splits:
            skipped.append((dataset_id, "no_train_shard"))
            continue
        if not task or task == "none":
            skipped.append((dataset_id, "missing_task"))
            continue
        validation_split = (
            "validation" if "validation" in available_splits else "dev" if "dev" in available_splits else None
        )
        selected.append(dataset_id)
        tasks.append((dataset_id, task))
        validation_splits.append((dataset_id, validation_split))

    if not selected:
        raise ValueError("Inventory selection produced no training datasets")
    return InventoryPolicy(tuple(selected), tuple(tasks), tuple(validation_splits), tuple(skipped))


def prepare_inventory(
    workbook_path: Path,
    source_root: Path,
    output_dir: Path,
    *,
    max_train_records_per_dataset: int = 10_000,
    max_validation_records_per_dataset: int = 1_000,
    resume_after_dataset: str | None = None,
) -> dict[str, object]:
    """Materialize the capped Top-40 mixture from all eligible inventory families.

    Args:
        workbook_path: Enriched AfriGemma inventory workbook.
        source_root: Read-only bucket mount containing source parquet shards.
        output_dir: New local output directory.
        max_train_records_per_dataset: Maximum accepted training records from each family.
        max_validation_records_per_dataset: Maximum accepted validation records from each family.
        resume_after_dataset: Rebuild all validation, but process training only after this family.

    Returns:
        JSON-compatible preparation summary.
    """
    policy = load_inventory_policy(workbook_path.resolve(), source_root.resolve())
    train_dataset_ids = policy.dataset_ids
    if resume_after_dataset is not None:
        if resume_after_dataset not in policy.dataset_ids:
            raise ValueError(f"Resume dataset is not selected by the inventory: {resume_after_dataset}")
        train_dataset_ids = policy.dataset_ids[policy.dataset_ids.index(resume_after_dataset) + 1 :]
        if not train_dataset_ids:
            raise ValueError(f"No training datasets remain after {resume_after_dataset}")
    summary = prepare_dataset(
        PreparationConfig(
            source_root=source_root,
            output_dir=output_dir,
            validation_fraction=0.1,
            dataset_ids=policy.dataset_ids,
            dataset_tasks=policy.dataset_tasks,
            approved_languages=TOP40_LANGUAGES,
            validation_source_splits=policy.validation_source_splits,
            max_train_records_per_dataset=max_train_records_per_dataset,
            max_validation_records_per_dataset=max_validation_records_per_dataset,
            train_dataset_ids=train_dataset_ids,
        )
    )
    summary.update(
        {
            "inventory_workbook": str(workbook_path.resolve()),
            "inventory_workbook_sha256": hashlib.sha256(workbook_path.read_bytes()).hexdigest(),
            "inventory_selected_datasets": len(policy.dataset_ids),
            "resume_after_dataset": resume_after_dataset,
            "inventory_skipped": [
                {"dataset_id": dataset_id, "reason": reason} for dataset_id, reason in policy.skipped
            ],
        }
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-train-records-per-dataset", type=_positive_int, default=10_000)
    parser.add_argument("--max-validation-records-per-dataset", type=_positive_int, default=1_000)
    parser.add_argument("--resume-after-dataset")
    return parser


def main() -> int:
    """Prepare the capped all-inventory Top-40 mixture."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = _build_parser().parse_args()
    summary = prepare_inventory(
        args.workbook,
        args.source_root,
        args.output_dir,
        max_train_records_per_dataset=args.max_train_records_per_dataset,
        max_validation_records_per_dataset=args.max_validation_records_per_dataset,
        resume_after_dataset=args.resume_after_dataset,
    )
    logger.info(
        "Prepared %d train and %d validation records from %d inventory datasets",
        summary["stats"]["written_by_split"]["train"],
        summary["stats"]["written_by_split"]["validation"],
        summary["inventory_selected_datasets"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
