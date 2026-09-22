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

"""Merge interrupted and resumed AfriGemma inventory materializations."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from tools.prepare_top40_gold import DatasetShardWriter, ExactDeduplicator, _conversation_digest


def _iter_manifest_records(root: Path, split: str) -> Iterable[dict[str, object]]:
    manifest = json.loads((root / f"{split}_meta.json").read_text(encoding="utf-8"))
    for entry in manifest.values():
        path = root / entry["file_name"]
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                yield json.loads(line)


def _iter_source_records(root: Path, dataset_id: str) -> Iterable[dict[str, object]]:
    for path in sorted((root / "processed" / "train").glob(f"{dataset_id}-*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                yield json.loads(line)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def merge_inventory_outputs(
    partial_root: Path,
    continuation_root: Path,
    output_dir: Path,
    *,
    shard_size: int = 100_000,
) -> dict[str, object]:
    """Merge a completed prefix with a resumed suffix under fresh deduplication.

    Args:
        partial_root: Interrupted output containing the completed training prefix.
        continuation_root: Completed resume output containing rebuilt validation and training suffix.
        output_dir: New final output directory.
        shard_size: Maximum records per output JSONL shard.

    Returns:
        JSON-compatible merge summary.
    """
    partial_root = partial_root.resolve()
    continuation_root = continuation_root.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    continuation_summary_path = continuation_root / "summary.json"
    continuation_summary = json.loads(continuation_summary_path.read_text(encoding="utf-8"))
    dataset_ids = tuple(continuation_summary["dataset_ids"])
    suffix_ids = tuple(continuation_summary["train_dataset_ids"])
    if not suffix_ids or dataset_ids[-len(suffix_ids) :] != suffix_ids:
        raise ValueError("Continuation training datasets must be a suffix of the selected inventory")
    prefix_ids = dataset_ids[: -len(suffix_ids)]
    resume_after = continuation_summary.get("resume_after_dataset")
    if not prefix_ids or prefix_ids[-1] != resume_after:
        raise ValueError("Continuation resume boundary does not match the completed prefix")

    output_dir.mkdir(parents=True, exist_ok=True)
    writer = DatasetShardWriter(output_dir, shard_size)
    database_path = output_dir / ".dedup.sqlite3"
    deduplicator = ExactDeduplicator(database_path)
    counts: Counter[str] = Counter()
    duplicates: Counter[str] = Counter()
    by_dataset: Counter[str] = Counter()
    missing_prefix_sources = []
    try:
        for record in _iter_manifest_records(continuation_root, "validation"):
            digest = _conversation_digest(record["messages"])
            if not deduplicator.add(digest):
                duplicates["validation"] += 1
                continue
            writer.write("validation", str(record["source"]), record)
            counts["validation"] += 1
            by_dataset[f"validation:{record['source']}"] += 1

        for dataset_id in dataset_ids:
            source_root = partial_root if dataset_id in prefix_ids else continuation_root
            source_records = list(_iter_source_records(source_root, dataset_id))
            if dataset_id in prefix_ids and not source_records:
                missing_prefix_sources.append(dataset_id)
            for record in source_records:
                digest = _conversation_digest(record["messages"])
                if not deduplicator.add(digest):
                    duplicates["train"] += 1
                    continue
                writer.write("train", dataset_id, record)
                counts["train"] += 1
                by_dataset[f"train:{dataset_id}"] += 1
    finally:
        writer.close()
        deduplicator.close()
        database_path.unlink(missing_ok=True)

    if not counts["train"] or not counts["validation"]:
        raise ValueError("Merged output must contain train and validation records")
    for split, manifest in writer.manifests.items():
        (output_dir / f"{split}_meta.json").write_text(
            json.dumps(dict(sorted(manifest.items())), indent=2) + "\n",
            encoding="utf-8",
        )
    benchmark_source = continuation_root / "benchmark_blocklist.sqlite3"
    benchmark_destination = output_dir / "benchmark_blocklist.sqlite3"
    shutil.copy2(benchmark_source, benchmark_destination)
    summary = {
        "partial_root": str(partial_root),
        "continuation_root": str(continuation_root),
        "partial_boundary": resume_after,
        "prefix_dataset_ids": list(prefix_ids),
        "suffix_dataset_ids": list(suffix_ids),
        "dataset_ids": list(dataset_ids),
        "written_by_split": dict(sorted(counts.items())),
        "duplicates_removed_by_split": dict(sorted(duplicates.items())),
        "written_by_dataset": dict(sorted(by_dataset.items())),
        "prefix_sources_without_records": missing_prefix_sources,
        "continuation_summary_sha256": _sha256(continuation_summary_path),
        "benchmark_blocklist_sha256": _sha256(benchmark_destination),
    }
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
    parser.add_argument("--partial-root", type=Path, required=True)
    parser.add_argument("--continuation-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-size", type=_positive_int, default=100_000)
    return parser


def main() -> int:
    """Merge the interrupted prefix and completed continuation."""
    args = _build_parser().parse_args()
    summary = merge_inventory_outputs(
        args.partial_root,
        args.continuation_root,
        args.output_dir,
        shard_size=args.shard_size,
    )
    print(json.dumps(summary["written_by_split"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
