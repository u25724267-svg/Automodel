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

import argparse
import hashlib
import json
import logging
import os
from collections import Counter
from pathlib import Path

from nemo_automodel.shared.import_utils import safe_import
from tools.benchmark_contamination import BenchmarkBlocklist

LOGGER = logging.getLogger(__name__)
PYARROW_AVAILABLE, pq = safe_import("pyarrow.parquet")
MOUNTED_RESERVED_DATASETS = (
    "afri_mgsm",
    "afri_mmlu",
    "afriqa",
    "afriqa_gold_passages",
    "uhura_arc_easy",
    "uhura_eval",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        while chunk := source_file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require_read_only_mount(source_root: Path) -> None:
    if not os.statvfs(source_root).f_flag & os.ST_RDONLY:
        raise RuntimeError(f"Reserved source root must be read-only: {source_root}")


def _decode_messages(raw_messages: object) -> list[dict[str, str]]:
    if isinstance(raw_messages, str):
        raw_messages = json.loads(raw_messages)
    if not isinstance(raw_messages, list):
        raise TypeError("messages must be a list or JSON-encoded list")
    messages = []
    for message in raw_messages:
        if not isinstance(message, dict):
            raise TypeError("message must be an object")
        role = message.get("role")
        content = message.get("content")
        if isinstance(role, str) and isinstance(content, str):
            messages.append({"role": role, "content": content})
    return messages


def _augment_blocklist(source_root: Path, blocklist_path: Path) -> dict[str, object]:
    origins_processed = 0
    source_manifests = {}
    with BenchmarkBlocklist(blocklist_path) as blocklist:
        for dataset_id in MOUNTED_RESERVED_DATASETS:
            manifest_path = source_root / dataset_id / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            source_manifests[dataset_id] = _sha256(manifest_path)
            blocklist.set_metadata(f"mounted:{dataset_id}", source_manifests[dataset_id])
            for shard in manifest.get("shards", []):
                split = shard["split"]
                shard_path = source_root / dataset_id / split / shard["filename"]
                parquet_file = pq.ParquetFile(shard_path)
                for batch in parquet_file.iter_batches(batch_size=2_048, columns=["messages"]):
                    for record in batch.to_pylist():
                        messages = _decode_messages(record["messages"])
                        for message in messages:
                            if message["role"] == "user":
                                blocklist.add_text(
                                    message["content"],
                                    benchmark=f"mounted:{dataset_id}:{split}",
                                    field="user",
                                )
                                origins_processed += 1
            blocklist.commit()
        counts = blocklist.counts()
    return {
        "origins_processed": origins_processed,
        "source_manifests": source_manifests,
        "counts_after_augmentation": counts,
    }


def audit_and_filter(
    data_root: Path,
    reserved_source_root: Path,
    blocklist_path: Path,
) -> dict[str, object]:
    """Augment benchmark fingerprints and remove contaminated prepared rows.

    Args:
        data_root: Prepared candidate directory with train and validation manifests.
        reserved_source_root: Read-only mounted directory containing reserved Parquet families.
        blocklist_path: Existing pinned benchmark blocklist to augment and query.

    Returns:
        JSON-serializable contamination and filtering summary.
    """
    if not PYARROW_AVAILABLE:
        raise RuntimeError("pyarrow is required for mounted benchmark auditing")
    data_root = data_root.resolve()
    reserved_source_root = reserved_source_root.resolve()
    blocklist_path = blocklist_path.resolve()
    report_path = data_root / "contamination_audit.json"
    if report_path.exists():
        raise FileExistsError(f"Contamination audit was already finalized: {report_path}")
    _require_read_only_mount(reserved_source_root)
    augmentation = _augment_blocklist(reserved_source_root, blocklist_path)

    staged_paths: list[tuple[Path, Path]] = []
    before_by_split: Counter[str] = Counter()
    after_by_split: Counter[str] = Counter()
    removed_by_split: Counter[str] = Counter()
    removed_by_source: Counter[str] = Counter()
    matches_by_benchmark: Counter[str] = Counter()
    try:
        with BenchmarkBlocklist(blocklist_path, mode="read-only") as blocklist:
            for split in ("train", "validation"):
                manifest = json.loads((data_root / f"{split}_meta.json").read_text(encoding="utf-8"))
                for entry in manifest.values():
                    source_path = data_root / entry["file_name"]
                    staged_path = source_path.with_suffix(source_path.suffix + ".contamination-filter.tmp")
                    if staged_path.exists():
                        raise FileExistsError(f"Staged audit output already exists: {staged_path}")
                    with (
                        source_path.open(encoding="utf-8") as source_file,
                        staged_path.open("w", encoding="utf-8") as output_file,
                    ):
                        for line_number, line in enumerate(source_file, start=1):
                            record = json.loads(line)
                            messages = _decode_messages(record.get("messages"))
                            texts = [message["content"] for message in messages]
                            if not texts:
                                raise ValueError(f"No auditable messages in {source_path}:{line_number}")
                            before_by_split[split] += 1
                            match = blocklist.find_match(texts)
                            if match is not None:
                                removed_by_split[split] += 1
                                removed_by_source[str(record.get("source_dataset", "unknown"))] += 1
                                matches_by_benchmark[f"{match.benchmark}:{match.field}"] += 1
                                continue
                            output_file.write(line)
                            after_by_split[split] += 1
                    staged_paths.append((source_path, staged_path))
    except BaseException:
        for _, staged_path in staged_paths:
            staged_path.unlink(missing_ok=True)
        raise

    for source_path, staged_path in staged_paths:
        os.replace(staged_path, source_path)

    report = {
        "data_root": str(data_root),
        "reserved_source_root": str(reserved_source_root),
        "reserved_source_access": "read-only mount verified with statvfs ST_RDONLY",
        "blocklist_path": str(blocklist_path),
        "blocklist_sha256": _sha256(blocklist_path),
        "augmentation": augmentation,
        "before_by_split": dict(sorted(before_by_split.items())),
        "after_by_split": dict(sorted(after_by_split.items())),
        "removed_by_split": dict(sorted(removed_by_split.items())),
        "removed_by_source": dict(sorted(removed_by_source.items())),
        "matches_by_benchmark": dict(sorted(matches_by_benchmark.items())),
        "coverage": {
            "mounted_reserved_datasets": list(MOUNTED_RESERVED_DATASETS),
            "pinned_upstream_datasets": ["afrimmlu", "afrixnli", "afrimgsm", "belebele"],
            "unresolved_reserved_datasets": ["flores_plus"],
        },
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    LOGGER.info(
        "Removed %d contaminated records; %d train and %d validation records remain",
        sum(removed_by_split.values()),
        after_by_split["train"],
        after_by_split["validation"],
    )
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit Top-40 gold data against pinned and mounted benchmarks.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--reserved-source-root", type=Path, required=True)
    parser.add_argument("--blocklist", type=Path, required=True)
    return parser


def main() -> int:
    """Audit and filter prepared Top-40 gold data for benchmark contamination."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = _build_parser().parse_args()
    audit_and_filter(args.data_root, args.reserved_source_root, args.blocklist)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
