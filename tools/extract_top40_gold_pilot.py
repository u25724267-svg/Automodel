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
import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import pyarrow.parquet as pq

LOGGER = logging.getLogger(__name__)

CANDIDATE_DATASETS = (
    "aya_dataset",
    "afrisenti",
    "mafand_mt",
    "masakhaner1",
    "masakhaner2",
    "masakhanews",
    "masakhapos",
    "menyo20k_mt",
    "naijasenti",
)
RESERVED_DATASETS = (
    "afri_mgsm",
    "afri_mmlu",
    "afrixnli",
    "afriqa",
    "afriqa_gold_passages",
    "flores_plus",
    "uhura_arc_easy",
    "uhura_eval",
)
CANDIDATE_SPLIT_ORDER = ("train", "validation", "dev", "test")
RESERVED_SPLIT_ORDER = ("test", "validation", "dev", "train")


@dataclass(frozen=True)
class DatasetResult:
    """Summary of one dataset's pilot extraction."""

    dataset_id: str
    disposition: str
    status: str
    source_split: str | None
    source_shards: list[str]
    samples_written: int
    output_file: str | None


@dataclass(frozen=True)
class PreparedResult:
    """Summary of one disposition's loader-ready preprocessing."""

    disposition: str
    datasets_written: int
    records_written: int
    system_messages_removed: int
    meta_file: str


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _require_read_only_mount(source_root: Path) -> None:
    mount_flags = os.statvfs(source_root).f_flag
    if not mount_flags & os.ST_RDONLY:
        raise RuntimeError(f"Source root must be on a read-only filesystem: {source_root}")


def _select_shards(dataset_root: Path, split_order: tuple[str, ...]) -> tuple[str, list[Path]]:
    manifest_path = dataset_root / "manifest.json"
    with manifest_path.open(encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)

    shards_by_split: dict[str, list[Path]] = {}
    for shard in manifest.get("shards", []):
        split = shard["split"]
        shard_path = dataset_root / split / shard["filename"]
        shards_by_split.setdefault(split, []).append(shard_path)

    for split in split_order:
        if split in shards_by_split:
            return split, sorted(shards_by_split[split])
    raise ValueError(f"No supported split found in {manifest_path}")


def _sample_records(shards: list[Path], sample_count: int) -> tuple[list[dict[str, object]], list[str]]:
    records: list[dict[str, object]] = []
    shards_read: list[str] = []
    for shard_path in shards:
        parquet_file = pq.ParquetFile(shard_path)
        remaining = sample_count - len(records)
        for batch in parquet_file.iter_batches(batch_size=remaining):
            records.extend(batch.to_pylist())
            break
        shards_read.append(shard_path.name)
        if len(records) >= sample_count:
            break
    return records[:sample_count], shards_read


def _extract_dataset(
    source_root: Path,
    output_root: Path,
    dataset_id: str,
    disposition: str,
    split_order: tuple[str, ...],
    sample_count: int,
) -> DatasetResult:
    dataset_root = source_root / dataset_id
    if not dataset_root.is_dir():
        return DatasetResult(dataset_id, disposition, "missing", None, [], 0, None)

    split, shards = _select_shards(dataset_root, split_order)
    records, shards_read = _sample_records(shards, sample_count)
    destination = output_root / disposition / f"{dataset_id}.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as output_file:
        for sample_index, record in enumerate(records):
            sample = {
                "dataset_id": dataset_id,
                "disposition": disposition,
                "source_split": split,
                "sample_index": sample_index,
                "record": record,
            }
            output_file.write(json.dumps(sample, ensure_ascii=False, default=str) + "\n")

    status = "extracted" if len(records) == sample_count else "insufficient_rows"
    return DatasetResult(
        dataset_id=dataset_id,
        disposition=disposition,
        status=status,
        source_split=split,
        source_shards=shards_read,
        samples_written=len(records),
        output_file=str(destination.relative_to(output_root)),
    )


def extract_pilot(source_root: Path, output_root: Path, sample_count: int) -> list[DatasetResult]:
    """Extract a bounded local pilot from the mounted Top-40 gold datasets.

    Args:
        source_root: Read-only directory containing the materialized dataset families.
        output_root: Local directory where pilot JSONL files and metadata are written.
        sample_count: Maximum number of records to extract from each dataset family.

    Returns:
        Extraction result for every candidate and benchmark-reserved dataset family.
    """
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    _require_read_only_mount(source_root)
    if output_root == source_root or output_root.is_relative_to(source_root):
        raise ValueError("Output directory must not be inside the read-only source mount")

    results = [
        _extract_dataset(
            source_root,
            output_root,
            dataset_id,
            "candidate",
            CANDIDATE_SPLIT_ORDER,
            sample_count,
        )
        for dataset_id in CANDIDATE_DATASETS
    ]
    results.extend(
        _extract_dataset(
            source_root,
            output_root,
            dataset_id,
            "reserved",
            RESERVED_SPLIT_ORDER,
            sample_count,
        )
        for dataset_id in RESERVED_DATASETS
    )

    manifest = {
        "source_root": str(source_root),
        "source_access": "read-only mount verified with statvfs ST_RDONLY",
        "samples_per_dataset": sample_count,
        "reserved_training_policy": "exclude from training; FLORES+ training is prohibited",
        "datasets": [asdict(result) for result in results],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "pilot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return results


def _prepare_messages(raw_messages: object, source_path: Path, line_number: int) -> tuple[list[dict[str, str]], int]:
    if isinstance(raw_messages, str):
        raw_messages = json.loads(raw_messages)
    if not isinstance(raw_messages, list):
        raise TypeError(f"{source_path}:{line_number}: messages must be a list or JSON-encoded list")

    messages: list[dict[str, str]] = []
    system_messages_removed = 0
    for message in raw_messages:
        if not isinstance(message, dict):
            raise TypeError(f"{source_path}:{line_number}: each message must be an object")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str) or not content.strip():
            raise ValueError(f"{source_path}:{line_number}: each message requires a non-empty string role and content")
        if role == "system":
            system_messages_removed += 1
        elif role in {"user", "assistant"}:
            messages.append({"role": role, "content": content})
        else:
            raise ValueError(f"{source_path}:{line_number}: unsupported message role {role!r}")

    expected_roles = ["user" if index % 2 == 0 else "assistant" for index in range(len(messages))]
    actual_roles = [message["role"] for message in messages]
    if actual_roles != expected_roles or not messages or messages[-1]["role"] != "assistant":
        raise ValueError(
            f"{source_path}:{line_number}: messages must alternate user/assistant and end with assistant; "
            f"got {actual_roles}"
        )
    return messages, system_messages_removed


def _prepare_disposition(pilot_root: Path, prepared_root: Path, disposition: str) -> PreparedResult:
    source_files = sorted((pilot_root / disposition).glob("*.jsonl"))
    if not source_files:
        raise FileNotFoundError(f"No pilot JSONL files found for {disposition}: {pilot_root / disposition}")

    output_dir = prepared_root / disposition
    output_dir.mkdir(parents=True)
    meta: dict[str, dict[str, object]] = {}
    records_written = 0
    system_messages_removed = 0

    for source_path in source_files:
        destination = output_dir / source_path.name
        with source_path.open(encoding="utf-8") as source_file, destination.open("w", encoding="utf-8") as output_file:
            for line_number, line in enumerate(source_file, start=1):
                sample = json.loads(line)
                record = sample.get("record")
                if not isinstance(record, dict):
                    raise TypeError(f"{source_path}:{line_number}: record must be an object")
                messages, removed = _prepare_messages(record.get("messages"), source_path, line_number)
                system_messages_removed += removed
                prepared_record = {key: value for key, value in record.items() if key != "messages"}
                prepared_record.update(
                    {
                        "messages": messages,
                        "pilot_dataset_id": sample["dataset_id"],
                        "pilot_disposition": disposition,
                        "pilot_sample_index": sample["sample_index"],
                    }
                )
                output_file.write(json.dumps(prepared_record, ensure_ascii=False, default=str) + "\n")
                records_written += 1

        meta[source_path.stem] = {
            "file_name": destination.relative_to(prepared_root).as_posix(),
            "columns": {"messages": "messages"},
            "sample_ratio": 1.0,
        }

    meta_name = "train_meta.json" if disposition == "candidate" else "reserved_meta.json"
    (prepared_root / meta_name).write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return PreparedResult(
        disposition=disposition,
        datasets_written=len(source_files),
        records_written=records_written,
        system_messages_removed=system_messages_removed,
        meta_file=meta_name,
    )


def prepare_pilot(pilot_root: Path, prepared_root: Path) -> list[PreparedResult]:
    """Convert inspection pilot records into NeMo AutoModel meta-dataset input.

    Args:
        pilot_root: Directory containing candidate and reserved inspection JSONL files.
        prepared_root: New local directory for loader-ready JSONL files and meta manifests.

    Returns:
        Preprocessing summaries for candidate and reserved records.
    """
    pilot_root = pilot_root.resolve()
    prepared_root = prepared_root.resolve()
    if prepared_root.exists():
        raise FileExistsError(f"Prepared output already exists: {prepared_root}")
    if prepared_root == pilot_root:
        raise ValueError("Prepared output must differ from the inspection pilot root")

    prepared_root.mkdir(parents=True)
    results = [
        _prepare_disposition(pilot_root, prepared_root, "candidate"),
        _prepare_disposition(pilot_root, prepared_root, "reserved"),
    ]
    manifest = {
        "source_pilot": str(pilot_root),
        "format": "NeMo AutoModel meta dataset with native OpenAI-format messages",
        "candidate_training_manifest": "train_meta.json",
        "reserved_training_policy": "exclude from training",
        "results": [asdict(result) for result in results],
    }
    (prepared_root / "preprocessing_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract a bounded pilot from the Top-40 gold datasets.")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--source-root", type=Path)
    source_group.add_argument("--pilot-input-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--samples-per-dataset", type=_positive_int, default=3)
    return parser


def main() -> int:
    """Run the Top-40 gold pilot extractor."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = _build_parser().parse_args()
    if args.pilot_input_root is not None:
        results = prepare_pilot(args.pilot_input_root, args.output_root)
        for result in results:
            LOGGER.info(
                "%s: %d datasets, %d records, %d system messages removed",
                result.disposition,
                result.datasets_written,
                result.records_written,
                result.system_messages_removed,
            )
        return 0

    results = extract_pilot(args.source_root, args.output_root, args.samples_per_dataset)
    for result in results:
        LOGGER.info(
            "%s: %s (%d samples)",
            result.dataset_id,
            result.status,
            result.samples_written,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
