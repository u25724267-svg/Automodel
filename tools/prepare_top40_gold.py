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
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from nemo_automodel.shared.import_utils import safe_import

LOGGER = logging.getLogger(__name__)
PYARROW_AVAILABLE, pq = safe_import("pyarrow.parquet")

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
VALIDATION_SOURCE_SPLITS = {
    "aya_dataset": None,
    "afrisenti": "validation",
    "mafand_mt": "dev",
    "masakhaner1": "dev",
    "masakhaner2": "dev",
    "masakhanews": "validation",
    "masakhapos": "dev",
    "menyo20k_mt": "validation",
    "naijasenti": "validation",
}


@dataclass(frozen=True)
class PreparationConfig:
    """Configuration for full Top-40 gold candidate materialization."""

    source_root: Path
    output_dir: Path
    validation_fraction: float = 0.01
    seed: int = 42
    shard_size: int = 100_000
    batch_size: int = 2_048
    dataset_ids: tuple[str, ...] = CANDIDATE_DATASETS


@dataclass
class PreparationStats:
    """Mutable counters collected during materialization."""

    written_by_split: Counter[str] = field(default_factory=Counter)
    written_by_dataset: Counter[str] = field(default_factory=Counter)
    duplicate_by_split: Counter[str] = field(default_factory=Counter)
    malformed_by_dataset: Counter[str] = field(default_factory=Counter)
    system_messages_removed: int = 0
    bom_characters_removed: int = 0


class ExactDeduplicator:
    """Disk-backed exact conversation deduplicator."""

    def __init__(self, database_path: Path) -> None:
        self._connection = sqlite3.connect(database_path)
        self._connection.execute("PRAGMA journal_mode=OFF")
        self._connection.execute("PRAGMA synchronous=OFF")
        self._connection.execute("CREATE TABLE seen (digest BLOB PRIMARY KEY) WITHOUT ROWID")
        self._pending = 0

    def add(self, digest: bytes) -> bool:
        """Record a digest and return whether it had not been seen before."""
        cursor = self._connection.execute("INSERT OR IGNORE INTO seen (digest) VALUES (?)", (digest,))
        self._pending += 1
        if self._pending >= 10_000:
            self._connection.commit()
            self._pending = 0
        return cursor.rowcount == 1

    def close(self) -> None:
        """Commit pending entries and close the database."""
        self._connection.commit()
        self._connection.close()


class DatasetShardWriter:
    """Write bounded per-dataset JSONL shards and build meta manifests."""

    def __init__(self, output_dir: Path, shard_size: int) -> None:
        self._output_dir = output_dir
        self._shard_size = shard_size
        self._states: dict[tuple[str, str], tuple[int, int]] = {}
        self._handles: dict[tuple[str, str], object] = {}
        self.manifests: dict[str, dict[str, dict[str, object]]] = {"train": {}, "validation": {}}

    def write(self, split: str, dataset_id: str, record: dict[str, object]) -> None:
        """Write one record to its dataset and output split."""
        key = (split, dataset_id)
        shard_index, records_in_shard = self._states.get(key, (0, 0))
        if records_in_shard >= self._shard_size:
            self._handles.pop(key).close()
            shard_index += 1
            records_in_shard = 0

        relative_path = Path("processed") / split / f"{dataset_id}-{shard_index:05d}.jsonl"
        destination = self._output_dir / relative_path
        if records_in_shard == 0:
            destination.parent.mkdir(parents=True, exist_ok=True)
            manifest_name = f"{dataset_id}-{shard_index:05d}"
            self.manifests[split][manifest_name] = {
                "file_name": relative_path.as_posix(),
                "columns": {"messages": "messages"},
                "sample_ratio": 1.0,
            }
            self._handles[key] = destination.open("w", encoding="utf-8")

        self._handles[key].write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self._states[key] = (shard_index, records_in_shard + 1)

    def close(self) -> None:
        """Close all open shard handles."""
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _fraction(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed < 1.0:
        raise argparse.ArgumentTypeError("value must be between zero and one")
    return parsed


def _require_read_only_mount(source_root: Path) -> None:
    if not os.statvfs(source_root).f_flag & os.ST_RDONLY:
        raise RuntimeError(f"Source root must be on a read-only filesystem: {source_root}")


def _manifest(source_root: Path, dataset_id: str) -> dict[str, object]:
    manifest_path = source_root / dataset_id / "manifest.json"
    with manifest_path.open(encoding="utf-8") as manifest_file:
        return json.load(manifest_file)


def _split_shards(source_root: Path, dataset_id: str, source_split: str) -> list[Path]:
    manifest = _manifest(source_root, dataset_id)
    shards = [
        source_root / dataset_id / source_split / shard["filename"]
        for shard in manifest.get("shards", [])
        if shard.get("split") == source_split
    ]
    if not shards:
        raise ValueError(f"No {source_split!r} shards found for {dataset_id}")
    return sorted(shards)


def _iter_parquet_rows(shards: list[Path], batch_size: int):
    for shard_path in shards:
        parquet_file = pq.ParquetFile(shard_path)
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            yield from batch.to_pylist()


def _normalize_messages(raw_messages: object) -> tuple[list[dict[str, str]], int, int]:
    if isinstance(raw_messages, str):
        raw_messages = json.loads(raw_messages)
    if not isinstance(raw_messages, list):
        raise TypeError("messages must be a list or JSON-encoded list")

    messages: list[dict[str, str]] = []
    system_messages_removed = 0
    bom_characters_removed = 0
    for message in raw_messages:
        if not isinstance(message, dict):
            raise TypeError("each message must be an object")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise TypeError("each message requires string role and content")
        bom_characters_removed += content.count("\ufeff")
        content = content.replace("\ufeff", "").strip()
        if not content:
            raise ValueError("message content must not be empty")
        if role == "system":
            system_messages_removed += 1
        elif role in {"user", "assistant"}:
            messages.append({"role": role, "content": content})
        else:
            raise ValueError(f"unsupported message role {role!r}")

    roles = [message["role"] for message in messages]
    expected_roles = ["user" if index % 2 == 0 else "assistant" for index in range(len(messages))]
    if not roles or roles != expected_roles or roles[-1] != "assistant":
        raise ValueError(f"messages must alternate user/assistant and end with assistant; got {roles}")
    return messages, system_messages_removed, bom_characters_removed


def _conversation_digest(messages: list[dict[str, str]]) -> bytes:
    serialized = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.blake2b(serialized.encode("utf-8"), digest_size=16).digest()


def _is_derived_validation(digest: bytes, validation_fraction: float, seed: int) -> bool:
    person = f"validation:{seed}".encode("utf-8")[:16]
    split_digest = hashlib.blake2b(digest, digest_size=8, person=person).digest()
    return int.from_bytes(split_digest, byteorder="big") / 2**64 < validation_fraction


def _prepare_record(
    raw_record: dict[str, object], dataset_id: str, output_split: str
) -> tuple[dict[str, object], bytes, int, int]:
    messages, systems_removed, boms_removed = _normalize_messages(raw_record.get("messages"))
    prepared = {key: value for key, value in raw_record.items() if key != "messages"}
    prepared.update(
        {
            "messages": messages,
            "source_dataset": str(raw_record.get("source_dataset") or raw_record.get("dataset_id") or dataset_id),
            "prepared_split": output_split,
        }
    )
    return prepared, _conversation_digest(messages), systems_removed, boms_removed


def _process_source_split(
    config: PreparationConfig,
    dataset_id: str,
    source_split: str,
    output_split: str,
    writer: DatasetShardWriter,
    deduplicator: ExactDeduplicator,
    stats: PreparationStats,
    derived_validation: bool | None = None,
) -> None:
    shards = _split_shards(config.source_root, dataset_id, source_split)
    for raw_record in _iter_parquet_rows(shards, config.batch_size):
        try:
            prepared, digest, systems_removed, boms_removed = _prepare_record(raw_record, dataset_id, output_split)
        except (json.JSONDecodeError, TypeError, ValueError):
            stats.malformed_by_dataset[dataset_id] += 1
            continue

        if derived_validation is not None:
            is_validation = _is_derived_validation(digest, config.validation_fraction, config.seed)
            if is_validation != derived_validation:
                continue
        if not deduplicator.add(digest):
            stats.duplicate_by_split[output_split] += 1
            continue

        writer.write(output_split, dataset_id, prepared)
        stats.written_by_split[output_split] += 1
        stats.written_by_dataset[dataset_id] += 1
        stats.system_messages_removed += systems_removed
        stats.bom_characters_removed += boms_removed


def prepare_dataset(config: PreparationConfig) -> dict[str, object]:
    """Materialize full candidate train and validation data from read-only Parquet sources.

    Args:
        config: Source, destination, split, and sharding settings.

    Returns:
        JSON-serializable preparation summary.
    """
    if not PYARROW_AVAILABLE:
        raise RuntimeError("pyarrow is required to prepare Top-40 gold data")
    config = PreparationConfig(
        source_root=config.source_root.resolve(),
        output_dir=config.output_dir.resolve(),
        validation_fraction=config.validation_fraction,
        seed=config.seed,
        shard_size=config.shard_size,
        batch_size=config.batch_size,
        dataset_ids=config.dataset_ids,
    )
    _require_read_only_mount(config.source_root)
    if config.output_dir.exists() and any(config.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {config.output_dir}")
    if config.output_dir == config.source_root or config.output_dir.is_relative_to(config.source_root):
        raise ValueError("Output directory must not be inside the read-only source mount")
    unknown = set(config.dataset_ids) - set(VALIDATION_SOURCE_SPLITS)
    if unknown:
        raise ValueError(f"Unknown candidate dataset IDs: {sorted(unknown)}")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    database_path = config.output_dir / ".dedup.sqlite3"
    writer = DatasetShardWriter(config.output_dir, config.shard_size)
    deduplicator = ExactDeduplicator(database_path)
    stats = PreparationStats()
    try:
        for dataset_id in config.dataset_ids:
            validation_source = VALIDATION_SOURCE_SPLITS[dataset_id]
            if validation_source is None:
                _process_source_split(
                    config,
                    dataset_id,
                    "train",
                    "validation",
                    writer,
                    deduplicator,
                    stats,
                    derived_validation=True,
                )
            else:
                _process_source_split(
                    config,
                    dataset_id,
                    validation_source,
                    "validation",
                    writer,
                    deduplicator,
                    stats,
                )

        for dataset_id in config.dataset_ids:
            derived_validation = False if VALIDATION_SOURCE_SPLITS[dataset_id] is None else None
            _process_source_split(
                config,
                dataset_id,
                "train",
                "train",
                writer,
                deduplicator,
                stats,
                derived_validation=derived_validation,
            )
    finally:
        writer.close()
        deduplicator.close()
        database_path.unlink(missing_ok=True)

    if not stats.written_by_split["train"] or not stats.written_by_split["validation"]:
        raise ValueError("Both train and validation splits must contain records")
    for split, manifest in writer.manifests.items():
        (config.output_dir / f"{split}_meta.json").write_text(
            json.dumps(dict(sorted(manifest.items())), indent=2) + "\n", encoding="utf-8"
        )

    source_manifests = {}
    for dataset_id in config.dataset_ids:
        manifest_path = config.source_root / dataset_id / "manifest.json"
        source_manifests[dataset_id] = {
            "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "splits": _manifest(config.source_root, dataset_id).get("splits", {}),
            "validation_source_split": VALIDATION_SOURCE_SPLITS[dataset_id],
        }
    summary = {
        "source_root": str(config.source_root),
        "source_access": "read-only mount verified with statvfs ST_RDONLY",
        "output_dir": str(config.output_dir),
        "dataset_ids": list(config.dataset_ids),
        "validation_fraction_for_unsplit_sources": config.validation_fraction,
        "seed": config.seed,
        "shard_size": config.shard_size,
        "test_splits_materialized": False,
        "reserved_datasets_materialized": False,
        "stats": {
            "written_by_split": dict(sorted(stats.written_by_split.items())),
            "written_by_dataset": dict(sorted(stats.written_by_dataset.items())),
            "duplicate_by_split": dict(sorted(stats.duplicate_by_split.items())),
            "malformed_by_dataset": dict(sorted(stats.malformed_by_dataset.items())),
            "system_messages_removed": stats.system_messages_removed,
            "bom_characters_removed": stats.bom_characters_removed,
        },
        "source_manifests": source_manifests,
    }
    (config.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    LOGGER.info(
        "Prepared %d train and %d validation records in %s",
        stats.written_by_split["train"],
        stats.written_by_split["validation"],
        config.output_dir,
    )
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare full Top-40 gold candidate data for Gemma 4.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=_fraction, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shard-size", type=_positive_int, default=100_000)
    parser.add_argument("--batch-size", type=_positive_int, default=2_048)
    return parser


def main() -> int:
    """Prepare full candidate data from the read-only bucket mount."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = _build_parser().parse_args()
    prepare_dataset(
        PreparationConfig(
            source_root=args.source_root,
            output_dir=args.output_dir,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
            shard_size=args.shard_size,
            batch_size=args.batch_size,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
