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
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from nemo_automodel.shared.import_utils import safe_import, safe_import_from
from tools.benchmark_contamination import FRAGMENT_TOKENS, BenchmarkBlocklist, canonicalize_text

LOGGER = logging.getLogger(__name__)
PYARROW_AVAILABLE, pq = safe_import("pyarrow.parquet")
TRANSFORMERS_AVAILABLE, AutoTokenizer = safe_import_from("transformers", "AutoTokenizer")

# Exact generic prompt used by the bucket preparation pass. Gemma 4 chat
# serialization is applied later by the processor, so this prompt is removed.
LEGACY_SYSTEM_PROMPT = (
    "You are afirka, a helpful multilingual assistant fluent in African languages.\n"
    "Always respond in the language the user wrote in unless explicitly asked to translate."
)
APPROVED_LANGUAGES = frozenset(
    {
        "amh",
        "bam",
        "eng",
        "fon",
        "fra",
        "hau",
        "ibo",
        "kin",
        "lin",
        "lug",
        "nya",
        "pcm",
        "plt",
        "run",
        "sna",
        "som",
        "swh",
        "tso",
        "tsn",
        "twi",
        "wol",
        "xho",
        "yor",
        "zul",
    }
)
LANGUAGE_ALIASES = {"swa": "swh"}
DATASET_TASKS = {
    "aya_dataset": "instruction",
    "afrisenti": "classification",
    "mafand_mt": "translation",
    "masakhaner1": "ner",
    "masakhaner2": "ner",
    "masakhanews": "topic_classification",
    "masakhapos": "pos",
    "menyo20k_mt": "translation",
    "naijasenti": "classification",
}
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

# Earlier sources own exact duplicates. Prefer primary focused datasets over
# aggregate families, and the newer MasakhaNER release over its predecessor.
CANDIDATE_DATASETS = (
    "aya_dataset",
    "naijasenti",
    "afrisenti",
    "menyo20k_mt",
    "mafand_mt",
    "masakhaner2",
    "masakhaner1",
    "masakhanews",
    "masakhapos",
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


class ChatTokenizer(Protocol):
    """Tokenizer contract required for template-aware length filtering."""

    model_max_length: int

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> object:
        """Render one conversation and optionally return its token IDs."""
        ...


@dataclass(frozen=True)
class PreparationConfig:
    """Configuration for full Top-40 gold candidate materialization."""

    source_root: Path
    output_dir: Path
    validation_fraction: float = 0.01
    seed: int = 42
    shard_size: int = 100_000
    batch_size: int = 2_048
    tokenizer_name_or_path: str = "google/gemma-4-E2B-it"
    max_seq_length: int = 4_096
    dataset_ids: tuple[str, ...] = CANDIDATE_DATASETS
    dataset_tasks: tuple[tuple[str, str], ...] = tuple(DATASET_TASKS.items())
    approved_languages: frozenset[str] = APPROVED_LANGUAGES
    validation_source_splits: tuple[tuple[str, str | None], ...] = tuple(VALIDATION_SOURCE_SPLITS.items())
    max_train_records_per_dataset: int | None = None
    max_validation_records_per_dataset: int | None = None


@dataclass
class PreparationStats:
    """Mutable counters collected during materialization."""

    written_by_split: Counter[str] = field(default_factory=Counter)
    written_by_dataset: Counter[str] = field(default_factory=Counter)
    duplicate_by_split: Counter[str] = field(default_factory=Counter)
    malformed_by_dataset: Counter[str] = field(default_factory=Counter)
    rejected_by_reason: Counter[str] = field(default_factory=Counter)
    contaminated_by_benchmark: Counter[str] = field(default_factory=Counter)
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


def _raw_messages(raw_messages: object) -> list[dict[str, object]]:
    if isinstance(raw_messages, str):
        raw_messages = json.loads(raw_messages)
    if not isinstance(raw_messages, list) or any(not isinstance(message, dict) for message in raw_messages):
        raise ValueError("messages_not_list")
    return raw_messages


def _build_reserved_blocklist(
    source_root: Path,
    database_path: Path,
    batch_size: int,
) -> tuple[list[str], dict[str, int]]:
    missing = []
    fields_by_dataset: Counter[str] = Counter()
    with BenchmarkBlocklist(database_path) as blocklist:
        for dataset_id in RESERVED_DATASETS:
            manifest_path = source_root / dataset_id / "manifest.json"
            if not manifest_path.is_file():
                missing.append(dataset_id)
                continue
            manifest = _manifest(source_root, dataset_id)
            for shard in manifest.get("shards", []):
                source_split = shard.get("split")
                filename = shard.get("filename")
                if not isinstance(source_split, str) or not isinstance(filename, str):
                    raise ValueError(f"Invalid reserved shard entry for {dataset_id}: {shard}")
                path = source_root / dataset_id / source_split / filename
                for raw_record in _iter_parquet_rows([path], batch_size):
                    try:
                        messages = _raw_messages(raw_record.get("messages"))
                    except (json.JSONDecodeError, ValueError):
                        continue
                    for message in messages:
                        role = message.get("role")
                        content = message.get("content")
                        if role == "system" or not isinstance(role, str) or not isinstance(content, str):
                            continue
                        if len(canonicalize_text(content).split()) < FRAGMENT_TOKENS:
                            continue
                        if blocklist.add_text(content, benchmark=f"{dataset_id}:{source_split}", field=role):
                            fields_by_dataset[dataset_id] += 1
        blocklist.set_metadata("source_root", str(source_root))
        blocklist.set_metadata("missing_reserved_datasets", ",".join(missing))
    return missing, dict(sorted(fields_by_dataset.items()))


def _normalize_messages(raw_messages: object) -> tuple[list[dict[str, str]], int, int]:
    try:
        raw_messages = _raw_messages(raw_messages)
    except json.JSONDecodeError as exc:
        raise ValueError("invalid_messages_json") from exc

    messages: list[dict[str, str]] = []
    system_messages_removed = 0
    bom_characters_removed = 0
    for message in raw_messages:
        if not isinstance(message, dict):
            raise ValueError("message_not_mapping")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError("invalid_message_fields")
        bom_characters_removed += content.count("\ufeff")
        content = content.replace("\ufeff", "").strip()
        if not content:
            raise ValueError("empty_message_content")
        if role == "system":
            if content != LEGACY_SYSTEM_PROMPT:
                raise ValueError("unexpected_system_prompt")
            system_messages_removed += 1
        elif role in {"user", "assistant"}:
            messages.append({"role": role, "content": content})
        else:
            raise ValueError("unsupported_message_role")

    roles = [message["role"] for message in messages]
    expected_roles = ["user" if index % 2 == 0 else "assistant" for index in range(len(messages))]
    if not roles or roles != expected_roles or roles[-1] != "assistant":
        raise ValueError("invalid_role_sequence")
    return messages, system_messages_removed, bom_characters_removed


def _conversation_digest(messages: list[dict[str, str]]) -> bytes:
    serialized = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.blake2b(serialized.encode("utf-8"), digest_size=16).digest()


def _token_count(tokenizer: ChatTokenizer, messages: list[dict[str, str]]) -> int:
    tokenized = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    input_ids = tokenized.get("input_ids") if isinstance(tokenized, Mapping) else tokenized
    if not isinstance(input_ids, list) or any(not isinstance(token_id, int) for token_id in input_ids):
        raise TypeError("chat template must return a flat input_ids list for one conversation")
    return len(input_ids)


def _is_derived_validation(digest: bytes, validation_fraction: float, seed: int) -> bool:
    person = f"validation:{seed}".encode("utf-8")[:16]
    split_digest = hashlib.blake2b(digest, digest_size=8, person=person).digest()
    return int.from_bytes(split_digest, byteorder="big") / 2**64 < validation_fraction


def _prepare_record(
    raw_record: dict[str, object],
    dataset_id: str,
    source_split: str,
    output_split: str,
    tokenizer: ChatTokenizer,
    max_seq_length: int,
    expected_task: str,
    approved_languages: frozenset[str],
) -> tuple[dict[str, object], bytes, int, int]:
    messages, systems_removed, boms_removed = _normalize_messages(raw_record.get("messages"))
    raw_language = raw_record.get("language", raw_record.get("lang"))
    if not isinstance(raw_language, str):
        raise ValueError("missing_language")
    language = LANGUAGE_ALIASES.get(raw_language.strip().casefold(), raw_language.strip().casefold())
    if language not in approved_languages:
        raise ValueError("language_outside_report")

    task = raw_record.get("task") or expected_task
    if task != expected_task:
        raise ValueError("unexpected_task")

    # Avoid sending clearly corrupt multi-megabyte rows through the tokenizer.
    if sum(len(message["content"]) for message in messages) > max_seq_length * 32:
        raise ValueError("overlength_sequence")
    text_tokens = _token_count(tokenizer, messages)
    if text_tokens > max_seq_length:
        raise ValueError("overlength_sequence")

    prepared = {
        key: value
        for key, value in raw_record.items()
        if key not in {"dataset_id", "language", "lang", "messages", "source", "source_dataset", "source_split", "task"}
    }
    prepared.update(
        {
            "messages": messages,
            "lang": language,
            "task": expected_task,
            "source": dataset_id,
            "source_dataset": dataset_id,
            "source_split": source_split,
            "prepared_split": output_split,
            "_text_tokens": text_tokens,
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
    tokenizer: ChatTokenizer,
    benchmark_blocklist: BenchmarkBlocklist,
    expected_task: str,
    approved_languages: frozenset[str],
    max_records: int | None = None,
    derived_validation: bool | None = None,
) -> None:
    shards = _split_shards(config.source_root, dataset_id, source_split)
    written_for_dataset = 0
    for raw_record in _iter_parquet_rows(shards, config.batch_size):
        try:
            prepared, digest, systems_removed, boms_removed = _prepare_record(
                raw_record,
                dataset_id,
                source_split,
                output_split,
                tokenizer,
                config.max_seq_length,
                expected_task,
                approved_languages,
            )
        except (TypeError, ValueError) as exc:
            stats.malformed_by_dataset[dataset_id] += 1
            stats.rejected_by_reason[str(exc)] += 1
            continue

        if derived_validation is not None:
            is_validation = _is_derived_validation(digest, config.validation_fraction, config.seed)
            if is_validation != derived_validation:
                continue
        substantive_texts = [
            message["content"]
            for message in prepared["messages"]
            if len(canonicalize_text(message["content"]).split()) >= FRAGMENT_TOKENS
        ]
        match = benchmark_blocklist.find_match(substantive_texts)
        if match is not None:
            stats.contaminated_by_benchmark[match.benchmark] += 1
            continue
        if not deduplicator.add(digest):
            stats.duplicate_by_split[output_split] += 1
            continue

        writer.write(output_split, dataset_id, prepared)
        written_for_dataset += 1
        stats.written_by_split[output_split] += 1
        stats.written_by_dataset[dataset_id] += 1
        stats.system_messages_removed += systems_removed
        stats.bom_characters_removed += boms_removed
        if max_records is not None and written_for_dataset >= max_records:
            break


def prepare_dataset(config: PreparationConfig, *, tokenizer: ChatTokenizer | None = None) -> dict[str, object]:
    """Materialize full candidate train and validation data from read-only Parquet sources.

    Args:
        config: Source, destination, split, and sharding settings.

    Returns:
        JSON-serializable preparation summary.
    """
    if not PYARROW_AVAILABLE:
        raise RuntimeError("pyarrow is required to prepare Top-40 gold data")
    if tokenizer is None:
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers is required to prepare Top-40 gold data")
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name_or_path)
    config = PreparationConfig(
        source_root=config.source_root.resolve(),
        output_dir=config.output_dir.resolve(),
        validation_fraction=config.validation_fraction,
        seed=config.seed,
        shard_size=config.shard_size,
        batch_size=config.batch_size,
        tokenizer_name_or_path=config.tokenizer_name_or_path,
        max_seq_length=config.max_seq_length,
        dataset_ids=config.dataset_ids,
        dataset_tasks=config.dataset_tasks,
        approved_languages=config.approved_languages,
        validation_source_splits=config.validation_source_splits,
        max_train_records_per_dataset=config.max_train_records_per_dataset,
        max_validation_records_per_dataset=config.max_validation_records_per_dataset,
    )
    _require_read_only_mount(config.source_root)
    if config.output_dir.exists() and any(config.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {config.output_dir}")
    if config.output_dir == config.source_root or config.output_dir.is_relative_to(config.source_root):
        raise ValueError("Output directory must not be inside the read-only source mount")
    dataset_tasks = dict(config.dataset_tasks)
    validation_source_splits = dict(config.validation_source_splits)
    unknown = set(config.dataset_ids) - set(dataset_tasks) | (set(config.dataset_ids) - set(validation_source_splits))
    if unknown:
        raise ValueError(f"Unknown candidate dataset IDs: {sorted(unknown)}")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    database_path = config.output_dir / ".dedup.sqlite3"
    benchmark_path = config.output_dir / "benchmark_blocklist.sqlite3"
    missing_reserved_datasets, reserved_fields_by_dataset = _build_reserved_blocklist(
        config.source_root,
        benchmark_path,
        config.batch_size,
    )
    writer = DatasetShardWriter(config.output_dir, config.shard_size)
    deduplicator = ExactDeduplicator(database_path)
    benchmark_blocklist = BenchmarkBlocklist(benchmark_path, mode="read-only")
    stats = PreparationStats()
    try:
        for dataset_id in config.dataset_ids:
            validation_source = validation_source_splits[dataset_id]
            if validation_source is None:
                _process_source_split(
                    config,
                    dataset_id,
                    "train",
                    "validation",
                    writer,
                    deduplicator,
                    stats,
                    tokenizer,
                    benchmark_blocklist,
                    dataset_tasks[dataset_id],
                    config.approved_languages,
                    max_records=config.max_validation_records_per_dataset,
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
                    tokenizer,
                    benchmark_blocklist,
                    dataset_tasks[dataset_id],
                    config.approved_languages,
                    max_records=config.max_validation_records_per_dataset,
                )

        for dataset_id in config.dataset_ids:
            derived_validation = False if validation_source_splits[dataset_id] is None else None
            _process_source_split(
                config,
                dataset_id,
                "train",
                "train",
                writer,
                deduplicator,
                stats,
                tokenizer,
                benchmark_blocklist,
                dataset_tasks[dataset_id],
                config.approved_languages,
                max_records=config.max_train_records_per_dataset,
                derived_validation=derived_validation,
            )
    finally:
        writer.close()
        deduplicator.close()
        benchmark_blocklist.close()
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
            "validation_source_split": validation_source_splits[dataset_id],
            "task": dataset_tasks[dataset_id],
        }
    summary = {
        "source_root": str(config.source_root),
        "source_access": "read-only mount verified with statvfs ST_RDONLY",
        "output_dir": str(config.output_dir),
        "dataset_ids": list(config.dataset_ids),
        "validation_fraction_for_unsplit_sources": config.validation_fraction,
        "seed": config.seed,
        "shard_size": config.shard_size,
        "tokenizer_name_or_path": config.tokenizer_name_or_path,
        "max_seq_length": config.max_seq_length,
        "approved_languages": sorted(config.approved_languages),
        "max_train_records_per_dataset": config.max_train_records_per_dataset,
        "max_validation_records_per_dataset": config.max_validation_records_per_dataset,
        "test_splits_materialized": False,
        "reserved_datasets_materialized": False,
        "missing_reserved_datasets_for_contamination_audit": missing_reserved_datasets,
        "reserved_blocklist_fields_by_dataset": reserved_fields_by_dataset,
        "stats": {
            "written_by_split": dict(sorted(stats.written_by_split.items())),
            "written_by_dataset": dict(sorted(stats.written_by_dataset.items())),
            "duplicate_by_split": dict(sorted(stats.duplicate_by_split.items())),
            "malformed_by_dataset": dict(sorted(stats.malformed_by_dataset.items())),
            "rejected_by_reason": dict(sorted(stats.rejected_by_reason.items())),
            "contaminated_by_benchmark": dict(sorted(stats.contaminated_by_benchmark.items())),
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
    parser.add_argument("--tokenizer-name-or-path", default="google/gemma-4-E2B-it")
    parser.add_argument("--max-seq-length", type=_positive_int, default=4_096)
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
            tokenizer_name_or_path=args.tokenizer_name_or_path,
            max_seq_length=args.max_seq_length,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
