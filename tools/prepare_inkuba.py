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

"""Profile and prepare a benchmark-safe, balanced subset of Inkuba-Instruct."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sqlite3
import time
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

if __package__:
    from tools.benchmark_contamination import BenchmarkBlocklist, canonicalize_text
else:
    from benchmark_contamination import BenchmarkBlocklist, canonicalize_text

logger = logging.getLogger(__name__)

INKUBA_DATASET_ID = "lelapa/Inkuba-instruct"
INKUBA_REVISION = "a417af0dd9fff92950f037857b0cfa6980cfc743"
LANGUAGE_SPLITS = {
    "hau": "hausa",
    "swa": "swahili",
    "xho": "xhosa",
    "yor": "yoruba",
    "zul": "isizulu",
}
TASK_WEIGHTS = {
    "mmt": 0.50,
    "sentiment": 0.15,
    "topic": 0.15,
    "ner": 0.10,
    "pos": 0.10,
}
TASK_ALIASES = {
    "mmt": "mmt",
    "machine translation": "mmt",
    "translation": "mmt",
    "sentiment": "sentiment",
    "sentiment analysis": "sentiment",
    "ner": "ner",
    "named entity recognition": "ner",
    "pos": "pos",
    "part of speech": "pos",
    "parts of speech": "pos",
    "topic": "topic",
    "topic classification": "topic",
    "news classification": "topic",
}
BLOCKED_SOURCE_MARKERS = ("afriqa", "sib200", "sib 200", "sib-200", "sib_200")


@dataclass(frozen=True)
class InkubaRecord:
    """Normalized Inkuba instruction record with provenance."""

    instruction: str
    inputs: str
    target: str
    language: str
    task: str
    source: str
    source_split: str

    @property
    def user_text(self) -> str:
        """Return the user turn while preserving the upstream field boundary."""
        return f"{self.instruction}\n\n{self.inputs}" if self.inputs else self.instruction


@dataclass
class PreparationStats:
    """Track accepted and rejected Inkuba records."""

    input_records: int = 0
    written_records: int = 0
    written_tokens: int = 0
    rejections: Counter[str] = field(default_factory=Counter)
    by_language: Counter[str] = field(default_factory=Counter)
    by_task: Counter[str] = field(default_factory=Counter)
    by_source: Counter[str] = field(default_factory=Counter)
    by_contamination_source: Counter[str] = field(default_factory=Counter)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable preparation statistics."""
        return {
            "input_records": self.input_records,
            "written_records": self.written_records,
            "written_tokens": self.written_tokens,
            "rejections": dict(sorted(self.rejections.items())),
            "by_language": dict(sorted(self.by_language.items())),
            "by_task": dict(sorted(self.by_task.items())),
            "by_source": dict(sorted(self.by_source.items())),
            "by_contamination_source": dict(sorted(self.by_contamination_source.items())),
        }


class _ExactDeduplicator:
    def __init__(self, path: Path) -> None:
        self._connection = sqlite3.connect(path)
        self._connection.execute("PRAGMA journal_mode=OFF")
        self._connection.execute("PRAGMA synchronous=OFF")
        self._connection.execute("CREATE TABLE seen (digest BLOB PRIMARY KEY) WITHOUT ROWID")

    def add(self, record: InkubaRecord) -> bool:
        identity = f"{canonicalize_text(record.user_text)}\x1f{canonicalize_text(record.target)}"
        digest = hashlib.blake2b(identity.encode("utf-8"), digest_size=20).digest()
        cursor = self._connection.execute("INSERT OR IGNORE INTO seen (digest) VALUES (?)", (digest,))
        return cursor.rowcount == 1

    def close(self) -> None:
        self._connection.close()


class _ShardWriter:
    def __init__(self, output_dir: Path, shard_size: int) -> None:
        self._output_dir = output_dir
        self._shard_size = shard_size
        self._states: dict[tuple[str, str, str], tuple[int, int]] = {}
        self._handles: dict[Path, TextIO] = {}
        self.manifests: dict[str, dict[str, dict[str, Any]]] = {"train": {}, "validation": {}}

    def write(self, split: str, record: InkubaRecord, text_tokens: int) -> None:
        key = (split, record.language, record.task)
        shard_index, records_in_shard = self._states.get(key, (0, 0))
        if records_in_shard >= self._shard_size:
            shard_index += 1
            records_in_shard = 0
        partition = f"{record.language}__{record.task}"
        relative_path = Path("processed") / split / f"{partition}-{shard_index:05d}.jsonl"
        path = self._output_dir / relative_path
        if records_in_shard == 0:
            path.parent.mkdir(parents=True, exist_ok=True)
            name = f"inkuba__{partition}-{shard_index:05d}"
            self.manifests[split][name] = {
                "file_name": relative_path.as_posix(),
                "columns": {"messages": "messages"},
                "sample_ratio": 1.0,
            }
        handle = self._handles.get(path)
        if handle is None:
            handle = path.open("a", encoding="utf-8")
            self._handles[path] = handle
        payload = {
            "messages": [
                {"role": "user", "content": record.user_text},
                {"role": "assistant", "content": record.target},
            ],
            "lang": record.language,
            "task": record.task,
            "source": f"inkuba:{record.source}",
            "source_split": record.source_split,
            "_text_tokens": text_tokens,
        }
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._states[key] = (shard_index, records_in_shard + 1)

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()


def _normalize_task(task: str) -> str | None:
    normalized = re.sub(r"[_-]+", " ", task).strip().casefold()
    return TASK_ALIASES.get(normalized)


def _source_is_blocked(source: str) -> bool:
    normalized = re.sub(r"[_-]+", " ", source).casefold()
    return any(marker in normalized for marker in BLOCKED_SOURCE_MARKERS)


def _has_expected_script(text: str) -> bool:
    letters = [character for character in text if character.isalpha()]
    if len(letters) < 10:
        return True
    latin_letters = sum("LATIN" in unicodedata.name(character, "") for character in letters)
    return latin_letters / len(letters) >= 0.80


def normalize_record(
    raw: Mapping[str, Any], *, language: str, source_split: str
) -> tuple[InkubaRecord | None, str | None]:
    """Normalize one upstream row and return a rejection reason when invalid."""
    required = ("instruction", "inputs", "targets", "task", "data_source")
    if any(not isinstance(raw.get(field), str) for field in required):
        return None, "malformed"
    instruction = raw["instruction"].strip()
    inputs = raw["inputs"].strip()
    target = raw["targets"].strip()
    source = raw["data_source"].strip()
    task = _normalize_task(raw["task"])
    if not instruction or not target or not source:
        return None, "empty"
    if task is None:
        return None, "unsupported_task"
    if _source_is_blocked(source):
        return None, "blocked_source"
    if canonicalize_text(target) in {canonicalize_text(instruction), canonicalize_text(inputs)}:
        return None, "copied_target"
    if not _has_expected_script(target):
        return None, "unexpected_script"
    if len(instruction) + len(inputs) > 100_000 or len(target) > 100_000:
        return None, "excessive_length"
    return InkubaRecord(instruction, inputs, target, language, task, source, source_split), None


def _record_fraction(record: InkubaRecord, seed: int, namespace: str) -> float:
    identity = f"{record.language}\x1f{record.task}\x1f{record.source}\x1f{record.user_text}\x1f{record.target}"
    digest = hashlib.blake2b(
        identity.encode("utf-8"),
        digest_size=8,
        person=f"{namespace}:{seed}".encode("utf-8")[:16],
    ).digest()
    return int.from_bytes(digest, "big") / 2**64


def _cell_key(partition: str, record: InkubaRecord) -> str:
    return "|".join((partition, record.language, record.task, record.source.casefold()))


def profile_records(
    records_by_split: Mapping[str, Iterable[Mapping[str, Any]]],
    *,
    token_counter: Callable[[str], int],
    benchmark_blocklist: Path,
    token_sample_fraction: float = 0.001,
    seed: int = 42,
) -> dict[str, Any]:
    """Profile clean Inkuba cells used to calculate deterministic sampling quotas."""
    cells: dict[str, dict[str, int]] = defaultdict(lambda: {"records": 0, "sampled_records": 0, "sampled_tokens": 0})
    stats = PreparationStats()
    started = time.monotonic()
    with BenchmarkBlocklist(benchmark_blocklist, mode="read-only") as blocklist:
        for source_split, rows in records_by_split.items():
            language, partition = source_split.split("_", maxsplit=1)
            for raw in rows:
                stats.input_records += 1
                if stats.input_records % 1_000_000 == 0:
                    elapsed = time.monotonic() - started
                    logger.info(
                        "Profiled %d records at %.0f records/s",
                        stats.input_records,
                        stats.input_records / max(elapsed, 1e-8),
                    )
                record, reason = normalize_record(raw, language=language, source_split=source_split)
                if record is None:
                    stats.rejections[reason or "unknown"] += 1
                    continue
                match = blocklist.find_match([record.instruction, record.inputs, record.target, record.user_text])
                if match is not None:
                    stats.rejections["benchmark_contamination"] += 1
                    stats.by_contamination_source[match.benchmark] += 1
                    continue
                key = _cell_key(partition, record)
                cells[key]["records"] += 1
                if _record_fraction(record, seed, "token-profile") < token_sample_fraction:
                    cells[key]["sampled_records"] += 1
                    cells[key]["sampled_tokens"] += token_counter(f"{record.user_text}\n{record.target}")

    result_cells = {}
    for key, values in sorted(cells.items()):
        sampled_records = values["sampled_records"]
        average_tokens = values["sampled_tokens"] / sampled_records if sampled_records else 0.0
        result_cells[key] = {**values, "average_tokens": average_tokens}
    return {
        "dataset_id": INKUBA_DATASET_ID,
        "revision": INKUBA_REVISION,
        "token_sample_fraction": token_sample_fraction,
        "seed": seed,
        "cells": result_cells,
        "stats": stats.to_dict(),
    }


def _waterfill_budget(total: float, capacities: Mapping[str, float], weights: Mapping[str, float]) -> dict[str, float]:
    allocations = {key: 0.0 for key in capacities}
    active = {key for key, capacity in capacities.items() if capacity > 0 and weights.get(key, 0) > 0}
    remaining = min(total, sum(capacities.values()))
    while active and remaining > 0:
        weight_total = sum(weights[key] for key in active)
        capped = []
        proposals = {key: remaining * weights[key] / weight_total for key in active}
        for key, proposal in proposals.items():
            available = capacities[key] - allocations[key]
            if available <= proposal:
                allocations[key] += available
                remaining -= available
                capped.append(key)
        if capped:
            active.difference_update(capped)
            continue
        for key, proposal in proposals.items():
            allocations[key] += proposal
        remaining = 0
    return allocations


def _cell_token_targets(profile: Mapping[str, Any], total_tokens: int) -> dict[str, int]:
    targets: dict[str, int] = {}
    language_budget = total_tokens / len(LANGUAGE_SPLITS)
    for language in LANGUAGE_SPLITS:
        task_cells: dict[str, dict[str, float]] = defaultdict(dict)
        for key, values in profile["cells"].items():
            partition, cell_language, task, _ = key.split("|", maxsplit=3)
            if partition != "train" or cell_language != language or task not in TASK_WEIGHTS:
                continue
            estimated_tokens = values["records"] * values["average_tokens"]
            if estimated_tokens > 0:
                task_cells[task][key] = estimated_tokens
        task_capacities = {task: sum(cells.values()) for task, cells in task_cells.items()}
        task_allocations = _waterfill_budget(
            language_budget,
            task_capacities,
            {task: TASK_WEIGHTS[task] for task in task_cells},
        )
        for task, task_budget in task_allocations.items():
            source_capacities = task_cells[task]
            source_allocations = _waterfill_budget(
                task_budget,
                source_capacities,
                {key: 1.0 for key in source_capacities},
            )
            targets.update({key: int(tokens) for key, tokens in source_allocations.items()})
    return targets


def _allocate_counts(total: int, weights: Mapping[str, float]) -> dict[str, int]:
    weight_total = sum(weights.values())
    exact = {key: total * weight / weight_total for key, weight in weights.items()}
    allocated = {key: int(value) for key, value in exact.items()}
    remainder = total - sum(allocated.values())
    order = sorted(exact, key=lambda key: (exact[key] - allocated[key], key), reverse=True)
    for key in order[:remainder]:
        allocated[key] += 1
    return allocated


def _cell_validation_targets(profile: Mapping[str, Any], records_per_language: int) -> dict[str, int]:
    targets: dict[str, int] = {}
    for language in LANGUAGE_SPLITS:
        task_sources: dict[str, list[str]] = defaultdict(list)
        for key, values in profile["cells"].items():
            partition, cell_language, task, _ = key.split("|", maxsplit=3)
            if partition == "dev" and cell_language == language and values["records"] > 0 and task in TASK_WEIGHTS:
                task_sources[task].append(key)
        task_counts = _allocate_counts(
            records_per_language,
            {task: TASK_WEIGHTS[task] for task in task_sources},
        )
        for task, source_keys in task_sources.items():
            source_counts = _allocate_counts(task_counts[task], {key: 1.0 for key in source_keys})
            targets.update(source_counts)
    return targets


def prepare_records(
    records_by_split: Mapping[str, Iterable[Mapping[str, Any]]],
    *,
    profile: Mapping[str, Any],
    token_counter: Callable[[str], int],
    benchmark_blocklist: Path,
    output_dir: Path,
    train_token_budget: int,
    validation_records_per_language: int = 500,
    shard_size: int = 100_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Prepare balanced Inkuba shards from a previously generated profile."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = _cell_token_targets(profile, train_token_budget)
    validation_targets = _cell_validation_targets(profile, validation_records_per_language)
    written_by_cell: Counter[str] = Counter()
    validation_by_cell: Counter[str] = Counter()
    stats = PreparationStats()
    writer = _ShardWriter(output_dir, shard_size)
    dedup_path = output_dir / ".dedup.sqlite3"
    deduplicator = _ExactDeduplicator(dedup_path)
    blocklist = BenchmarkBlocklist(benchmark_blocklist, mode="read-only")
    started = time.monotonic()

    try:
        for source_split, rows in records_by_split.items():
            language, partition = source_split.split("_", maxsplit=1)
            for raw in rows:
                stats.input_records += 1
                if stats.input_records % 1_000_000 == 0:
                    elapsed = time.monotonic() - started
                    logger.info(
                        "Prepared scan %d records at %.0f records/s (%d records written)",
                        stats.input_records,
                        stats.input_records / max(elapsed, 1e-8),
                        stats.written_records,
                    )
                record, reason = normalize_record(raw, language=language, source_split=source_split)
                if record is None:
                    stats.rejections[reason or "unknown"] += 1
                    continue
                match = blocklist.find_match([record.instruction, record.inputs, record.target, record.user_text])
                if match is not None:
                    stats.rejections["benchmark_contamination"] += 1
                    stats.by_contamination_source[match.benchmark] += 1
                    continue
                if not deduplicator.add(record):
                    stats.rejections["duplicate"] += 1
                    continue

                key = _cell_key(partition, record)
                if partition == "train":
                    target = targets.get(key, 0)
                    if target == 0 or written_by_cell[key] >= target:
                        stats.rejections["outside_quota"] += 1
                        continue
                    cell = profile["cells"].get(key, {})
                    estimated_tokens = cell.get("records", 0) * cell.get("average_tokens", 0.0)
                    probability = min(1.0, 1.25 * target / estimated_tokens) if estimated_tokens else 1.0
                    if _record_fraction(record, seed, "prepare") >= probability:
                        stats.rejections["sampled_out"] += 1
                        continue
                    output_split = "train"
                else:
                    validation_target = validation_targets.get(key, 0)
                    if validation_target == 0 or validation_by_cell[key] >= validation_target:
                        stats.rejections["outside_quota"] += 1
                        continue
                    output_split = "validation"

                text_tokens = token_counter(f"{record.user_text}\n{record.target}")
                if text_tokens <= 0 or text_tokens > 4096:
                    stats.rejections["token_length"] += 1
                    continue
                writer.write(output_split, record, text_tokens)
                stats.written_records += 1
                stats.written_tokens += text_tokens
                stats.by_language[record.language] += 1
                stats.by_task[record.task] += 1
                stats.by_source[record.source] += 1
                if output_split == "train":
                    written_by_cell[key] += text_tokens
                else:
                    validation_by_cell[key] += 1
    finally:
        writer.close()
        deduplicator.close()
        blocklist.close()
        dedup_path.unlink(missing_ok=True)

    for split, manifest in writer.manifests.items():
        (output_dir / f"{split}_meta.json").write_text(
            json.dumps(dict(sorted(manifest.items())), indent=2) + "\n",
            encoding="utf-8",
        )
    summary = {
        "dataset_id": INKUBA_DATASET_ID,
        "revision": INKUBA_REVISION,
        "train_token_budget": train_token_budget,
        "task_weights": TASK_WEIGHTS,
        "cell_token_targets": dict(sorted(targets.items())),
        "cell_written_tokens": dict(sorted(written_by_cell.items())),
        "validation_records_per_language": validation_records_per_language,
        "validation_cell_targets": dict(sorted(validation_targets.items())),
        "validation_cell_written": dict(sorted(validation_by_cell.items())),
        "stats": stats.to_dict(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _load_inkuba_splits(partition: str) -> dict[str, Iterable[Mapping[str, Any]]]:
    from datasets import load_dataset

    return {
        f"{language}_{partition}": load_dataset(
            INKUBA_DATASET_ID,
            split=f"{split_name}_{partition}",
            revision=INKUBA_REVISION,
            streaming=True,
        )
        for language, split_name in LANGUAGE_SPLITS.items()
    }


def _load_token_counter(model_id: str) -> Callable[[str], int]:
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    tokenizer = getattr(processor, "tokenizer", processor)
    return lambda text: len(tokenizer.encode(text, add_special_tokens=False))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    profile = subparsers.add_parser("profile")
    profile.add_argument("--output", type=Path, required=True)
    profile.add_argument("--benchmark-blocklist", type=Path, required=True)
    profile.add_argument("--model-id", default="google/gemma-4-E2B-it")
    profile.add_argument("--token-sample-fraction", type=float, default=0.001)
    profile.add_argument("--seed", type=int, default=42)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--profile", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--benchmark-blocklist", type=Path, required=True)
    prepare.add_argument("--model-id", default="google/gemma-4-E2B-it")
    prepare.add_argument("--train-token-budget", type=int, default=100_000_000)
    prepare.add_argument("--validation-records-per-language", type=int, default=500)
    prepare.add_argument("--shard-size", type=int, default=100_000)
    prepare.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    """Profile or prepare Inkuba-Instruct from command-line arguments."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = _build_parser().parse_args()
    token_counter = _load_token_counter(args.model_id)
    if args.command == "profile":
        records = {**_load_inkuba_splits("train"), **_load_inkuba_splits("dev")}
        result = profile_records(
            records,
            token_counter=token_counter,
            benchmark_blocklist=args.benchmark_blocklist.resolve(),
            token_sample_fraction=args.token_sample_fraction,
            seed=args.seed,
        )
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    else:
        records = {**_load_inkuba_splits("train"), **_load_inkuba_splits("dev")}
        profile = json.loads(args.profile.read_text(encoding="utf-8"))
        prepare_records(
            records,
            profile=profile,
            token_counter=token_counter,
            benchmark_blocklist=args.benchmark_blocklist.resolve(),
            output_dir=args.output_dir.resolve(),
            train_token_budget=args.train_token_budget,
            validation_records_per_language=args.validation_records_per_language,
            shard_size=args.shard_size,
            seed=args.seed,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
