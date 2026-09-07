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

"""Profile normalized SFT manifests and simulate task-aware mixture policies."""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import logging
import math
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

if __package__:
    from tools.benchmark_contamination import BenchmarkBlocklist, canonicalize_text
else:
    from benchmark_contamination import BenchmarkBlocklist, canonicalize_text

logger = logging.getLogger(__name__)

LENGTH_BOUNDS = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
QUALITY_WEIGHTS = {"human": 1.0, "curated": 0.8, "mixed": 0.5, "silver": 0.25, "synthetic": 0.2}


@dataclass(frozen=True)
class TokenCounts:
    """Token counts for one normalized training record."""

    prompt: int
    label: int
    text: int


@dataclass(frozen=True)
class PoolConfig:
    """Declarative metadata for one prepared source pool."""

    name: str
    manifest: Path
    source_family: str
    revision: str
    license_name: str
    quality_tier: str


@dataclass(frozen=True)
class PlanningConfig:
    """Static controls for allocation simulations."""

    packed_token_budget: int
    language_token_budget: int = 0
    temperature: float = 2.0
    max_epochs: float = 4.0
    task_example_caps: Mapping[str, int] = field(default_factory=dict)
    source_family_caps: Mapping[str, float] = field(default_factory=dict)
    default_source_family_cap: float = 0.5
    fallback_source_families: tuple[str, ...] = ()
    fallback_base_share: float = 0.0
    task_token_shares: Mapping[str, float] = field(default_factory=dict)
    minimum_label_tokens: int = 0
    label_token_planning_margin: int = 0
    sampling_seed: int = 42
    minimum_cell_token_share: float = 0.6


@dataclass(frozen=True)
class MixtureConfig:
    """Configuration for profiling normalized SFT pools."""

    model_id: str
    max_seq_length: int
    languages: tuple[str, ...]
    tasks: tuple[str, ...]
    pools: tuple[PoolConfig, ...]
    planning: PlanningConfig
    fixed_validation_manifest: Path | None = None
    locked_train_manifest: Path | None = None
    locked_train_max_repetitions: int = 4
    reference_train_summary: Path | None = None
    review_samples_per_cell: int = 20


@dataclass(frozen=True)
class LockedTrainStats:
    """Validated identity and token totals for a locked training manifest."""

    digests: frozenset[bytes]
    records: int
    text_tokens: int
    label_tokens: int
    max_repetitions: int


@dataclass
class CellStats:
    """Streaming statistics for one pool, language, task, and source cell."""

    records: int = 0
    prompt_tokens: int = 0
    label_tokens: int = 0
    text_tokens: int = 0
    max_text_tokens: int = 0
    length_histogram: Counter[str] = field(default_factory=Counter)
    target_labels: Counter[str] = field(default_factory=Counter)
    directions: Counter[str] = field(default_factory=Counter)
    entity_types: Counter[str] = field(default_factory=Counter)

    def add(self, record: Mapping[str, Any], counts: TokenCounts) -> None:
        """Add one accepted record to this cell."""
        self.records += 1
        self.prompt_tokens += counts.prompt
        self.label_tokens += counts.label
        self.text_tokens += counts.text
        self.max_text_tokens = max(self.max_text_tokens, counts.text)
        self.length_histogram[_length_bucket(counts.text)] += 1
        task = str(record["task"])
        if task == "classification":
            self.target_labels[_assistant_text(record)] += 1
        direction = record.get("direction")
        if isinstance(direction, str) and direction.strip():
            self.directions[direction.strip()] += 1
        entity_types = record.get("entity_types")
        if isinstance(entity_types, list):
            self.entity_types.update(str(value) for value in entity_types if str(value).strip())

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable exact and derived statistics."""
        return {
            "records": self.records,
            "prompt_tokens": self.prompt_tokens,
            "label_tokens": self.label_tokens,
            "text_tokens": self.text_tokens,
            "average_prompt_tokens": self.prompt_tokens / self.records if self.records else 0.0,
            "average_label_tokens": self.label_tokens / self.records if self.records else 0.0,
            "average_text_tokens": self.text_tokens / self.records if self.records else 0.0,
            "max_text_tokens": self.max_text_tokens,
            "length_histogram": dict(self.length_histogram),
            "target_labels": dict(self.target_labels.most_common()),
            "directions": dict(self.directions.most_common()),
            "entity_types": dict(self.entity_types.most_common()),
        }


class _ReviewSampler:
    def __init__(self, samples_per_cell: int) -> None:
        self._samples_per_cell = samples_per_cell
        self._heaps: dict[str, list[tuple[int, str]]] = defaultdict(list)

    def add(self, cell: str, digest: bytes, record: Mapping[str, Any], counts: TokenCounts) -> None:
        payload = json.dumps(
            {
                "pool": cell.split("|", maxsplit=1)[0],
                "lang": record["lang"],
                "task": record["task"],
                "source": record["source"],
                "messages": record["messages"],
                "_prompt_tokens": counts.prompt,
                "_label_tokens": counts.label,
                "_text_tokens": counts.text,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        score = int.from_bytes(digest[:8], "big")
        heap = self._heaps[cell]
        item = (-score, payload)
        if len(heap) < self._samples_per_cell:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)

    def records(self) -> Iterable[str]:
        for cell in sorted(self._heaps):
            for _, payload in sorted(self._heaps[cell], reverse=True):
                yield payload


def load_config(path: Path) -> MixtureConfig:
    """Load and validate a profiling configuration from YAML."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a mapping")
    if raw.get("version") != 1:
        raise ValueError("Configuration version must be 1")
    base = path.parent
    pool_configs = []
    for value in raw.get("pools", []):
        if not isinstance(value, dict):
            raise ValueError("Each pool must be a mapping")
        manifest = Path(str(value["manifest"]))
        if not manifest.is_absolute():
            manifest = base / manifest
        quality_tier = str(value["quality_tier"])
        if quality_tier not in QUALITY_WEIGHTS:
            raise ValueError(f"Unsupported quality tier: {quality_tier}")
        pool_configs.append(
            PoolConfig(
                name=str(value["name"]),
                manifest=manifest,
                source_family=str(value.get("source_family", value["name"])),
                revision=str(value["revision"]),
                license_name=str(value["license"]),
                quality_tier=quality_tier,
            )
        )
    planning_raw = raw.get("planning", {})
    if not isinstance(planning_raw, dict):
        raise ValueError("planning must be a mapping")
    fixed_validation_value = raw.get("fixed_validation_manifest")
    fixed_validation_manifest = (
        (path.parent / str(fixed_validation_value)).resolve() if fixed_validation_value is not None else None
    )
    locked_train_value = raw.get("locked_train_manifest")
    locked_train_manifest = (
        (path.parent / str(locked_train_value)).resolve() if locked_train_value is not None else None
    )
    reference_summary_value = raw.get("reference_train_summary")
    reference_train_summary = (
        (path.parent / str(reference_summary_value)).resolve() if reference_summary_value is not None else None
    )
    config = MixtureConfig(
        model_id=str(raw.get("model_id", "google/gemma-4-E2B-it")),
        max_seq_length=int(raw.get("max_seq_length", 4096)),
        languages=tuple(str(value) for value in raw.get("languages", [])),
        tasks=tuple(str(value) for value in raw.get("tasks", [])),
        pools=tuple(pool_configs),
        planning=PlanningConfig(
            packed_token_budget=int(planning_raw["packed_token_budget"]),
            language_token_budget=int(planning_raw.get("language_token_budget", 0)),
            temperature=float(planning_raw.get("temperature", 2.0)),
            max_epochs=float(planning_raw.get("max_epochs", 4.0)),
            task_example_caps={
                str(key): int(value) for key, value in planning_raw.get("task_example_caps", {}).items()
            },
            source_family_caps={
                str(key): float(value) for key, value in planning_raw.get("source_family_caps", {}).items()
            },
            default_source_family_cap=float(planning_raw.get("default_source_family_cap", 0.5)),
            fallback_source_families=tuple(str(value) for value in planning_raw.get("fallback_source_families", [])),
            fallback_base_share=float(planning_raw.get("fallback_base_share", 0.0)),
            task_token_shares={
                str(key): float(value) for key, value in planning_raw.get("task_token_shares", {}).items()
            },
            minimum_label_tokens=int(planning_raw.get("minimum_label_tokens", 0)),
            label_token_planning_margin=int(planning_raw.get("label_token_planning_margin", 0)),
            sampling_seed=int(planning_raw.get("sampling_seed", 42)),
            minimum_cell_token_share=float(planning_raw.get("minimum_cell_token_share", 0.6)),
        ),
        fixed_validation_manifest=fixed_validation_manifest,
        locked_train_manifest=locked_train_manifest,
        locked_train_max_repetitions=int(raw.get("locked_train_max_repetitions", 4)),
        reference_train_summary=reference_train_summary,
        review_samples_per_cell=int(raw.get("review_samples_per_cell", 20)),
    )
    _validate_config(config)
    return config


def _validate_config(config: MixtureConfig) -> None:
    if not config.languages or not config.tasks or not config.pools:
        raise ValueError("languages, tasks, and pools must be non-empty")
    if len(set(config.languages)) != len(config.languages) or len(set(config.tasks)) != len(config.tasks):
        raise ValueError("languages and tasks must not contain duplicates")
    if len({pool.name for pool in config.pools}) != len(config.pools):
        raise ValueError("pool names must be unique")
    if config.max_seq_length <= 0 or config.planning.packed_token_budget <= 0:
        raise ValueError("max_seq_length and packed_token_budget must be positive")
    if config.planning.language_token_budget < 0:
        raise ValueError("language_token_budget must be non-negative")
    if config.planning.language_token_budget:
        expected_budget = config.planning.language_token_budget * len(config.languages)
        if config.planning.packed_token_budget != expected_budget:
            raise ValueError(
                f"packed_token_budget must equal language_token_budget * number of languages ({expected_budget})"
            )
        if not config.planning.task_token_shares:
            raise ValueError("language_token_budget requires task_token_shares")
    if config.planning.temperature < 1 or config.planning.max_epochs <= 0:
        raise ValueError("temperature must be at least 1 and max_epochs must be positive")
    if not 0 < config.planning.default_source_family_cap <= 1:
        raise ValueError("default_source_family_cap must be in (0, 1]")
    if any(not 0 < share <= 1 for share in config.planning.source_family_caps.values()):
        raise ValueError("source_family_caps must be in (0, 1]")
    if not 0 <= config.planning.fallback_base_share <= 1:
        raise ValueError("fallback_base_share must be in [0, 1]")
    source_families = {pool.source_family for pool in config.pools}
    unknown_fallbacks = set(config.planning.fallback_source_families) - source_families
    if unknown_fallbacks:
        raise ValueError(f"Unknown fallback source families: {sorted(unknown_fallbacks)}")
    unknown_tasks = set(config.planning.task_token_shares) - set(config.tasks)
    if unknown_tasks:
        raise ValueError(f"Unknown task_token_shares tasks: {sorted(unknown_tasks)}")
    if config.planning.task_token_shares and not math.isclose(
        sum(config.planning.task_token_shares.values()), 1.0, abs_tol=1e-9
    ):
        raise ValueError("task_token_shares must sum to 1")
    if any(share <= 0 for share in config.planning.task_token_shares.values()):
        raise ValueError("task_token_shares must be positive")
    if config.planning.minimum_label_tokens < 0:
        raise ValueError("minimum_label_tokens must be non-negative")
    if config.planning.label_token_planning_margin < 0:
        raise ValueError("label_token_planning_margin must be non-negative")
    if config.locked_train_max_repetitions <= 0:
        raise ValueError("locked_train_max_repetitions must be positive")
    if not 0 < config.planning.minimum_cell_token_share <= 1:
        raise ValueError("minimum_cell_token_share must be in (0, 1]")


def _iter_manifest_records(manifest_path: Path) -> Iterable[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"Manifest must be a mapping: {manifest_path}")
    for entry in manifest.values():
        path = Path(entry["file_name"])
        if not path.is_absolute():
            path = manifest_path.parent / path
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def _normalize_record(raw: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    messages = raw.get("messages")
    if not isinstance(messages, list) or not messages:
        return None, "malformed_messages"
    normalized_messages = []
    for message in messages:
        if not isinstance(message, dict) or not isinstance(message.get("role"), str):
            return None, "malformed_messages"
        if not isinstance(message.get("content"), str) or not message["content"].strip():
            return None, "malformed_messages"
        normalized_messages.append({"role": message["role"].strip(), "content": message["content"].strip()})
    if not any(message["role"] == "assistant" for message in normalized_messages):
        return None, "missing_assistant"
    values = {field: raw.get(field) for field in ("lang", "task", "source")}
    if any(not isinstance(value, str) or not value.strip() for value in values.values()):
        return None, "missing_provenance"
    return {
        **raw,
        "messages": normalized_messages,
        "lang": str(values["lang"]).strip(),
        "task": str(values["task"]).strip(),
        "source": str(values["source"]).strip(),
    }, None


def _record_digest(record: Mapping[str, Any]) -> bytes:
    identity = "\x1e".join(
        f"{message['role']}\x1f{canonicalize_text(message['content'])}" for message in record["messages"]
    )
    return hashlib.blake2b(identity.encode("utf-8"), digest_size=20).digest()


def _fixed_validation_digests(manifest_path: Path | None) -> set[bytes]:
    if manifest_path is None:
        return set()
    digests: set[bytes] = set()
    for raw in _iter_manifest_records(manifest_path):
        record, reason = _normalize_record(raw)
        if record is None:
            raise ValueError(f"Invalid fixed validation record in {manifest_path}: {reason or 'invalid'}")
        digest = _record_digest(record)
        if digest in digests:
            raise ValueError(f"Duplicate fixed validation record in {manifest_path}")
        digests.add(digest)
    return digests


def _locked_train_stats(manifest_path: Path | None) -> LockedTrainStats:
    if manifest_path is None:
        return LockedTrainStats(frozenset(), 0, 0, 0, 0)
    repetitions: Counter[bytes] = Counter()
    token_counts: dict[bytes, tuple[int, int]] = {}
    records = 0
    text_tokens = 0
    label_tokens = 0
    for raw in _iter_manifest_records(manifest_path):
        record, reason = _normalize_record(raw)
        if record is None:
            raise ValueError(f"Invalid locked training record in {manifest_path}: {reason or 'invalid'}")
        text = raw.get("_text_tokens")
        label = raw.get("_label_tokens")
        if not isinstance(text, int) or text <= 0 or not isinstance(label, int) or label <= 0:
            raise ValueError(f"Invalid token metadata in locked training manifest {manifest_path}")
        digest = _record_digest(record)
        previous_counts = token_counts.setdefault(digest, (text, label))
        if previous_counts != (text, label):
            raise ValueError(f"Inconsistent token metadata for a repeated locked record in {manifest_path}")
        repetitions[digest] += 1
        records += 1
        text_tokens += text
        label_tokens += label
    return LockedTrainStats(
        digests=frozenset(repetitions),
        records=records,
        text_tokens=text_tokens,
        label_tokens=label_tokens,
        max_repetitions=max(repetitions.values(), default=0),
    )


def _record_texts(record: Mapping[str, Any]) -> list[str]:
    return [str(message["content"]) for message in record["messages"]]


def _assistant_text(record: Mapping[str, Any]) -> str:
    return "\n".join(
        str(message["content"]).strip() for message in record["messages"] if message["role"] == "assistant"
    )


def _length_bucket(length: int) -> str:
    for bound in LENGTH_BOUNDS:
        if length <= bound:
            return f"<={bound}"
    return ">4096"


def _make_token_counter(model_id: str) -> Callable[[list[dict[str, str]]], TokenCounts]:
    from transformers import AutoProcessor

    from nemo_automodel.components.datasets.llm.formatting_utils import format_chat_template

    processor = AutoProcessor.from_pretrained(model_id)
    tokenizer = getattr(processor, "tokenizer", processor)
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_token_id

    def count(messages: list[dict[str, str]]) -> TokenCounts:
        formatted = format_chat_template(
            tokenizer,
            messages,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            answer_only_loss_mask=True,
        )
        labels = formatted["labels"]
        text_tokens = len(formatted["input_ids"])
        label_tokens = sum(label != -100 for label in labels)
        return TokenCounts(prompt=text_tokens - label_tokens, label=label_tokens, text=text_tokens)

    return count


def profile_mixture(
    config: MixtureConfig,
    *,
    token_counter: Callable[[list[dict[str, str]]], TokenCounts],
    output_dir: Path,
    benchmark_blocklist: Path | None = None,
) -> dict[str, Any]:
    """Profile all configured pools and write reviewable preprocessing artifacts."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    database_path = output_dir / ".profile.sqlite3"
    connection = sqlite3.connect(database_path)
    connection.execute("CREATE TABLE seen (digest BLOB PRIMARY KEY) WITHOUT ROWID")
    blocklist = BenchmarkBlocklist(benchmark_blocklist, mode="read-only") if benchmark_blocklist else None
    fixed_validation_digests = _fixed_validation_digests(config.fixed_validation_manifest)
    locked_train = _locked_train_stats(config.locked_train_manifest)
    if locked_train.digests & fixed_validation_digests:
        raise ValueError("Locked training and fixed validation manifests overlap")
    stats: dict[str, CellStats] = defaultdict(CellStats)
    rejections: Counter[str] = Counter()
    sampler = _ReviewSampler(config.review_samples_per_cell)
    processed_records = 0
    try:
        ordered_pools = sorted(config.pools, key=lambda pool: (-QUALITY_WEIGHTS[pool.quality_tier], pool.name))
        for pool in ordered_pools:
            logger.info("Profiling %s", pool.name)
            for raw in _iter_manifest_records(pool.manifest):
                processed_records += 1
                if processed_records % 100_000 == 0:
                    logger.info("Profiled %d source records", processed_records)
                record, reason = _normalize_record(raw)
                if record is None:
                    rejections[reason or "invalid"] += 1
                    continue
                if record["lang"] not in config.languages or record["task"] not in config.tasks:
                    rejections["outside_scope"] += 1
                    continue
                match = blocklist.find_match(_record_texts(record)) if blocklist else None
                if match is not None:
                    rejections[f"benchmark:{match.benchmark}"] += 1
                    continue
                digest = _record_digest(record)
                if digest in fixed_validation_digests:
                    rejections["fixed_validation"] += 1
                    continue
                if digest in locked_train.digests:
                    rejections["locked_train"] += 1
                    continue
                cursor = connection.execute("INSERT OR IGNORE INTO seen VALUES (?)", (digest,))
                if cursor.rowcount == 0:
                    rejections["duplicate"] += 1
                    continue
                counts = token_counter(record["messages"])
                if counts.label <= 0 or counts.text <= 0 or counts.prompt < 0:
                    rejections["invalid_token_counts"] += 1
                    continue
                if counts.text > config.max_seq_length:
                    rejections["overlength"] += 1
                    continue
                cell = "|".join((pool.name, record["lang"], record["task"], record["source"]))
                stats[cell].add(record, counts)
                sampler.add(cell, digest, record, counts)
        connection.commit()
    finally:
        connection.close()
        if blocklist is not None:
            blocklist.close()
        database_path.unlink(missing_ok=True)

    pools = {
        pool.name: {
            "manifest": str(pool.manifest),
            "source_family": pool.source_family,
            "revision": pool.revision,
            "license": pool.license_name,
            "quality_tier": pool.quality_tier,
        }
        for pool in config.pools
    }
    profile = {
        "model_id": config.model_id,
        "max_seq_length": config.max_seq_length,
        "languages": list(config.languages),
        "tasks": list(config.tasks),
        "pools": pools,
        "fixed_validation_manifest": (
            str(config.fixed_validation_manifest) if config.fixed_validation_manifest is not None else None
        ),
        "fixed_validation_records": len(fixed_validation_digests),
        "locked_train_manifest": str(config.locked_train_manifest)
        if config.locked_train_manifest is not None
        else None,
        "locked_train_records": locked_train.records,
        "locked_train_unique_records": len(locked_train.digests),
        "locked_train_text_tokens": locked_train.text_tokens,
        "locked_train_label_tokens": locked_train.label_tokens,
        "locked_train_max_repetitions": locked_train.max_repetitions,
        "rejections": dict(rejections.most_common()),
        "cells": {key: value.to_dict() for key, value in sorted(stats.items())},
        "coverage": {
            "missing_language_task_cells": [
                f"{language}|{task}"
                for language in config.languages
                for task in config.tasks
                if not any(key.split("|")[1:3] == [language, task] for key in stats)
            ],
            "unresolved_license_pools": sorted(
                pool.name
                for pool in config.pools
                if any(marker in pool.license_name.casefold() for marker in ("unknown", "review", "unspecified"))
            ),
            "unpinned_revision_pools": sorted(
                pool.name
                for pool in config.pools
                if any(marker in pool.revision.casefold() for marker in ("pin", "latest", "unknown", "required"))
            ),
        },
    }
    _write_profile_artifacts(profile, sampler, output_dir)
    return profile


def _waterfill(total: float, capacities: Mapping[str, float], weights: Mapping[str, float]) -> dict[str, float]:
    allocations = {key: 0.0 for key in capacities}
    active = {key for key, capacity in capacities.items() if capacity > 0 and weights.get(key, 0) > 0}
    remaining = min(total, sum(capacities.values()))
    while active and remaining > 1e-9:
        weight_total = sum(weights[key] for key in active)
        proposals = {key: remaining * weights[key] / weight_total for key in active}
        capped = []
        for key, proposal in proposals.items():
            available = capacities[key] - allocations[key]
            if available <= proposal:
                allocations[key] += available
                remaining -= available
                capped.append(key)
        if capped:
            active.difference_update(capped)
        else:
            for key, proposal in proposals.items():
                allocations[key] += proposal
            remaining = 0
    return allocations


def build_mixture_plans(config: MixtureConfig, profile: Mapping[str, Any]) -> dict[str, Any]:
    """Simulate equal-token, capped-example, and quality-constrained policies."""
    aggregates = _aggregate_profile(config, profile)
    equal = _build_equal_token_plan(config, aggregates)
    capped = _build_capped_example_plan(config, aggregates, quality_constrained=False)
    quality = _build_capped_example_plan(config, aggregates, quality_constrained=True)
    plans = {
        "p0_equal_packed_tokens": equal,
        "p1_capped_examples_t2": capped,
        "p2_quality_constrained": quality,
    }
    if config.planning.task_token_shares:
        plans["p3_token_stratified"] = _build_token_stratified_plan(config, aggregates)
        if config.planning.language_token_budget:
            plans["p4_fixed_language_tokens"] = _build_fixed_language_token_plan(config, aggregates)
    if config.reference_train_summary is not None:
        reference_summary = json.loads(config.reference_train_summary.read_text(encoding="utf-8"))
        plans["p2_nested_extension"] = _build_nested_extension_plan(config, aggregates, reference_summary)
    return plans


def _build_nested_extension_plan(
    config: MixtureConfig,
    aggregates: Mapping[str, Any],
    reference_summary: Mapping[str, Any],
) -> dict[str, Any]:
    raw_allocations = reference_summary.get("train_allocations")
    if not isinstance(raw_allocations, dict) or not raw_allocations:
        raise ValueError("reference_train_summary must contain non-empty train_allocations")
    pool_map = {pool.name: pool for pool in config.pools}
    reference_tokens: dict[str, float] = {}
    for key, values in raw_allocations.items():
        pool, language, task = key.split("|", maxsplit=2)
        if pool not in pool_map or language not in config.languages or task not in config.tasks:
            raise ValueError(f"Reference allocation is outside the configured registry: {key}")
        tokens = float(values.get("tokens", 0))
        if tokens > 0:
            reference_tokens[key] = tokens
    reference_total = sum(reference_tokens.values())
    if reference_total <= 0:
        raise ValueError("reference_train_summary has no positive allocation tokens")
    scale = config.planning.packed_token_budget / reference_total
    desired_pool_tokens = {key: tokens * scale for key, tokens in reference_tokens.items()}
    desired_cell_tokens: Counter[str] = Counter()
    desired_language_tokens: Counter[str] = Counter()
    desired_task_tokens: Counter[str] = Counter()
    for key, tokens in desired_pool_tokens.items():
        _, language, task = key.split("|", maxsplit=2)
        desired_cell_tokens[f"{language}|{task}"] += tokens
        desired_language_tokens[language] += tokens
        desired_task_tokens[task] += tokens

    pool_capacities = {
        f"{pool.name}|{language}|{task}": float(
            aggregates["by_pool_language_task"].get(f"{pool.name}|{language}|{task}", {}).get("text_tokens", 0)
        )
        * config.planning.max_epochs
        for pool in config.pools
        for language in config.languages
        for task in config.tasks
    }
    allocation_capacities = {
        key: (
            capacity
            if pool_map[key.split("|", maxsplit=1)[0]].quality_tier in {"human", "curated"}
            else min(capacity, desired_pool_tokens.get(key, 0.0))
        )
        for key, capacity in pool_capacities.items()
    }
    label_ratios = {
        key: (
            float(aggregates["by_pool_language_task"].get(key, {}).get("label_tokens", 0))
            / float(aggregates["by_pool_language_task"].get(key, {}).get("text_tokens", 1))
            if float(aggregates["by_pool_language_task"].get(key, {}).get("text_tokens", 0)) > 0
            else 0.0
        )
        for key in pool_capacities
    }
    cell_capacities = {
        f"{language}|{task}": sum(allocation_capacities[f"{pool.name}|{language}|{task}"] for pool in config.pools)
        for language in config.languages
        for task in config.tasks
    }
    cell_targets: dict[str, float] = {}
    language_targets: dict[str, dict[str, float]] = {}
    for language in config.languages:
        language_cells = [f"{language}|{task}" for task in config.tasks]
        for cell in language_cells:
            cell_targets[cell] = min(desired_cell_tokens.get(cell, 0.0), cell_capacities[cell])
        remaining = desired_language_tokens.get(language, 0.0) - sum(cell_targets[cell] for cell in language_cells)
        residual_capacities = {cell: max(cell_capacities[cell] - cell_targets[cell], 0.0) for cell in language_cells}
        weights = {cell: desired_cell_tokens.get(cell, 0.0) for cell in language_cells}
        extras = _waterfill(remaining, residual_capacities, weights)
        for cell, tokens in extras.items():
            cell_targets[cell] += tokens
        planned_tokens = sum(cell_targets[cell] for cell in language_cells)
        shortfall = max(desired_language_tokens.get(language, 0.0) - planned_tokens, 0.0)
        language_targets[language] = {
            "target_tokens": desired_language_tokens.get(language, 0.0),
            "planned_tokens": planned_tokens,
            "shortfall_tokens": shortfall,
        }

    remaining_global = config.planning.packed_token_budget - sum(cell_targets.values())
    if remaining_global > 1e-9:
        current_task_tokens = {
            task: sum(cell_targets[f"{language}|{task}"] for language in config.languages) for task in config.tasks
        }
        task_residual_capacities = {
            task: sum(
                max(cell_capacities[f"{language}|{task}"] - cell_targets[f"{language}|{task}"], 0.0)
                for language in config.languages
            )
            for task in config.tasks
        }
        task_extras = _waterfill(
            remaining_global,
            task_residual_capacities,
            {task: max(desired_task_tokens[task] - current_task_tokens[task], 0.0) for task in config.tasks},
        )
        for task, task_extra in task_extras.items():
            task_cells = [f"{language}|{task}" for language in config.languages]
            cell_extras = _waterfill(
                task_extra,
                {cell: max(cell_capacities[cell] - cell_targets[cell], 0.0) for cell in task_cells},
                {cell: desired_cell_tokens.get(cell, 0.0) for cell in task_cells},
            )
            for cell, tokens in cell_extras.items():
                cell_targets[cell] += tokens

    for language in config.languages:
        planned_tokens = sum(cell_targets[f"{language}|{task}"] for task in config.tasks)
        target_tokens = desired_language_tokens.get(language, 0.0)
        language_targets[language] = {
            "target_tokens": target_tokens,
            "planned_tokens": planned_tokens,
            "shortfall_tokens": max(target_tokens - planned_tokens, 0.0),
            "deviation_tokens": planned_tokens - target_tokens,
        }
    coverage_shortfall_tokens = max(config.planning.packed_token_budget - sum(cell_targets.values()), 0.0)

    cells: dict[str, dict[str, float]] = {}
    pool_allocations: dict[str, dict[str, Any]] = {}
    estimated_tokens = 0.0
    estimated_label_tokens = 0.0
    for language in config.languages:
        for task in config.tasks:
            cell = f"{language}|{task}"
            target_tokens = cell_targets[cell]
            cell_pool_keys = [f"{pool.name}|{language}|{task}" for pool in config.pools]
            allocations = {
                key: min(desired_pool_tokens.get(key, 0.0), allocation_capacities[key]) for key in cell_pool_keys
            }
            remaining = target_tokens - sum(allocations.values())
            original_keys = [key for key in cell_pool_keys if desired_pool_tokens.get(key, 0.0) > 0]
            original_residual = {key: max(allocation_capacities[key] - allocations[key], 0.0) for key in original_keys}
            original_extras = _allocate_by_quality_tier(
                remaining,
                original_residual,
                pool_map,
                label_ratios,
            )
            for key, tokens in original_extras.items():
                allocations[key] += tokens
            remaining = target_tokens - sum(allocations.values())
            fallback_keys = [key for key in cell_pool_keys if key not in original_keys]
            fallback_residual = {key: allocation_capacities[key] for key in fallback_keys}
            fallback_extras = _allocate_by_quality_tier(remaining, fallback_residual, pool_map, label_ratios)
            for key, tokens in fallback_extras.items():
                allocations[key] += tokens

            cell_records = 0.0
            cell_tokens = 0.0
            cell_labels = 0.0
            for key, tokens in allocations.items():
                if tokens <= 0:
                    continue
                values = aggregates["by_pool_language_task"].get(key, {})
                records = float(values.get("records", 0))
                available_tokens = float(values.get("text_tokens", 0))
                if records <= 0 or available_tokens <= 0:
                    continue
                target_records = tokens / (available_tokens / records)
                target_labels = target_records * (float(values.get("label_tokens", 0)) / records)
                pool = pool_map[key.split("|", maxsplit=1)[0]]
                pool_allocations[key] = {
                    "target_records": target_records,
                    "target_tokens": tokens,
                    "estimated_label_tokens": target_labels,
                    "source_family": pool.source_family,
                    "quality_tier": pool.quality_tier,
                    "reference_tokens": desired_pool_tokens.get(key, 0.0),
                }
                cell_records += target_records
                cell_tokens += tokens
                cell_labels += target_labels
            cells[cell] = {
                "target_records": cell_records,
                "target_tokens": target_tokens,
                "estimated_packed_tokens": cell_tokens,
                "estimated_label_tokens": cell_labels,
                "shortfall_tokens": max(target_tokens - cell_tokens, 0.0),
                "reference_target_tokens": desired_cell_tokens.get(cell, 0.0),
            }
            estimated_tokens += cell_tokens
            estimated_label_tokens += cell_labels

    label_reconciliation = _reconcile_nested_label_floor(
        pool_allocations,
        allocation_capacities,
        desired_pool_tokens,
        label_ratios,
        config.planning.minimum_label_tokens + config.planning.label_token_planning_margin,
    )
    cells = {
        f"{language}|{task}": {
            "target_records": 0.0,
            "target_tokens": 0.0,
            "estimated_packed_tokens": 0.0,
            "estimated_label_tokens": 0.0,
            "shortfall_tokens": 0.0,
            "reference_target_tokens": desired_cell_tokens.get(f"{language}|{task}", 0.0),
        }
        for language in config.languages
        for task in config.tasks
    }
    for key, values in pool_allocations.items():
        _, language, task = key.split("|", maxsplit=2)
        cell = cells[f"{language}|{task}"]
        cell["target_records"] += values["target_records"]
        cell["target_tokens"] += values["target_tokens"]
        cell["estimated_packed_tokens"] += values["target_tokens"]
        cell["estimated_label_tokens"] += values["estimated_label_tokens"]
    for cell in cells.values():
        cell["shortfall_tokens"] = max(cell["reference_target_tokens"] - cell["target_tokens"], 0.0)
    estimated_tokens = sum(values["target_tokens"] for values in pool_allocations.values())
    estimated_label_tokens = sum(values["estimated_label_tokens"] for values in pool_allocations.values())
    for language in config.languages:
        planned_tokens = sum(cells[f"{language}|{task}"]["target_tokens"] for task in config.tasks)
        target_tokens = desired_language_tokens.get(language, 0.0)
        language_targets[language] = {
            "target_tokens": target_tokens,
            "planned_tokens": planned_tokens,
            "shortfall_tokens": max(target_tokens - planned_tokens, 0.0),
            "deviation_tokens": planned_tokens - target_tokens,
        }
    coverage_shortfall_tokens = max(config.planning.packed_token_budget - estimated_tokens, 0.0)

    return {
        "reference_train_summary": str(config.reference_train_summary),
        "reference_scale": scale,
        "lower_quality_excess_allowed": False,
        "reference_task_token_shares": {
            task: desired_task_tokens[task] / config.planning.packed_token_budget for task in config.tasks
        },
        "estimated_packed_tokens": estimated_tokens,
        "estimated_label_tokens": estimated_label_tokens,
        "minimum_label_tokens": config.planning.minimum_label_tokens,
        "label_token_planning_margin": config.planning.label_token_planning_margin,
        "planning_label_target": config.planning.minimum_label_tokens + config.planning.label_token_planning_margin,
        "label_token_shortfall": max(config.planning.minimum_label_tokens - estimated_label_tokens, 0.0),
        "label_reconciliation": label_reconciliation,
        "coverage_shortfall_tokens": coverage_shortfall_tokens,
        "language_targets": language_targets,
        "cells": cells,
        "pool_allocations": pool_allocations,
    }


def _allocate_by_quality_tier(
    total: float,
    capacities: Mapping[str, float],
    pool_map: Mapping[str, PoolConfig],
    label_ratios: Mapping[str, float],
) -> dict[str, float]:
    allocations = {key: 0.0 for key in capacities}
    remaining = total
    for quality_tier, _ in sorted(QUALITY_WEIGHTS.items(), key=lambda item: (-item[1], item[0])):
        tier_capacities = {
            key: capacity
            for key, capacity in capacities.items()
            if pool_map[key.split("|", maxsplit=1)[0]].quality_tier == quality_tier and capacity > 0
        }
        for key in sorted(tier_capacities, key=lambda value: (-label_ratios.get(value, 0.0), value)):
            tokens = min(remaining, tier_capacities[key])
            allocations[key] += tokens
            remaining -= tokens
            if remaining <= 1e-9:
                break
        if remaining <= 1e-9:
            break
    return allocations


def _reconcile_nested_label_floor(
    pool_allocations: dict[str, dict[str, Any]],
    capacities: Mapping[str, float],
    reference_targets: Mapping[str, float],
    label_ratios: Mapping[str, float],
    minimum_label_tokens: int,
) -> dict[str, float]:
    average_text_tokens = {
        key: (
            float(values["target_tokens"]) / float(values["target_records"])
            if float(values["target_records"]) > 0
            else 0.0
        )
        for key, values in pool_allocations.items()
    }
    before_label_tokens = sum(
        float(values["target_tokens"]) * label_ratios.get(key, 0.0) for key, values in pool_allocations.items()
    )
    remaining_label_tokens = max(minimum_label_tokens - before_label_tokens, 0.0)
    moved_tokens = 0.0
    recipients = sorted(pool_allocations, key=lambda key: (-label_ratios.get(key, 0.0), key))
    donors = sorted(pool_allocations, key=lambda key: (label_ratios.get(key, 0.0), key))
    for recipient in recipients:
        recipient_capacity = capacities.get(recipient, 0.0) - float(pool_allocations[recipient]["target_tokens"])
        if recipient_capacity <= 1e-9:
            continue
        for donor in donors:
            ratio_gain = label_ratios.get(recipient, 0.0) - label_ratios.get(donor, 0.0)
            if ratio_gain <= 1e-12:
                break
            donor_excess = float(pool_allocations[donor]["target_tokens"]) - min(
                reference_targets.get(donor, 0.0), capacities.get(donor, 0.0)
            )
            if donor_excess <= 1e-9:
                continue
            transfer = min(recipient_capacity, donor_excess, remaining_label_tokens / ratio_gain)
            pool_allocations[donor]["target_tokens"] -= transfer
            pool_allocations[recipient]["target_tokens"] += transfer
            moved_tokens += transfer
            recipient_capacity -= transfer
            remaining_label_tokens -= transfer * ratio_gain
            if remaining_label_tokens <= 1e-6 or recipient_capacity <= 1e-9:
                break
        if remaining_label_tokens <= 1e-6:
            break

    for key, values in pool_allocations.items():
        target_tokens = float(values["target_tokens"])
        average_tokens = average_text_tokens[key]
        if average_tokens <= 0:
            continue
        values["target_records"] = target_tokens / average_tokens
        values["estimated_label_tokens"] = target_tokens * label_ratios.get(key, 0.0)
    after_label_tokens = sum(float(values["estimated_label_tokens"]) for values in pool_allocations.values())
    return {
        "before_label_tokens": before_label_tokens,
        "moved_tokens": moved_tokens,
        "after_label_tokens": after_label_tokens,
        "remaining_shortfall": max(minimum_label_tokens - after_label_tokens, 0.0),
    }


def _build_token_stratified_plan(config: MixtureConfig, aggregates: Mapping[str, Any]) -> dict[str, Any]:
    cells: dict[str, dict[str, float]] = {}
    pool_allocations: dict[str, dict[str, Any]] = {}
    estimated_tokens = 0.0
    estimated_label_tokens = 0.0
    shortfall_tokens = 0.0
    pool_map = {pool.name: pool for pool in config.pools}
    for task in config.tasks:
        task_budget = config.planning.packed_token_budget * config.planning.task_token_shares[task]
        equal_target = task_budget / len(config.languages)
        cell_capacities = {
            language: sum(
                float(
                    aggregates["by_pool_language_task"].get(f"{pool.name}|{language}|{task}", {}).get("text_tokens", 0)
                )
                * config.planning.max_epochs
                for pool in config.pools
            )
            for language in config.languages
        }
        representation_floor = equal_target * config.planning.minimum_cell_token_share
        cell_targets = {language: min(representation_floor, capacity) for language, capacity in cell_capacities.items()}
        residual_budget = task_budget - sum(cell_targets.values())
        residual_capacities = {
            language: max(cell_capacities[language] - cell_targets[language], 0.0) for language in config.languages
        }
        extra_targets = _waterfill(
            residual_budget,
            residual_capacities,
            {language: 1.0 for language in config.languages},
        )
        for language, tokens in extra_targets.items():
            cell_targets[language] += tokens
        for language in config.languages:
            cell_target = cell_targets[language]
            cell_key = f"{language}|{task}"
            pool_values = {
                pool.name: aggregates["by_pool_language_task"].get(f"{pool.name}|{language}|{task}", {})
                for pool in config.pools
            }
            capacities = {
                pool_name: float(values.get("text_tokens", 0)) * config.planning.max_epochs
                for pool_name, values in pool_values.items()
                if values.get("records", 0) > 0
            }
            capped_capacities = {}
            for pool_name, capacity in capacities.items():
                pool = pool_map[pool_name]
                family_cap = config.planning.source_family_caps.get(
                    pool.source_family,
                    1.0 if len(capacities) == 1 else config.planning.default_source_family_cap,
                )
                capped_capacities[pool_name] = min(capacity, cell_target * family_cap)
            weights = {
                pool_name: QUALITY_WEIGHTS[pool_map[pool_name].quality_tier]
                * math.sqrt(max(float(pool_values[pool_name].get("text_tokens", 0)), 1.0))
                for pool_name in capacities
            }
            token_allocations = _waterfill(cell_target, capped_capacities, weights)
            remaining = cell_target - sum(token_allocations.values())
            if remaining > 1e-9:
                residual = {
                    pool_name: capacities[pool_name] - token_allocations.get(pool_name, 0.0) for pool_name in capacities
                }
                extras = _waterfill(remaining, residual, weights)
                for pool_name, tokens in extras.items():
                    token_allocations[pool_name] = token_allocations.get(pool_name, 0.0) + tokens

            cell_tokens = 0.0
            cell_labels = 0.0
            cell_records = 0.0
            for pool_name, target_tokens in token_allocations.items():
                if target_tokens <= 0:
                    continue
                values = pool_values[pool_name]
                average_tokens = float(values["text_tokens"]) / float(values["records"])
                average_labels = float(values["label_tokens"]) / float(values["records"])
                target_records = target_tokens / average_tokens
                target_labels = target_records * average_labels
                pool = pool_map[pool_name]
                pool_allocations[f"{pool_name}|{language}|{task}"] = {
                    "target_records": target_records,
                    "target_tokens": target_tokens,
                    "estimated_label_tokens": target_labels,
                    "source_family": pool.source_family,
                    "quality_tier": pool.quality_tier,
                }
                cell_tokens += target_tokens
                cell_labels += target_labels
                cell_records += target_records
            cell_shortfall = max(cell_target - cell_tokens, 0.0)
            cells[cell_key] = {
                "target_records": cell_records,
                "target_tokens": cell_target,
                "estimated_packed_tokens": cell_tokens,
                "estimated_label_tokens": cell_labels,
                "shortfall_tokens": cell_shortfall,
            }
            estimated_tokens += cell_tokens
            estimated_label_tokens += cell_labels
            shortfall_tokens += cell_shortfall
    return {
        "task_token_shares": dict(config.planning.task_token_shares),
        "estimated_packed_tokens": estimated_tokens,
        "estimated_label_tokens": estimated_label_tokens,
        "minimum_label_tokens": config.planning.minimum_label_tokens,
        "label_token_shortfall": max(config.planning.minimum_label_tokens - estimated_label_tokens, 0.0),
        "coverage_shortfall_tokens": shortfall_tokens,
        "cells": cells,
        "pool_allocations": pool_allocations,
    }


def _build_fixed_language_token_plan(config: MixtureConfig, aggregates: Mapping[str, Any]) -> dict[str, Any]:
    cells: dict[str, dict[str, Any]] = {}
    pool_allocations: dict[str, dict[str, Any]] = {}
    language_targets: dict[str, dict[str, float]] = {}
    estimated_tokens = 0.0
    estimated_label_tokens = 0.0
    coverage_shortfall_tokens = 0.0

    for language in config.languages:
        desired_targets = {
            task: config.planning.language_token_budget * config.planning.task_token_shares[task]
            for task in config.tasks
        }
        capacities = {
            task: sum(
                float(
                    aggregates["by_pool_language_task"].get(f"{pool.name}|{language}|{task}", {}).get("text_tokens", 0)
                )
                * config.planning.max_epochs
                for pool in config.pools
            )
            for task in config.tasks
        }
        targets = {task: min(desired_targets[task], capacities[task]) for task in config.tasks}
        remaining = config.planning.language_token_budget - sum(targets.values())
        residual_capacities = {task: max(capacities[task] - targets[task], 0.0) for task in config.tasks}
        extras = _waterfill(remaining, residual_capacities, config.planning.task_token_shares)
        for task, tokens in extras.items():
            targets[task] += tokens

        adjusted_shares = {task: targets[task] / config.planning.language_token_budget for task in config.tasks}
        language_planning = replace(
            config.planning,
            packed_token_budget=config.planning.language_token_budget,
            language_token_budget=0,
            task_token_shares=adjusted_shares,
            minimum_label_tokens=0,
        )
        language_config = replace(config, languages=(language,), planning=language_planning)
        language_plan = _build_token_stratified_plan(language_config, aggregates)

        for cell_key, values in language_plan["cells"].items():
            task = cell_key.split("|", maxsplit=1)[1]
            desired_target = desired_targets[task]
            values["desired_target_tokens"] = desired_target
            values["target_share"] = values["target_tokens"] / config.planning.language_token_budget
            values["below_minimum_cell_share"] = (
                values["target_tokens"] + 1e-9 < desired_target * config.planning.minimum_cell_token_share
            )
            cells[cell_key] = values
        pool_allocations.update(language_plan["pool_allocations"])
        estimated_tokens += language_plan["estimated_packed_tokens"]
        estimated_label_tokens += language_plan["estimated_label_tokens"]
        actual_tokens = language_plan["estimated_packed_tokens"]
        language_shortfall = max(config.planning.language_token_budget - actual_tokens, 0.0)
        language_targets[language] = {
            "target_tokens": float(config.planning.language_token_budget),
            "planned_tokens": actual_tokens,
            "shortfall_tokens": language_shortfall,
        }
        coverage_shortfall_tokens += language_shortfall

    return {
        "language_token_budget": config.planning.language_token_budget,
        "task_token_shares": dict(config.planning.task_token_shares),
        "estimated_packed_tokens": estimated_tokens,
        "estimated_label_tokens": estimated_label_tokens,
        "minimum_label_tokens": config.planning.minimum_label_tokens,
        "label_token_shortfall": max(config.planning.minimum_label_tokens - estimated_label_tokens, 0.0),
        "coverage_shortfall_tokens": coverage_shortfall_tokens,
        "language_targets": language_targets,
        "cells": cells,
        "pool_allocations": pool_allocations,
    }


def _aggregate_profile(config: MixtureConfig, profile: Mapping[str, Any]) -> dict[str, Any]:
    pools = {pool.name: pool for pool in config.pools}
    by_task: dict[str, Counter[str]] = defaultdict(Counter)
    by_language_task: dict[str, Counter[str]] = defaultdict(Counter)
    by_pool_language_task: dict[str, Counter[str]] = defaultdict(Counter)
    for key, values in profile["cells"].items():
        pool, language, task, _ = key.split("|", maxsplit=3)
        metrics = {name: float(values[name]) for name in ("records", "text_tokens", "label_tokens")}
        by_task[task].update(metrics)
        by_language_task[f"{language}|{task}"].update(metrics)
        pool_key = f"{pool}|{language}|{task}"
        by_pool_language_task[pool_key].update(metrics)
        by_pool_language_task[pool_key]["quality_weight"] = QUALITY_WEIGHTS[pools[pool].quality_tier]
    return {
        "by_task": by_task,
        "by_language_task": by_language_task,
        "by_pool_language_task": by_pool_language_task,
    }


def _build_equal_token_plan(config: MixtureConfig, aggregates: Mapping[str, Any]) -> dict[str, Any]:
    target = config.planning.packed_token_budget / len(config.tasks) / len(config.languages)
    cells = {}
    estimated_total = 0.0
    for language in config.languages:
        for task in config.tasks:
            key = f"{language}|{task}"
            values = aggregates["by_language_task"].get(key, {})
            average = values.get("text_tokens", 0) / values.get("records", 1) if values.get("records", 0) else 0
            capacity = values.get("records", 0) * config.planning.max_epochs
            records = min(capacity, target / average) if average else 0
            packed = records * average
            cells[key] = {
                "target_records": records,
                "estimated_packed_tokens": packed,
                "shortfall_tokens": target - packed,
            }
            estimated_total += packed
    return {"estimated_packed_tokens": estimated_total, "cells": cells}


def _build_capped_example_plan(
    config: MixtureConfig, aggregates: Mapping[str, Any], *, quality_constrained: bool
) -> dict[str, Any]:
    task_scores = {}
    for task in config.tasks:
        records = aggregates["by_task"].get(task, {}).get("records", 0)
        cap = config.planning.task_example_caps.get(task, math.ceil(records))
        task_scores[task] = min(records, cap) ** (1 / config.planning.temperature)
    score_total = sum(task_scores.values())
    task_probabilities = {task: score / score_total if score_total else 0.0 for task, score in task_scores.items()}

    def allocate(total_records: float) -> tuple[dict[str, Any], dict[str, Any], float]:
        if quality_constrained:
            return _allocate_quality_cells(config, aggregates, task_probabilities, total_records)
        cells, estimated_total = _allocate_capped_cells(config, aggregates, task_probabilities, total_records)
        return cells, {}, estimated_total

    upper_records = float(config.planning.packed_token_budget)
    cells, pool_allocations, estimated_total = allocate(upper_records)
    while estimated_total < config.planning.packed_token_budget:
        previous_total = estimated_total
        upper_records *= 2
        cells, pool_allocations, estimated_total = allocate(upper_records)
        if math.isclose(estimated_total, previous_total):
            break
    if estimated_total >= config.planning.packed_token_budget:
        lower_records = 0.0
        for _ in range(64):
            total_records = (lower_records + upper_records) / 2
            cells, pool_allocations, estimated_total = allocate(total_records)
            if estimated_total < config.planning.packed_token_budget:
                lower_records = total_records
            else:
                upper_records = total_records
        cells, pool_allocations, estimated_total = allocate(upper_records)
    return {
        "temperature": config.planning.temperature,
        "task_example_probabilities": task_probabilities,
        "estimated_packed_tokens": estimated_total,
        "cells": cells,
        "pool_allocations": pool_allocations,
    }


def _allocate_capped_cells(
    config: MixtureConfig,
    aggregates: Mapping[str, Any],
    task_probabilities: Mapping[str, float],
    total_records: float,
) -> tuple[dict[str, dict[str, float]], float]:
    cells = {}
    estimated_total = 0.0
    for task in config.tasks:
        task_records = total_records * task_probabilities[task]
        capacities = {
            language: aggregates["by_language_task"].get(f"{language}|{task}", {}).get("records", 0)
            * config.planning.max_epochs
            for language in config.languages
        }
        language_records = _waterfill(task_records, capacities, {language: 1.0 for language in config.languages})
        for language, records in language_records.items():
            cell_key = f"{language}|{task}"
            values = aggregates["by_language_task"].get(cell_key, {})
            average = values.get("text_tokens", 0) / values.get("records", 1) if values.get("records", 0) else 0
            packed = records * average
            cells[cell_key] = {"target_records": records, "estimated_packed_tokens": packed}
            estimated_total += packed
    return cells, estimated_total


def _allocate_quality_cells(
    config: MixtureConfig,
    aggregates: Mapping[str, Any],
    task_probabilities: Mapping[str, float],
    total_records: float,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]], float]:
    cells, _ = _allocate_capped_cells(config, aggregates, task_probabilities, total_records)
    pool_allocations = {}
    estimated_total = 0.0
    for cell_key, values in cells.items():
        language, task = cell_key.split("|", maxsplit=1)
        records = values["target_records"]
        allocations, fallback_records = _allocate_pools(config, aggregates, language, task, records)
        pool_allocations.update(allocations)
        packed = 0.0
        for allocation_key, allocation in allocations.items():
            pool, _, _ = allocation_key.split("|", maxsplit=2)
            pool_values = aggregates["by_pool_language_task"].get(f"{pool}|{language}|{task}", {})
            average = (
                pool_values.get("text_tokens", 0) / pool_values.get("records", 1)
                if pool_values.get("records", 0)
                else 0.0
            )
            packed += allocation["target_records"] * average
        values["estimated_packed_tokens"] = packed
        values["fallback_records"] = fallback_records
        values["fallback_share"] = fallback_records / records if records else 0.0
        estimated_total += packed
    return cells, pool_allocations, estimated_total


def _allocate_pools(
    config: MixtureConfig, aggregates: Mapping[str, Any], language: str, task: str, target_records: float
) -> tuple[dict[str, dict[str, float]], float]:
    pool_map = {pool.name: pool for pool in config.pools}
    fallback_families = set(config.planning.fallback_source_families)
    raw_capacities = {}
    for pool in config.pools:
        values = aggregates["by_pool_language_task"].get(f"{pool.name}|{language}|{task}", {})
        raw_capacities[pool.name] = values.get("records", 0) * config.planning.max_epochs

    primary_names = [
        pool.name
        for pool in config.pools
        if pool.source_family not in fallback_families and raw_capacities[pool.name] > 0
    ]
    primary_capacities = {}
    for pool_name in primary_names:
        pool = pool_map[pool_name]
        if pool.source_family in config.planning.source_family_caps:
            family_cap = config.planning.source_family_caps[pool.source_family]
        elif len(primary_names) == 1:
            family_cap = 1.0
        else:
            family_cap = config.planning.default_source_family_cap
        primary_capacities[pool_name] = min(raw_capacities[pool_name], target_records * family_cap)

    primary_target = target_records * (1 - config.planning.fallback_base_share)
    primary_allocations = _waterfill(
        primary_target,
        primary_capacities,
        {pool_name: QUALITY_WEIGHTS[pool_map[pool_name].quality_tier] for pool_name in primary_names},
    )
    fallback_target = target_records - sum(primary_allocations.values())
    fallback_names = [
        pool.name for pool in config.pools if pool.source_family in fallback_families and raw_capacities[pool.name] > 0
    ]
    fallback_allocations = _waterfill(
        fallback_target,
        {pool_name: raw_capacities[pool_name] for pool_name in fallback_names},
        {pool_name: QUALITY_WEIGHTS[pool_map[pool_name].quality_tier] for pool_name in fallback_names},
    )
    allocations = {**primary_allocations, **fallback_allocations}
    remaining = target_records - sum(allocations.values())
    if remaining > 1e-9:
        primary_residual = {
            pool_name: primary_capacities[pool_name] - primary_allocations.get(pool_name, 0.0)
            for pool_name in primary_names
        }
        extra_primary = _waterfill(
            remaining,
            primary_residual,
            {pool_name: QUALITY_WEIGHTS[pool_map[pool_name].quality_tier] for pool_name in primary_names},
        )
        for pool_name, records in extra_primary.items():
            allocations[pool_name] = allocations.get(pool_name, 0.0) + records

    result = {
        f"{pool}|{language}|{task}": {
            "target_records": records,
            "source_family": pool_map[pool].source_family,
            "quality_tier": pool_map[pool].quality_tier,
            "allocation_role": "fallback" if pool_map[pool].source_family in fallback_families else "primary",
        }
        for pool, records in allocations.items()
        if records > 0
    }
    return result, sum(fallback_allocations.values())


def _write_profile_artifacts(profile: Mapping[str, Any], sampler: _ReviewSampler, output_dir: Path) -> None:
    (output_dir / "profile.json").write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "cells.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "pool",
                "lang",
                "task",
                "source",
                "records",
                "prompt_tokens",
                "label_tokens",
                "text_tokens",
                "average_prompt_tokens",
                "average_label_tokens",
                "average_text_tokens",
                "max_text_tokens",
            ),
        )
        writer.writeheader()
        for key, values in profile["cells"].items():
            pool, language, task, source = key.split("|", maxsplit=3)
            writer.writerow(
                {
                    "pool": pool,
                    "lang": language,
                    "task": task,
                    "source": source,
                    **{
                        field: values[field]
                        for field in (
                            "records",
                            "prompt_tokens",
                            "label_tokens",
                            "text_tokens",
                            "average_prompt_tokens",
                            "average_label_tokens",
                            "average_text_tokens",
                            "max_text_tokens",
                        )
                    },
                }
            )
    with (output_dir / "review_samples.jsonl").open("w", encoding="utf-8") as handle:
        for payload in sampler.records():
            handle.write(payload + "\n")
    (output_dir / "profile.md").write_text(_profile_markdown(profile), encoding="utf-8")


def _profile_markdown(profile: Mapping[str, Any]) -> str:
    totals: dict[str, Counter[str]] = defaultdict(Counter)
    for key, values in profile["cells"].items():
        _, language, task, _ = key.split("|", maxsplit=3)
        totals[f"{language}|{task}"].update(
            {name: int(values[name]) for name in ("records", "text_tokens", "label_tokens")}
        )
    lines = [
        "# SFT mixture profile",
        "",
        "## Language-task capacity",
        "",
        "| Language | Task | Records | Packed tokens | Label tokens |",
        "|---|---|---:|---:|---:|",
    ]
    for key, values in sorted(totals.items()):
        language, task = key.split("|", maxsplit=1)
        lines.append(
            f"| {language} | {task} | {values['records']:,} | {values['text_tokens']:,} | {values['label_tokens']:,} |"
        )
    lines.extend(("", "## Rejections", "", "| Reason | Records |", "|---|---:|"))
    for reason, count in profile["rejections"].items():
        lines.append(f"| {reason} | {count:,} |")
    lines.extend(("", "## Coverage issues", ""))
    missing = profile["coverage"]["missing_language_task_cells"]
    lines.append(f"Missing language-task cells: {', '.join(missing) if missing else 'none'}.")
    unresolved = profile["coverage"]["unresolved_license_pools"]
    lines.append(f"Pools requiring license resolution: {', '.join(unresolved) if unresolved else 'none'}.")
    unpinned = profile["coverage"]["unpinned_revision_pools"]
    lines.append(f"Pools requiring an immutable revision: {', '.join(unpinned) if unpinned else 'none'}.")
    return "\n".join(lines) + "\n"


def _write_plans(plans: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plans.json").write_text(json.dumps(plans, indent=2) + "\n", encoding="utf-8")
    lines = ["# SFT mixture allocation simulations", "", "| Policy | Estimated packed tokens |", "|---|---:|"]
    for name, plan in plans.items():
        lines.append(f"| {name} | {plan['estimated_packed_tokens']:,.0f} |")
    for name, plan in plans.items():
        lines.extend(("", f"## {name}", "", "| Task | Target records | Estimated packed tokens |", "|---|---:|---:|"))
        task_totals: dict[str, Counter[str]] = defaultdict(Counter)
        for cell, values in plan["cells"].items():
            _, task = cell.split("|", maxsplit=1)
            task_totals[task].update(
                {
                    "target_records": values["target_records"],
                    "estimated_packed_tokens": values["estimated_packed_tokens"],
                }
            )
        for task, values in sorted(task_totals.items()):
            lines.append(f"| {task} | {values['target_records']:,.0f} | {values['estimated_packed_tokens']:,.0f} |")
        if "language_targets" in plan:
            lines.extend(
                (
                    "",
                    "### Language budgets",
                    "",
                    "| Language | Target tokens | Planned tokens | Shortfall |",
                    "|---|---:|---:|---:|",
                )
            )
            for language, values in sorted(plan["language_targets"].items()):
                lines.append(
                    f"| {language} | {values['target_tokens']:,.0f} | "
                    f"{values['planned_tokens']:,.0f} | {values['shortfall_tokens']:,.0f} |"
                )
            if name == "p2_nested_extension":
                lines.extend(
                    (
                        "",
                        "### Language-task targets",
                        "",
                        "| Language | Task | Reference tokens | Planned tokens | Deviation |",
                        "|---|---|---:|---:|---:|",
                    )
                )
                for cell, values in sorted(plan["cells"].items()):
                    language, task = cell.split("|", maxsplit=1)
                    deviation = values["target_tokens"] - values["reference_target_tokens"]
                    lines.append(
                        f"| {language} | {task} | {values['reference_target_tokens']:,.0f} | "
                        f"{values['target_tokens']:,.0f} | {deviation:+,.0f} |"
                    )
            else:
                lines.extend(
                    (
                        "",
                        "### Language-task targets",
                        "",
                        "| Language | Task | Desired tokens | Planned tokens | Planned share | Below 60% target |",
                        "|---|---|---:|---:|---:|---|",
                    )
                )
                for cell, values in sorted(plan["cells"].items()):
                    language, task = cell.split("|", maxsplit=1)
                    lines.append(
                        f"| {language} | {task} | {values['desired_target_tokens']:,.0f} | "
                        f"{values['target_tokens']:,.0f} | {values['target_share']:.2%} | "
                        f"{'yes' if values['below_minimum_cell_share'] else 'no'} |"
                    )
            lines.extend(
                (
                    "",
                    f"Coverage shortfall: {plan['coverage_shortfall_tokens']:,.0f} packed tokens.",
                    f"Supervised-token shortfall: {plan['label_token_shortfall']:,.0f} tokens.",
                )
            )
        fallback_cells = {cell: values for cell, values in plan["cells"].items() if "fallback_share" in values}
        if fallback_cells:
            lines.extend(
                (
                    "",
                    "### Fallback usage",
                    "",
                    "| Language | Task | Fallback records | Fallback share |",
                    "|---|---|---:|---:|",
                )
            )
            for cell, values in sorted(fallback_cells.items()):
                language, task = cell.split("|", maxsplit=1)
                lines.append(
                    f"| {language} | {task} | {values['fallback_records']:,.0f} | {values['fallback_share']:.1%} |"
                )
    (output_dir / "plans.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    profile = subparsers.add_parser("profile")
    profile.add_argument("--config", type=Path, required=True)
    profile.add_argument("--output-dir", type=Path, required=True)
    profile.add_argument("--benchmark-blocklist", type=Path)
    plan = subparsers.add_parser("plan")
    plan.add_argument("--config", type=Path, required=True)
    plan.add_argument("--profile", type=Path, required=True)
    plan.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    """Profile prepared pools or simulate allocation policies."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = _build_parser().parse_args()
    config = load_config(args.config.resolve())
    if args.command == "profile":
        profile_mixture(
            config,
            token_counter=_make_token_counter(config.model_id),
            output_dir=args.output_dir.resolve(),
            benchmark_blocklist=args.benchmark_blocklist.resolve() if args.benchmark_blocklist else None,
        )
    else:
        profile = json.loads(args.profile.read_text(encoding="utf-8"))
        _write_plans(build_mixture_plans(config, profile), args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
