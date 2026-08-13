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

"""Materialize a deterministic K6 SFT mixture from a profiling plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TextIO

if __package__:
    from tools.benchmark_contamination import BenchmarkBlocklist
    from tools.profile_sft_mixture import (
        QUALITY_WEIGHTS,
        MixtureConfig,
        TokenCounts,
        _iter_manifest_records,
        _make_token_counter,
        _normalize_record,
        _record_digest,
        _record_texts,
        load_config,
    )
else:
    from benchmark_contamination import BenchmarkBlocklist
    from profile_sft_mixture import (
        QUALITY_WEIGHTS,
        MixtureConfig,
        TokenCounts,
        _iter_manifest_records,
        _make_token_counter,
        _normalize_record,
        _record_digest,
        _record_texts,
        load_config,
    )

logger = logging.getLogger(__name__)

TokenCounter = Callable[[list[dict[str, str]]], TokenCounts]


class _OutputWriter:
    def __init__(self, output_dir: Path, shard_size: int) -> None:
        self._output_dir = output_dir
        self._shard_size = shard_size
        self._states: dict[tuple[str, str], tuple[int, int]] = {}
        self._handles: dict[Path, TextIO] = {}
        self.manifests: dict[str, dict[str, dict[str, Any]]] = {"train": {}, "validation": {}}

    def write(self, partition: str, pool: str, payload: str) -> None:
        key = (partition, pool)
        shard_index, records_in_shard = self._states.get(key, (0, 0))
        if records_in_shard >= self._shard_size:
            shard_index += 1
            records_in_shard = 0
        relative_path = Path("processed") / partition / f"{pool}-{shard_index:05d}.jsonl"
        path = self._output_dir / relative_path
        if records_in_shard == 0:
            path.parent.mkdir(parents=True, exist_ok=True)
            name = f"{pool}-{shard_index:05d}"
            self.manifests[partition][name] = {
                "file_name": relative_path.as_posix(),
                "columns": {"messages": "messages"},
                "sample_ratio": 1.0,
            }
        handle = self._handles.get(path)
        if handle is None:
            handle = path.open("a", encoding="utf-8")
            self._handles[path] = handle
        handle.write(payload + "\n")
        self._states[key] = (shard_index, records_in_shard + 1)

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()


def _write_manifest(output_dir: Path, partition: str, manifest: Mapping[str, Mapping[str, Any]]) -> None:
    (output_dir / f"{partition}_meta.json").write_text(
        json.dumps(dict(sorted(manifest.items())), indent=2) + "\n",
        encoding="utf-8",
    )


def _assistant_text(record: Mapping[str, Any]) -> str:
    return "\n".join(
        str(message["content"]).strip() for message in record["messages"] if message["role"] == "assistant"
    )


def _candidate_stratum(record: Mapping[str, Any]) -> str:
    task = record["task"]
    if task == "classification":
        return f"label:{_assistant_text(record)}"
    if task == "translation" and isinstance(record.get("direction"), str):
        return f"direction:{record['direction']}"
    if task == "qa":
        for key in ("source_dataset", "qa_source", "qa_subtype", "grade", "domain"):
            value = record.get(key)
            if isinstance(value, str) and value.strip():
                return f"{key}:{value.strip()}"
    if task == "ner" and isinstance(record.get("entity_types"), list):
        count = len(record["entity_types"])
        return "entities:none" if count == 0 else "entities:one-type" if count == 1 else "entities:multi-type"
    return "__all__"


def _sample_key(digest: bytes, seed: int) -> bytes:
    person = f"kseries:{seed}".encode("utf-8")[:16]
    return hashlib.blake2b(digest, digest_size=20, person=person).digest()


def _build_candidates(
    config: MixtureConfig,
    benchmark_blocklist: Path,
    output_dir: Path,
    token_counter: TokenCounter,
) -> dict[str, Any]:
    database_path = output_dir / ".mixture.sqlite3"
    connection = sqlite3.connect(database_path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute(
        "CREATE TABLE candidates ("
        "digest BLOB PRIMARY KEY, sample_key BLOB NOT NULL, pool TEXT NOT NULL, lang TEXT NOT NULL, "
        "task TEXT NOT NULL, stratum TEXT NOT NULL, tokens INTEGER NOT NULL, labels INTEGER NOT NULL, "
        "payload TEXT NOT NULL, selected INTEGER NOT NULL DEFAULT 0"
        ") WITHOUT ROWID"
    )
    rejections: Counter[str] = Counter()
    available: dict[str, Counter[str]] = defaultdict(Counter)
    with BenchmarkBlocklist(benchmark_blocklist, mode="read-only") as blocklist:
        ordered_pools = sorted(config.pools, key=lambda pool: (-QUALITY_WEIGHTS[pool.quality_tier], pool.name))
        for pool in ordered_pools:
            logger.info("Indexing %s", pool.name)
            for raw in _iter_manifest_records(pool.manifest):
                record, reason = _normalize_record(raw)
                if record is None:
                    rejections[reason or "invalid"] += 1
                    continue
                if record["lang"] not in config.languages or record["task"] not in config.tasks:
                    rejections["outside_scope"] += 1
                    continue
                match = blocklist.find_match(_record_texts(record))
                if match is not None:
                    rejections[f"benchmark:{match.benchmark}"] += 1
                    continue
                counts = token_counter(record["messages"])
                if counts.label <= 0 or counts.text <= 0 or counts.prompt < 0:
                    rejections["invalid_token_counts"] += 1
                    continue
                if counts.text > config.max_seq_length:
                    rejections["overlength"] += 1
                    continue
                payload_record = dict(record)
                payload_record["_prompt_tokens"] = counts.prompt
                payload_record["_label_tokens"] = counts.label
                payload_record["_text_tokens"] = counts.text
                payload = json.dumps(payload_record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                digest = _record_digest(record)
                cursor = connection.execute(
                    "INSERT OR IGNORE INTO candidates VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)",
                    (
                        digest,
                        _sample_key(digest, config.planning.sampling_seed),
                        pool.name,
                        record["lang"],
                        record["task"],
                        _candidate_stratum(record),
                        counts.text,
                        counts.label,
                        payload,
                    ),
                )
                if cursor.rowcount == 0:
                    rejections["duplicate"] += 1
                    continue
                cell = f"{pool.name}|{record['lang']}|{record['task']}"
                available[cell]["records"] += 1
                available[cell]["tokens"] += counts.text
            connection.commit()
    connection.execute("CREATE INDEX candidates_cell ON candidates (pool, lang, task, stratum, selected, sample_key)")
    connection.commit()
    connection.close()
    return {
        "rejections": dict(rejections.most_common()),
        "available": {cell: dict(values) for cell, values in sorted(available.items())},
    }


def _round_allocations(policy_plan: Mapping[str, Any]) -> dict[str, int]:
    grouped: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for key, values in policy_plan["pool_allocations"].items():
        pool, language, task = key.split("|", maxsplit=2)
        grouped[f"{language}|{task}"].append((f"{pool}|{language}|{task}", float(values["target_records"])))

    rounded: dict[str, int] = {}
    for cell, allocations in grouped.items():
        target_total = round(float(policy_plan["cells"][cell]["target_records"]))
        floors = {key: math.floor(target) for key, target in allocations}
        remainder = target_total - sum(floors.values())
        ranked = sorted(allocations, key=lambda item: (-(item[1] - math.floor(item[1])), item[0]))
        if remainder < 0 or remainder > len(ranked):
            raise ValueError(f"Cannot round pool allocations for {cell}")
        for key, _ in allocations:
            rounded[key] = floors[key]
        for key, _ in ranked[:remainder]:
            rounded[key] += 1
    return rounded


def _balanced_targets(total: int, capacities: Mapping[str, int]) -> dict[str, int]:
    if total > sum(capacities.values()):
        raise ValueError(f"Requested {total} records from capacity {sum(capacities.values())}")
    allocations = {key: 0 for key in capacities}
    active = sorted(key for key, capacity in capacities.items() if capacity > 0)
    remaining = total
    while active and remaining:
        share, extra = divmod(remaining, len(active))
        capped = [key for key in active if capacities[key] <= share]
        if capped:
            for key in capped:
                allocations[key] = capacities[key]
                remaining -= capacities[key]
            active = [key for key in active if key not in capped]
            continue
        for index, key in enumerate(active):
            allocations[key] = share + int(index < extra)
        remaining = 0
    return allocations


def _token_balanced_targets(
    total_records: int, candidates: Mapping[str, list[tuple[bytes, int, int, str]]], max_epochs: float
) -> dict[str, int]:
    capacities = {stratum: math.floor(len(rows) * max_epochs + 1e-9) for stratum, rows in candidates.items()}
    if total_records > sum(capacities.values()):
        raise ValueError(f"Requested {total_records} records from capacity {sum(capacities.values())}")
    if len(candidates) == 1:
        return {next(iter(candidates)): total_records}

    averages = {
        stratum: sum(tokens for _, tokens, _, _ in rows) / len(rows) for stratum, rows in candidates.items() if rows
    }
    raw_records = _weighted_record_targets(
        total_records,
        capacities,
        {stratum: 1.0 / average for stratum, average in averages.items()},
    )
    targets = {stratum: math.floor(records) for stratum, records in raw_records.items()}
    remainder = total_records - sum(targets.values())
    ranked = sorted(raw_records.items(), key=lambda item: (-(item[1] - math.floor(item[1])), item[0]))
    for stratum, _ in ranked:
        if remainder <= 0:
            break
        if targets[stratum] < capacities[stratum]:
            targets[stratum] += 1
            remainder -= 1
    if remainder:
        residual = sorted(
            (stratum for stratum in candidates if targets[stratum] < capacities[stratum]), key=lambda value: value
        )
        for stratum in residual:
            while remainder and targets[stratum] < capacities[stratum]:
                targets[stratum] += 1
                remainder -= 1
    if remainder:
        raise ValueError(f"Could not distribute {remainder} records across strata")
    return targets


def _weighted_record_targets(
    total: int, capacities: Mapping[str, int], weights: Mapping[str, float]
) -> dict[str, float]:
    allocations = {key: 0.0 for key in capacities}
    active = {key for key, capacity in capacities.items() if capacity > 0 and weights.get(key, 0) > 0}
    remaining = float(total)
    while active and remaining > 1e-9:
        weight_total = sum(weights[key] for key in active)
        proposals = {key: remaining * weights[key] / weight_total for key in active}
        capped = []
        for key in active:
            available = capacities[key] - allocations[key]
            if available <= proposals[key]:
                allocations[key] += available
                remaining -= available
                capped.append(key)
        if capped:
            active.difference_update(capped)
        else:
            for key, proposal in proposals.items():
                allocations[key] += proposal
            remaining = 0.0
    return allocations


def _select_train_records(
    output_dir: Path,
    policy_plan: Mapping[str, Any],
    writer: _OutputWriter,
    max_epochs: float,
) -> tuple[dict[str, dict[str, Any]], int, int, int]:
    connection = sqlite3.connect(output_dir / ".mixture.sqlite3")
    selected: dict[str, dict[str, Any]] = {}
    total_records = 0
    total_tokens = 0
    total_label_tokens = 0
    try:
        for allocation_key, target_records in sorted(_round_allocations(policy_plan).items()):
            if target_records <= 0:
                continue
            pool, language, task = allocation_key.split("|", maxsplit=2)
            rows = connection.execute(
                "SELECT digest, stratum, tokens, labels, payload FROM candidates "
                "WHERE pool = ? AND lang = ? AND task = ? ORDER BY stratum, sample_key",
                (pool, language, task),
            ).fetchall()
            by_stratum: dict[str, list[tuple[bytes, int, int, str]]] = defaultdict(list)
            for digest, stratum, tokens, labels, payload in rows:
                by_stratum[stratum].append((digest, tokens, labels, payload))
            stratum_targets = _token_balanced_targets(target_records, by_stratum, max_epochs)
            allocation_tokens = 0
            allocation_label_tokens = 0
            unique_digests: set[bytes] = set()
            max_repetitions = 0
            realized_strata = {}
            for stratum, stratum_target in sorted(stratum_targets.items()):
                candidates = by_stratum.get(stratum, [])
                capacity = math.floor(len(candidates) * max_epochs + 1e-9)
                if not candidates or stratum_target > capacity:
                    raise ValueError(
                        f"Pool {pool} provides capacity {capacity} for {language}|{task}|{stratum}; "
                        f"requested {stratum_target}"
                    )
                repetitions: Counter[bytes] = Counter()
                stratum_tokens = 0
                stratum_label_tokens = 0
                for index in range(stratum_target):
                    digest, tokens, labels, payload = candidates[index % len(candidates)]
                    writer.write("train", pool, payload)
                    repetitions[digest] += 1
                    unique_digests.add(digest)
                    stratum_tokens += tokens
                    stratum_label_tokens += labels
                max_repetitions = max(max_repetitions, max(repetitions.values(), default=0))
                allocation_tokens += stratum_tokens
                allocation_label_tokens += stratum_label_tokens
                realized_strata[stratum] = {
                    "records": stratum_target,
                    "tokens": stratum_tokens,
                    "label_tokens": stratum_label_tokens,
                }
            connection.executemany(
                "UPDATE candidates SET selected = 1 WHERE digest = ?",
                ((digest,) for digest in unique_digests),
            )
            selected[allocation_key] = {
                "records": target_records,
                "unique_records": len(unique_digests),
                "tokens": allocation_tokens,
                "label_tokens": allocation_label_tokens,
                "max_repetitions": max_repetitions,
                "strata": realized_strata,
            }
            total_records += target_records
            total_tokens += allocation_tokens
            total_label_tokens += allocation_label_tokens
        connection.commit()
    finally:
        connection.close()
    return selected, total_records, total_tokens, total_label_tokens


def _select_validation_records(
    output_dir: Path,
    records_per_pool: int,
    writer: _OutputWriter,
    pool_names: list[str],
) -> tuple[dict[str, dict[str, int]], int, int]:
    connection = sqlite3.connect(output_dir / ".mixture.sqlite3")
    selected: dict[str, dict[str, int]] = {}
    total_records = 0
    total_tokens = 0
    try:
        for pool in pool_names:
            rows = connection.execute(
                "SELECT tokens, payload FROM candidates WHERE pool = ? AND selected = 0 ORDER BY sample_key LIMIT ?",
                (pool, records_per_pool),
            ).fetchall()
            pool_tokens = 0
            for tokens, payload in rows:
                writer.write("validation", pool, payload)
                pool_tokens += tokens
            selected[pool] = {"records": len(rows), "tokens": pool_tokens}
            total_records += len(rows)
            total_tokens += pool_tokens
    finally:
        connection.close()
    return selected, total_records, total_tokens


def build_mixture(
    *,
    config_path: Path,
    plan_path: Path,
    policy: str,
    benchmark_blocklist: Path,
    output_dir: Path,
    validation_records_per_pool: int = 2_500,
    shard_size: int = 100_000,
    token_counter: TokenCounter | None = None,
) -> dict[str, Any]:
    """Build a deterministic training mixture and disjoint validation monitor."""
    config = load_config(config_path)
    plans = json.loads(plan_path.read_text(encoding="utf-8"))
    if policy not in plans:
        raise ValueError(f"Unknown plan policy: {policy}")
    if validation_records_per_pool < 0 or shard_size <= 0:
        raise ValueError("validation_records_per_pool must be non-negative and shard_size must be positive")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    database_path = output_dir / ".mixture.sqlite3"
    writer = _OutputWriter(output_dir, shard_size)
    try:
        candidate_summary = _build_candidates(
            config,
            benchmark_blocklist,
            output_dir,
            token_counter or _make_token_counter(config.model_id),
        )
        policy_plan = plans[policy]
        if policy_plan.get("coverage_shortfall_tokens", 0) > 1e-6:
            raise ValueError(
                f"Policy {policy} has {policy_plan['coverage_shortfall_tokens']:.0f} uncovered packed tokens"
            )
        train, train_records, train_tokens, train_label_tokens = _select_train_records(
            output_dir,
            policy_plan,
            writer,
            config.planning.max_epochs,
        )
        if train_label_tokens < config.planning.minimum_label_tokens:
            raise ValueError(
                f"Materialized {train_label_tokens} label tokens; "
                f"requires at least {config.planning.minimum_label_tokens}"
            )
        validation, validation_records, validation_tokens = _select_validation_records(
            output_dir,
            validation_records_per_pool,
            writer,
            [pool.name for pool in config.pools],
        )
    finally:
        writer.close()
        database_path.unlink(missing_ok=True)

    _write_manifest(output_dir, "train", writer.manifests["train"])
    _write_manifest(output_dir, "validation", writer.manifests["validation"])
    planned_tokens = float(plans[policy]["estimated_packed_tokens"])
    summary = {
        "policy": policy,
        "config_path": str(config_path),
        "plan_path": str(plan_path),
        "planned_train_tokens": planned_tokens,
        "train_total_records": train_records,
        "train_total_tokens": train_tokens,
        "train_total_label_tokens": train_label_tokens,
        "train_token_deviation": train_tokens - planned_tokens,
        "validation_total_records": validation_records,
        "validation_total_tokens": validation_tokens,
        "train_allocations": train,
        "validation_per_pool": validation,
        **candidate_summary,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--policy", default="p2_quality_constrained")
    parser.add_argument("--benchmark-blocklist", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-records-per-pool", type=int, default=2_500)
    parser.add_argument("--shard-size", type=int, default=100_000)
    return parser


def main() -> int:
    """Build the configured K6 mixture."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = _build_parser().parse_args()
    build_mixture(
        config_path=args.config.resolve(),
        plan_path=args.plan.resolve(),
        policy=args.policy,
        benchmark_blocklist=args.benchmark_blocklist.resolve(),
        output_dir=args.output_dir.resolve(),
        validation_records_per_pool=args.validation_records_per_pool,
        shard_size=args.shard_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
