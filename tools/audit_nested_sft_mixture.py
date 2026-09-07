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

"""Audit a locked-base SFT mixture and its disjoint extension."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

if __package__:
    from tools.benchmark_contamination import BenchmarkBlocklist
    from tools.profile_sft_mixture import (
        TokenCounts,
        _make_token_counter,
        _normalize_record,
        _record_digest,
        _record_texts,
        build_mixture_plans,
        load_config,
    )
else:
    from benchmark_contamination import BenchmarkBlocklist
    from profile_sft_mixture import (
        TokenCounts,
        _make_token_counter,
        _normalize_record,
        _record_digest,
        _record_texts,
        build_mixture_plans,
        load_config,
    )

TokenCounter = Callable[[list[dict[str, str]]], TokenCounts]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Manifest must be a mapping: {path}")
    return value


def _manifest_rows(
    manifest_path: Path, manifest: Mapping[str, Mapping[str, Any]]
) -> Iterable[tuple[str, Path, int, dict[str, Any]]]:
    root = manifest_path.parent.resolve()
    for name, entry in sorted(manifest.items()):
        relative_path = Path(str(entry["file_name"]))
        path = (root / relative_path).resolve()
        if relative_path.is_absolute() or root not in path.parents or not path.is_file():
            raise ValueError(f"Invalid manifest shard path: {relative_path}")
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if line.strip():
                    yield name, path, line_number, json.loads(line)


def _validated_row(
    row: Mapping[str, Any], path: Path, line_number: int, token_counter: TokenCounter
) -> tuple[dict[str, Any], int, int]:
    record, reason = _normalize_record(row)
    if record is None:
        raise ValueError(f"Invalid record in {path}:{line_number}: {reason or 'invalid'}")
    text_tokens = row.get("_text_tokens")
    label_tokens = row.get("_label_tokens")
    if not isinstance(text_tokens, int) or text_tokens <= 0:
        raise ValueError(f"Invalid _text_tokens in {path}:{line_number}")
    if not isinstance(label_tokens, int) or label_tokens <= 0:
        raise ValueError(f"Invalid _label_tokens in {path}:{line_number}")
    prompt_tokens = row.get("_prompt_tokens")
    counts = token_counter(record["messages"])
    if (prompt_tokens, label_tokens, text_tokens) != (counts.prompt, counts.label, counts.text):
        raise ValueError(f"Token metadata mismatch in {path}:{line_number}")
    return record, text_tokens, label_tokens


def audit_nested_mixture(
    *,
    output_dir: Path,
    locked_train_manifest: Path,
    fixed_validation_manifest: Path,
    config_path: Path,
    plan_path: Path,
    profile_path: Path,
    profile_path_b: Path,
    benchmark_blocklist: Path,
    max_combined_repetitions: int = 4,
    token_counter: TokenCounter | None = None,
) -> dict[str, Any]:
    """Audit a materialized nested mixture and write its integrity record."""
    config = load_config(config_path)
    counter = token_counter or _make_token_counter(config.model_id)
    plans = json.loads(plan_path.read_text(encoding="utf-8"))
    plan = plans.get("p2_nested_extension")
    if not isinstance(plan, dict):
        raise ValueError("Plan does not contain p2_nested_extension")
    if plan.get("coverage_shortfall_tokens", 0) > 1e-6 or plan.get("label_token_shortfall", 0) > 1e-6:
        raise ValueError("Nested plan has an unresolved token or label shortfall")
    if profile_path.resolve() == profile_path_b.resolve():
        raise ValueError("Independent profile paths must be distinct")
    if profile_path.read_bytes() != profile_path_b.read_bytes():
        raise ValueError("Independent profile passes are not byte-identical")
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if build_mixture_plans(config, profile)["p2_nested_extension"] != plan:
        raise ValueError("Supplied nested plan does not match the profile and config")
    summary_path = output_dir / "summary.json"
    train_manifest_path = output_dir / "train_meta.json"
    validation_manifest_path = output_dir / "validation_meta.json"
    if (output_dir / ".mixture.sqlite3").exists():
        raise ValueError("Temporary materialization database still exists")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if Path(str(summary.get("config_path"))).resolve() != config_path.resolve():
        raise ValueError("Materialized summary references a different config")
    if Path(str(summary.get("plan_path"))).resolve() != plan_path.resolve():
        raise ValueError("Materialized summary references a different plan")
    train_manifest = _load_manifest(train_manifest_path)
    validation_manifest = _load_manifest(validation_manifest_path)
    locked_manifest = _load_manifest(locked_train_manifest)
    fixed_validation = _load_manifest(fixed_validation_manifest)
    if validation_manifest != fixed_validation:
        raise ValueError("Validation manifest differs from the frozen monitor")

    artifact_hashes = {
        "config": _sha256(config_path),
        "plan": _sha256(plan_path),
        "profile_a": _sha256(profile_path),
        "profile_b": _sha256(profile_path_b),
        "benchmark_blocklist": _sha256(benchmark_blocklist),
        "locked_train_manifest": _sha256(locked_train_manifest),
        "fixed_validation_manifest": _sha256(fixed_validation_manifest),
        "summary": _sha256(summary_path),
        "train_manifest": _sha256(train_manifest_path),
        "validation_manifest": _sha256(validation_manifest_path),
        "materializer": _sha256(Path(__file__).with_name("build_k6_sft_mixture.py")),
        "auditor": _sha256(Path(__file__)),
    }
    base_shard_hashes: dict[str, str] = {}
    for name, entry in locked_manifest.items():
        if train_manifest.get(name) != entry:
            raise ValueError(f"Locked manifest entry changed: {name}")
        relative_path = Path(str(entry["file_name"]))
        source = locked_train_manifest.parent / relative_path
        destination = output_dir / relative_path
        if source.read_bytes() != destination.read_bytes():
            raise ValueError(f"Locked shard bytes changed: {relative_path}")
        base_shard_hashes[relative_path.as_posix()] = _sha256(destination)
    for name, entry in fixed_validation.items():
        relative_path = Path(str(entry["file_name"]))
        source = fixed_validation_manifest.parent / relative_path
        destination = output_dir / relative_path
        if source.read_bytes() != destination.read_bytes():
            raise ValueError(f"Frozen validation shard bytes changed: {name}")

    base_digests: Counter[bytes] = Counter()
    base_tokens = 0
    base_label_tokens = 0
    for _, path, line_number, row in _manifest_rows(locked_train_manifest, locked_manifest):
        record, text_tokens, label_tokens = _validated_row(row, path, line_number, counter)
        base_digests[_record_digest(record)] += 1
        base_tokens += text_tokens
        base_label_tokens += label_tokens

    train_digests: Counter[bytes] = Counter()
    extension_digests: Counter[bytes] = Counter()
    train_records = 0
    train_tokens = 0
    train_label_tokens = 0
    extension_records = 0
    extension_tokens = 0
    extension_label_tokens = 0
    benchmark_matches: Counter[str] = Counter()
    extension_allocations: dict[str, Counter[str]] = {}
    extension_language_task_tokens: Counter[str] = Counter()
    extension_pool_tokens: Counter[str] = Counter()
    train_shard_hashes: dict[str, str] = {}
    validation_shard_hashes: dict[str, str] = {}
    extension_names = set(train_manifest) - set(locked_manifest)
    for entry in train_manifest.values():
        relative_path = Path(str(entry["file_name"]))
        train_shard_hashes[relative_path.as_posix()] = _sha256(output_dir / relative_path)
    for entry in validation_manifest.values():
        relative_path = Path(str(entry["file_name"]))
        validation_shard_hashes[relative_path.as_posix()] = _sha256(output_dir / relative_path)
    with BenchmarkBlocklist(benchmark_blocklist, mode="read-only") as blocklist:
        for name, path, line_number, row in _manifest_rows(train_manifest_path, train_manifest):
            record, text_tokens, label_tokens = _validated_row(row, path, line_number, counter)
            digest = _record_digest(record)
            train_digests[digest] += 1
            train_records += 1
            train_tokens += text_tokens
            train_label_tokens += label_tokens
            if name in extension_names:
                matching_pools = [pool.name for pool in config.pools if name.startswith(f"extension-{pool.name}-")]
                if len(matching_pools) != 1:
                    raise ValueError(f"Cannot identify extension pool from manifest entry: {name}")
                pool = matching_pools[0]
                extension_digests[digest] += 1
                extension_records += 1
                extension_tokens += text_tokens
                extension_label_tokens += label_tokens
                allocation = extension_allocations.setdefault(f"{pool}|{record['lang']}|{record['task']}", Counter())
                allocation["records"] += 1
                allocation["tokens"] += text_tokens
                allocation["label_tokens"] += label_tokens
                extension_language_task_tokens[f"{record['lang']}|{record['task']}"] += text_tokens
                extension_pool_tokens[pool] += text_tokens
            match = blocklist.find_match(_record_texts(record))
            if match is not None:
                benchmark_matches[match.benchmark] += 1

        validation_digests: set[bytes] = set()
        for _, path, line_number, row in _manifest_rows(validation_manifest_path, validation_manifest):
            record, _, _ = _validated_row(row, path, line_number, counter)
            digest = _record_digest(record)
            validation_digests.add(digest)
            match = blocklist.find_match(_record_texts(record))
            if match is not None:
                benchmark_matches[match.benchmark] += 1

    if set(base_digests) & set(extension_digests):
        raise ValueError("Locked base and extension overlap")
    if any(repetitions != 1 for repetitions in extension_digests.values()):
        raise ValueError("Extension contains repeated records")
    if set(train_digests) & validation_digests:
        raise ValueError("Training and frozen validation overlap")
    if max(train_digests.values(), default=0) > max_combined_repetitions:
        raise ValueError("Combined training repetitions exceed the accepted cap")
    if benchmark_matches:
        raise ValueError(f"Benchmark matches found: {dict(benchmark_matches)}")
    if summary.get("policy") != "p2_nested_extension":
        raise ValueError(f"Unexpected materialization policy: {summary.get('policy')}")
    if extension_tokens != config.planning.packed_token_budget:
        raise ValueError(f"Extension has {extension_tokens} tokens; expected {config.planning.packed_token_budget}")
    if extension_label_tokens < config.planning.minimum_label_tokens:
        raise ValueError(
            f"Extension has {extension_label_tokens} label tokens; requires {config.planning.minimum_label_tokens}"
        )
    if train_tokens != summary.get("base_train_tokens", 0) + config.planning.packed_token_budget:
        raise ValueError("Combined token total does not equal locked base plus extension target")
    expected = {
        "base_train_records": sum(base_digests.values()),
        "base_train_unique_records": len(base_digests),
        "base_train_tokens": base_tokens,
        "base_train_label_tokens": base_label_tokens,
        "train_total_records": train_records,
        "train_total_tokens": train_tokens,
        "train_total_label_tokens": train_label_tokens,
        "extension_train_records": extension_records,
        "extension_train_tokens": extension_tokens,
        "extension_train_label_tokens": extension_label_tokens,
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise ValueError(f"Summary mismatch for {key}: {summary.get(key)} != {value}")
    accounting_expected = {
        "planned_extension_tokens": config.planning.packed_token_budget,
        "planned_train_tokens": base_tokens + config.planning.packed_token_budget,
        "extension_token_deviation": 0,
        "train_token_deviation": 0,
    }
    for key, value in accounting_expected.items():
        if summary.get(key) != value:
            raise ValueError(f"Summary accounting mismatch for {key}: {summary.get(key)} != {value}")
    for key, values in extension_allocations.items():
        summary_values = summary["train_allocations"].get(key)
        if summary_values is None or any(summary_values.get(metric) != value for metric, value in values.items()):
            raise ValueError(f"Summary allocation mismatch for {key}")
    if set(summary["train_allocations"]) != set(extension_allocations):
        raise ValueError("Summary and audited extension allocation keys differ")

    quality_by_pool = {pool.name: pool.quality_tier for pool in config.pools}
    quality_tokens: Counter[str] = Counter()
    for pool, tokens in extension_pool_tokens.items():
        quality_tokens[quality_by_pool[pool]] += tokens
    target_language_task_tokens: Counter[str] = Counter()
    target_source_tokens: Counter[str] = Counter()
    for key, values in plan["pool_allocations"].items():
        pool, language, task = key.split("|", maxsplit=2)
        target_language_task_tokens[f"{language}|{task}"] += round(float(values["target_tokens"]))
        target_source_tokens[pool] += round(float(values["target_tokens"]))
    language_task_deviations = {
        key: {
            "planned_tokens": target_language_task_tokens.get(key, 0),
            "realized_tokens": extension_language_task_tokens.get(key, 0),
            "deviation_tokens": extension_language_task_tokens.get(key, 0) - target_language_task_tokens.get(key, 0),
        }
        for key in sorted(set(target_language_task_tokens) | set(extension_language_task_tokens))
    }
    source_deviations = {
        pool: {
            "planned_tokens": target_source_tokens.get(pool, 0),
            "realized_tokens": extension_pool_tokens.get(pool, 0),
            "deviation_tokens": extension_pool_tokens.get(pool, 0) - target_source_tokens.get(pool, 0),
        }
        for pool in sorted(set(target_source_tokens) | set(extension_pool_tokens))
    }
    audit = {
        "status": "accepted",
        "output_dir": str(output_dir),
        "base_records": sum(base_digests.values()),
        "base_unique_records": len(base_digests),
        "extension_records": extension_records,
        "extension_unique_records": len(extension_digests),
        "extension_tokens": extension_tokens,
        "extension_label_tokens": extension_label_tokens,
        "combined_records": train_records,
        "combined_unique_records": len(train_digests),
        "combined_tokens": train_tokens,
        "combined_label_tokens": train_label_tokens,
        "maximum_combined_repetitions": max(train_digests.values(), default=0),
        "train_validation_overlap": 0,
        "base_extension_overlap": 0,
        "benchmark_matches": 0,
        "extension_quality_tokens": dict(sorted(quality_tokens.items())),
        "artifact_hashes": artifact_hashes,
        "base_shard_hashes": dict(sorted(base_shard_hashes.items())),
        "train_shard_hashes": dict(sorted(train_shard_hashes.items())),
        "validation_shard_hashes": dict(sorted(validation_shard_hashes.items())),
        "extension_language_task_tokens": dict(sorted(extension_language_task_tokens.items())),
        "planned_language_task_tokens": dict(sorted(target_language_task_tokens.items())),
        "language_task_deviations": language_task_deviations,
        "extension_source_tokens": dict(sorted(extension_pool_tokens.items())),
        "planned_source_tokens": dict(sorted(target_source_tokens.items())),
        "source_deviations": source_deviations,
    }
    (output_dir / "final_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--locked-train-manifest", type=Path, required=True)
    parser.add_argument("--fixed-validation-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--profile-b", type=Path, required=True)
    parser.add_argument("--benchmark-blocklist", type=Path, required=True)
    parser.add_argument("--max-combined-repetitions", type=int, default=4)
    return parser


def main() -> int:
    """Audit a nested SFT mixture from command-line arguments."""
    args = _build_parser().parse_args()
    audit_nested_mixture(
        output_dir=args.output_dir.resolve(),
        locked_train_manifest=args.locked_train_manifest.resolve(),
        fixed_validation_manifest=args.fixed_validation_manifest.resolve(),
        config_path=args.config.resolve(),
        plan_path=args.plan.resolve(),
        profile_path=args.profile.resolve(),
        profile_path_b=args.profile_b.resolve(),
        benchmark_blocklist=args.benchmark_blocklist.resolve(),
        max_combined_repetitions=args.max_combined_repetitions,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
