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

import json
from collections import Counter
from pathlib import Path

import pytest

from tools.benchmark_contamination import BenchmarkBlocklist
from tools.build_k6_sft_mixture import _find_exact_token_subset, _token_balanced_targets, build_mixture
from tools.profile_sft_mixture import TokenCounts


def _write_manifest(root: Path, pool_name: str, rows: list[dict[str, object]]) -> Path:
    pool_dir = root / pool_name
    pool_dir.mkdir(parents=True, exist_ok=True)
    shard = pool_dir / "data.jsonl"
    shard.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    manifest = pool_dir / "train_meta.json"
    manifest.write_text(
        json.dumps({"data": {"file_name": shard.name, "columns": {"messages": "messages"}}}),
        encoding="utf-8",
    )
    return manifest


def _config(root: Path, pools: list[tuple[str, Path]], max_epochs: float = 4.0) -> Path:
    config_path = root / "config.yaml"
    config_path.write_text(
        json.dumps(
            {
                "version": 1,
                "model_id": "test-model",
                "max_seq_length": 4096,
                "languages": ["hau"],
                "tasks": ["classification"],
                "pools": [
                    {
                        "name": name,
                        "manifest": str(manifest),
                        "source_family": name,
                        "revision": f"{name}-revision",
                        "license": "Apache-2.0",
                        "quality_tier": "human",
                    }
                    for name, manifest in pools
                ],
                "planning": {
                    "packed_token_budget": 1_000,
                    "temperature": 2.0,
                    "max_epochs": max_epochs,
                    "task_example_caps": {"classification": 100},
                    "default_source_family_cap": 0.5,
                    "fallback_source_families": [],
                    "fallback_base_share": 0.0,
                },
            }
        ),
        encoding="utf-8",
    )
    return config_path


def _plan(root: Path, allocations: dict[str, float], target_records: float) -> Path:
    plan_path = root / "plans.json"
    plan_path.write_text(
        json.dumps(
            {
                "p2_quality_constrained": {
                    "cells": {
                        "hau|classification": {
                            "target_records": target_records,
                            "estimated_packed_tokens": 100.0,
                        }
                    },
                    "pool_allocations": {key: {"target_records": value} for key, value in allocations.items()},
                    "estimated_packed_tokens": 100.0,
                }
            }
        ),
        encoding="utf-8",
    )
    return plan_path


def _row(index: int, pool: str, label: str = "Normal") -> dict[str, object]:
    return {
        "messages": [
            {"role": "user", "content": f"{pool} prompt {index}"},
            {"role": "assistant", "content": label},
        ],
        "lang": "hau",
        "task": "classification",
        "source": pool,
    }


def _counter(messages: list[dict[str, str]]) -> TokenCounts:
    prompt = len(messages[0]["content"].split())
    label = len(messages[-1]["content"].split())
    return TokenCounts(prompt=prompt, label=label, text=prompt + label)


def _blocklist(path: Path) -> Path:
    with BenchmarkBlocklist(path) as blocklist:
        blocklist.add_text("irrelevant", benchmark="test:bench", field="text")
    return path


def test_build_k6_sft_mixture_writes_disjoint_manifests(tmp_path: Path) -> None:
    pool_a = _write_manifest(tmp_path, "pool_a", [_row(0, "pool_a"), _row(1, "pool_a")])
    pool_b = _write_manifest(tmp_path, "pool_b", [_row(0, "pool_b"), _row(1, "pool_b")])
    config = _config(tmp_path, [("pool_a", pool_a), ("pool_b", pool_b)])
    plan = _plan(
        tmp_path,
        {"pool_a|hau|classification": 1.4, "pool_b|hau|classification": 0.6},
        target_records=2.0,
    )

    output = tmp_path / "mixture"
    summary = build_mixture(
        config_path=config,
        plan_path=plan,
        policy="p2_quality_constrained",
        benchmark_blocklist=_blocklist(tmp_path / "benchmarks.sqlite3"),
        output_dir=output,
        validation_records_per_pool=1,
        shard_size=10,
        token_counter=_counter,
    )

    train_rows = [json.loads(line) for path in (output / "processed" / "train").glob("*.jsonl") for line in path.open()]
    validation_rows = [
        json.loads(line) for path in (output / "processed" / "validation").glob("*.jsonl") for line in path.open()
    ]
    train_messages = {json.dumps(row["messages"], sort_keys=True) for row in train_rows}
    validation_messages = {json.dumps(row["messages"], sort_keys=True) for row in validation_rows}

    assert summary["train_total_records"] == 2
    assert summary["train_total_label_tokens"] == 2
    assert summary["validation_total_records"] == 2
    assert train_messages.isdisjoint(validation_messages)
    assert (output / "train_meta.json").is_file()
    assert (output / "validation_meta.json").is_file()


def test_build_k6_sft_mixture_repeats_and_balances_afrihate_labels(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path,
        "afrihate",
        [_row(0, "afrihate", "Normal"), _row(1, "afrihate", "Normal"), _row(2, "afrihate", "Hate")],
    )
    config = _config(tmp_path, [("afrihate", manifest)])
    plan = _plan(tmp_path, {"afrihate|hau|classification": 8.0}, target_records=8.0)

    output = tmp_path / "mixture"
    summary = build_mixture(
        config_path=config,
        plan_path=plan,
        policy="p2_quality_constrained",
        benchmark_blocklist=_blocklist(tmp_path / "benchmarks.sqlite3"),
        output_dir=output,
        validation_records_per_pool=1,
        token_counter=_counter,
    )

    labels = Counter(
        json.loads(line)["messages"][-1]["content"]
        for path in (output / "processed" / "train").glob("*.jsonl")
        for line in path.open()
    )
    allocation = summary["train_allocations"]["afrihate|hau|classification"]

    assert labels == {"Hate": 4, "Normal": 4}
    assert allocation["unique_records"] == 3
    assert allocation["max_repetitions"] == 4
    assert summary["validation_total_records"] == 0


def test_build_mixture_balances_translation_directions(tmp_path: Path) -> None:
    rows = []
    for index in range(4):
        forward = _row(index, "source")
        forward["task"] = "translation"
        forward["direction"] = "eng-hau"
        reverse = _row(index + 10, "source")
        reverse["task"] = "translation"
        reverse["direction"] = "hau-eng"
        rows.extend((forward, reverse))
    manifest = _write_manifest(tmp_path, "source", rows)
    config = _config(tmp_path, [("source", manifest)])
    config_data = json.loads(config.read_text(encoding="utf-8"))
    config_data["tasks"] = ["translation"]
    config.write_text(json.dumps(config_data), encoding="utf-8")
    plan = _plan(tmp_path, {"source|hau|translation": 6.0}, target_records=6.0)
    plan_data = json.loads(plan.read_text(encoding="utf-8"))
    plan_data["p2_quality_constrained"]["cells"] = {
        "hau|translation": {"target_records": 6.0, "estimated_packed_tokens": 100.0}
    }
    plan.write_text(json.dumps(plan_data), encoding="utf-8")

    output = tmp_path / "mixture"
    summary = build_mixture(
        config_path=config,
        plan_path=plan,
        policy="p2_quality_constrained",
        benchmark_blocklist=_blocklist(tmp_path / "benchmarks.sqlite3"),
        output_dir=output,
        validation_records_per_pool=0,
        token_counter=_counter,
    )

    strata = summary["train_allocations"]["source|hau|translation"]["strata"]
    assert strata["direction:eng-hau"]["records"] == 3
    assert strata["direction:hau-eng"]["records"] == 3


def test_build_mixture_accepts_p4_and_rejects_language_shortfall(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, "source", [_row(0, "source"), _row(1, "source"), _row(2, "source")])
    config = _config(tmp_path, [("source", manifest)])
    config_data = json.loads(config.read_text(encoding="utf-8"))
    config_data["planning"]["packed_token_budget"] = 12
    config_data["planning"]["language_token_budget"] = 12
    config_data["planning"]["task_token_shares"] = {"classification": 1.0}
    config.write_text(json.dumps(config_data), encoding="utf-8")
    plan = _plan(tmp_path, {"source|hau|classification": 2.0}, target_records=2.0)
    plan_data = json.loads(plan.read_text(encoding="utf-8"))
    plan_data["p4_fixed_language_tokens"] = plan_data.pop("p2_quality_constrained")
    plan_data["p4_fixed_language_tokens"]["estimated_packed_tokens"] = 12.0
    plan_data["p4_fixed_language_tokens"]["language_targets"] = {
        "hau": {"target_tokens": 12.0, "planned_tokens": 12.0, "shortfall_tokens": 0.0}
    }
    plan_data["p4_fixed_language_tokens"]["pool_allocations"]["source|hau|classification"]["target_tokens"] = 12.0
    plan_data["p4_fixed_language_tokens"]["coverage_shortfall_tokens"] = 0.0
    plan.write_text(json.dumps(plan_data), encoding="utf-8")

    output = tmp_path / "mixture"
    summary = build_mixture(
        config_path=config,
        plan_path=plan,
        policy="p4_fixed_language_tokens",
        benchmark_blocklist=_blocklist(tmp_path / "benchmarks.sqlite3"),
        output_dir=output,
        validation_records_per_pool=0,
        token_counter=_counter,
    )

    assert summary["policy"] == "p4_fixed_language_tokens"
    assert summary["train_total_records"] == 3
    assert summary["train_total_tokens"] == 12
    assert summary["train_token_deviation"] == 0
    assert summary["language_reconciliation"]["hau"] == {
        "target_tokens": 12,
        "before_tokens": 8,
        "added_records": 1,
        "removed_records": 0,
        "after_tokens": 12,
    }

    shortfall_plan = tmp_path / "shortfall-plans.json"
    plan_data["p4_fixed_language_tokens"]["coverage_shortfall_tokens"] = 1.0
    shortfall_plan.write_text(json.dumps(plan_data), encoding="utf-8")
    with pytest.raises(ValueError, match="uncovered packed tokens"):
        build_mixture(
            config_path=config,
            plan_path=shortfall_plan,
            policy="p4_fixed_language_tokens",
            benchmark_blocklist=_blocklist(tmp_path / "shortfall-benchmarks.sqlite3"),
            output_dir=tmp_path / "shortfall-mixture",
            validation_records_per_pool=0,
            token_counter=_counter,
        )


def test_build_mixture_preserves_fixed_validation_and_excludes_it_from_train(tmp_path: Path) -> None:
    reserved = _row(0, "source")
    reserved_counts = _counter(reserved["messages"])
    reserved["_prompt_tokens"] = reserved_counts.prompt
    reserved["_label_tokens"] = reserved_counts.label
    reserved["_text_tokens"] = reserved_counts.text
    train_manifest = _write_manifest(
        tmp_path,
        "source",
        [reserved, _row(1, "source"), _row(2, "source")],
    )
    validation_manifest = _write_manifest(tmp_path / "fixed", "validation", [reserved])
    config = _config(tmp_path, [("source", train_manifest)])
    config_data = json.loads(config.read_text(encoding="utf-8"))
    config_data["fixed_validation_manifest"] = str(validation_manifest)
    config_data["planning"]["packed_token_budget"] = 8
    config_data["planning"]["language_token_budget"] = 8
    config_data["planning"]["task_token_shares"] = {"classification": 1.0}
    config.write_text(json.dumps(config_data), encoding="utf-8")
    plan = _plan(tmp_path, {"source|hau|classification": 2.0}, target_records=2.0)
    plan_data = json.loads(plan.read_text(encoding="utf-8"))
    plan_data["p4_fixed_language_tokens"] = plan_data.pop("p2_quality_constrained")
    plan_data["p4_fixed_language_tokens"]["estimated_packed_tokens"] = 8.0
    plan_data["p4_fixed_language_tokens"]["language_targets"] = {
        "hau": {"target_tokens": 8.0, "planned_tokens": 8.0, "shortfall_tokens": 0.0}
    }
    plan_data["p4_fixed_language_tokens"]["pool_allocations"]["source|hau|classification"]["target_tokens"] = 8.0
    plan_data["p4_fixed_language_tokens"]["coverage_shortfall_tokens"] = 0.0
    plan.write_text(json.dumps(plan_data), encoding="utf-8")

    output = tmp_path / "mixture"
    summary = build_mixture(
        config_path=config,
        plan_path=plan,
        policy="p4_fixed_language_tokens",
        benchmark_blocklist=_blocklist(tmp_path / "benchmarks.sqlite3"),
        output_dir=output,
        validation_records_per_pool=0,
        token_counter=_counter,
    )

    train_rows = [json.loads(line) for path in (output / "processed" / "train").glob("*.jsonl") for line in path.open()]
    validation_rows = [json.loads(line) for line in (output / "data.jsonl").open()]
    train_messages = {json.dumps(row["messages"], sort_keys=True) for row in train_rows}
    validation_messages = {json.dumps(row["messages"], sort_keys=True) for row in validation_rows}
    assert train_messages.isdisjoint(validation_messages)
    assert summary["train_total_records"] == 2
    assert summary["validation_total_records"] == 1
    assert summary["rejections"]["fixed_validation"] == 1
    assert summary["fixed_validation_manifest"] == str(validation_manifest.resolve())
    assert (output / "validation_meta.json").read_bytes() == validation_manifest.read_bytes()
    assert (output / "data.jsonl").read_bytes() == (validation_manifest.parent / "data.jsonl").read_bytes()


def test_build_mixture_rejects_unrealizable_fixed_language_budget(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, "source", [_row(0, "source"), _row(1, "source"), _row(2, "source")])
    config = _config(tmp_path, [("source", manifest)])
    config_data = json.loads(config.read_text(encoding="utf-8"))
    config_data["planning"]["packed_token_budget"] = 10
    config_data["planning"]["language_token_budget"] = 10
    config_data["planning"]["task_token_shares"] = {"classification": 1.0}
    config.write_text(json.dumps(config_data), encoding="utf-8")
    plan = _plan(tmp_path, {"source|hau|classification": 2.0}, target_records=2.0)
    plan_data = json.loads(plan.read_text(encoding="utf-8"))
    plan_data["p4_fixed_language_tokens"] = plan_data.pop("p2_quality_constrained")
    plan_data["p4_fixed_language_tokens"]["estimated_packed_tokens"] = 10.0
    plan_data["p4_fixed_language_tokens"]["language_targets"] = {
        "hau": {"target_tokens": 10.0, "planned_tokens": 10.0, "shortfall_tokens": 0.0}
    }
    plan_data["p4_fixed_language_tokens"]["pool_allocations"]["source|hau|classification"]["target_tokens"] = 10.0
    plan_data["p4_fixed_language_tokens"]["coverage_shortfall_tokens"] = 0.0
    plan.write_text(json.dumps(plan_data), encoding="utf-8")

    with pytest.raises(ValueError, match="Cannot reconcile hau from 8 to 10 tokens"):
        build_mixture(
            config_path=config,
            plan_path=plan,
            policy="p4_fixed_language_tokens",
            benchmark_blocklist=_blocklist(tmp_path / "benchmarks.sqlite3"),
            output_dir=tmp_path / "mixture",
            validation_records_per_pool=0,
            token_counter=_counter,
        )


def test_build_mixture_removes_records_to_reconcile_fixed_language_budget(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, "source", [_row(0, "source"), _row(1, "source"), _row(2, "source")])
    config = _config(tmp_path, [("source", manifest)])
    config_data = json.loads(config.read_text(encoding="utf-8"))
    config_data["planning"]["packed_token_budget"] = 8
    config_data["planning"]["language_token_budget"] = 8
    config_data["planning"]["task_token_shares"] = {"classification": 1.0}
    config.write_text(json.dumps(config_data), encoding="utf-8")
    plan = _plan(tmp_path, {"source|hau|classification": 3.0}, target_records=3.0)
    plan_data = json.loads(plan.read_text(encoding="utf-8"))
    plan_data["p4_fixed_language_tokens"] = plan_data.pop("p2_quality_constrained")
    plan_data["p4_fixed_language_tokens"]["estimated_packed_tokens"] = 8.0
    plan_data["p4_fixed_language_tokens"]["language_targets"] = {
        "hau": {"target_tokens": 8.0, "planned_tokens": 8.0, "shortfall_tokens": 0.0}
    }
    plan_data["p4_fixed_language_tokens"]["pool_allocations"]["source|hau|classification"]["target_tokens"] = 8.0
    plan_data["p4_fixed_language_tokens"]["coverage_shortfall_tokens"] = 0.0
    plan.write_text(json.dumps(plan_data), encoding="utf-8")

    summary = build_mixture(
        config_path=config,
        plan_path=plan,
        policy="p4_fixed_language_tokens",
        benchmark_blocklist=_blocklist(tmp_path / "benchmarks.sqlite3"),
        output_dir=tmp_path / "mixture",
        validation_records_per_pool=0,
        token_counter=_counter,
    )

    assert summary["train_total_records"] == 2
    assert summary["train_total_tokens"] == 8
    assert summary["language_reconciliation"]["hau"] == {
        "target_tokens": 8,
        "before_tokens": 12,
        "added_records": 0,
        "removed_records": 1,
        "after_tokens": 8,
    }


def test_token_balanced_targets_avoid_length_skew() -> None:
    candidates = {
        "long": [(b"a", 100, 10, "{}") for _ in range(10)],
        "short": [(b"b", 10, 1, "{}") for _ in range(10)],
    }

    targets = _token_balanced_targets(10, candidates, max_epochs=1)

    assert targets == {"long": 1, "short": 9}


def test_find_exact_token_subset_reconstructs_or_rejects_target() -> None:
    candidates = [("first", 7), ("second", 11), ("third", 13), ("fourth", 17)]

    assert _find_exact_token_subset(candidates, 31) == ["first", "second", "third"]
    assert _find_exact_token_subset(candidates, 5) is None
