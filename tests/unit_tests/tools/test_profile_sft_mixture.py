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

import csv
import json
from pathlib import Path

import pytest

from tools.profile_sft_mixture import (
    MixtureConfig,
    PlanningConfig,
    PoolConfig,
    TokenCounts,
    build_mixture_plans,
    load_config,
    profile_mixture,
)


def _write_manifest(root: Path, name: str, rows: list[dict[str, object]]) -> Path:
    shard = root / f"{name}.jsonl"
    shard.parent.mkdir(parents=True, exist_ok=True)
    shard.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    manifest = root / f"{name}_meta.json"
    manifest.write_text(
        json.dumps({name: {"file_name": shard.name, "columns": {"messages": "messages"}}}),
        encoding="utf-8",
    )
    return manifest


def _row(
    index: int,
    *,
    language: str = "hau",
    task: str = "classification",
    source: str = "afrihate",
    answer: str = "normal",
) -> dict[str, object]:
    return {
        "messages": [
            {"role": "user", "content": f"Prompt number {index}"},
            {"role": "assistant", "content": answer},
        ],
        "lang": language,
        "task": task,
        "source": source,
    }


def _counter(messages: list[dict[str, str]]) -> TokenCounts:
    prompt = sum(len(message["content"].split()) for message in messages if message["role"] != "assistant")
    label = sum(len(message["content"].split()) for message in messages if message["role"] == "assistant")
    return TokenCounts(prompt=prompt, label=label, text=prompt + label)


def _config(tmp_path: Path, new_manifest: Path, afri_manifest: Path) -> MixtureConfig:
    return MixtureConfig(
        model_id="test-tokenizer",
        max_seq_length=10,
        languages=("hau", "yor"),
        tasks=("classification", "translation"),
        pools=(
            PoolConfig("afri", afri_manifest, "afriinstruct", "afri-revision", "mixed", "mixed"),
            PoolConfig("new", new_manifest, "new", "new-revision", "Apache-2.0", "human"),
        ),
        planning=PlanningConfig(
            packed_token_budget=1_000,
            temperature=2.0,
            max_epochs=4.0,
            task_example_caps={"classification": 100, "translation": 100},
            default_source_family_cap=0.5,
            fallback_source_families=("afriinstruct",),
            fallback_base_share=0.25,
        ),
        review_samples_per_cell=2,
    )


def test_profile_reports_tokens_deduplication_and_overlength(tmp_path: Path) -> None:
    duplicate = _row(1)
    new_manifest = _write_manifest(
        tmp_path,
        "new",
        [
            duplicate,
            _row(2, language="yor", task="translation", answer="translated sentence"),
            _row(3, language="zul"),
            _row(4, answer="one two three four five six seven eight"),
        ],
    )
    afri_manifest = _write_manifest(tmp_path, "afri", [duplicate, _row(5)])
    config = _config(tmp_path, new_manifest, afri_manifest)

    output_dir = tmp_path / "profile"
    profile = profile_mixture(config, token_counter=_counter, output_dir=output_dir)

    assert profile["rejections"] == {"outside_scope": 1, "overlength": 1, "duplicate": 1}
    assert profile["cells"]["new|hau|classification|afrihate"]["records"] == 1
    assert profile["cells"]["new|hau|classification|afrihate"]["label_tokens"] == 1
    assert profile["cells"]["new|yor|translation|afrihate"]["text_tokens"] == 5
    assert profile["cells"]["afri|hau|classification|afrihate"]["records"] == 1
    assert profile["coverage"]["missing_language_task_cells"] == ["hau|translation", "yor|classification"]
    assert len((output_dir / "review_samples.jsonl").read_text(encoding="utf-8").splitlines()) == 3
    with (output_dir / "cells.csv").open(encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 3


def test_plans_use_capped_examples_and_afriinstruct_anchor(tmp_path: Path) -> None:
    new_manifest = _write_manifest(tmp_path, "new", [])
    afri_manifest = _write_manifest(tmp_path, "afri", [])
    config = _config(tmp_path, new_manifest, afri_manifest)
    profile = {
        "cells": {
            "new|hau|classification|afrihate": {"records": 100, "text_tokens": 400, "label_tokens": 100},
            "afri|hau|classification|afrisenti": {"records": 100, "text_tokens": 400, "label_tokens": 100},
            "new|yor|classification|afrihate": {"records": 100, "text_tokens": 400, "label_tokens": 100},
            "afri|yor|classification|afrisenti": {"records": 100, "text_tokens": 400, "label_tokens": 100},
            "new|hau|translation|pontoon": {"records": 25, "text_tokens": 1_000, "label_tokens": 500},
            "new|yor|translation|pontoon": {"records": 25, "text_tokens": 1_000, "label_tokens": 500},
        }
    }

    plans = build_mixture_plans(config, profile)

    capped = plans["p1_capped_examples_t2"]
    assert capped["estimated_packed_tokens"] == pytest.approx(config.planning.packed_token_budget)
    classification_share = capped["task_example_probabilities"]["classification"]
    translation_share = capped["task_example_probabilities"]["translation"]
    assert classification_share == pytest.approx(0.5857864376)
    assert translation_share == pytest.approx(0.4142135624)
    assert classification_share / translation_share < 2
    assert (
        capped["cells"]["hau|translation"]["estimated_packed_tokens"]
        > capped["cells"]["hau|classification"]["estimated_packed_tokens"]
    )
    quality = plans["p2_quality_constrained"]
    assert quality["estimated_packed_tokens"] == pytest.approx(config.planning.packed_token_budget)
    for cell, values in quality["cells"].items():
        if values["target_records"] > 0 and cell.endswith("|classification"):
            assert values["fallback_share"] == pytest.approx(0.25)


def test_quality_plan_uses_allocated_pool_lengths(tmp_path: Path) -> None:
    new_manifest = _write_manifest(tmp_path, "new", [])
    afri_manifest = _write_manifest(tmp_path, "afri", [])
    config = _config(tmp_path, new_manifest, afri_manifest)
    profile = {
        "cells": {
            "new|hau|classification|primary": {"records": 100, "text_tokens": 10_000, "label_tokens": 100},
            "afri|hau|classification|fallback": {"records": 100, "text_tokens": 100, "label_tokens": 100},
            "new|yor|classification|primary": {"records": 100, "text_tokens": 10_000, "label_tokens": 100},
            "afri|yor|classification|fallback": {"records": 100, "text_tokens": 100, "label_tokens": 100},
            "new|hau|translation|primary": {"records": 100, "text_tokens": 1_000, "label_tokens": 100},
            "new|yor|translation|primary": {"records": 100, "text_tokens": 1_000, "label_tokens": 100},
        }
    }

    plans = build_mixture_plans(config, profile)

    capped = plans["p1_capped_examples_t2"]
    quality = plans["p2_quality_constrained"]
    assert quality["estimated_packed_tokens"] == pytest.approx(config.planning.packed_token_budget)
    assert (
        quality["cells"]["hau|classification"]["estimated_packed_tokens"]
        > capped["cells"]["hau|classification"]["estimated_packed_tokens"]
    )


def test_afriinstruct_expands_to_fill_partial_and_empty_cells(tmp_path: Path) -> None:
    new_manifest = _write_manifest(tmp_path, "new", [])
    afri_manifest = _write_manifest(tmp_path, "afri", [])
    base = _config(tmp_path, new_manifest, afri_manifest)
    config = MixtureConfig(
        model_id=base.model_id,
        max_seq_length=base.max_seq_length,
        languages=("hau",),
        tasks=("classification",),
        pools=base.pools,
        planning=base.planning,
    )
    partial_profile = {
        "cells": {
            "new|hau|classification|afrihate": {"records": 10, "text_tokens": 40, "label_tokens": 10},
            "afri|hau|classification|afrisenti": {"records": 100, "text_tokens": 400, "label_tokens": 100},
        }
    }
    empty_profile = {
        "cells": {"afri|hau|classification|afrisenti": {"records": 100, "text_tokens": 400, "label_tokens": 100}}
    }

    partial = build_mixture_plans(config, partial_profile)["p2_quality_constrained"]
    empty = build_mixture_plans(config, empty_profile)["p2_quality_constrained"]

    assert partial["cells"]["hau|classification"]["fallback_share"] > 0.25
    assert empty["cells"]["hau|classification"]["fallback_share"] == pytest.approx(1.0)
    assert empty["pool_allocations"]["afri|hau|classification"]["allocation_role"] == "fallback"


def test_token_stratified_plan_targets_every_cell_and_reports_label_tokens(tmp_path: Path) -> None:
    new_manifest = _write_manifest(tmp_path, "new", [])
    afri_manifest = _write_manifest(tmp_path, "afri", [])
    base = _config(tmp_path, new_manifest, afri_manifest)
    config = MixtureConfig(
        model_id=base.model_id,
        max_seq_length=base.max_seq_length,
        languages=base.languages,
        tasks=base.tasks,
        pools=base.pools,
        planning=PlanningConfig(
            packed_token_budget=1_000,
            max_epochs=4,
            default_source_family_cap=0.5,
            task_token_shares={"classification": 0.6, "translation": 0.4},
            minimum_label_tokens=100,
        ),
    )
    profile = {
        "cells": {
            "new|hau|classification|primary": {"records": 100, "text_tokens": 1_000, "label_tokens": 200},
            "afri|hau|classification|fallback": {"records": 100, "text_tokens": 1_000, "label_tokens": 100},
            "new|yor|classification|primary": {"records": 100, "text_tokens": 1_000, "label_tokens": 200},
            "afri|yor|classification|fallback": {"records": 100, "text_tokens": 1_000, "label_tokens": 100},
            "new|hau|translation|primary": {"records": 100, "text_tokens": 1_000, "label_tokens": 500},
            "new|yor|translation|primary": {"records": 100, "text_tokens": 1_000, "label_tokens": 500},
        }
    }

    plan = build_mixture_plans(config, profile)["p3_token_stratified"]

    assert plan["estimated_packed_tokens"] == pytest.approx(1_000)
    assert plan["coverage_shortfall_tokens"] == pytest.approx(0)
    assert plan["label_token_shortfall"] == pytest.approx(0)
    assert plan["cells"]["hau|classification"]["target_tokens"] == pytest.approx(300)
    assert plan["cells"]["yor|translation"]["target_tokens"] == pytest.approx(200)
    assert set(plan["cells"]) == {
        "hau|classification",
        "hau|translation",
        "yor|classification",
        "yor|translation",
    }


def test_token_stratified_plan_redistributes_low_capacity_cells(tmp_path: Path) -> None:
    new_manifest = _write_manifest(tmp_path, "new", [])
    afri_manifest = _write_manifest(tmp_path, "afri", [])
    base = _config(tmp_path, new_manifest, afri_manifest)
    config = MixtureConfig(
        model_id=base.model_id,
        max_seq_length=base.max_seq_length,
        languages=("hau", "yor"),
        tasks=("classification",),
        pools=base.pools,
        planning=PlanningConfig(
            packed_token_budget=1_000,
            max_epochs=1,
            task_token_shares={"classification": 1.0},
            minimum_cell_token_share=0.6,
        ),
    )
    profile = {
        "cells": {
            "new|hau|classification|small": {"records": 10, "text_tokens": 100, "label_tokens": 20},
            "new|yor|classification|large": {"records": 100, "text_tokens": 1_000, "label_tokens": 200},
        }
    }

    plan = build_mixture_plans(config, profile)["p3_token_stratified"]

    assert plan["estimated_packed_tokens"] == pytest.approx(1_000)
    assert plan["coverage_shortfall_tokens"] == pytest.approx(0)
    assert plan["cells"]["hau|classification"]["target_tokens"] == pytest.approx(100)
    assert plan["cells"]["yor|classification"]["target_tokens"] == pytest.approx(900)


def test_load_config_rejects_unknown_quality_tier(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
version: 1
languages: [hau]
tasks: [classification]
pools:
  - name: source
    manifest: train_meta.json
    revision: abc123
    license: Apache-2.0
    quality_tier: unknown
planning:
  packed_token_budget: 1000
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported quality tier"):
        load_config(config_path)
