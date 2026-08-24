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
    _write_plans,
    build_mixture_plans,
    load_config,
    profile_mixture,
)

PROFILE_CONFIG_DIR = Path(__file__).resolve().parents[3] / "examples" / "vlm_finetune" / "gemma4" / "data"


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


def test_profile_excludes_fixed_validation_records(tmp_path: Path) -> None:
    reserved = _row(1)
    train_manifest = _write_manifest(tmp_path, "train", [reserved, _row(2)])
    validation_manifest = _write_manifest(tmp_path, "validation", [reserved])
    config = MixtureConfig(
        model_id="test-tokenizer",
        max_seq_length=10,
        languages=("hau",),
        tasks=("classification",),
        pools=(PoolConfig("train", train_manifest, "train", "revision", "Apache-2.0", "human"),),
        planning=PlanningConfig(packed_token_budget=10),
        fixed_validation_manifest=validation_manifest,
    )

    profile = profile_mixture(config, token_counter=_counter, output_dir=tmp_path / "profile")

    assert profile["fixed_validation_manifest"] == str(validation_manifest)
    assert profile["fixed_validation_records"] == 1
    assert profile["rejections"]["fixed_validation"] == 1
    assert profile["cells"]["train|hau|classification|afrihate"]["records"] == 1


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


def test_fixed_language_plan_redistributes_only_within_language(tmp_path: Path) -> None:
    new_manifest = _write_manifest(tmp_path, "new", [])
    afri_manifest = _write_manifest(tmp_path, "afri", [])
    base = _config(tmp_path, new_manifest, afri_manifest)
    config = MixtureConfig(
        model_id=base.model_id,
        max_seq_length=base.max_seq_length,
        languages=("hau", "yor"),
        tasks=("classification", "translation"),
        pools=base.pools,
        planning=PlanningConfig(
            packed_token_budget=1_000,
            language_token_budget=500,
            max_epochs=1,
            task_token_shares={"classification": 0.6, "translation": 0.4},
            minimum_cell_token_share=0.6,
        ),
    )
    profile = {
        "cells": {
            "new|hau|classification|small": {"records": 10, "text_tokens": 100, "label_tokens": 20},
            "new|hau|translation|large": {"records": 100, "text_tokens": 1_000, "label_tokens": 500},
            "new|yor|classification|large": {"records": 100, "text_tokens": 1_000, "label_tokens": 200},
            "new|yor|translation|large": {"records": 100, "text_tokens": 1_000, "label_tokens": 500},
        }
    }

    plans = build_mixture_plans(config, profile)
    plan = plans["p4_fixed_language_tokens"]

    assert plan["estimated_packed_tokens"] == pytest.approx(1_000)
    assert plan["coverage_shortfall_tokens"] == pytest.approx(0)
    assert plan["language_targets"]["hau"] == pytest.approx(
        {"target_tokens": 500, "planned_tokens": 500, "shortfall_tokens": 0}
    )
    assert plan["language_targets"]["yor"] == pytest.approx(
        {"target_tokens": 500, "planned_tokens": 500, "shortfall_tokens": 0}
    )
    assert plan["cells"]["hau|classification"]["target_tokens"] == pytest.approx(100)
    assert plan["cells"]["hau|translation"]["target_tokens"] == pytest.approx(400)
    assert plan["cells"]["yor|classification"]["target_tokens"] == pytest.approx(300)
    assert plan["cells"]["yor|translation"]["target_tokens"] == pytest.approx(200)
    assert plan["cells"]["hau|classification"]["below_minimum_cell_share"] is True
    assert plan["cells"]["hau|classification"]["desired_target_tokens"] == pytest.approx(300)
    assert plans["p3_token_stratified"]["cells"]["hau|classification"]["target_tokens"] == pytest.approx(100)
    assert plans["p3_token_stratified"]["cells"]["yor|classification"]["target_tokens"] == pytest.approx(500)

    output_dir = tmp_path / "plans"
    _write_plans({"p4_fixed_language_tokens": plan}, output_dir)
    report = (output_dir / "plans.md").read_text(encoding="utf-8")
    assert "### Language budgets" in report
    assert "| hau | 500 | 500 | 0 |" in report
    assert "| hau | classification | 300 | 100 | 20.00% | yes |" in report
    assert "Coverage shortfall: 0 packed tokens." in report


def test_fixed_language_plan_reports_language_capacity_shortfall(tmp_path: Path) -> None:
    new_manifest = _write_manifest(tmp_path, "new", [])
    afri_manifest = _write_manifest(tmp_path, "afri", [])
    base = _config(tmp_path, new_manifest, afri_manifest)
    config = MixtureConfig(
        model_id=base.model_id,
        max_seq_length=base.max_seq_length,
        languages=("hau",),
        tasks=("classification", "translation"),
        pools=base.pools,
        planning=PlanningConfig(
            packed_token_budget=500,
            language_token_budget=500,
            max_epochs=1,
            task_token_shares={"classification": 0.6, "translation": 0.4},
        ),
    )
    profile = {
        "cells": {
            "new|hau|classification|small": {"records": 10, "text_tokens": 100, "label_tokens": 20},
            "new|hau|translation|small": {"records": 20, "text_tokens": 200, "label_tokens": 100},
        }
    }

    plan = build_mixture_plans(config, profile)["p4_fixed_language_tokens"]

    assert plan["estimated_packed_tokens"] == pytest.approx(300)
    assert plan["coverage_shortfall_tokens"] == pytest.approx(200)
    assert plan["language_targets"]["hau"] == pytest.approx(
        {"target_tokens": 500, "planned_tokens": 300, "shortfall_tokens": 200}
    )


def test_load_config_rejects_inconsistent_language_token_budget(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
version: 1
languages: [hau, yor]
tasks: [classification]
pools:
  - name: source
    manifest: train_meta.json
    revision: abc123
    license: Apache-2.0
    quality_tier: human
planning:
  packed_token_budget: 999
  language_token_budget: 500
  task_token_shares: {classification: 1.0}
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="packed_token_budget must equal language_token_budget"):
        load_config(config_path)


@pytest.mark.parametrize(
    ("filename", "language_count", "packed_budget", "minimum_labels", "validation_manifest"),
    (
        (
            "k10_p4_5m_per_language_profile.yaml",
            10,
            50_000_000,
            16_585_053,
            "/data/gemma4-k10/mixture-p2-v1/validation_meta.json",
        ),
        (
            "k14_p4_5m_per_language_profile.yaml",
            14,
            70_000_000,
            23_219_074,
            "/data/gemma4-k14/mixture-p3-final-v2/validation_meta.json",
        ),
    ),
)
def test_fixed_language_profile_configs_use_unified_registry(
    filename: str,
    language_count: int,
    packed_budget: int,
    minimum_labels: int,
    validation_manifest: str,
) -> None:
    config = load_config(PROFILE_CONFIG_DIR / filename)

    assert len(config.languages) == language_count
    assert config.planning.packed_token_budget == packed_budget
    assert config.planning.language_token_budget == 5_000_000
    assert config.planning.minimum_label_tokens == minimum_labels
    assert config.fixed_validation_manifest == Path(validation_manifest)
    assert config.planning.max_epochs == 4.0
    assert config.planning.task_token_shares == {
        "instruction": 0.33,
        "qa": 0.15,
        "translation": 0.10,
        "classification": 0.17,
        "ner": 0.25,
    }
    pool_names = {pool.name for pool in config.pools}
    assert "finerweb_k14" in pool_names
    assert "wolof_sentiment_filtered" in pool_names


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
