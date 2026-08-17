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
from pathlib import Path

from tools.run_author_benchmarks import LANGUAGES, BenchmarkRun, build_command, run_benchmarks, task_names


def test_default_author_matrix_covers_k6_languages_and_controls(tmp_path: Path) -> None:
    run = BenchmarkRun(
        suite="all",
        base_model="google/gemma-4-E2B-it",
        output_dir=tmp_path,
    )

    tasks = task_names(run)

    assert LANGUAGES == ("eng", "hau", "ibo", "kin", "swa", "xho", "yor", "zul")
    assert len(tasks) == 160
    assert "afrimmlu_direct_ibo_prompt_1" in tasks
    assert "afrixnli_kin_prompt_5" in tasks
    assert "afrimgsm_cot_ibo_prompt_5" in tasks
    assert "belebele_kin_prompt_1" in tasks


def test_author_task_names_cover_upstream_prompts_and_suites(tmp_path: Path) -> None:
    run = BenchmarkRun(
        suite="all",
        base_model="google/gemma-4-E2B-it",
        output_dir=tmp_path,
        languages=("hau", "zul"),
        prompt_ids=(1, 5),
    )

    tasks = task_names(run)

    assert len(tasks) == 16
    assert "afrimmlu_direct_hau_prompt_1" in tasks
    assert "afrixnli_zul_prompt_5" in tasks
    assert "afrimgsm_cot_hau_prompt_5" in tasks
    assert "belebele_zul_prompt_1" in tasks


def test_author_command_uses_multimodal_model_and_standard_peft(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    model_dir = checkpoint / "model"
    model_dir.mkdir(parents=True)
    (model_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    (model_dir / "adapter_model.safetensors").write_bytes(b"adapter")
    run = BenchmarkRun(
        suite="irokobench",
        base_model="google/gemma-4-E2B-it",
        checkpoint_path=checkpoint,
        output_dir=tmp_path / "results",
        languages=("swa",),
        prompt_ids=(1,),
        limit=2,
    )

    command = build_command(run, command_prefix=("python", "tools/lm_eval_compat.py"))

    assert command[:3] == ["python", "tools/lm_eval_compat.py", "run"]
    assert command[command.index("--model") + 1] == "hf-multimodal"
    model_args = command[command.index("--model_args") + 1]
    assert "attn_implementation=sdpa" in model_args
    assert f"peft={model_dir}" in model_args
    assert command[-2:] == ["--limit", "2"]

    provenance = run_benchmarks(run, dry_run=True)
    saved = json.loads((run.output_dir / "run_provenance.json").read_text(encoding="utf-8"))
    assert provenance["task_count"] == 3
    assert saved["upstream_revisions"]["lm_evaluation_harness"]