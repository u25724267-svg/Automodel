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

from tools.benchmark_contamination import BenchmarkBlocklist
from tools.build_instruction_mixture import build_mixture


def _write_pool(root: Path, source: str) -> tuple[Path, Path]:
    manifests = []
    for partition, count in (("train", 30), ("validation", 5)):
        shard = root / source / f"{partition}.jsonl"
        shard.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "messages": [
                    {"role": "user", "content": f"{source} question {index}"},
                    {"role": "assistant", "content": f"{source} answer {index}"},
                ],
                "_text_tokens": 10,
            }
            for index in range(count)
        ]
        shard.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        meta = root / source / f"{partition}_meta.json"
        meta.write_text(
            json.dumps({partition: {"file_name": shard.name, "columns": {"messages": "messages"}}}),
            encoding="utf-8",
        )
        manifests.append(meta)
    return manifests[0], manifests[1]


def test_build_mixture_materializes_equal_token_sources(tmp_path: Path) -> None:
    afri_train, afri_validation = _write_pool(tmp_path, "afri")
    inkuba_train, inkuba_validation = _write_pool(tmp_path, "inkuba")
    blocklist_path = tmp_path / "benchmarks.sqlite3"
    with BenchmarkBlocklist(blocklist_path) as blocklist:
        blocklist.add_text("afri question 0", benchmark="irokobench:eng:test", field="question")

    summary = build_mixture(
        afri_train_meta=afri_train,
        afri_validation_meta=afri_validation,
        inkuba_train_meta=inkuba_train,
        inkuba_validation_meta=inkuba_validation,
        benchmark_blocklist=blocklist_path,
        output_dir=tmp_path / "mixture",
        tokens_per_source=100,
        validation_records_per_source=3,
        shard_size=4,
    )

    assert summary["selected"]["train"]["afriinstruct"]["tokens"] == 100
    assert summary["selected"]["train"]["inkuba"]["tokens"] == 100
    assert summary["realized_train_token_share"] == {"afriinstruct": 0.5, "inkuba": 0.5}
    assert summary["selected"]["validation"]["afriinstruct"]["records"] == 3
    assert summary["selected"]["validation"]["inkuba"]["records"] == 3
    assert summary["benchmark_contamination_excluded"]["afriinstruct"] == {"irokobench:eng:test": 2}
    assert (tmp_path / "mixture" / "train_meta.json").is_file()
    assert (tmp_path / "mixture" / "validation_meta.json").is_file()