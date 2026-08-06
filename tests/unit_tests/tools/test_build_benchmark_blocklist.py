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

from pathlib import Path

from tools.benchmark_contamination import BenchmarkBlocklist
from tools.build_benchmark_blocklist import BENCHMARK_DATASETS, build_blocklist


def test_build_blocklist_uses_every_pinned_dataset_partition(tmp_path: Path) -> None:
    dataset_records = {}
    for dataset in BENCHMARK_DATASETS:
        for config in dataset.configs:
            for split in dataset.splits:
                if dataset.dataset_id.endswith("afrimmlu"):
                    row = {"question": f"MMLU {config} {split}", "choices": "['a', 'b', 'c', 'd']"}
                elif dataset.dataset_id.endswith("afrixnli"):
                    row = {"premise": f"premise {config} {split}", "hypothesis": f"hypothesis {config} {split}"}
                elif dataset.dataset_id.endswith("afrimgsm"):
                    row = {"question": f"MGSM {config} {split}"}
                else:
                    row = {
                        "flores_passage": f"passage {config} {split}",
                        "question": f"question {config} {split}",
                    }
                dataset_records[(dataset.dataset_id, config, split)] = [row]

    path = tmp_path / "benchmarks.sqlite3"
    summary = build_blocklist(path, dataset_records=dataset_records)

    assert summary["origins_processed"] > 0
    with BenchmarkBlocklist(path, mode="read-only") as blocklist:
        assert blocklist.find_match(["MMLU hau test"]) is not None
        assert blocklist.find_match(["premise zul test"]) is not None
        assert blocklist.find_match(["MGSM swa test"]) is not None
        assert blocklist.find_match(["passage yor_Latn test"]) is not None