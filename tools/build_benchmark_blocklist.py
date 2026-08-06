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

"""Build a pinned blocklist from the benchmarks used for Gemma 4 evaluation."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__:
    from tools.benchmark_contamination import BenchmarkBlocklist
else:
    from benchmark_contamination import BenchmarkBlocklist

logger = logging.getLogger(__name__)

LANGUAGE_CONFIGS = ("eng", "hau", "swa", "xho", "yor", "zul")
BELEBELE_CONFIGS = ("eng_Latn", "hau_Latn", "swh_Latn", "xho_Latn", "yor_Latn", "zul_Latn")


@dataclass(frozen=True)
class BenchmarkDataset:
    """Describe one pinned upstream benchmark dataset."""

    dataset_id: str
    revision: str
    configs: tuple[str, ...]
    splits: tuple[str, ...]
    fields: tuple[str, ...]


BENCHMARK_DATASETS = (
    BenchmarkDataset(
        dataset_id="masakhane/afrimmlu",
        revision="96f247619673906ae4c321a3232afc67b98ea57e",
        configs=LANGUAGE_CONFIGS,
        splits=("validation", "dev", "test"),
        fields=("question", "question_and_choices"),
    ),
    BenchmarkDataset(
        dataset_id="masakhane/afrixnli",
        revision="e3ca06b30f3e7af2a86f6c8609ea76fee326bc56",
        configs=LANGUAGE_CONFIGS,
        splits=("validation", "test"),
        fields=("premise", "hypothesis", "premise_and_hypothesis"),
    ),
    BenchmarkDataset(
        dataset_id="masakhane/afrimgsm",
        revision="8e4268d2b94941f18f63f694cf48e4ae26fbec65",
        configs=LANGUAGE_CONFIGS,
        splits=("train", "test"),
        fields=("question",),
    ),
    BenchmarkDataset(
        dataset_id="facebook/belebele",
        revision="7899cdfa4e1e0d733fd77c848e2c273cb1d32be2",
        configs=BELEBELE_CONFIGS,
        splits=("test",),
        fields=("flores_passage", "question", "passage_and_question"),
    ),
)

UPSTREAM_REVISIONS = {
    "masakhane_nlu": "1f8be590da6699aee3dc23de6f63e801e2352eff",
    "lm_evaluation_harness": "f4d4b3de3ee6741a7151a9fe74945ee515262f4c",
    "belebele": "918890beb2290a8d3ef2d7a90369925959e1bacf",
}


def _field_values(record: Mapping[str, Any], fields: tuple[str, ...]) -> Iterable[tuple[str, str]]:
    for field in fields:
        if field == "question_and_choices":
            choices = ast.literal_eval(str(record["choices"]))
            yield field, "\n".join((str(record["question"]), *(str(choice) for choice in choices)))
        elif field == "premise_and_hypothesis":
            yield field, f"{record['premise']}\n{record['hypothesis']}"
        elif field == "passage_and_question":
            yield field, f"{record['flores_passage']}\n{record['question']}"
        else:
            value = record.get(field)
            if isinstance(value, str):
                yield field, value


def _load_dataset_records(dataset: BenchmarkDataset, config: str, split: str) -> Iterable[Mapping[str, Any]]:
    from datasets import load_dataset

    return load_dataset(dataset.dataset_id, config, split=split, revision=dataset.revision)


def _load_local_records(path: Path) -> Iterable[Mapping[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            yield from (json.loads(line) for line in handle if line.strip())
        return
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON array in {path}")
    yield from rows


def build_blocklist(
    output_path: Path,
    *,
    local_benchmark_paths: tuple[Path, ...] = (),
    dataset_records: Mapping[tuple[str, str, str], Iterable[Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Build the evaluation blocklist and return its provenance summary."""
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite benchmark blocklist: {output_path}")

    origins = 0
    with BenchmarkBlocklist(output_path) as blocklist:
        for dataset in BENCHMARK_DATASETS:
            blocklist.set_metadata(f"dataset:{dataset.dataset_id}", dataset.revision)
            for config in dataset.configs:
                for split in dataset.splits:
                    key = (dataset.dataset_id, config, split)
                    records = (
                        dataset_records[key]
                        if dataset_records is not None
                        else _load_dataset_records(dataset, config, split)
                    )
                    benchmark = f"{dataset.dataset_id}:{config}:{split}"
                    for record in records:
                        for field, text in _field_values(record, dataset.fields):
                            blocklist.add_text(text, benchmark=benchmark, field=field)
                            origins += 1

        for path in local_benchmark_paths:
            file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            blocklist.set_metadata(f"local:{path.name}", file_hash)
            for record in _load_local_records(path):
                value = record.get("instruction")
                if isinstance(value, str):
                    blocklist.add_text(value, benchmark=f"afriinstruct-paper:{path.name}", field="instruction")
                    origins += 1

        for name, revision in UPSTREAM_REVISIONS.items():
            blocklist.set_metadata(f"upstream:{name}", revision)
        counts = blocklist.counts()

    summary = {
        "output_path": str(output_path),
        "origins_processed": origins,
        "counts": counts,
        "upstream_revisions": UPSTREAM_REVISIONS,
        "datasets": [
            {
                "dataset_id": dataset.dataset_id,
                "revision": dataset.revision,
                "configs": list(dataset.configs),
                "splits": list(dataset.splits),
                "fields": list(dataset.fields),
            }
            for dataset in BENCHMARK_DATASETS
        ],
    }
    output_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a pinned benchmark contamination blocklist.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--afriinstruct-benchmark", type=Path, nargs="+", default=[])
    return parser


def main() -> int:
    """Build the benchmark blocklist from command-line arguments."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = _build_parser().parse_args()
    summary = build_blocklist(
        args.output.resolve(),
        local_benchmark_paths=tuple(path.resolve() for path in args.afriinstruct_benchmark),
    )
    logger.info("Wrote %d full-text and %d fragment fingerprints", *summary["counts"].values())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
