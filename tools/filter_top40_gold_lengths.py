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

import argparse
import json
import logging
import os
from collections import Counter
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def filter_precomputed_lengths(data_root: Path, max_text_tokens: int) -> dict[str, object]:
    """Remove records whose precomputed text length exceeds a safe threshold.

    Args:
        data_root: Prepared dataset directory containing train and validation manifests.
        max_text_tokens: Largest accepted precomputed content-token count.

    Returns:
        JSON-serializable filtering summary.
    """
    data_root = data_root.resolve()
    summary_path = data_root / "length_filter_summary.json"
    if summary_path.exists():
        raise FileExistsError(f"Length filtering was already finalized: {summary_path}")

    staged_paths: list[tuple[Path, Path]] = []
    before_by_split: Counter[str] = Counter()
    after_by_split: Counter[str] = Counter()
    removed_by_split: Counter[str] = Counter()
    removed_by_dataset: Counter[str] = Counter()
    try:
        for split in ("train", "validation"):
            manifest_path = data_root / f"{split}_meta.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for name, entry in sorted(manifest.items()):
                source_path = data_root / entry["file_name"]
                staged_path = source_path.with_suffix(source_path.suffix + ".length-filter.tmp")
                if staged_path.exists():
                    raise FileExistsError(f"Staged filter output already exists: {staged_path}")
                dataset_id = name.rsplit("-", 1)[0]
                with (
                    source_path.open(encoding="utf-8") as source_file,
                    staged_path.open("w", encoding="utf-8") as output_file,
                ):
                    for line_number, line in enumerate(source_file, start=1):
                        record = json.loads(line)
                        text_tokens = record.get("_text_tokens")
                        if not isinstance(text_tokens, int) or text_tokens <= 0:
                            raise ValueError(f"Invalid _text_tokens in {source_path}:{line_number}")
                        before_by_split[split] += 1
                        if text_tokens > max_text_tokens:
                            removed_by_split[split] += 1
                            removed_by_dataset[dataset_id] += 1
                            continue
                        output_file.write(line)
                        after_by_split[split] += 1
                staged_paths.append((source_path, staged_path))
    except BaseException:
        for _, staged_path in staged_paths:
            staged_path.unlink(missing_ok=True)
        raise

    for source_path, staged_path in staged_paths:
        os.replace(staged_path, source_path)

    summary = {
        "data_root": str(data_root),
        "max_text_tokens": max_text_tokens,
        "before_by_split": dict(sorted(before_by_split.items())),
        "after_by_split": dict(sorted(after_by_split.items())),
        "removed_by_split": dict(sorted(removed_by_split.items())),
        "removed_by_dataset": dict(sorted(removed_by_dataset.items())),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    LOGGER.info(
        "Removed %d records above %d tokens; %d train and %d validation records remain",
        sum(removed_by_split.values()),
        max_text_tokens,
        after_by_split["train"],
        after_by_split["validation"],
    )
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Filter precomputed Top-40 gold records by token length.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--max-text-tokens", type=_positive_int, default=4_080)
    return parser


def main() -> int:
    """Filter overlong records from prepared train and validation data."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = _build_parser().parse_args()
    filter_precomputed_lengths(args.data_root, args.max_text_tokens)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
