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

"""Materialize an exact-token AfriInstruct and Inkuba-Instruct mixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any, TextIO

if __package__:
    from tools.benchmark_contamination import BenchmarkBlocklist
else:
    from benchmark_contamination import BenchmarkBlocklist


def _iter_manifest_records(meta_path: Path) -> Iterable[dict[str, Any]]:
    manifest = json.loads(meta_path.read_text(encoding="utf-8"))
    for entry in manifest.values():
        path = Path(entry["file_name"])
        if not path.is_absolute():
            path = meta_path.parent / path
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def _record_digest(record: dict[str, Any]) -> bytes:
    payload = json.dumps(record.get("messages", []), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=20).digest()


def _record_texts(record: dict[str, Any]) -> list[str]:
    return [
        content
        for message in record.get("messages", [])
        if isinstance(message, dict) and isinstance((content := message.get("content")), str)
    ]


class _OutputWriter:
    def __init__(self, output_dir: Path, shard_size: int) -> None:
        self._output_dir = output_dir
        self._shard_size = shard_size
        self._states: dict[tuple[str, str], tuple[int, int]] = {}
        self._handles: dict[Path, TextIO] = {}
        self.manifests: dict[str, dict[str, dict[str, Any]]] = {"train": {}, "validation": {}}

    def write(self, partition: str, source: str, payload: str) -> None:
        key = (partition, source)
        shard_index, records_in_shard = self._states.get(key, (0, 0))
        if records_in_shard >= self._shard_size:
            shard_index += 1
            records_in_shard = 0
        relative_path = Path("processed") / partition / f"{source}-{shard_index:05d}.jsonl"
        path = self._output_dir / relative_path
        if records_in_shard == 0:
            path.parent.mkdir(parents=True, exist_ok=True)
            name = f"{source}-{shard_index:05d}"
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


def build_mixture(
    *,
    afri_train_meta: Path,
    afri_validation_meta: Path,
    inkuba_train_meta: Path,
    inkuba_validation_meta: Path,
    benchmark_blocklist: Path,
    output_dir: Path,
    tokens_per_source: int,
    validation_records_per_source: int = 2_500,
    shard_size: int = 100_000,
) -> dict[str, Any]:
    """Build a stable 50/50 token mixture and fixed validation monitor."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    database_path = output_dir / ".mixture.sqlite3"
    connection = sqlite3.connect(database_path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute(
        "CREATE TABLE candidates (partition TEXT, source TEXT, digest BLOB, tokens INTEGER, payload TEXT, "
        "PRIMARY KEY (partition, source, digest)) WITHOUT ROWID"
    )

    inputs = {
        ("train", "afriinstruct"): afri_train_meta,
        ("validation", "afriinstruct"): afri_validation_meta,
        ("train", "inkuba"): inkuba_train_meta,
        ("validation", "inkuba"): inkuba_validation_meta,
    }
    available: dict[str, dict[str, dict[str, int]]] = {"train": {}, "validation": {}}
    contamination: dict[str, dict[str, int]] = {"afriinstruct": {}, "inkuba": {}}
    with BenchmarkBlocklist(benchmark_blocklist, mode="read-only") as blocklist:
        for (partition, source), meta_path in inputs.items():
            records = 0
            tokens = 0
            source_contamination: dict[str, int] = {}
            for record in _iter_manifest_records(meta_path):
                match = blocklist.find_match(_record_texts(record))
                if match is not None:
                    source_contamination[match.benchmark] = source_contamination.get(match.benchmark, 0) + 1
                    continue
                text_tokens = record.get("_text_tokens")
                if not isinstance(text_tokens, int) or text_tokens <= 0:
                    raise ValueError(f"Record in {meta_path} is missing a positive integer _text_tokens value")
                payload = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                connection.execute(
                    "INSERT OR IGNORE INTO candidates VALUES (?, ?, ?, ?, ?)",
                    (partition, source, _record_digest(record), text_tokens, payload),
                )
                records += 1
                tokens += text_tokens
            available[partition][source] = {"records": records, "tokens": tokens}
            for benchmark, count in source_contamination.items():
                contamination[source][benchmark] = contamination[source].get(benchmark, 0) + count
    connection.commit()

    selected: dict[str, dict[str, dict[str, int]]] = {"train": {}, "validation": {}}
    writer = _OutputWriter(output_dir, shard_size)
    try:
        for source in ("afriinstruct", "inkuba"):
            selected_records = 0
            selected_tokens = 0
            rows = connection.execute(
                "SELECT tokens, payload FROM candidates WHERE partition = 'train' AND source = ? ORDER BY digest",
                (source,),
            )
            for text_tokens, payload in rows:
                if selected_tokens >= tokens_per_source:
                    break
                without_distance = tokens_per_source - selected_tokens
                with_distance = abs(tokens_per_source - selected_tokens - text_tokens)
                if text_tokens > without_distance and without_distance < with_distance:
                    break
                writer.write("train", source, payload)
                selected_records += 1
                selected_tokens += text_tokens
            if selected_tokens < tokens_per_source * 0.999:
                raise ValueError(
                    f"{source} provides only {selected_tokens} selected tokens; requested {tokens_per_source}"
                )
            selected["train"][source] = {"records": selected_records, "tokens": selected_tokens}

            validation_rows = connection.execute(
                "SELECT tokens, payload FROM candidates WHERE partition = 'validation' AND source = ? "
                "ORDER BY digest LIMIT ?",
                (source, validation_records_per_source),
            ).fetchall()
            if len(validation_rows) < validation_records_per_source:
                raise ValueError(
                    f"{source} provides only {len(validation_rows)} validation records; "
                    f"requested {validation_records_per_source}"
                )
            validation_tokens = 0
            for text_tokens, payload in validation_rows:
                writer.write("validation", source, payload)
                validation_tokens += text_tokens
            selected["validation"][source] = {
                "records": len(validation_rows),
                "tokens": validation_tokens,
            }
    finally:
        writer.close()
        connection.close()
        database_path.unlink(missing_ok=True)

    for partition, manifest in writer.manifests.items():
        (output_dir / f"{partition}_meta.json").write_text(
            json.dumps(dict(sorted(manifest.items())), indent=2) + "\n",
            encoding="utf-8",
        )
    total_train_tokens = sum(source["tokens"] for source in selected["train"].values())
    summary = {
        "tokens_per_source": tokens_per_source,
        "validation_records_per_source": validation_records_per_source,
        "available": available,
        "benchmark_contamination_excluded": contamination,
        "selected": selected,
        "realized_train_token_share": {
            source: values["tokens"] / total_train_tokens for source, values in selected["train"].items()
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--afri-train-meta", type=Path, required=True)
    parser.add_argument("--afri-validation-meta", type=Path, required=True)
    parser.add_argument("--inkuba-train-meta", type=Path, required=True)
    parser.add_argument("--inkuba-validation-meta", type=Path, required=True)
    parser.add_argument("--benchmark-blocklist", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokens-per-source", type=int, default=100_000_000)
    parser.add_argument("--validation-records-per-source", type=int, default=2_500)
    parser.add_argument("--shard-size", type=int, default=100_000)
    return parser


def main() -> int:
    """Build the configured instruction mixture."""
    args = _build_parser().parse_args()
    build_mixture(
        afri_train_meta=args.afri_train_meta.resolve(),
        afri_validation_meta=args.afri_validation_meta.resolve(),
        inkuba_train_meta=args.inkuba_train_meta.resolve(),
        inkuba_validation_meta=args.inkuba_validation_meta.resolve(),
        benchmark_blocklist=args.benchmark_blocklist.resolve(),
        output_dir=args.output_dir.resolve(),
        tokens_per_source=args.tokens_per_source,
        validation_records_per_source=args.validation_records_per_source,
        shard_size=args.shard_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
