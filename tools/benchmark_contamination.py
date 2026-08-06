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

"""Persistent text fingerprints for benchmark contamination prevention."""

from __future__ import annotations

import hashlib
import sqlite3
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

CANONICALIZATION_VERSION = "nfkc-casefold-whitespace-v1"
FRAGMENT_TOKENS = 12


def canonicalize_text(text: str) -> str:
    """Return the stable representation used for contamination checks."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(normalized.split())


def text_digest(text: str) -> bytes:
    """Return a compact digest of canonicalized text."""
    return hashlib.blake2b(canonicalize_text(text).encode("utf-8"), digest_size=20).digest()


def fragment_digests(text: str) -> set[bytes]:
    """Return digests for fixed-width token windows in canonicalized text."""
    tokens = canonicalize_text(text).split()
    if len(tokens) < FRAGMENT_TOKENS:
        return set()
    return {
        hashlib.blake2b(" ".join(tokens[index : index + FRAGMENT_TOKENS]).encode("utf-8"), digest_size=20).digest()
        for index in range(len(tokens) - FRAGMENT_TOKENS + 1)
    }


@dataclass(frozen=True)
class BlocklistMatch:
    """Describe the benchmark field that matched a candidate text."""

    benchmark: str
    field: str


class BenchmarkBlocklist:
    """Store benchmark text digests and their provenance in SQLite."""

    def __init__(self, path: Path, *, mode: Literal["create", "read-only"] = "create") -> None:
        self.path = path
        self._read_only = mode == "read-only"
        if self._read_only:
            if not path.is_file():
                raise FileNotFoundError(f"Benchmark blocklist does not exist: {path}")
            self._connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            self._digest_cache = {row[0] for row in self._connection.execute("SELECT digest FROM digests")}
            self._fragment_cache = {row[0] for row in self._connection.execute("SELECT DISTINCT digest FROM fragments")}
        else:
            self._digest_cache = None
            self._fragment_cache = None
            path.parent.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(path)
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=NORMAL")
            self._connection.execute("CREATE TABLE IF NOT EXISTS digests (digest BLOB PRIMARY KEY) WITHOUT ROWID")
            self._connection.execute(
                "CREATE TABLE IF NOT EXISTS origins ("
                "digest BLOB NOT NULL, benchmark TEXT NOT NULL, field TEXT NOT NULL, "
                "PRIMARY KEY (digest, benchmark, field), "
                "FOREIGN KEY (digest) REFERENCES digests(digest)) WITHOUT ROWID"
            )
            self._connection.execute(
                "CREATE TABLE IF NOT EXISTS fragments ("
                "digest BLOB NOT NULL, benchmark TEXT NOT NULL, field TEXT NOT NULL, "
                "PRIMARY KEY (digest, benchmark, field)) WITHOUT ROWID"
            )
            self._connection.execute(
                "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID"
            )
            self.set_metadata("canonicalization_version", CANONICALIZATION_VERSION)

    def add_text(self, text: str, *, benchmark: str, field: str) -> bool:
        """Add a benchmark text field and return whether its digest was new."""
        self._require_writable()
        canonical = canonicalize_text(text)
        if not canonical:
            return False
        digest = text_digest(canonical)
        cursor = self._connection.execute("INSERT OR IGNORE INTO digests (digest) VALUES (?)", (digest,))
        self._connection.execute(
            "INSERT OR IGNORE INTO origins (digest, benchmark, field) VALUES (?, ?, ?)",
            (digest, benchmark, field),
        )
        self._connection.executemany(
            "INSERT OR IGNORE INTO fragments (digest, benchmark, field) VALUES (?, ?, ?)",
            ((fragment, benchmark, field) for fragment in fragment_digests(canonical)),
        )
        return cursor.rowcount == 1

    def find_match(self, texts: list[str]) -> BlocklistMatch | None:
        """Return provenance for the first candidate text present in the blocklist."""
        for text in texts:
            if not canonicalize_text(text):
                continue
            digest = text_digest(text)
            if self._digest_cache is None or digest in self._digest_cache:
                row = self._connection.execute(
                    "SELECT benchmark, field FROM origins WHERE digest = ? ORDER BY benchmark, field LIMIT 1",
                    (digest,),
                ).fetchone()
                if row is not None:
                    return BlocklistMatch(benchmark=row[0], field=row[1])
            for fragment in fragment_digests(text):
                if self._fragment_cache is not None and fragment not in self._fragment_cache:
                    continue
                row = self._connection.execute(
                    "SELECT benchmark, field FROM fragments WHERE digest = ? ORDER BY benchmark, field LIMIT 1",
                    (fragment,),
                ).fetchone()
                if row is not None:
                    return BlocklistMatch(benchmark=row[0], field=f"{row[1]}:fragment")
        return None

    def counts(self) -> dict[str, int]:
        """Return the number of unique full-text and fragment fingerprints."""
        full_texts = self._connection.execute("SELECT COUNT(*) FROM digests").fetchone()[0]
        fragments = self._connection.execute("SELECT COUNT(DISTINCT digest) FROM fragments").fetchone()[0]
        return {"full_texts": full_texts, "fragments": fragments}

    def set_metadata(self, key: str, value: str) -> None:
        """Store one provenance value on a writable blocklist."""
        self._require_writable()
        self._connection.execute(
            "INSERT INTO metadata (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    def get_metadata(self) -> dict[str, str]:
        """Return all blocklist provenance metadata."""
        return dict(self._connection.execute("SELECT key, value FROM metadata ORDER BY key"))

    def commit(self) -> None:
        """Commit pending writes."""
        if not self._read_only:
            self._connection.commit()

    def close(self) -> None:
        """Commit pending writes and close the database."""
        self.commit()
        self._connection.close()

    def _require_writable(self) -> None:
        if self._read_only:
            raise PermissionError("Benchmark blocklist was opened read-only")

    def __enter__(self) -> BenchmarkBlocklist:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
