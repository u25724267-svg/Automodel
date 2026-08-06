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

import pytest

from tools.benchmark_contamination import BenchmarkBlocklist, canonicalize_text


def test_blocklist_persists_canonicalized_text_and_provenance(tmp_path: Path) -> None:
    path = tmp_path / "benchmarks.sqlite3"
    with BenchmarkBlocklist(path) as blocklist:
        assert blocklist.add_text(
            "  Àwọn   ọmọ\nń kàwé. ",
            benchmark="belebele:yor_Latn:test",
            field="passage",
        )
        assert not blocklist.add_text(
            "àwọn ọmọ ń kàwé.",
            benchmark="belebele:yor_Latn:test",
            field="passage",
        )
        blocklist.set_metadata("upstream_commit", "abc123")

    with BenchmarkBlocklist(path, mode="read-only") as blocklist:
        match = blocklist.find_match(["ÀWỌN ỌMỌ   Ń KÀWÉ."])
        assert match is not None
        assert match.benchmark == "belebele:yor_Latn:test"
        assert match.field == "passage"
        assert blocklist.get_metadata()["upstream_commit"] == "abc123"
        with pytest.raises(PermissionError, match="read-only"):
            blocklist.add_text("new", benchmark="test", field="question")


def test_blocklist_matches_long_fragments(tmp_path: Path) -> None:
    path = tmp_path / "benchmarks.sqlite3"
    passage = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen"
    with BenchmarkBlocklist(path) as blocklist:
        blocklist.add_text(passage, benchmark="belebele", field="passage")

    with BenchmarkBlocklist(path, mode="read-only") as blocklist:
        match = blocklist.find_match(["zero two three four five six seven eight nine ten eleven twelve thirteen end"])
        assert match is not None
        assert match.field == "passage:fragment"


def test_canonicalize_text_normalizes_compatibility_and_whitespace() -> None:
    assert canonicalize_text("  Ｔｅｓｔ\nVALUE  ") == "test value"