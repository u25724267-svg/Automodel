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

"""Prepare pinned K-series sources as normalized, provenance-rich SFT manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import re
import urllib.request
import zipfile
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, TextIO

if __package__:
    from tools.benchmark_contamination import canonicalize_text
else:
    from benchmark_contamination import canonicalize_text

logger = logging.getLogger(__name__)

SOURCE_REVISIONS = {
    "afriinstruct": "0a6325e6806ffcd7b3baefcc3c6b2de435ac6e1e",
    "aya": "f9ea04583f02a8f86404ff6c58bf75fe637df8a2",
    "kenswquad": "a7f211970d5929c438c7cf3e971d8d34a7292d6e",
    "afrihate": "bb341d2234e404c1efa355ad2d7973f74ca4c140",
    "kinnews": "97dce354a363f853fa94c439dd42135fc31618f0",
    "swahili_news": "24fcf066e6b96f9e0d743e8b79184e0c599f73c3",
    "afridocmt": "8ae06a3f73aa9194a0d29b32e6c1037b3f9420fb",
    "pontoon": "7318c7cac7cdd5e2e7efc7e86d84256aaf54c8dc",
    "digital_umuganda": "a69005be8d20e2982fedbe0474c5a653adf290f8",
    "hausa_voa_ner": "a6cbe619a7a18309bbd7a3813033b180a043c1a8",
    "yoruba_gv_ner": "5f5c76d9ea6a6a7205e53c842c61af719aa8b8c5",
    "nchlt_zulu_ner": "4f9fee744c6d5ca9fcc921c621386a81ccfa2837",
    "afrisenti": "eb42667d2e83d0081864767b47681fbaf00144fb",
    "masakhanews": "fa3b5fff8a91d187bf0c5900a39c4271d08cf7fe",
    "masakhaner": "ba5843cd08aa491d5f96a5e809e71eb9ec461391",
    "wolof_sentiment": "1b41f866b9d9c0f4bcae710f41932bca9ae260fb",
    "alpaca_twi": "e80f712d7b2d09a515329081e56c5504cdedeee4",
    "alpaca_luganda": "8a42b29321ff1eaafd342d575de8e336d7d41a97",
    "alpaca_sesotho": "2e2fdb4fac4183d531fd3b94c59543f3f93a3de4",
    "ghana_farmer_qa_twi": "66fbd932df7b86a6b1c6aeab7f9fbd981010a0ed",
    "mafand": "b215cd3c701dd16e447a0a2132fb73181acd6c53",
    "crane_luganda": "6dff8980c02d8f2e41ed49167c1e0b854cc285e4",
    "crane_luganda_v2": "6dff8980c02d8f2e41ed49167c1e0b854cc285e4",
    "crane_literacy": "7516cbe1c044347571fec41ed98c3515319c3394",
    "vukuzenzele": "8e21ffdc5b1340d86150acf562386973d9fb6fb7",
    "aya_collection": "a3af2fde4b4cb5b2775830b11244a1a20b5f004f",
    "finerweb": "8454782ddcbe121767b8a5a7ec71111445f9749f",
    "nchlt_sesotho_ner": "42115922ccdebd3252365807ef16878ff757713f",
    "alpaca_k14": (
        "twi@e80f712d7b2d09a515329081e56c5504cdedeee4;"
        "lug@8a42b29321ff1eaafd342d575de8e336d7d41a97;"
        "sot@2e2fdb4fac4183d531fd3b94c59543f3f93a3de4"
    ),
    "mafand_k14": "b215cd3c701dd16e447a0a2132fb73181acd6c53",
    "sesotho_news": "zenodo:10531959",
    "aya_collection_k14": "a3af2fde4b4cb5b2775830b11244a1a20b5f004f",
    "aya_somali": "f9ea04583f02a8f86404ff6c58bf75fe637df8a2",
    "finerweb_k14": "8454782ddcbe121767b8a5a7ec71111445f9749f",
    "infopankki": "8812cd4bd395d7bf709b2ebf614e74370756ee04",
    "afriqa_twi": "01add28d7f2e32b9605de08a3a3c4cce1e0a4e5e",
    "aya_kinyarwanda_qa": "a3af2fde4b4cb5b2775830b11244a1a20b5f004f",
}

LANGUAGE_NAMES = {
    "hau": "Hausa",
    "ibo": "Igbo",
    "kin": "Kinyarwanda",
    "swh": "Swahili",
    "yor": "Yoruba",
    "zul": "isiZulu",
    "amh": "Amharic",
    "xho": "isiXhosa",
    "sna": "Shona",
    "wol": "Wolof",
    "twi": "Twi",
    "lug": "Luganda",
    "sot": "Sesotho",
    "som": "Somali",
}

AYA_LANGUAGE_CODES = {
    "hau": "hau",
    "ibo": "ibo",
    "swh": "swh",
    "yor": "yor",
    "zul": "zul",
    "amh": "amh",
    "xho": "xho",
    "sna": "sna",
    "wol": "wol",
    "som": "som",
}
AYA_QA_TERMS = {
    "hau": ("menene", "waye", "ina ", "yaushe", "yaya", "nawa", "wace", "wane", "shin "),
    "ibo": ("gịnị", "kedu", "kedụ", "onye", "ebe", "olee", "mgbe"),
    "swh": ("nini", "nani", "wapi", "lini", "kwa nini", "jinsi gani", "je "),
    "yor": ("kí ni", "kini", "ta ni", "níbo", "nigbawo", "báwo", "mélòó"),
    "zul": ("yini", "ubani", "kuphi", "nini", "kanjani", "kungani", "ingabe"),
    "amh": ("ምን", "ማን", "የት", "መቼ", "እንዴት", "ለምን", "ስንት"),
    "xho": ("yintoni", "ngubani", "phi", "nini", "njani", "kutheni", "ingaba"),
    "sna": ("chii", "ndiani", "kupi", "rini", "sei", "nei", "mangani"),
    "wol": ("lan", "kan", "fan", "kañ", "naka", "lu tax", "ndax"),
    "som": ("maxay", "waa maxay", "yaa", "xaggee", "goorma", "sidee", "sababta"),
}

AFRIINSTRUCT_LANGUAGE_ALIASES = {
    "hau": "hau",
    "ibo": "ibo",
    "kin": "kin",
    "swa": "swh",
    "swh": "swh",
    "yor": "yor",
    "zul": "zul",
    "amh": "amh",
    "xho": "xho",
    "sna": "sna",
    "wol": "wol",
    "twi": "twi",
    "lug": "lug",
    "sot": "sot",
    "som": "som",
}
AFRIINSTRUCT_TASK_ALIASES = {
    "multitask": "instruction",
    "summarization": "instruction",
    "qa": "qa",
    "translation": "translation",
    "sentiment analysis": "classification",
    "sentiment-analysis": "classification",
    "news topic classification": "classification",
    "news-topic-classification": "classification",
    "ner": "ner",
}

PONTOON_CONFIGS = {
    "hau": "en-ha",
    "ibo": "en-ig",
    "kin": "en-rw",
    "swh": "en-sw",
    "yor": "en-yo",
    "zul": "en-zu",
    "amh": "en-am",
    "xho": "en-xh",
    "sna": "en-sn",
    "wol": "en-wo",
}
AFRIDOC_COLUMNS = {"hau": "ha", "swh": "sw", "yor": "yo", "zul": "zu", "amh": "am"}

KINNEWS_LABELS = (
    "politics",
    "sport",
    "economy",
    "health",
    "entertainment",
    "history",
    "technology",
    "tourism",
    "culture",
    "fashion",
    "religion",
    "environment",
    "education",
    "relationship",
)

NER_URLS = {
    "hausa_voa_ner": "https://raw.githubusercontent.com/uds-lsv/transfer-distant-transformer-african/master/data/hausa_ner/train_clean.tsv",
    "yoruba_gv_ner": "https://raw.githubusercontent.com/ajesujoba/YorubaTwi-Embedding/master/Yoruba/Yoruba-NER/train.tsv",
}

NCHLT_URL = (
    "https://repo.sadilar.org/bitstream/handle/20.500.12185/319/"
    "nchlt_isizulu_named_entity_annotated_corpus.zip?sequence=3&isAllowed=y"
)
NCHLT_MEMBER = "NCHLT isiZulu Named Entity Annotated Corpus/Dataset.NCHLT-II.zu.NER.Full.txt"
NCHLT_SESOTHO_URL = (
    "https://repo.sadilar.org/bitstream/handle/20.500.12185/334/"
    "nchlt_sesotho_named_entity_annotated_corpus.zip?sequence=3&isAllowed=y"
)
NCHLT_SESOTHO_MEMBER = "NCHLT Sesotho Named Entity Annotated Corpus/Dataset.NCHLT-II.st.NER.Full.txt"
SESOTHO_NEWS_URL = "https://zenodo.org/records/10531959/files/NewsSA.txt?download=1"
AYA_QA_DATASETS = frozenset(("NQ-Open (T)", "Mintaka-inst (T)", "Adversarial QA (T)", "WIKI QA (T)"))
MOJIBAKE_MARKERS = ("Ã", "Â", "â€", "�")


class SourceUnavailableError(RuntimeError):
    """Signal that a source cannot be accessed in the current environment."""


class _ShardWriter:
    def __init__(self, output_dir: Path, source_name: str, revision: str, shard_size: int) -> None:
        self._output_dir = output_dir
        self._source_name = source_name
        self._revision = revision
        self._shard_size = shard_size
        self._shard_index = 0
        self._records_in_shard = 0
        self._handle: TextIO | None = None
        self._seen: set[bytes] = set()
        self._manifest: dict[str, dict[str, Any]] = {}
        self.records = 0
        self.duplicates = 0
        self.by_language: Counter[str] = Counter()
        self.by_task: Counter[str] = Counter()

    def write(self, record: Mapping[str, Any]) -> None:
        payload = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=20).digest()
        if digest in self._seen:
            self.duplicates += 1
            return
        self._seen.add(digest)
        if self._handle is None or self._records_in_shard >= self._shard_size:
            self._open_next_shard()
        assert self._handle is not None
        self._handle.write(payload + "\n")
        self._records_in_shard += 1
        self.records += 1
        self.by_language[str(record["lang"])] += 1
        self.by_task[str(record["task"])] += 1

    def _open_next_shard(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._shard_index += 1
        self._records_in_shard = 0
        relative_path = Path("processed") / "train" / f"{self._source_name}-{self._shard_index:05d}.jsonl"
        path = self._output_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("w", encoding="utf-8")
        name = f"{self._source_name}-{self._shard_index:05d}"
        self._manifest[name] = {
            "file_name": relative_path.as_posix(),
            "columns": {"messages": "messages"},
            "sample_ratio": 1.0,
        }

    def close(self) -> dict[str, Any]:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        if self.records == 0:
            raise ValueError(f"No records were prepared for {self._source_name}")
        (self._output_dir / "train_meta.json").write_text(
            json.dumps(dict(sorted(self._manifest.items())), indent=2) + "\n",
            encoding="utf-8",
        )
        summary = {
            "source": self._source_name,
            "revision": self._revision,
            "records": self.records,
            "duplicates": self.duplicates,
            "by_language": dict(sorted(self.by_language.items())),
            "by_task": dict(sorted(self.by_task.items())),
        }
        (self._output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return summary

    def abort(self) -> None:
        """Close an open shard after preparation fails."""
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def _record(
    *,
    user: str,
    assistant: str,
    language: str,
    task: str,
    source: str,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    user = user.strip()
    assistant = assistant.strip()
    if not user or not assistant:
        return None
    result: dict[str, Any] = {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "lang": language,
        "task": task,
        "source": source,
    }
    if metadata:
        result.update(metadata)
    return result


def _is_aya_qa(language: str, text: str) -> bool:
    folded = text.casefold()
    return "?" in text or "፧" in text or any(term in folded for term in AYA_QA_TERMS[language])


def _normalize_afriinstruct_record(raw: Mapping[str, Any]) -> dict[str, Any] | None:
    raw_language = raw.get("lang")
    raw_task = raw.get("task")
    messages = raw.get("messages")
    source = raw.get("source")
    if not isinstance(raw_language, str) or not isinstance(raw_task, str):
        return None
    if not isinstance(messages, list) or not isinstance(source, str):
        return None
    languages = {
        AFRIINSTRUCT_LANGUAGE_ALIASES[code]
        for code in raw_language.casefold().split("-")
        if code in AFRIINSTRUCT_LANGUAGE_ALIASES
    }
    task = AFRIINSTRUCT_TASK_ALIASES.get(raw_task.casefold().replace("_", " "))
    if len(languages) != 1 or task is None:
        return None
    normalized_messages = []
    for message in messages:
        if not isinstance(message, dict):
            return None
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str) or not content.strip():
            return None
        normalized_messages.append({"role": role.strip(), "content": content.strip()})
    if not any(message["role"] == "assistant" for message in normalized_messages):
        return None
    record: dict[str, Any] = {
        "messages": normalized_messages,
        "lang": languages.pop(),
        "task": task,
        "source": source.strip(),
        "source_task": raw_task,
    }
    if task == "ner":
        assistant = "\n".join(message["content"] for message in normalized_messages if message["role"] == "assistant")
        record["entity_types"] = sorted(set(re.findall(r"\b[BI]-([A-Z][A-Z0-9_]*)\b", assistant)))
    if "-" in raw_language:
        record["direction"] = raw_language.casefold().replace("swa", "swh")
    return record


def _iter_afriinstruct(manifest_path: Path) -> Iterable[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"Manifest must be a mapping: {manifest_path}")
    for entry in manifest.values():
        path = Path(entry["file_name"])
        if not path.is_absolute():
            path = manifest_path.parent / path
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = _normalize_afriinstruct_record(json.loads(line))
                if record is not None:
                    yield record


def _iter_aya() -> Iterable[dict[str, Any]]:
    from datasets import load_dataset

    rows = load_dataset(
        "CohereLabs/aya_dataset",
        split="train",
        revision=SOURCE_REVISIONS["aya"],
        streaming=True,
    )
    for row in rows:
        language = AYA_LANGUAGE_CODES.get(row["language_code"])
        if language is None:
            continue
        task = "qa" if _is_aya_qa(language, row["inputs"]) else "instruction"
        metadata = {"annotation_type": row["annotation_type"]}
        if task == "qa":
            metadata["qa_subtype"] = "open_ended"
        record = _record(
            user=row["inputs"],
            assistant=row["targets"],
            language=language,
            task=task,
            source="aya_dataset",
            metadata=metadata,
        )
        if record is not None:
            yield record


def _iter_kenswquad() -> Iterable[dict[str, Any]]:
    from datasets import load_dataset

    rows = load_dataset(
        "Kencorpus/KenSwQuAD",
        split="train",
        revision=SOURCE_REVISIONS["kenswquad"],
        streaming=True,
    )
    for row in rows:
        record = _record(
            user=f"Muktadha:\n{row['context']}\n\nSwali: {row['question']}",
            assistant=row["answer"],
            language="swh",
            task="qa",
            source="kenswquad",
            metadata={"qa_subtype": "extractive", "story_id": row["story_id"]},
        )
        if record is not None:
            yield record


def _iter_afrihate() -> Iterable[dict[str, Any]]:
    from datasets import get_dataset_config_names, load_dataset

    repo_id = "afrihate/afrihate"
    try:
        configs = set(get_dataset_config_names(repo_id, revision=SOURCE_REVISIONS["afrihate"]))
    except Exception as error:
        raise SourceUnavailableError("AfriHate access has not been granted") from error
    aliases = {
        "hau": "hau",
        "ibo": "ibo",
        "kin": "kin",
        "swa": "swh",
        "swh": "swh",
        "yor": "yor",
        "zul": "zul",
        "amh": "amh",
        "xho": "xho",
        "twi": "twi",
        "som": "som",
    }
    for config_name, language in aliases.items():
        if config_name not in configs:
            continue
        rows = load_dataset(
            repo_id,
            name=config_name,
            split="train",
            revision=SOURCE_REVISIONS["afrihate"],
            streaming=True,
        )
        for row in rows:
            text = row.get("text") or row.get("tweet")
            label = row.get("label")
            if not isinstance(text, str):
                continue
            record = _record(
                user=f"Classify this text as hate, abusive, or normal:\n\n{text}",
                assistant=str(label),
                language=language,
                task="classification",
                source="afrihate",
                metadata={"classification_subtype": "hate_speech"},
            )
            if record is not None:
                yield record


def _download_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "nemo-automodel-k6-preparer"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def _iter_kinnews() -> Iterable[dict[str, Any]]:
    archive = zipfile.ZipFile(
        io.BytesIO(_download_bytes("https://github.com/saradhix/kinnews_kirnews/raw/master/KINNEWS.zip"))
    )
    with archive.open("KINNEWS/cleaned/train.csv") as raw_handle:
        text_handle = io.TextIOWrapper(raw_handle, encoding="utf-8")
        reader = csv.reader(text_handle, quotechar='"', delimiter=",", quoting=csv.QUOTE_ALL, skipinitialspace=True)
        next(reader)
        for row in reader:
            if len(row) != 3:
                continue
            label, title, content = row
            label_name = KINNEWS_LABELS[int(label) - 1]
            record = _record(
                user=f"Classify the Kinyarwanda news article by topic:\n\n{title}\n\n{content}",
                assistant=label_name,
                language="kin",
                task="classification",
                source="kinnews",
                metadata={"classification_subtype": "news_topic"},
            )
            if record is not None:
                yield record


def _iter_swahili_news() -> Iterable[dict[str, Any]]:
    from datasets import load_dataset

    rows = load_dataset(
        "Mollel/SwahiliNewsClassification",
        split="train",
        revision=SOURCE_REVISIONS["swahili_news"],
        streaming=True,
    )
    for row in rows:
        record = _record(
            user=f"Classify the Swahili news article by topic:\n\n{row['content']}",
            assistant=row["category"],
            language="swh",
            task="classification",
            source="swahili_news",
            metadata={"classification_subtype": "news_topic"},
        )
        if record is not None:
            yield record


def _iter_afrisenti() -> Iterable[dict[str, Any]]:
    from datasets import load_dataset

    for language in ("amh", "twi"):
        rows = load_dataset(
            "masakhane/afrisenti",
            name=language,
            split="train",
            revision=SOURCE_REVISIONS["afrisenti"],
            streaming=True,
        )
        for row in rows:
            record = _record(
                user=(
                    f"Classify the sentiment of this {LANGUAGE_NAMES[language]} text "
                    f"as positive, negative, or neutral:\n\n{row['tweet']}"
                ),
                assistant=row["label"],
                language=language,
                task="classification",
                source="afrisenti",
                metadata={"classification_subtype": "sentiment"},
            )
            if record is not None:
                yield record


def _iter_masakhanews() -> Iterable[dict[str, Any]]:
    from datasets import load_dataset

    for language in ("amh", "sna", "xho", "lug", "som"):
        rows = load_dataset(
            "masakhane/masakhanews",
            name=language,
            split="train",
            revision=SOURCE_REVISIONS["masakhanews"],
            streaming=True,
        )
        for row in rows:
            record = _record(
                user=f"Classify this {LANGUAGE_NAMES[language]} news article by topic:\n\n{row['headline']}\n\n{row['text']}",
                assistant=row["category"],
                language=language,
                task="classification",
                source="masakhanews",
                metadata={"classification_subtype": "news_topic"},
            )
            if record is not None:
                yield record


def _iter_wolof_sentiment() -> Iterable[dict[str, Any]]:
    from datasets import load_dataset

    rows = load_dataset(
        "michsethowusu/wolof-sentiments-corpus",
        split="train",
        revision=SOURCE_REVISIONS["wolof_sentiment"],
        streaming=True,
    )
    for row in rows:
        record = _record(
            user=f"Classify the sentiment of this Wolof text as positive or negative:\n\n{row['Wolof']}",
            assistant=row["sentiment"],
            language="wol",
            task="classification",
            source="wolof_sentiment",
            metadata={"classification_subtype": "silver_sentiment"},
        )
        if record is not None:
            yield record


def _translation_record(
    english: Any, target: Any, language: str, source: str, metadata: Mapping[str, Any]
) -> dict[str, Any] | None:
    if not isinstance(english, str) or not isinstance(target, str):
        return None
    if canonicalize_text(english) == canonicalize_text(target):
        return None
    return _record(
        user=f"Translate the following English text to {LANGUAGE_NAMES[language]}:\n\n{english}",
        assistant=target,
        language=language,
        task="translation",
        source=source,
        metadata={**metadata, "direction": f"eng-{language}"},
    )


def _translation_pair_records(
    english: Any, target: Any, language: str, source: str, metadata: Mapping[str, Any]
) -> tuple[dict[str, Any], ...]:
    forward = _translation_record(english, target, language, source, metadata)
    if not isinstance(english, str) or not isinstance(target, str):
        return (forward,) if forward is not None else ()
    reverse = _record(
        user=f"Translate the following {LANGUAGE_NAMES[language]} text to English:\n\n{target}",
        assistant=english,
        language=language,
        task="translation",
        source=source,
        metadata={**metadata, "direction": f"{language}-eng"},
    )
    return tuple(record for record in (forward, reverse) if record is not None)


def _alpaca_record(raw: Mapping[str, Any], language: str, source: str) -> dict[str, Any] | None:
    instruction = raw.get("instruction")
    target = raw.get("output")
    inputs = raw.get("input")
    if not isinstance(instruction, str) or not isinstance(target, str):
        return None
    input_text = inputs.strip() if isinstance(inputs, str) and inputs.strip().casefold() != "nan" else ""
    user = instruction.strip() if not input_text else f"{instruction.strip()}\n\n{input_text}"
    return _record(user=user, assistant=target, language=language, task="instruction", source=source)


def _iter_alpaca_k14() -> Iterable[dict[str, Any]]:
    from datasets import load_dataset

    datasets = (
        ("saillab/alpaca-twi-cleaned", "twi", "alpaca_twi"),
        ("saillab/alpaca-luganda-cleaned", "lug", "alpaca_luganda"),
        ("saillab/alpaca-sesotho-cleaned", "sot", "alpaca_sesotho"),
    )
    for repo_id, language, source in datasets:
        rows = load_dataset(
            repo_id,
            split="train",
            revision=SOURCE_REVISIONS[source],
            streaming=True,
        )
        for row in rows:
            record = _alpaca_record(row, language, source)
            if record is not None:
                yield record


def _iter_ghana_farmer_qa_twi() -> Iterable[dict[str, Any]]:
    from datasets import load_dataset

    rows = load_dataset(
        "ghanaopendata/ghana-farmer-qa-twi",
        split="train",
        revision=SOURCE_REVISIONS["ghana_farmer_qa_twi"],
        streaming=True,
    )
    for row in rows:
        record = _record(
            user=row.get("question_ak", ""),
            assistant=row.get("answer_ak", ""),
            language="twi",
            task="qa",
            source="ghana_farmer_qa_twi",
            metadata={"domain": "agriculture", "category": row.get("category"), "group_id": row.get("file_name")},
        )
        if record is not None:
            yield record


def _iter_afriqa_twi() -> Iterable[dict[str, Any]]:
    url = (
        "https://github.com/masakhane-io/afriqa/raw/main/data/gold_passages/twi/"
        "gold_span_passages.afriqa.twi.en.train.json"
    )
    for line in _download_bytes(url).decode("utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        answer = row.get("answer_lang")
        if not isinstance(answer, str):
            continue
        record = _record(
            user=f"Context:\n{row.get('context', '')}\n\nQuestion:\n{row.get('question_lang', '')}",
            assistant=answer,
            language="twi",
            task="qa",
            source="afriqa_twi",
            metadata={"qa_subtype": "cross_lingual_extractive", "group_id": row.get("id")},
        )
        if record is not None:
            yield record


def _iter_mafand_k14() -> Iterable[dict[str, Any]]:
    from datasets import load_dataset

    configs = {"twi": "en-twi", "lug": "en-lug"}
    for language, config_name in configs.items():
        url = (
            "https://huggingface.co/datasets/masakhane/mafand/resolve/"
            f"refs%2Fconvert%2Fparquet/{config_name}/train/0000.parquet"
        )
        rows = load_dataset("parquet", data_files=url, split="train", streaming=True)
        for row in rows:
            translation = row.get("translation")
            if not isinstance(translation, dict):
                continue
            for record in _translation_pair_records(
                translation.get("en"),
                translation.get(language),
                language,
                "mafand",
                {"domain": "news"},
            ):
                yield record


def _parse_gemma_chat(text: Any) -> tuple[str, str] | None:
    if not isinstance(text, str):
        return None
    match = re.search(
        r"<start_of_turn>user\s*(.*?)<end_of_turn>\s*<start_of_turn>model\s*(.*?)<end_of_turn>",
        text,
        flags=re.DOTALL,
    )
    return (match.group(1).strip(), match.group(2).strip()) if match else None


def _iter_crane_luganda() -> Iterable[dict[str, Any]]:
    from datasets import load_dataset

    for config_name in ("fln_content", "fln_shortform", "fln_synthetic", "fln_weak_sections"):
        rows = load_dataset(
            "CraneAILabs/luganda-fln-training-data",
            name=config_name,
            split="train",
            revision=SOURCE_REVISIONS["crane_luganda"],
            streaming=True,
        )
        for row in rows:
            conversation = _parse_gemma_chat(row.get("text"))
            if conversation is None:
                continue
            record = _record(
                user=conversation[0],
                assistant=conversation[1],
                language="lug",
                task="instruction" if config_name == "fln_content" else "qa",
                source="crane_luganda",
                metadata={
                    "qa_source": "fln",
                    "qa_subtype": row.get("format"),
                    "domain": "foundational_literacy",
                },
            )
            if record is not None:
                yield record

    rows = load_dataset(
        "CraneAILabs/luganda-bilingual-literacy-exercises",
        name="all",
        split="train",
        revision=SOURCE_REVISIONS["crane_literacy"],
        streaming=True,
    )
    for row in rows:
        answer = str(row.get("luganda_answer", "")).strip()
        explanation = str(row.get("luganda_explanation", "")).strip()
        record = _record(
            user=row.get("luganda_question", ""),
            assistant="\n\n".join(value for value in (answer, explanation) if value),
            language="lug",
            task="qa",
            source="crane_literacy",
            metadata={
                "qa_source": "literacy_exercises",
                "qa_subtype": row.get("type"),
                "grade": row.get("grade"),
            },
        )
        if record is not None:
            yield record


def _iter_vukuzenzele() -> Iterable[dict[str, Any]]:
    from datasets import load_dataset

    rows = load_dataset(
        "dsfsi/vukuzenzele-sentence-aligned",
        name="eng-sot",
        split="train",
        revision=SOURCE_REVISIONS["vukuzenzele"],
        streaming=True,
    )
    for row in rows:
        for record in _translation_pair_records(
            row.get("eng"), row.get("sot"), "sot", "vukuzenzele", {"alignment_score": row.get("score")}
        ):
            yield record


def _iter_sesotho_news() -> Iterable[dict[str, Any]]:
    lines = [line.strip() for line in _download_bytes(SESOTHO_NEWS_URL).decode("utf-8").splitlines() if line.strip()]
    labels = {"-1": "negative", "0": "neutral", "1": "positive"}
    for index in range(0, len(lines) - 1, 2):
        if lines[index + 1] not in labels:
            continue
        record = _record(
            user=f"Classify the sentiment of this Sesotho news headline:\n\n{lines[index]}",
            assistant=labels[lines[index + 1]],
            language="sot",
            task="classification",
            source="sesotho_news",
            metadata={"classification_subtype": "sentiment"},
        )
        if record is not None:
            yield record


def _iter_aya_collection_k14() -> Iterable[dict[str, Any]]:
    from datasets import load_dataset

    for config_name, language in (("kinyarwanda", "kin"), ("southern_sotho", "sot"), ("wolof", "wol")):
        rows = load_dataset(
            "CohereLabs/aya_collection_language_split",
            name=config_name,
            split="train",
            revision=SOURCE_REVISIONS["aya_collection"],
            streaming=True,
        )
        for row in rows:
            task_type = row.get("task_type")
            dataset_name = row.get("dataset_name")
            task = None
            if language in {"kin", "sot"} and task_type == "question-answering" and dataset_name in AYA_QA_DATASETS:
                task = "qa"
            elif language == "wol" and task_type == "text-classification":
                task = "classification"
            if task is None:
                continue
            record = _record(
                user=row.get("inputs", ""),
                assistant=row.get("targets", ""),
                language=language,
                task=task,
                source="aya_collection",
                metadata={"source_dataset": dataset_name, "template_id": row.get("template_id")},
            )
            if record is not None:
                yield record


def _iter_aya_kinyarwanda_qa() -> Iterable[dict[str, Any]]:
    from datasets import load_dataset

    rows = load_dataset(
        "CohereLabs/aya_collection_language_split",
        name="kinyarwanda",
        split="train",
        revision=SOURCE_REVISIONS["aya_collection"],
        streaming=True,
    )
    for row in rows:
        if row.get("task_type") != "question-answering" or row.get("dataset_name") != "AfriQA-inst":
            continue
        record = _record(
            user=row.get("inputs", ""),
            assistant=row.get("targets", ""),
            language="kin",
            task="qa",
            source="aya_kinyarwanda_qa",
            metadata={"source_dataset": "AfriQA-inst", "template_id": row.get("template_id")},
        )
        if record is not None:
            yield record


def _iter_aya_somali() -> Iterable[dict[str, Any]]:
    for record in _iter_aya():
        if record["lang"] == "som":
            yield record


def _finerweb_record(raw: Mapping[str, Any], language: str) -> dict[str, Any] | None:
    text = raw.get("text")
    spans = raw.get("char_spans")
    if isinstance(spans, str):
        try:
            spans = json.loads(spans)
        except json.JSONDecodeError:
            return None
    if not isinstance(text, str) or not isinstance(spans, list):
        return None
    entities = []
    for span in spans:
        if not isinstance(span, dict):
            continue
        start, end, label = span.get("start"), span.get("end"), span.get("label")
        if isinstance(start, int) and isinstance(end, int) and isinstance(label, str) and 0 <= start < end <= len(text):
            entities.append({"type": label, "text": text[start:end]})
    return _record(
        user=f"Extract the named entities from this {LANGUAGE_NAMES[language]} text as JSON objects with type and text:\n\n{text}",
        assistant=json.dumps(entities, ensure_ascii=False, separators=(",", ":")),
        language=language,
        task="ner",
        source="finerweb",
        metadata={"entity_types": sorted({entity["type"] for entity in entities})},
    )


def _iter_finerweb_k14() -> Iterable[dict[str, Any]]:
    from datasets import load_dataset

    for language in ("amh", "som"):
        rows = load_dataset(
            "whoisjones/fiNERweb",
            name=language,
            split="train",
            revision=SOURCE_REVISIONS["finerweb"],
            streaming=True,
        )
        for row in rows:
            record = _finerweb_record(row, language)
            if record is not None:
                yield record


def _iter_nchlt_sesotho_ner() -> Iterable[dict[str, Any]]:
    archive = zipfile.ZipFile(io.BytesIO(_download_bytes(NCHLT_SESOTHO_URL)))
    with archive.open(NCHLT_SESOTHO_MEMBER) as raw_handle:
        text_handle = io.TextIOWrapper(raw_handle, encoding="utf-8")
        for tokens, tags in _iter_conll(text_handle):
            record = _ner_record(tokens, tags, "sot", "nchlt_sesotho_ner")
            if record is not None:
                yield record


def _iter_infopankki() -> Iterable[dict[str, Any]]:
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        "tufaax/somali-multilingual-infopankki",
        "en-so/train-00000-of-00001.parquet",
        repo_type="dataset",
        revision="8812cd4bd395d7bf709b2ebf614e74370756ee04",
    )
    rows = load_dataset("parquet", data_files=path, split="train", streaming=True)
    for row in rows:
        translation = row.get("translation")
        if not isinstance(translation, dict):
            continue
        for record in _translation_pair_records(
            translation.get("en"), translation.get("so"), "som", "infopankki", {"domain": "civic_information"}
        ):
            yield record


def _iter_afridocmt() -> Iterable[dict[str, Any]]:
    from datasets import load_dataset

    for config_name in ("tech", "health"):
        rows = load_dataset(
            "masakhane/AfriDocMT",
            name=config_name,
            split="train",
            revision=SOURCE_REVISIONS["afridocmt"],
            streaming=True,
        )
        for row in rows:
            for language, column in AFRIDOC_COLUMNS.items():
                record = _translation_record(
                    row["en"],
                    row[column],
                    language,
                    "afridocmt",
                    {"domain": config_name},
                )
                if record is not None:
                    yield record


def _iter_pontoon() -> Iterable[dict[str, Any]]:
    from datasets import load_dataset

    for language, config_name in PONTOON_CONFIGS.items():
        rows = load_dataset(
            "ayymen/Pontoon-Translations",
            name=config_name,
            split="train",
            revision=SOURCE_REVISIONS["pontoon"],
            streaming=True,
        )
        for row in rows:
            record = _translation_record(
                row["source_string"],
                row["target_string"],
                language,
                "pontoon",
                {"domain": "software_localization"},
            )
            if record is not None:
                yield record


def _iter_digital_umuganda() -> Iterable[dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    filenames = (
        "kinyarwanda-english-corpus.tsv",
        "kinyarwanda-english-corpus2.tsv",
        "kinyarwanda-english-corpus3.tsv",
    )
    for filename in filenames:
        path = hf_hub_download(
            "DigitalUmuganda/kinyarwanda-english-machine-translation-dataset",
            filename,
            repo_type="dataset",
            revision=SOURCE_REVISIONS["digital_umuganda"],
        )
        raw = Path(path).read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("cp1252")
        for row in csv.reader(io.StringIO(text), delimiter="\t"):
            if len(row) < 2:
                continue
            kinyarwanda, english = row[0], row[1]
            record = _translation_record(
                english,
                kinyarwanda,
                "kin",
                "digital_umuganda",
                {"domain": "general"},
            )
            if record is not None:
                yield record


def _iter_conll(lines: Iterable[str]) -> Iterable[tuple[list[str], list[str]]]:
    tokens: list[str] = []
    tags: list[str] = []
    for line in lines:
        line = line.strip()
        if not line:
            if tokens:
                yield tokens, tags
                tokens, tags = [], []
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        tokens.append(parts[0])
        tags.append(parts[1])
    if tokens:
        yield tokens, tags


def _entity_spans(tokens: list[str], tags: list[str]) -> list[dict[str, str]]:
    spans: list[dict[str, str]] = []
    current_tokens: list[str] = []
    current_type: str | None = None
    for token, raw_tag in zip(tokens, tags):
        tag = raw_tag.replace("PERS", "PER")
        if tag in {"O", "OUT"} or "-" not in tag:
            if current_tokens and current_type:
                spans.append({"type": current_type, "text": " ".join(current_tokens)})
            current_tokens, current_type = [], None
            continue
        prefix, entity_type = tag.split("-", maxsplit=1)
        if prefix == "B" or entity_type != current_type:
            if current_tokens and current_type:
                spans.append({"type": current_type, "text": " ".join(current_tokens)})
            current_tokens, current_type = [token], entity_type
        else:
            current_tokens.append(token)
    if current_tokens and current_type:
        spans.append({"type": current_type, "text": " ".join(current_tokens)})
    return spans


def _ner_record(tokens: list[str], tags: list[str], language: str, source: str) -> dict[str, Any] | None:
    spans = _entity_spans(tokens, tags)
    assistant = json.dumps(spans, ensure_ascii=False, separators=(",", ":"))
    return _record(
        user=f"Extract the named entities from this {LANGUAGE_NAMES[language]} text as JSON objects with type and text:\n\n{' '.join(tokens)}",
        assistant=assistant,
        language=language,
        task="ner",
        source=source,
        metadata={"entity_types": sorted({span["type"] for span in spans})},
    )


def _iter_remote_ner(source: str, language: str) -> Iterable[dict[str, Any]]:
    text = _download_bytes(NER_URLS[source]).decode("utf-8")
    for tokens, tags in _iter_conll(io.StringIO(text)):
        record = _ner_record(tokens, tags, language, source)
        if record is not None:
            yield record


def _iter_masakhaner() -> Iterable[dict[str, Any]]:
    revision = SOURCE_REVISIONS["masakhaner"]
    base = f"https://raw.githubusercontent.com/masakhane-io/masakhane-ner/{revision}/MasakhaNER2.0/data"
    for language in ("sna", "wol", "xho", "twi", "lug"):
        text = _download_bytes(f"{base}/{language}/train.txt").decode("utf-8")
        lines = (line.replace(" ", "\t", 1) if " " in line else line for line in io.StringIO(text))
        for tokens, tags in _iter_conll(lines):
            record = _ner_record(tokens, tags, language, "masakhaner")
            if record is not None:
                yield record


def _iter_nchlt_zulu_ner() -> Iterable[dict[str, Any]]:
    archive = zipfile.ZipFile(io.BytesIO(_download_bytes(NCHLT_URL)))
    with archive.open(NCHLT_MEMBER) as raw_handle:
        text_handle = io.TextIOWrapper(raw_handle, encoding="utf-8")
        for tokens, tags in _iter_conll(text_handle):
            if any(marker in token for token in tokens for marker in MOJIBAKE_MARKERS):
                continue
            record = _ner_record(tokens, tags, "zul", "nchlt_zulu_ner")
            if record is not None:
                yield record


SOURCE_LOADERS = {
    "aya": _iter_aya,
    "kenswquad": _iter_kenswquad,
    "afrihate": _iter_afrihate,
    "kinnews": _iter_kinnews,
    "swahili_news": _iter_swahili_news,
    "afridocmt": _iter_afridocmt,
    "pontoon": _iter_pontoon,
    "digital_umuganda": _iter_digital_umuganda,
    "hausa_voa_ner": lambda: _iter_remote_ner("hausa_voa_ner", "hau"),
    "yoruba_gv_ner": lambda: _iter_remote_ner("yoruba_gv_ner", "yor"),
    "nchlt_zulu_ner": _iter_nchlt_zulu_ner,
    "afrisenti": _iter_afrisenti,
    "masakhanews": _iter_masakhanews,
    "masakhaner": _iter_masakhaner,
    "wolof_sentiment": _iter_wolof_sentiment,
    "alpaca_k14": _iter_alpaca_k14,
    "ghana_farmer_qa_twi": _iter_ghana_farmer_qa_twi,
    "afriqa_twi": _iter_afriqa_twi,
    "mafand_k14": _iter_mafand_k14,
    "crane_luganda": _iter_crane_luganda,
    "crane_luganda_v2": _iter_crane_luganda,
    "vukuzenzele": _iter_vukuzenzele,
    "sesotho_news": _iter_sesotho_news,
    "aya_collection_k14": _iter_aya_collection_k14,
    "aya_kinyarwanda_qa": _iter_aya_kinyarwanda_qa,
    "aya_somali": _iter_aya_somali,
    "finerweb_k14": _iter_finerweb_k14,
    "nchlt_sesotho_ner": _iter_nchlt_sesotho_ner,
    "infopankki": _iter_infopankki,
}
SOURCE_NAMES = ("afriinstruct", *SOURCE_LOADERS)


def prepare_source(
    source_name: str,
    *,
    output_root: Path,
    shard_size: int = 100_000,
    afriinstruct_manifest: Path | None = None,
) -> dict[str, Any]:
    """Prepare one pinned K6 source and return its summary."""
    if source_name not in SOURCE_NAMES:
        raise ValueError(f"Unknown source: {source_name}")
    if source_name == "afriinstruct" and afriinstruct_manifest is None:
        raise ValueError("afriinstruct_manifest is required for the AfriInstruct source")
    output_dir = output_root / source_name.replace("_", "-")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = _ShardWriter(output_dir, source_name, SOURCE_REVISIONS[source_name], shard_size)
    try:
        records = (
            _iter_afriinstruct(afriinstruct_manifest)
            if source_name == "afriinstruct" and afriinstruct_manifest is not None
            else SOURCE_LOADERS[source_name]()
        )
        for record in records:
            writer.write(record)
        return writer.close()
    except BaseException:
        writer.abort()
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sources", nargs="+", choices=SOURCE_NAMES, default=tuple(SOURCE_LOADERS))
    parser.add_argument("--afriinstruct-manifest", type=Path)
    parser.add_argument("--shard-size", type=int, default=100_000)
    return parser


def main() -> int:
    """Prepare requested K6 sources into independent normalized pools."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = _build_parser().parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    completed = {}
    unavailable = {}
    failed = {}
    for source_name in args.sources:
        logger.info("Preparing %s", source_name)
        try:
            completed[source_name] = prepare_source(
                source_name,
                output_root=args.output_root.resolve(),
                shard_size=args.shard_size,
                afriinstruct_manifest=(
                    args.afriinstruct_manifest.resolve() if args.afriinstruct_manifest is not None else None
                ),
            )
        except SourceUnavailableError as error:
            logger.warning("Skipping unavailable source %s: %s", source_name, error)
            unavailable[source_name] = str(error)
        except Exception as error:
            logger.exception("Failed to prepare %s", source_name)
            failed[source_name] = f"{type(error).__name__}: {error}"
    summary = {"completed": completed, "unavailable": unavailable, "failed": failed}
    (args.output_root / "execution_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
