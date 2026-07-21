# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Generate and score AfriInstruct paper benchmark responses."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import logging
import re
import string
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

PAPER_PREFIX = "You are very proficient in African languages, and you are very good at responding in those languages."
TOPICS = (
    "science/technology",
    "travel",
    "politics",
    "sports",
    "health",
    "entertainment",
    "geography",
)
TOPIC_MATCHING_MODES = ("canonical", "paper-fuzzy")

logger = logging.getLogger(__name__)


def record_id(record: dict[str, Any]) -> str:
    """Return a stable ID for a benchmark record."""
    identity = {key: record.get(key) for key in ("instruction", "output", "lang", "source", "task")}
    payload = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_benchmarks(paths: Iterable[Path], limit: int | None = None) -> list[dict[str, Any]]:
    """Load benchmark JSON arrays and remove exact duplicate records."""
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError(f"Expected a JSON array in {path}")
        for row in rows:
            identifier = record_id(row)
            if identifier in seen:
                continue
            seen.add(identifier)
            records.append({**row, "record_id": identifier})
            if limit is not None and len(records) >= limit:
                return records
    return records


def build_prompt(instruction: str, prefix: str = PAPER_PREFIX) -> str:
    """Build the semantic prompt shared by base and adapter generation."""
    return f"{prefix}\n\n{instruction}" if prefix else instruction


def normalize_text(value: str) -> str:
    """Apply standard QA-style normalization."""
    value = value.casefold()
    value = "".join(character for character in value if character not in string.punctuation)
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def parse_references(value: Any) -> list[str]:
    """Parse a benchmark reference that may contain a serialized answer list."""
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            parsed = value
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
        return [str(parsed)]
    return [str(value)]


def token_f1(prediction: str, reference: str) -> float:
    """Calculate normalized token F1 in the range 0-100."""
    prediction_tokens = normalize_text(prediction).split()
    reference_tokens = normalize_text(reference).split()
    if not prediction_tokens or not reference_tokens:
        return 100.0 if prediction_tokens == reference_tokens else 0.0
    common = Counter(prediction_tokens) & Counter(reference_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 200.0 * precision * recall / (precision + recall)


def best_qa_f1(prediction: str, reference: Any) -> float:
    """Return the best token F1 across accepted QA references."""
    candidates = parse_references(reference)
    if not candidates:
        return 0.0
    return max(token_f1(prediction, candidate) for candidate in candidates)


def extract_topic(prediction: str) -> str | None:
    """Extract one canonical topic label from generated text."""
    normalized = normalize_text(prediction).replace(" ", "")
    for topic in TOPICS:
        if normalize_text(topic).replace(" ", "") in normalized:
            return topic
    return None


def extract_topic_paper_fuzzy(prediction: str) -> str | None:
    """Reproduce the AfriInstruct paper's fuzzy topic extraction."""
    try:
        from fuzzywuzzy import fuzz
    except ImportError as error:
        raise ImportError(
            "Paper-compatible topic scoring requires fuzzywuzzy. "
            "Install the test dependencies or run: uv pip install fuzzywuzzy==0.18.0"
        ) from error

    prediction = prediction.strip().casefold()
    best_score = 0
    closest_topic = None
    for topic in TOPICS:
        score = fuzz.partial_ratio(prediction, topic)
        if score > best_score:
            best_score = score
            closest_topic = topic
    return closest_topic if best_score > 80 else None


def macro_f1(true_labels: list[str], predicted_labels: list[str | None]) -> float:
    """Calculate macro F1 over the fixed topic label set."""
    scores = []
    for label in TOPICS:
        true_positive = sum(
            truth == label and prediction == label for truth, prediction in zip(true_labels, predicted_labels)
        )
        false_positive = sum(
            truth != label and prediction == label for truth, prediction in zip(true_labels, predicted_labels)
        )
        false_negative = sum(
            truth == label and prediction != label for truth, prediction in zip(true_labels, predicted_labels)
        )
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return 100.0 * sum(scores) / len(scores)


def observed_macro_f1(true_labels: list[str], predicted_labels: list[str]) -> float:
    """Calculate macro F1 over labels observed by the paper evaluator."""
    labels = tuple(set(true_labels) | set(predicted_labels))
    if not labels:
        return 0.0
    scores = []
    for label in labels:
        true_positive = sum(
            truth == label and prediction == label for truth, prediction in zip(true_labels, predicted_labels)
        )
        false_positive = sum(
            truth != label and prediction == label for truth, prediction in zip(true_labels, predicted_labels)
        )
        false_negative = sum(
            truth == label and prediction != label for truth, prediction in zip(true_labels, predicted_labels)
        )
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return 100.0 * sum(scores) / len(scores)


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    """Load resumable JSONL predictions keyed by record ID."""
    if not path.exists():
        return {}
    predictions: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                predictions[row["record_id"]] = row
    return predictions


def score_records(
    records: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    topic_matching: str = "canonical",
) -> dict[str, Any]:
    """Score cached predictions with standardized task metrics."""
    if topic_matching not in TOPIC_MATCHING_MODES:
        raise ValueError(f"Unknown topic matching mode: {topic_matching}")

    qa_scores: list[float] = []
    translation_predictions: list[str] = []
    translation_references: list[str] = []
    topic_truth: list[str] = []
    topic_predictions: list[str | None] = []
    missing: list[str] = []

    for record in records:
        prediction_row = predictions.get(record["record_id"])
        if prediction_row is None:
            missing.append(record["record_id"])
            continue
        prediction = prediction_row.get("prediction", "")
        task = record.get("task")
        if task == "QA":
            qa_scores.append(best_qa_f1(prediction, record.get("output", "")))
        elif task == "translation":
            translation_predictions.append(prediction)
            translation_references.append(str(record.get("output", "")))
        elif task == "topic-classification":
            topic_truth.append(str(record.get("output", "")).strip().casefold())
            extractor = extract_topic_paper_fuzzy if topic_matching == "paper-fuzzy" else extract_topic
            topic_predictions.append(extractor(prediction))

    matched_topic_pairs = [
        (truth, prediction)
        for truth, prediction in zip(topic_truth, topic_predictions)
        if prediction is not None
    ]
    if topic_matching == "paper-fuzzy":
        scored_topic_truth = [truth for truth, _ in matched_topic_pairs]
        scored_topic_predictions = [prediction for _, prediction in matched_topic_pairs]
        topic_macro_f1 = observed_macro_f1(scored_topic_truth, scored_topic_predictions)
    else:
        scored_topic_truth = topic_truth
        scored_topic_predictions = topic_predictions
        topic_macro_f1 = macro_f1(topic_truth, topic_predictions) if topic_truth else None

    result: dict[str, Any] = {
        "records": len(records),
        "scored": len(records) - len(missing),
        "missing": len(missing),
        "qa_count": len(qa_scores),
        "qa_token_f1": sum(qa_scores) / len(qa_scores) if qa_scores else None,
        "translation_count": len(translation_predictions),
        "topic_count": len(topic_truth),
        "topic_matching": topic_matching,
        "topic_matched": len(matched_topic_pairs),
        "topic_unmatched": len(topic_truth) - len(matched_topic_pairs),
        "topic_match_rate": 100.0 * len(matched_topic_pairs) / len(topic_truth) if topic_truth else None,
        "topic_accuracy": (
            100.0
            * sum(truth == prediction for truth, prediction in zip(scored_topic_truth, scored_topic_predictions))
            / len(scored_topic_truth)
            if scored_topic_truth
            else None
        ),
        "topic_macro_f1": topic_macro_f1,
    }
    if translation_predictions:
        try:
            import sacrebleu
        except ImportError as error:
            raise ImportError(
                "Translation scoring requires sacrebleu. Install it with: uv pip install sacrebleu"
            ) from error
        result["translation_chrf"] = sacrebleu.corpus_chrf(
            translation_predictions, [translation_references], word_order=2
        ).score
    else:
        result["translation_chrf"] = None
    return result


def configure_sdpa(model: Any) -> None:
    """Force the SDPA backend used by the qualified Gemma 4 training recipe."""
    configs = [getattr(model, "config", None)]
    configs.append(getattr(configs[0], "text_config", None))
    for config in configs:
        if config is not None:
            config._attn_implementation = "sdpa"


def _load_model(base_model: str, checkpoint_path: str | None):
    import torch

    from nemo_automodel._transformers import NeMoAutoModelForImageTextToText

    if checkpoint_path:
        from examples.vlm_generate.generate import load_model_from_checkpoint

        model = load_model_from_checkpoint(checkpoint_path, base_model_path=base_model)
    else:
        model = NeMoAutoModelForImageTextToText.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to("cuda")
    configure_sdpa(model)
    model.eval()
    return model


def generate_predictions(args: argparse.Namespace) -> None:
    """Generate missing benchmark predictions and append them to JSONL."""
    import torch
    from transformers import AutoProcessor

    records = load_benchmarks(args.benchmark_file, args.limit)
    completed = load_predictions(args.output)
    pending = [record for record in records if record["record_id"] not in completed]
    logger.info("Loaded %d unique records; %d already complete; %d pending", len(records), len(completed), len(pending))
    if not pending:
        return

    processor = AutoProcessor.from_pretrained(args.base_model)
    model = _load_model(args.base_model, args.checkpoint_path)
    eos_token_ids = getattr(model.generation_config, "eos_token_id", None)
    if eos_token_ids is None:
        eos_token_ids = []
    elif isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as output_file:
        for index, record in enumerate(pending, start=1):
            prompt = build_prompt(record["instruction"], args.prefix)
            messages = [{"role": "user", "content": prompt}]
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            ).to("cuda")
            input_length = inputs["input_ids"].shape[-1]
            started = time.perf_counter()
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                )
            latency = time.perf_counter() - started
            generated_ids = outputs[0, input_length:]
            prediction = processor.decode(generated_ids, skip_special_tokens=True).strip()
            generated_tokens = int(generated_ids.numel())
            row = {
                **record,
                "model_label": args.model_label,
                "base_model": args.base_model,
                "checkpoint_path": args.checkpoint_path,
                "prediction": prediction,
                "input_tokens": int(input_length),
                "generated_tokens": generated_tokens,
                "eos_reached": any(int(token) in eos_token_ids for token in generated_ids),
                "hit_token_limit": generated_tokens >= args.max_new_tokens,
                "latency_seconds": latency,
            }
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            output_file.flush()
            logger.info(
                "[%d/%d] %s %s %s: %d tokens in %.2fs",
                index,
                len(pending),
                record.get("lang"),
                record.get("task"),
                record["record_id"][:8],
                generated_tokens,
                latency,
            )


def score_predictions(args: argparse.Namespace) -> None:
    """Score one cached prediction file and print/write JSON metrics."""
    records = load_benchmarks(args.benchmark_file, args.limit)
    result = score_records(records, load_predictions(args.predictions), topic_matching=args.topic_matching)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate resumable benchmark predictions")
    generate.add_argument("--benchmark-file", type=Path, action="append", required=True)
    generate.add_argument("--base-model", default="google/gemma-4-E2B-it")
    generate.add_argument("--checkpoint-path")
    generate.add_argument("--model-label", required=True)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--max-new-tokens", type=int, default=128)
    generate.add_argument("--limit", type=int)
    generate.add_argument("--prefix", default=PAPER_PREFIX)
    generate.set_defaults(func=generate_predictions)

    score = subparsers.add_parser("score", help="Score cached benchmark predictions")
    score.add_argument("--benchmark-file", type=Path, action="append", required=True)
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--output", type=Path)
    score.add_argument("--limit", type=int)
    score.add_argument("--topic-matching", choices=TOPIC_MATCHING_MODES, default="canonical")
    score.set_defaults(func=score_predictions)
    return parser


def main() -> int:
    """Run AfriInstruct benchmark generation or scoring."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = _build_parser().parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
