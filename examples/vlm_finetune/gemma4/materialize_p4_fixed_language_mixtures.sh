#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-nvcr.io/nvidia/nemo-automodel@sha256:7213eab8055a2029ce1ef9022384a780f9095b9577f32edca4c48d303907421f}"
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/ext_data/casper_neo/Casper/kseries-next-run}"
K10_OUTPUT_NAME="${K10_OUTPUT_NAME:-mixture-p4-5m-per-language-v2}"
K14_OUTPUT_NAME="${K14_OUTPUT_NAME:-mixture-p4-5m-per-language-v1}"

ACTION="${1:-audit}"
ARM="${2:-all}"

usage() {
    cat <<'EOF'
Usage: materialize_p4_fixed_language_mixtures.sh <materialize|audit> [k10|k14|all]

Environment overrides:
  REPO_ROOT        AutoModel checkout (derived from this script by default)
  EXPERIMENT_ROOT  Root containing data/k6-sft, data/k10-sft, data/k14-sft,
                   and hf-cache
  IMAGE            Immutable NeMo AutoModel image reference
  K10_OUTPUT_NAME  K10 output directory name
  K14_OUTPUT_NAME  K14 output directory name

The materialize action refuses to overwrite a nonempty output directory and
runs the exhaustive audit after each successful build. The audit action checks
an existing output and writes final_audit.json into that output directory.
EOF
}

if [[ "$ACTION" != "materialize" && "$ACTION" != "audit" ]]; then
    usage
    exit 2
fi
if [[ "$ARM" != "k10" && "$ARM" != "k14" && "$ARM" != "all" ]]; then
    usage
    exit 2
fi
if ! command -v docker >/dev/null 2>&1; then
    echo "docker is required" >&2
    exit 1
fi

DATA_ROOT="$EXPERIMENT_ROOT/data"
K6_ROOT="$DATA_ROOT/k6-sft"
K10_ROOT="$DATA_ROOT/k10-sft"
K14_ROOT="$DATA_ROOT/k14-sft"
HF_CACHE="$EXPERIMENT_ROOT/hf-cache"
ENV_FILE="$REPO_ROOT/.env"
BLOCKLIST="/data/gemma4-k10/benchmarks.sqlite3"

require_path() {
    if [[ ! -e "$1" ]]; then
        echo "Required path does not exist: $1" >&2
        exit 1
    fi
}

require_path "$REPO_ROOT/tools/build_k6_sft_mixture.py"
require_path "$K6_ROOT"
require_path "$K10_ROOT"
require_path "$K14_ROOT"
require_path "$K10_ROOT/benchmarks.sqlite3"

arm_values() {
    local arm="$1"
    if [[ "$arm" == "k10" ]]; then
        OUTPUT_ROOT="$K10_ROOT/$K10_OUTPUT_NAME"
        CONTAINER_OUTPUT="/data/gemma4-k10/$K10_OUTPUT_NAME"
        CONFIG="/opt/Automodel/examples/vlm_finetune/gemma4/data/k10_p4_5m_per_language_profile.yaml"
        PLAN="/data/gemma4-k10/plans-p4-5m/plans.json"
        FIXED_VALIDATION_ROOT="/data/gemma4-k10/mixture-p2-v1"
        EXPECTED_TOKENS=50000000
        EXPECTED_LANGUAGES=10
        MINIMUM_LABEL_TOKENS=16585053
    else
        OUTPUT_ROOT="$K14_ROOT/$K14_OUTPUT_NAME"
        CONTAINER_OUTPUT="/data/gemma4-k14/$K14_OUTPUT_NAME"
        CONFIG="/opt/Automodel/examples/vlm_finetune/gemma4/data/k14_p4_5m_per_language_profile.yaml"
        PLAN="/data/gemma4-k14/plans-p4-5m/plans.json"
        FIXED_VALIDATION_ROOT="/data/gemma4-k14/mixture-p3-final-v2"
        EXPECTED_TOKENS=70000000
        EXPECTED_LANGUAGES=14
        MINIMUM_LABEL_TOKENS=23219074
    fi
}

docker_mounts() {
    local writable_arm="$1"
    DOCKER_MOUNTS=(
        -v "$REPO_ROOT:/opt/Automodel:ro"
        -v "$K6_ROOT:/data/gemma4-k6:ro"
        -v "$HF_CACHE:/root/.cache/huggingface"
    )
    if [[ "$writable_arm" == "k10" ]]; then
        DOCKER_MOUNTS+=(
            -v "$K10_ROOT:/data/gemma4-k10"
            -v "$K14_ROOT:/data/gemma4-k14:ro"
        )
    else
        DOCKER_MOUNTS+=(
            -v "$K10_ROOT:/data/gemma4-k10:ro"
            -v "$K14_ROOT:/data/gemma4-k14"
        )
    fi
}

audit_arm() {
    local arm="$1"
    arm_values "$arm"
    require_path "$OUTPUT_ROOT/summary.json"
    require_path "$OUTPUT_ROOT/train_meta.json"
    require_path "$OUTPUT_ROOT/validation_meta.json"
    docker_mounts "$arm"

    echo "Auditing ${arm^^}: $OUTPUT_ROOT"
    docker run --rm -i --network none \
        "${DOCKER_MOUNTS[@]}" \
        -w /opt/Automodel \
        "$IMAGE" \
        /opt/venv/bin/python - \
        "$arm" "$CONTAINER_OUTPUT" "$FIXED_VALIDATION_ROOT" "$PLAN" "$CONFIG" "$BLOCKLIST" \
        "$EXPECTED_TOKENS" "$EXPECTED_LANGUAGES" "$MINIMUM_LABEL_TOKENS" <<'PY'
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

from tools.benchmark_contamination import BenchmarkBlocklist


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def message_digest(row: dict[str, object]) -> bytes:
    payload = json.dumps(
        row["messages"],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).digest()


(
    arm,
    output_arg,
    fixed_validation_arg,
    plan_arg,
    config_arg,
    blocklist_arg,
    expected_tokens_arg,
    expected_languages_arg,
    minimum_label_tokens_arg,
) = sys.argv[1:]
output_root = Path(output_arg)
fixed_validation_root = Path(fixed_validation_arg)
plan_path = Path(plan_arg)
config_path = Path(config_arg)
blocklist_path = Path(blocklist_arg)
expected_tokens = int(expected_tokens_arg)
expected_languages = int(expected_languages_arg)
minimum_label_tokens = int(minimum_label_tokens_arg)

if (output_root / ".mixture.sqlite3").exists():
    raise SystemExit("temporary materialization database still exists")

summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
train_manifest = json.loads((output_root / "train_meta.json").read_text(encoding="utf-8"))
validation_manifest = json.loads((output_root / "validation_meta.json").read_text(encoding="utf-8"))
accepted_validation_manifest = json.loads(
    (fixed_validation_root / "validation_meta.json").read_text(encoding="utf-8")
)
plan = json.loads(plan_path.read_text(encoding="utf-8"))["p4_fixed_language_tokens"]
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

if summary["policy"] != "p4_fixed_language_tokens":
    raise SystemExit(f"unexpected materialization policy: {summary['policy']}")
if plan["coverage_shortfall_tokens"] != 0 or plan["label_token_shortfall"] != 0:
    raise SystemExit("approved plan contains a coverage or supervised-token shortfall")
if plan["estimated_packed_tokens"] != expected_tokens:
    raise SystemExit(f"approved plan token budget mismatch: {plan['estimated_packed_tokens']}")
expected_validation_manifest = fixed_validation_root / "validation_meta.json"
if Path(config["fixed_validation_manifest"]) != expected_validation_manifest:
    raise SystemExit(
        f"configured validation manifest is {config['fixed_validation_manifest']}; "
        f"expected {expected_validation_manifest}"
    )
if Path(summary["fixed_validation_manifest"]) != expected_validation_manifest:
    raise SystemExit(
        f"materialized validation manifest is {summary['fixed_validation_manifest']}; "
        f"expected {expected_validation_manifest}"
    )
if (output_root / "validation_meta.json").read_bytes() != (
    expected_validation_manifest
).read_bytes():
    raise SystemExit("validation manifest is not byte-identical to the accepted monitor")
for entry in accepted_validation_manifest.values():
    relative_path = Path(entry["file_name"])
    if (output_root / relative_path).read_bytes() != (fixed_validation_root / relative_path).read_bytes():
        raise SystemExit(f"validation shard differs from accepted monitor: {relative_path}")

by_language: Counter[str] = Counter()
by_task: Counter[str] = Counter()
appearances: Counter[bytes] = Counter()
train_records = 0
label_tokens = 0
benchmark_matches: list[dict[str, str]] = []
with BenchmarkBlocklist(blocklist_path, mode="read-only") as blocklist:
    for entry in train_manifest.values():
        with (output_root / entry["file_name"]).open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                tokens = int(row["_text_tokens"])
                by_language[str(row["lang"])] += tokens
                by_task[str(row["task"])] += tokens
                label_tokens += int(row["_label_tokens"])
                train_records += 1
                appearances[message_digest(row)] += 1
                match = blocklist.find_match([str(message["content"]) for message in row["messages"]])
                if match is not None and len(benchmark_matches) < 20:
                    benchmark_matches.append(
                        {
                            "source": str(row.get("source", "")),
                            "benchmark": match.benchmark,
                            "field": match.field,
                        }
                    )

validation_digests: set[bytes] = set()
validation_records = 0
validation_tokens = 0
for entry in validation_manifest.values():
    with (output_root / entry["file_name"]).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            validation_digests.add(message_digest(row))
            validation_records += 1
            validation_tokens += int(row["_text_tokens"])

total_tokens = sum(by_language.values())
max_appearances = max(appearances.values(), default=0)
overlap = len(set(appearances) & validation_digests)
expected_language_set = set(plan["language_targets"])
if set(by_language) != expected_language_set or len(by_language) != expected_languages:
    raise SystemExit(f"unexpected languages: {sorted(by_language)}")
if any(tokens != 5_000_000 for tokens in by_language.values()):
    raise SystemExit(f"language token budget mismatch: {dict(sorted(by_language.items()))}")
if total_tokens != expected_tokens or summary["train_total_tokens"] != expected_tokens:
    raise SystemExit(f"total token mismatch: rows={total_tokens}, summary={summary['train_total_tokens']}")
if train_records != summary["train_total_records"]:
    raise SystemExit("train record count differs from summary")
if label_tokens != summary["train_total_label_tokens"] or label_tokens < minimum_label_tokens:
    raise SystemExit("supervised-token gate failed")
if max_appearances > 4:
    raise SystemExit(f"appearance cap exceeded: {max_appearances}")
if overlap:
    raise SystemExit(f"train-validation overlap: {overlap}")
if benchmark_matches:
    raise SystemExit(f"benchmark matches found: {benchmark_matches}")
if validation_records != summary["validation_total_records"]:
    raise SystemExit("validation record count differs from summary")
if validation_tokens != summary["validation_total_tokens"]:
    raise SystemExit("validation token count differs from summary")
if summary["train_token_deviation"] != 0:
    raise SystemExit(f"nonzero train token deviation: {summary['train_token_deviation']}")
if any(values["after_tokens"] != 5_000_000 for values in summary["language_reconciliation"].values()):
    raise SystemExit("reconciliation did not finish at 5M tokens per language")

quality_by_allocation = {
    key: values["quality_tier"] for key, values in plan["pool_allocations"].items()
}
by_pool: Counter[str] = Counter()
by_quality: Counter[str] = Counter()
unplanned_allocations: list[str] = []
for key, values in summary["train_allocations"].items():
    tokens = int(values["tokens"])
    by_pool[key.split("|", maxsplit=1)[0]] += tokens
    if key not in quality_by_allocation:
        unplanned_allocations.append(key)
    else:
        by_quality[quality_by_allocation[key]] += tokens
if unplanned_allocations:
    raise SystemExit(f"unplanned allocations: {unplanned_allocations}")

output_hashes = {
    path.relative_to(output_root).as_posix(): sha256(path)
    for path in sorted(output_root.rglob("*"))
    if path.is_file() and path.name != "final_audit.json"
}
source_manifests = {
    pool["name"]: {
        "path": pool["manifest"],
        "sha256": sha256(Path(pool["manifest"])),
        "revision": pool["revision"],
        "license": pool["license"],
        "quality_tier": pool["quality_tier"],
    }
    for pool in config["pools"]
}
audit = {
    "version": 1,
    "status": "accepted",
    "arm": arm,
    "policy": summary["policy"],
    "expected": {
        "total_tokens": expected_tokens,
        "language_tokens": 5_000_000,
        "languages": expected_languages,
        "minimum_label_tokens": minimum_label_tokens,
        "maximum_appearances": 4,
    },
    "computed": {
        "train_records": train_records,
        "unique_train_records": len(appearances),
        "train_tokens": total_tokens,
        "label_tokens": label_tokens,
        "maximum_appearances": max_appearances,
        "validation_records": validation_records,
        "validation_tokens": validation_tokens,
        "validation_unique_records": len(validation_digests),
        "train_validation_overlap": overlap,
        "benchmark_matches": 0,
    },
    "by_language": dict(sorted(by_language.items())),
    "by_task": {
        task: {"tokens": tokens, "share": tokens / total_tokens}
        for task, tokens in sorted(by_task.items())
    },
    "by_pool": {
        pool: {"tokens": tokens, "share": tokens / total_tokens}
        for pool, tokens in by_pool.most_common()
    },
    "by_quality": {
        quality: {"tokens": tokens, "share": tokens / total_tokens}
        for quality, tokens in sorted(by_quality.items())
    },
    "language_reconciliation": summary["language_reconciliation"],
    "source_manifests": source_manifests,
    "input_sha256": {
        "materializer": sha256(Path("/opt/Automodel/tools/build_k6_sft_mixture.py")),
        "config": sha256(config_path),
        "plan": sha256(plan_path),
        "benchmark_blocklist": sha256(blocklist_path),
        "fixed_validation_manifest": sha256(fixed_validation_root / "validation_meta.json"),
    },
    "output_sha256": output_hashes,
}
(output_root / "final_audit.json").write_text(
    json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(
    f"accepted {arm}: records={train_records} tokens={total_tokens} "
    f"labels={label_tokens} max_appearances={max_appearances} overlap={overlap} benchmark_matches=0"
)
print(f"wrote {output_root / 'final_audit.json'}")
PY
}

materialize_arm() {
    local arm="$1"
    arm_values "$arm"
    require_path "$ENV_FILE"
    require_path "$HF_CACHE"
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
    if [[ -z "${HF_TOKEN:-}" ]]; then
        echo "HF_TOKEN is required for materialization" >&2
        exit 1
    fi
    if [[ -e "$OUTPUT_ROOT" ]] && find "$OUTPUT_ROOT" -mindepth 1 -print -quit | grep -q .; then
        echo "Refusing to overwrite nonempty output: $OUTPUT_ROOT" >&2
        exit 1
    fi
    docker_mounts "$arm"

    echo "Materializing ${arm^^}: $OUTPUT_ROOT"
    docker run --rm --network host --env HF_TOKEN \
        "${DOCKER_MOUNTS[@]}" \
        -w /opt/Automodel \
        "$IMAGE" \
        /opt/venv/bin/python tools/build_k6_sft_mixture.py \
        --config "$CONFIG" \
        --plan "$PLAN" \
        --policy p4_fixed_language_tokens \
        --benchmark-blocklist "$BLOCKLIST" \
        --output-dir "$CONTAINER_OUTPUT" \
        --validation-records-per-pool 0
    audit_arm "$arm"
}

run_action() {
    local arm="$1"
    if [[ "$ACTION" == "materialize" ]]; then
        materialize_arm "$arm"
    else
        audit_arm "$arm"
    fi
}

if [[ "$ARM" == "all" ]]; then
    run_action k10
    run_action k14
else
    run_action "$ARM"
fi