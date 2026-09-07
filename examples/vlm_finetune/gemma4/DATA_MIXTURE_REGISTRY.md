# Gemma 4 African-language data mixture registry

The canonical machine-readable registry is
[`data/data_mixture_registry.yaml`](data/data_mixture_registry.yaml). It records
the data selection, filtering, preprocessing, provenance, training-run usage,
and exact language-by-task token allocations available as of 2026-09-01.

## Scope and evidence

This registry covers full and authoritative Gemma 4 E2B African-language runs
supported by checked-in reports or preserved local checkpoints. It includes 12
completed runs and separately records two stopped P4 qualifications. Smoke runs
and failed short qualifications are not claimed to be exhaustive.

Five K-series mixtures have locally verified token cubes. Older AfriInstruct
and Inkuba summaries remain on the original host; their known preparation and
run totals are registered, but unavailable language-task token cells are
explicitly `null` rather than estimated.

## Token definitions

All `language_task_tokens` values use **prepared text tokens**: exact Gemma
processor `_text_tokens` before the VLM causal shift. Do not compare them as if
they were model-input, supervised-label, or logged packed-training tokens.

- Prepared text tokens define mixture allocation and per-language exposure.
- Model-input tokens are lower by one token per record after causal shifting.
- Supervised-label tokens contribute to answer-only loss.
- Packed-training tokens are non-padding tokens actually processed by a run.

For a full multi-epoch pass, multiply every registered language-task cell by
the run's numeric `epochs`. One-thousand-step K6/K10/K14 runs are marked
`approximately_1` because they closely matched one materialized pass but did
not preserve a checked-in per-cell processed-token log.

## Completed runs

| Run | Mixture | W&B | Steps | Exposure |
|---|---|---|---:|---:|
| AfriInstruct full r16 | `afriinstruct_full_v1` | `k5sc78hr` | 1,000 | 31,353,533 packed tokens |
| Inkuba-only v1 r16 | `inkuba_clean_v2` | `qfa8rk2r` | 1,000 | unavailable |
| Inkuba + AfriInstruct r32 | `inkuba_afriinstruct_r32_v2` | `y33vdowm` | 3,000 | unavailable |
| K6 P2 r32 | `k6_p2_v2` | `fw1uvbfb` | 1,000 | approximately one pass |
| K10 P2 r32 | `k10_p2_v1` | `kje5g1e7` | 1,000 | approximately one pass |
| K14 P3 r32 | `k14_p3_final_v2` | `nvjz0y4j` | 1,000 | approximately one pass |
| K6 P2 r16 NVIDIA LR | `k6_p2_v2` | `m8r79yjp` | 2,116 | 62,697,980 prepared tokens |
| K10 P2 r32 original LR | `k10_p2_v1` | `w9knou92` | 2,116 | 62,698,624 prepared tokens |
| K10 P2 r16 NVIDIA LR | `k10_p2_v1` | `vebv31eh` | 2,116 | 62,698,624 prepared tokens |
| K14 P3 r16 NVIDIA LR | `k14_p3_final_v2` | `zawhmm8g` | 2,110 | 62,707,124 prepared tokens |
| K10 P4 fixed-language r16 | `k10_p4_5m_v2` | `dedr4hqq` | 3,362 | 100,000,000 prepared tokens |
| K14 P4 fixed-language r16 | `k14_p4_5m_v1` | `t1ifwwal` | 4,708 | 140,000,000 prepared tokens |

The final local records confirm steps 2,115, 2,109, 3,361, and 4,707 for the
later K10 P2 r16, K14 P3 r16, K10 P4, and K14 P4 runs respectively. This
supersedes the older `running` status in the phase handoff documents.

## Mixture inventory

| Mixture | Languages | Tasks | Train records | Prepared tokens | Cell detail |
|---|---:|---:|---:|---:|---|
| AfriInstruct full v1 | many | upstream labels | 8,708,033 | unavailable | external summary required |
| Inkuba clean v2 | 5 | MMT, sentiment, topic, NER, POS | unavailable | planned 100,000,000 | external summary required |
| Inkuba + AfriInstruct v2 | many | component labels | unavailable | planned 195,000,000 | external summary required |
| K6 P2 v2 | 6 | 5 | 153,273 | 31,348,990 | exact in registry |
| K10 P2 v1 | 10 | 5 | 143,579 | 31,349,312 | exact in registry |
| K10 P2 nested 50M v1 | 10 | 5 | 254,954 | 50,000,000 | exact, independently audited |
| K14 P3 final v2 | 14 | 5 | 268,251 | 31,353,562 | exact in registry |
| K10 P4 5M v2 | 10 | 5 | 446,727 | 50,000,000 | exact in registry |
| K14 P4 5M v1 | 14 | 5 | 604,116 | 70,000,000 | exact in registry |

The canonical tasks are instruction following, QA, translation,
classification, and NER. The K6 languages are Hausa, Igbo, Kinyarwanda,
Swahili, Yoruba, and isiZulu. K10 adds Amharic, Shona, Wolof, and isiXhosa. K14
adds Luganda, Somali, Sesotho, and Twi.

Export every exact language-task cell as CSV:

```bash
uv run python - <<'PY'
from pathlib import Path

import yaml

registry = yaml.safe_load(
    Path("examples/vlm_finetune/gemma4/data/data_mixture_registry.yaml").read_text()
)
tasks = tuple(registry["tasks"])
print("mixture,language," + ",".join(tasks) + ",total")
for mixture_name, mixture in registry["mixtures"].items():
    for language, allocation in (mixture.get("language_task_tokens") or {}).items():
        values = [mixture_name, language, *(allocation[task] for task in tasks), allocation["total"]]
        print(",".join(map(str, values)))
PY
```

## Selection and preprocessing

### AfriInstruct full v1

The pipeline streamed the complete `llama-lang-adapt/AfriInstruct-Data` train
split at revision `0a6325e6`, normalized source fields, converted records to
user/assistant conversations, removed 153,415 exact instruction/output
duplicates, and assigned 1% validation by deterministic content hash with seed
42. It wrote language/task-partitioned JSONL shards and added exact Gemma token
metadata. No language, task, source, or benchmark blocklist filter was used in
this first run; near duplicates and templated overlap may remain.

### Inkuba and the 50/50 mixture

Inkuba profiling covered Hausa, Swahili (`swa`), isiXhosa, Yoruba, and isiZulu.
It retained machine translation, sentiment, topic classification, NER, and POS
as distinct tasks with weights 50%, 15%, 15%, 10%, and 10%. It rejected
malformed records, unsupported tasks, script failures, blocklist matches,
AfriQA, and SIB-200. QA was excluded because its available source was AfriQA.
The 100M-token plan divided budget equally by language and shared each task
budget equally across observed sources.

AfriInstruct was re-prepared against the IrokoBench/Belebele blocklist. The
combined builder then selected each clean component by stable content hash to
within one record of 97.5M tokens and selected 2,500 validation records per
component. Exact realized cells require the external original-host summaries.

### K-series P2 and P3

Source-specific adapters normalized 29 source types to conversation records
with canonical language/task/source fields plus translation direction and NER
entity metadata. The profiler rejected malformed and out-of-scope records,
benchmark matches, exact cross-pool duplicates, records without supervised
assistant tokens, and records above the sequence limit.

P2 used capped examples-proportional sampling with temperature 2, quality
weights, a 50% default source-family cap, and a soft 25% AfriInstruct anchor.
It filled measured capacity shortfalls without exceeding four appearances,
rounded pool allocations by largest remainder, and selected by stable digest.
Validation used only unique records not selected for training. P3 added
token-stratified language/task allocation and deterministic strata. The K14 P3
table was computed from materialized shards so it includes donor supplement and
final-fit records omitted from its base `train_allocations` map.

### K-series P4 fixed-language

P4 began with a per-language task prior of 33% instruction, 15% QA, 10%
translation, 17% classification, and 25% NER. Capacity shortfalls could move
tokens only between tasks in the same language. Quality weights, source-family
caps, deterministic strata, benchmark exclusion, frozen-validation exclusion,
and the four-appearance limit remained active.

The materializer staged selections in SQLite and used an exact bounded subset
solver to add or remove records until every language contained exactly 5M
prepared tokens. K10 and K14 have zero train-validation overlap and zero
benchmark matches. Their frozen validation sets were copied byte-for-byte from
K10 P2 and K14 P3 respectively.

## Source and audit authority

The exhaustive source lists, immutable revisions, licenses, quality tiers,
manifest paths, and source-family caps remain in the checked-in profiles:

- [`data/k6_sft_profile.example.yaml`](data/k6_sft_profile.example.yaml)
- [`data/k10_p4_5m_per_language_profile.yaml`](data/k10_p4_5m_per_language_profile.yaml)
- [`data/k14_p4_5m_per_language_profile.yaml`](data/k14_p4_5m_per_language_profile.yaml)

Detailed procedures and accepted risks are in
[`README_k6_sft_preprocessing.md`](README_k6_sft_preprocessing.md),
[`README_inkuba_afriinstruct_v2.md`](README_inkuba_afriinstruct_v2.md), and
[`KSERIES_FIXED_LANGUAGE_DATA_REPRODUCTION.md`](KSERIES_FIXED_LANGUAGE_DATA_REPRODUCTION.md).
The registry points to each accepted local summary and audit where available.

## Maintenance rule

Add a mixture before adding a run. Record one explicit token metric, provenance
paths and hashes, selection policy, rejection gates, preprocessing, validation
contract, and either a complete language-task token cube or a documented `null`
with the missing artifact named. Run the focused recipe test after every update.