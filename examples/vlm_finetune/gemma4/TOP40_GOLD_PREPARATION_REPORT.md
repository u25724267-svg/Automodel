# Top-40 gold data preparation report

Date: 2026-09-22

## Status

The nine candidate families from `TOP40_GOLD_DATA_REPORT.md` were read from the
kernel-enforced read-only mount at `/home/casper/afrigemma-train` and
materialized under
`examples/vlm_finetune/gemma4/data/top40_gold_v1`. The materialized directory
is intentionally ignored by Git.

No training command has been run. Training remains blocked pending explicit
user approval.

## Split policy

- Source `train` records were used for training.
- Existing `validation` or `dev` records were preserved as validation.
- Aya has no validation split, so 1% of its training records were assigned by
  deterministic conversation hash with seed 42.
- Source `test` splits were not materialized.
- Reserved benchmark families were not materialized into either candidate
  split.
- Exact conversations were deduplicated globally with validation processed
  first, ensuring zero exact train-validation overlap.

## Preparation results

| Stage | Train | Validation | Total |
|---|---:|---:|---:|
| After normalization and deduplication | 407,843 | 70,077 | 477,920 |
| After 4,080-token safety filter | 407,607 | 70,054 | 477,661 |
| After benchmark contamination filter | 407,567 | 70,053 | 477,620 |

Normalization removed 477,920 generic system turns and 3,639 byte-order-mark
characters. Exact deduplication removed 59,345 candidate training duplicates
and 8,513 validation duplicates. NaijaSenti contributed no unique records after
global deduplication against the earlier-processed candidate sources.

## Token and loader validation

Gemma 4 token counts were precomputed for both manifests with
`google/gemma-4-E2B-it`.

| Split | Records | Text tokens | Maximum text tokens |
|---|---:|---:|---:|
| Train | 407,567 | 61,531,356 | 4,078 |
| Validation | 70,053 | 9,147,730 | 4,014 |
| **Total** | **477,620** | **70,679,086** | — |

Both manifests load through the production `make_meta_dataset` builder with all
token counts present and valid user/assistant conversations. A real sample was
processed by `Gemma4Processor` and `PreTokenizedDatasetWrapper`; its input,
attention, and label tensors aligned, with assistant-only supervised tokens.

## Contamination audit

The repository's pinned benchmark blocklist was built from AfriMMLU,
AfriXNLI, AfriMGSM, and Belebele, then augmented with the mounted AfriMGSM,
AfriMMLU, AfriQA, AfriQA gold passages, Uhura ARC Easy, and Uhura Eval data.
The combined blocklist contains 92,833 full-text and 2,130,460 fragment
fingerprints.

The audit removed 40 training records and one validation record. A complete
post-filter re-audit found zero remaining matches. Exact FLORES+ coverage is
unresolved because `flores_plus` is not present in the mounted bucket; Belebele
provides partial FLORES-derived passage coverage but is not a substitute.

## Evidence

The versioned data directory contains:

- `train_meta.json` and `validation_meta.json`;
- `summary.json` with source-manifest hashes and preparation counts;
- `token_profile.json` and `length_filter_summary.json`;
- `contamination_audit.json` and the pinned blocklist under `audit/`;
- `readiness_report.json` with final counts, token totals, zero-match results,
  and SHA-256 hashes for every prepared shard and governing artifact.

Licensing review was explicitly deferred for this preparation pass. It remains
a release and usage gate, but it did not block local materialization.

## Approval boundary

After approval, the prepared data can be selected by setting:

```bash
export AFRIINSTRUCT_DATA_DIR="$PWD/examples/vlm_finetune/gemma4/data/top40_gold_v1"
```

The first execution must be the handoff's 20-step single-GPU LoRA qualification
using `gemma4_e2b_afriinstruct_peft.yaml`, followed by the step-25 resume check.
Do not start full SFT before those gates pass.