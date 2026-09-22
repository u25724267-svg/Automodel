# Top-40 gold data preparation report

Date: 2026-09-22

## Status

The nine candidate families from `TOP40_GOLD_DATA_REPORT.md` were read from the
kernel-enforced read-only mount at `/home/casper/afrigemma-train` and
materialized under
`examples/vlm_finetune/gemma4/data/top40_gold_v1`. The materialized directory
is intentionally ignored by Git.

The handoff's 20-step single-GPU LoRA qualification and step-25 resume gate
completed successfully. No long LoRA run or full SFT has been started.

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

## LoRA qualification

Qualification used the complete 407,567-record training manifest and a bounded
466-record Aya validation shard. The eventual long run must use the complete
70,053-record validation manifest.

The locked environment used PyTorch 2.10.0 with CUDA 13.0, Transformers 5.12.1,
BF16, and one NVIDIA A100-SXM4-40GB GPU. The first attempt stopped before step 1
because the configured fused loss dependency was absent; installing the locked
`dev` dependency group supplied `cut-cross-entropy`. The preserved successful
run is `qualification/qual-20-v2`.

| Gate | Result |
|---|---:|
| Initial training steps | 0–19 |
| Resume training steps | 20–24 |
| Validation loss at step 9 | 3.7224 |
| Validation loss at step 19 | 3.4122 |
| Validation loss at step 24 | 3.3918 |
| Peak allocated memory | 30.83 GiB |
| Final training loss | 3.0008 |
| Final gradient norm | 3.4187 |

All recorded losses, perplexities, gradient norms, and learning rates were
finite. Complete adapter, optimizer, RNG, scheduler, and dataloader checkpoints
were written at zero-based steps 9, 19, and 24. The resume command loaded
`epoch_0_step_19`, began at 20/25, and continued without restarting.

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
- `qualification/qual-20-v2/qualification_summary.json`, training and validation
  logs, and adapter checkpoints for the completed qualification and resume gate.

Licensing review was explicitly deferred for this preparation pass. It remains
a release and usage gate, but it did not block local materialization.

## Next training boundary

After approval, the prepared data can be selected by setting:

```bash
export AFRIINSTRUCT_DATA_DIR="$PWD/examples/vlm_finetune/gemma4/data/top40_gold_v1"
```

The handoff's qualification and resume gates have passed. A long, versioned
LoRA run may now be configured, but it must use the complete validation manifest
and preserve the qualified environment and data hashes. Full SFT remains
blocked until a long LoRA run produces measurable held-out gains.
