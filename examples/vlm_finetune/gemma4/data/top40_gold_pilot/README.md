# Top-40 gold data pilot

This directory contains a bounded inspection pilot derived from
`TOP40_GOLD_DATA_REPORT.md`. It is not an approved training mixture.

## Extraction

- Source: `gs://afrigemma-sg/datasets/train/`, mounted at
  `/home/casper/afrigemma-train` with the kernel `ro` option.
- Extraction date: 2026-09-22.
- Sample size: three records from one preferred split per available dataset.
- Candidate policy: prefer `train`, then `validation`, `dev`, and `test`.
- Reserved policy: prefer `test`, then `validation`, `dev`, and `train`.
- Missing exact dataset IDs: `afrixnli` and `flores_plus`.
- No substitute was used for either missing dataset.

The nine potential training families are under `candidate/`. The benchmark
families are under `reserved/` and must remain excluded from training. FLORES+
training is prohibited by the source policy recorded in the report.

Each JSONL row contains `dataset_id`, `disposition`, `source_split`,
`sample_index`, and the unmodified Parquet row under `record`. See
`pilot_manifest.json` for the selected shard and status of every report ID.

## Reproduction

```bash
uv run --with pyarrow python tools/extract_top40_gold_pilot.py \
  --source-root /home/casper/afrigemma-train \
  --output-root examples/vlm_finetune/gemma4/data/top40_gold_pilot \
  --samples-per-dataset 3
```

The extractor refuses to run unless the source filesystem reports
`ST_RDONLY`, and it refuses to place output inside the source mount.

## Gemma 4 preprocessing

Loader-ready pilot files are under `prepared/`. Candidate files are registered
by `prepared/train_meta.json`; `prepared/reserved_meta.json` remains separate
and must never be configured as training data.

The preprocessing layer decodes the Parquet row's serialized `messages`,
removes the generic system turn that the VLM meta-dataset loader does not
accept, validates alternating user/assistant turns, and writes native message
lists. The Gemma 4 processor applies its model chat template during
tokenization; the JSONL files do not contain pre-rendered template tokens.

The generated `train_meta.json` was loaded through the repository's real
`make_meta_dataset` implementation: all 27 candidate records produced typed
user/assistant conversations. The loader warns that `_text_tokens` is not
precomputed and uses its character-count estimate for this pilot.

The exact `google/gemma-4-E2B-it` processor was then applied to all 27 candidate
records. Inspect `prepared/gemma4_chat_template_preview.jsonl` to see the
rendered text and token count for every sample. Validation confirmed the Gemma
4 user/model turn markers and token counts ranging from 43 to 1,029.

```bash
uv run --with pyarrow python tools/extract_top40_gold_pilot.py \
  --pilot-input-root examples/vlm_finetune/gemma4/data/top40_gold_pilot \
  --output-root examples/vlm_finetune/gemma4/data/top40_gold_pilot/prepared
```