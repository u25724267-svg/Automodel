# Gemma 4 E2B Inkuba–AfriInstruct LoRA v2

This pipeline starts from `google/gemma-4-E2B-it`, creates a benchmark-safe
50/50 token mixture of cleaned Inkuba-Instruct and AfriInstruct, trains a
rank-32 LoRA adapter, and evaluates base, v1, and v2 through the benchmark
authors' maintained EleutherAI LM Evaluation Harness tasks.

Inkuba-Instruct and the resulting adapter are governed by CC BY-NC 4.0. Confirm
that restriction is compatible with the intended use before preparing data or
distributing the adapter.

## Pinned sources

| Artifact | Revision |
|---|---|
| Inkuba-Instruct | `a417af0dd9fff92950f037857b0cfa6980cfc743` |
| IrokoBench repository | `1f8be590da6699aee3dc23de6f63e801e2352eff` |
| LM Evaluation Harness | `f4d4b3de3ee6741a7151a9fe74945ee515262f4c` |
| Belebele repository | `918890beb2290a8d3ef2d7a90369925959e1bacf` |

The blocklist additionally pins the individual Hugging Face dataset revisions
used by the upstream task definitions.

## 1. Install

```bash
uv sync --locked --all-groups --extra vlm --extra benchmark
export HF_HOME=/shared/hf-cache
```

Accept the gated Gemma, Inkuba-Instruct, and FLORES/Belebele terms on Hugging
Face before running the pipeline. Authenticate through the Hugging Face CLI or
an existing credential store; do not place access tokens in commands or YAML.

## 2. Build the evaluation blocklist

Build this artifact before preparing either training source:

```bash
uv run python tools/build_benchmark_blocklist.py \
  --output /data/gemma4-african-v2/benchmarks.sqlite3 \
  --afriinstruct-benchmark /data/afriinstruct/benchmarks/afriinstruct-paper/*.json
```

It indexes all official splits for:

- AfriMMLU, AfriXNLI, and AfriMGSM in English and the five Inkuba languages.
- Belebele in English and the five Inkuba languages.
- AfriQA, NTREX, and SIB-200 records from the existing AfriInstruct benchmark.

Both complete fields and 12-token fragments are indexed. Fragment matching
prevents a FLORES sentence from entering training merely because Belebele stores
it inside a longer passage.

Never use benchmark test or development sets for training, monitor validation,
prompt selection, quality-filter tuning, or early stopping.

## 3. Re-prepare AfriInstruct

The existing prepared pool predates the IrokoBench and Belebele blocklist, so
create a new version rather than reusing it blindly:

```bash
uv run python tools/prepare_afriinstruct.py \
  --output-dir /data/gemma4-african-v2/afri-clean-v2 \
  --benchmark-blocklist /data/gemma4-african-v2/benchmarks.sqlite3 \
  --validation-fraction 0.01 \
  --shard-size 100000
```

Add exact Gemma token counts to both manifests:

```bash
uv run python scripts/precompute_tokens.py \
  --meta /data/gemma4-african-v2/afri-clean-v2/train_meta.json \
  --processor google/gemma-4-E2B-it \
  --inplace \
  --workers 8

uv run python scripts/precompute_tokens.py \
  --meta /data/gemma4-african-v2/afri-clean-v2/validation_meta.json \
  --processor google/gemma-4-E2B-it \
  --inplace \
  --workers 8
```

Inspect `summary.json`. `contaminated_records` must be present, and every
excluded benchmark origin is reported under `by_contamination_source`.

## 4. Profile Inkuba-Instruct

Profiling is a complete streaming pass over the five African-language train and
dev splits. It rejects AfriQA and SIB-200 by source, rejects blocklist matches,
audits unsupported tasks and script failures, and samples exact Gemma token
lengths for each `(partition, language, task, source)` cell.

```bash
uv run python tools/prepare_inkuba.py profile \
  --output /data/gemma4-african-v2/inkuba-profile-v2.json \
  --benchmark-blocklist /data/gemma4-african-v2/benchmarks.sqlite3 \
  --model-id google/gemma-4-E2B-it \
  --token-sample-fraction 0.001
```

Review the profile before preparation. The expected task codes are machine
translation, sentiment, topic classification, NER, and POS. QA is not retained
because Inkuba's QA source is AfriQA.

## 5. Prepare the bounded Inkuba pool

```bash
uv run python tools/prepare_inkuba.py prepare \
  --profile /data/gemma4-african-v2/inkuba-profile-v2.json \
  --output-dir /data/gemma4-african-v2/inkuba-clean-v2 \
  --benchmark-blocklist /data/gemma4-african-v2/benchmarks.sqlite3 \
  --model-id google/gemma-4-E2B-it \
  --train-token-budget 100000000 \
  --validation-records-per-language 500 \
  --shard-size 100000
```

The train budget is divided equally among Hausa, Swahili, isiXhosa, Yoruba, and
isiZulu. Within each language, task weights cap machine translation at 50%, and
each observed source receives an equal share of its task budget. Every output
record includes exact `_text_tokens` and upstream provenance.

Preparation must be repeated from a new directory if the summary reports a
material token shortfall or an unexpected source/task distribution.

## 6. Materialize the 50/50 mixture

```bash
uv run python tools/build_instruction_mixture.py \
  --afri-train-meta /data/gemma4-african-v2/afri-clean-v2/train_meta.json \
  --afri-validation-meta /data/gemma4-african-v2/afri-clean-v2/validation_meta.json \
  --inkuba-train-meta /data/gemma4-african-v2/inkuba-clean-v2/train_meta.json \
  --inkuba-validation-meta /data/gemma4-african-v2/inkuba-clean-v2/validation_meta.json \
  --benchmark-blocklist /data/gemma4-african-v2/benchmarks.sqlite3 \
  --output-dir /data/gemma4-african-v2/mixture-r32-v2 \
  --tokens-per-source 97500000 \
  --validation-records-per-source 2500
```

The tool selects records by stable content hash and physically writes the
mixture. It refuses missing token metadata and reports realized source shares.
The 97.5M-token source budget reflects the clean unique Inkuba capacity measured
by the full profile and preparation pass. Each training source must be 50% of
selected tokens within one record's token length. Monitor validation contains
2,500 records from each source.

## 7. Qualification

```bash
export AFRICAN_V2_DATA_DIR=/data/gemma4-african-v2/mixture-r32-v2
export AFRICAN_V2_CHECKPOINT_DIR=/checkpoints/gemma4-e2b-african-v2
export AFRICAN_V2_WANDB_DIR=/logs/wandb

uv run automodel \
  examples/vlm_finetune/gemma4/gemma4_e2b_inkuba_afriinstruct_peft.yaml \
  --nproc-per-node 1 \
  --step_scheduler.max_steps 100 \
  --step_scheduler.ckpt_every_steps 50 \
  --step_scheduler.val_every_steps 50 \
  --lr_scheduler.lr_warmup_steps 10
```

Proceed only when loss and gradients remain finite, both checkpoints are
loadable, resume works, W&B receives metrics, and the benchmark smoke does not
show a catastrophic v1 regression.

## 8. Full training

```bash
uv run automodel \
  examples/vlm_finetune/gemma4/gemma4_e2b_inkuba_afriinstruct_peft.yaml \
  --nproc-per-node 1 \
  --wandb.enable true
```

The default recipe runs 3,000 optimizer steps, validates and checkpoints every
250 steps, and logs to the shared W&B project. W&B receives token-normalized
`loss` and `val_loss` together with the derived `perplexity` and
`val_perplexity` chart series.

## 9. Author-aligned evaluation

The benchmark wrapper delegates dataset loading, five author prompt variants,
log-likelihood multiple-choice scoring, AfriMGSM generation filters, exact
match, and aggregation to the pinned LM Evaluation Harness implementation.

Inspect the exact task expansion before consuming GPU time:

```bash
uv run python tools/run_author_benchmarks.py \
  --suite all \
  --output-dir /results/gemma4-base-author-suite \
  --dry-run
```

Run base Gemma:

```bash
uv run python tools/run_author_benchmarks.py \
  --suite all \
  --output-dir /results/gemma4-base-author-suite
```

Run an adapter by passing its epoch-level checkpoint directory:

```bash
uv run python tools/run_author_benchmarks.py \
  --suite all \
  --checkpoint-path /checkpoints/gemma4-e2b-african-v2/epoch_0_step_2999 \
  --output-dir /results/gemma4-v2-author-suite
```

Repeat the same command for the v1 checkpoint. Keep the old 4,946-record
AfriQA/NTREX/SIB-200 suite as a longitudinal control; do not use it to tune v2.

Every author-suite output directory contains `run_provenance.json` with the
exact task list, model/checkpoint identity, command, and upstream revisions.