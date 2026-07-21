# Gemma 4 E2B AfriInstruct full-data LoRA 1,000-step run report

## Executive summary

This report documents the first completed full-data LoRA fine-tuning run of
`google/gemma-4-E2B-it` on `llama-lang-adapt/AfriInstruct-Data` through NeMo
AutoModel's existing VLM fine-tuning pipeline.

The run completed all 1,000 planned optimizer steps successfully. It processed
31,353,533 packed training tokens, wrote ten resumable checkpoints, and
finished with finite losses and gradients throughout. Full-set validation loss
decreased monotonically from 2.9849 at step 99 to 2.5322 at step 999, a 15.2%
reduction. Under the standard natural-log cross-entropy interpretation, this
corresponds to approximate perplexity decreasing from 19.8 to 12.6.

The latest checkpoint, `epoch_0_step_999`, is also the lowest-validation-loss
checkpoint. There was no observed validation regression or numerical
instability. However, improvement slowed materially near the end, and the run
does not provide evidence that the recipe's original 10,000-step default is
necessary.

The experiment also established two important operational facts:

1. The complete 8.8-million-record prepared dataset fits in the target system.
   The production dataset builder peaked at 16.84 GiB resident memory, while
   training peaked at 30.32 GiB GPU/unified-memory allocation.
2. Full validation dominated wall-clock time. Ten validation passes consumed
   14.94 hours of the 24.23-hour run. Future experiments should use a smaller,
   stratified validation subset for frequent checks and reserve full validation
   for major milestones.

This run demonstrates stable in-distribution optimization. It does not yet
establish generation quality, source-disjoint generalization, factuality, or
retention of the pretrained model's general capabilities.

## Run identity

| Field | Value |
|---|---|
| Run name | `gemma4-e2b-afriinstruct-lora-full-1k-v1` |
| W&B entity | `dsfsi` |
| W&B project | `gemma4-afriinstruct` |
| W&B group | `gemma4-e2b-afriinstruct-lora-full` |
| W&B run ID | `k5sc78hr` |
| W&B URL | <https://wandb.ai/dsfsi/gemma4-afriinstruct/runs/k5sc78hr> |
| W&B state | `finished` |
| Start | 2026-07-15 07:46:21 UTC |
| Finish heartbeat | 2026-07-16 08:00:16 UTC |
| W&B runtime | 87,231.8 seconds (24.23 hours) |
| Checkpoint root | `/checkpoints/gemma4-e2b-afriinstruct/lora-full-1k-v1` |
| Host checkpoint root | `/home/casper/checkpoints/gemma4-e2b-afriinstruct/lora-full-1k-v1` |
| Dataset root | `/data/afriinstruct/e2b-full-v1` |
| Host dataset root | `/home/casper/afriinstruct/e2b-full-v1` |

## Objective and architecture decision

The objective was text-only supervised fine-tuning on AfriInstruct while
retaining Gemma 4 E2B's correct multimodal model path.

Gemma 4 E2B is a conditional-generation VLM even when examples contain only
text. The run therefore used:

- Recipe: `FinetuneRecipeForVLM`
- Model wrapper: `NeMoAutoModelForImageTextToText.from_pretrained`
- Recipe implementation: `nemo_automodel/recipes/vlm/finetune.py`
- Dataset builder:
  `nemo_automodel.components.datasets.vlm.datasets.make_meta_dataset`
- Model: `google/gemma-4-E2B-it`
- Observed Hugging Face model revision:
  `9dbdf8a839e4e9e0eb56ed80cc8886661d3817cf`

The project did not introduce a new model class, recipe class, trainer,
dataset component, CLI route, or notebook.

## Infrastructure

### Host

| Component | Value |
|---|---|
| Machine architecture | AArch64 |
| Kernel | Linux 6.17.0-1021-nvidia |
| GPU | NVIDIA GB10 |
| GPU compute capability | 12.1 |
| NVIDIA driver | 580.159.03 |
| Host memory | 121 GiB total, approximately 112 GiB available before training |
| Swap | 15 GiB |
| Filesystem | 3.7 TiB total, approximately 3.1 TiB free during setup |

The GB10 uses unified memory. Host dataset memory and CUDA allocations must
therefore be considered together rather than treated as fully independent
capacity pools.

### Container

| Component | Value |
|---|---|
| Image | `nvcr.io/nvidia/nemo-automodel:26.04` |
| Image ID | `sha256:c7111bca2dbc213aac84119b194a509e0b5dd7b0b6c37fe0d1bb2070ea24948a` |
| Image architecture | ARM64 |
| Image creation time | 2026-04-26T08:08:52Z |
| Internal NVIDIA PyTorch label | `2.11.0a0+eb65b36` |
| Persistent container name | `afriinstruct-gemma4` |

Although the image tag is `26.04`, its startup banner and package labels use
the NVIDIA 26.02 PyTorch base. The repository-provided
`docker/common/update_pyproject_pytorch.sh` patch was applied before `uv sync`
so that the baked PyTorch and torchvision pair remained intact.

The container used persistent mounts for:

- Source checkout: `/opt/Automodel`
- Dataset: `/data/afriinstruct`
- Hugging Face cache: `/root/.cache/huggingface`
- Checkpoints: `/checkpoints`
- W&B files: `/logs/wandb`

### Software

| Package | Version |
|---|---|
| Python | 3.12.3 |
| NeMo AutoModel | 0.5.0 |
| PyTorch | `2.11.0a0+eb65b36914.nv26.02` |
| torchvision | `0.25.0a0+1e53952f.nv26.02.44259020` |
| Transformers | 5.12.1 |
| huggingface_hub | 1.9.2 |
| W&B | 0.28.0 |

### Source provenance

The training worktree was a detached worktree at commit:

```text
6c5290ddd0d1585903c6610f0712ba8254f26ba6
```

The worktree contained uncommitted, qualification-driven framework fixes plus
container-local dependency metadata changes. This means the Git commit alone
is not sufficient to reproduce the exact run. The production-code fixes are
documented under "Qualification findings and framework fixes" below. The
container-local `pyproject.toml` and `uv.lock` modifications must not be
committed as general project changes.

## Data pipeline

### Source and preparation settings

| Field | Value |
|---|---|
| Dataset | `llama-lang-adapt/AfriInstruct-Data` |
| Observed source revision | `0a6325e6806ffcd7b3baefcc3c6b2de435ac6e1e` |
| Source split | `train` |
| Sample fraction | 1.0 (complete source) |
| Validation fraction | 0.01 |
| Seed | 42 |
| Maximum shard size | 50,000 records |
| Language filters | None |
| Task filters | None |
| Source filters | None |

Preparation used `tools/prepare_afriinstruct.py`. The utility:

- Streamed the Hugging Face source rather than materializing it first.
- Normalized `instruction`, `output`, `lang`, `task`, and `source`.
- Converted each pair to the existing conversation schema:
  `user` followed by `assistant`.
- Removed malformed records.
- Removed exact duplicate instruction/output pairs with disk-backed SQLite.
- Applied deterministic hash-based validation splitting.
- Partitioned bounded JSONL shards by language and task.
- Wrote train and validation meta manifests and `summary.json`.
- Refused to overwrite a nonempty output directory.

### Record counts

| Metric | Count |
|---|---:|
| Source records scanned | 8,949,088 |
| Malformed records | 0 |
| Exact duplicates removed | 153,415 |
| Duplicate rate | 1.71% of source records |
| Filtered records | 0 |
| Sampled-out records | 0 |
| Training records | 8,708,033 |
| Validation records | 87,640 |
| Total written records | 8,795,673 |
| Training shards | 288 |
| Validation shards | 128 |
| Prepared size after token metadata | 4.8 GiB |

### Source distribution

The complete dataset remained highly source-imbalanced.

| Source | Records | Share |
|---|---:|---:|
| xP3 | 8,166,124 | 92.84% |
| AfriSenti | 234,720 | 2.67% |
| MasakhaNEWS | 90,220 | 1.03% |
| XL-Sum | 72,124 | 0.82% |
| FLORES | 71,868 | 0.82% |
| MAFAND | 65,737 | 0.75% |
| MasakhaNER2.0 | 57,974 | 0.66% |
| MENYO | 15,123 | 0.17% |
| NollySenti | 14,961 | 0.17% |
| MasakhaPOS | 6,822 | 0.08% |

### Task distribution

| Task | Records | Share |
|---|---:|---:|
| Multitask | 7,773,312 | 88.38% |
| QA | 392,812 | 4.47% |
| Sentiment Analysis | 249,681 | 2.84% |
| translation | 152,728 | 1.74% |
| news_topic_classification | 90,220 | 1.03% |
| Summarization | 72,124 | 0.82% |
| NER | 57,974 | 0.66% |
| POS | 6,822 | 0.08% |

### Largest language groups

| Language label | Records | Share |
|---|---:|---:|
| swa | 1,149,181 | 13.07% |
| ibo | 1,022,122 | 11.62% |
| yor | 986,670 | 11.22% |
| kin | 939,803 | 10.68% |
| xho | 926,475 | 10.53% |
| nya | 921,945 | 10.48% |
| zul | 921,656 | 10.48% |
| sna | 878,444 | 9.99% |
| ara | 304,928 | 3.47% |
| sot | 265,063 | 3.01% |

Language labels such as `eng-swa` and `swa-eng` are separate from monolingual
labels and are not folded into the table above.

### Token metadata and loader validation

Exact Gemma token counts were added in place with
`scripts/precompute_tokens.py` for both manifests.

- Training records tokenized: 8,708,033
- Validation records tokenized: 87,640
- Missing `_text_tokens`: 0
- Invalid or nonpositive `_text_tokens`: 0
- Training token-precompute wall clock: 190.9 seconds

The exact production builder loaded all 8,708,033 train and 87,640 validation
records successfully. Peak process RSS during this model-free loader test was
16.84 GiB. No `_text_tokens` fallback warnings occurred.

### Data-split limitations

The split is deterministic and exact duplicates are removed globally before
train/validation assignment. However, the validation set is sampled from the
same sources and task templates as training. It is not source-disjoint, and
near-duplicate or templated examples may remain. The validation loss therefore
measures in-distribution AfriInstruct generalization, not robust transfer to
unseen sources.

The licenses of all aggregated upstream datasets must be reviewed before
publishing or redistributing derived model artifacts.

## Training configuration

### Model and precision

| Setting | Value |
|---|---|
| Model | `google/gemma-4-E2B-it` |
| Precision | BF16 |
| Attention | SDPA |
| Cache | Disabled |
| Hidden-state output | Enabled |
| Liger kernel | Disabled |
| SDPA patching | Disabled |
| Activation checkpointing | Disabled |

Activation checkpointing remained disabled because Gemma 4 E2B/E4B shares
keys and values across trailing layers and this recipe intentionally preserves
the model-owned cache-free training path.

### LoRA and freezing

| Setting | Value |
|---|---|
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.0 |
| Triton LoRA path | Enabled |
| Trainable parameters | 26,333,184 |
| Total parameters | 5,130,630,688 |
| Trainable share | 0.51% |

Excluded LoRA module patterns:

- Vision tower and visual modules
- Image encoder
- Audio modules
- LM head

The embeddings, vision tower, and audio tower were frozen. The language model
was the trainable logical component through LoRA adapters. Three dead
KV-sharing parameters in layers 15-34 were also frozen by model setup.

### Distributed topology

| Setting | Value |
|---|---|
| Strategy | FSDP2 |
| Processes / GPUs | 1 |
| TP size | 1 |
| CP size | 1 |
| PP size | 1 |
| Sequence parallelism | Disabled |

Additional processes were not used. The single process ran data-parallel size
one on the GB10.

### Batching and packing

| Setting | Value |
|---|---|
| Local batch size | 1 packed sequence |
| Global batch size | 8 packed sequences |
| Gradient accumulation | 8 microbatches per optimizer step |
| Maximum packed length | 4,096 tokens |
| Pack size | 4,096 tokens |
| Collate maximum length | 4,096 tokens |
| Packing ratio | 0.9 |
| Drop long samples | Enabled |
| Fake image injection | Disabled |

On one GPU:

```text
1 optimizer step = 1 global batch = 8 microbatches = 8 packed sequences
```

The number of original AfriInstruct records per packed sequence varies with
record length. Consequently, 1,000 steps do not correspond to one epoch or to
a known fixed number of original records.

### Loss and optimizer

| Setting | Value |
|---|---|
| Loss | `FusedLinearCrossEntropy` |
| Logit soft-capping | 30.0 |
| Reduction | Sum, normalized by supervised tokens in the recipe |
| Optimizer | AdamW |
| Maximum learning rate | 2.0e-4 |
| Minimum learning rate | 2.0e-5 |
| Betas | 0.9, 0.95 |
| Epsilon | 1.0e-8 |
| Weight decay | 0.01 |
| Gradient clipping | 1.0 |
| LR schedule | Cosine decay |
| Warmup | 50 steps (5% of planned steps) |

### Step, validation, logging, and checkpoint schedule

| Setting | Value |
|---|---:|
| Maximum steps | 1,000 |
| Validation interval | 100 steps |
| Checkpoint interval | 100 steps |
| Remote logging interval | 10 steps |
| Loss averaging window | 50 steps |
| Garbage collection interval | 50 steps |

The recipe file itself defaults to 10,000 steps, 500 warmup steps, and
500-step validation/checkpoint intervals. This run overrode those values from
the CLI. W&B's recorded config is authoritative for the actual run.

## Launch command

The completed experiment was launched in the persistent container with the
equivalent of:

```bash
docker exec -d \
  -e WANDB_MODE=online \
  -e AFRIINSTRUCT_DATA_DIR=/data/afriinstruct/e2b-full-v1 \
  -e AFRIINSTRUCT_CHECKPOINT_DIR=/checkpoints/gemma4-e2b-afriinstruct/lora-full-1k-v1 \
  -e AFRIINSTRUCT_WANDB_DIR=/logs/wandb \
  -w /opt/Automodel \
  afriinstruct-gemma4 \
  /opt/venv/bin/automodel \
  examples/vlm_finetune/gemma4/gemma4_e2b_afriinstruct_peft.yaml \
  --nproc-per-node 1 \
  --step_scheduler.max_steps 1000 \
  --step_scheduler.ckpt_every_steps 100 \
  --step_scheduler.val_every_steps 100 \
  --lr_scheduler.lr_warmup_steps 50 \
  --wandb.enable true \
  --wandb.entity dsfsi \
  --wandb.project gemma4-afriinstruct \
  --wandb.group gemma4-e2b-afriinstruct-lora-full \
  --wandb.name gemma4-e2b-afriinstruct-lora-full-1k-v1
```

Docker access on the host required the `docker` group. The actual host launch
wrapped Docker with `sg docker -c` because the long-lived VS Code process had
started before group membership was refreshed.

## Qualification before the full run

The full run was preceded by staged qualification:

1. A 1,000-record schema smoke preparation verified source access, schema,
   sharding, manifests, and conversation roles. As expected, this ordered
   prefix was not representative and was not used for training.
2. A deterministic 1% representative preparation produced 88,783 train and
   866 validation records.
3. A 20-step LoRA qualification completed with finite loss and gradients,
   validation at steps 9 and 19, and checkpoints at the requested cadence.
4. Explicit resume from step 19 through step 24 restored model, optimizer,
   scheduler, dataloader, RNG, and global-step state successfully.
5. A two-step W&B smoke run verified team-visible online logging to the
   `dsfsi/gemma4-afriinstruct` project and checkpoint creation.
6. An initial 1,000-step pilot on the 1% subset was stopped deliberately after
   the requirement changed to the complete dataset. Its W&B run ID was
   `j2er048b`; its metrics are not included in the full-run analysis.

## Qualification findings and framework fixes

Qualification exposed three framework defects. The fixes were applied to both
the active development checkout and the isolated training worktree.

### VLM sampler/shuffle collision

`nemo_automodel/recipes/vlm/finetune.py` forwarded `shuffle` to the
`StatefulDataLoader` while also supplying a `DistributedSampler`. PyTorch
rejects this combination.

The fix consumes `dataloader.shuffle` in the VLM builder and forwards it only
to `DistributedSampler`, matching existing LLM and retrieval patterns.

### Generic collater's unnecessary Qwen dependency

`nemo_automodel/components/datasets/vlm/collate_fns.py` made
`default_collate_fn` fail when optional `qwen_vl_utils` was unavailable, even
though the generic collater does not call that package.

The fix removed the dead guard while preserving dependency errors in the
Qwen-specific collaters.

### W&B config recursive-copy failure

`nemo_automodel/components/loggers/loggers.py` called `dataclasses.asdict()` on
`WandbConfig`, recursively copying config-resolved `_OrigValueStr` values and
failing before `wandb.init()`.

The fix reads named fields directly and forwards `extra` values verbatim, as
the config contract already documents.

### Regression evidence

After the run, the combined focused CPU regression set was rerun:

```text
229 passed, 28 warnings in 5.48s
```

The set covered:

- AfriInstruct preparation
- AfriInstruct recipe contracts
- VLM recipe helpers
- VLM collaters
- Remote logger configuration

All 438 example YAML files also passed the repository linter. Ruff checks and
`git diff --check` passed for the changed files. The warnings were dependency
deprecations, existing unknown CUDA pytest marks, and the container's
`pynvml` deprecation warning; no focused test failed.

## Results

### Completion and numerical health

| Metric | Result |
|---|---:|
| Planned optimizer steps | 1,000 |
| Completed optimizer steps | 1,000 |
| First step | 0 |
| Final step | 999 |
| Finite loss/gradient/LR/throughput records | 1,000 / 1,000 |
| First recorded step loss | 5.4841 |
| Final step loss | 2.4126 |
| Minimum individual step loss | 1.5472 |
| Mean loss over all steps | 2.7339 |
| First gradient norm | 74.0631 |
| Final gradient norm | 1.3034 |
| Maximum gradient norm | 74.0631 |

The high initial gradient norm fell rapidly and did not produce instability.
No NaN, infinity, gradient explosion, OOM, or process failure occurred.

### Smoothed training trend

| Steps | Mean loss | Loss standard deviation | Mean gradient norm |
|---:|---:|---:|---:|
| 0-99 | 3.3603 | 0.6319 | 11.2131 |
| 100-199 | 2.9176 | 0.2406 | 2.1476 |
| 200-299 | 2.8164 | 0.2036 | 1.8619 |
| 300-399 | 2.7145 | 0.2444 | 1.5900 |
| 400-499 | 2.6250 | 0.2760 | 1.6338 |
| 500-599 | 2.6372 | 0.2097 | 1.5748 |
| 600-699 | 2.6355 | 0.1951 | 1.9757 |
| 700-799 | 2.5678 | 0.2149 | 1.7505 |
| 800-899 | 2.5489 | 0.2191 | 1.9908 |
| 900-999 | 2.5161 | 0.2331 | 1.5775 |

Individual step loss fluctuated because batches contain different languages,
tasks, sources, sequence lengths, and supervised-token counts. The 100-step
means show a substantial early decline, a plateau around steps 400-699, and a
smaller late decline.

### Validation trend

Each validation pass evaluated 3,675,934 supervised label tokens.

| Step | Validation loss | Absolute improvement from prior check |
|---:|---:|---:|
| 99 | 2.9849 | - |
| 199 | 2.8534 | 0.1315 |
| 299 | 2.7793 | 0.0741 |
| 399 | 2.6946 | 0.0847 |
| 499 | 2.6292 | 0.0654 |
| 599 | 2.5952 | 0.0340 |
| 699 | 2.5608 | 0.0344 |
| 799 | 2.5460 | 0.0148 |
| 899 | 2.5390 | 0.0070 |
| 999 | 2.5322 | 0.0068 |

Validation loss decreased at every checkpoint. From step 99 to step 999:

- Absolute reduction: 0.4526
- Relative reduction: 15.16%
- Approximate perplexity: 19.78 to 12.58
- Approximate perplexity reduction: 36.41%

Perplexity is reported only under the conventional natural-log
cross-entropy interpretation. The first validation was after 100 optimizer
steps; there is no zero-shot pretrained validation measurement in this run.
Therefore, these reductions describe step 99 to step 999, not pretrained model
to final adapter.

### Diminishing returns

The final three validation improvements were 0.0148, 0.0070, and 0.0068.
Validation was still improving, and the latest checkpoint remained best, but
the marginal gain had become small. This result supports evaluating the
1,000-step adapter before allocating a substantially larger training budget.
It does not support an automatic jump to the recipe's 10,000-step default.

### Throughput, memory, and token volume

| Metric | Result |
|---|---:|
| Total packed training tokens | 31,353,533 |
| Total supervised training label tokens | 10,260,635 |
| Average packed tokens per optimizer step | 31,354 |
| Median recorded throughput | 946.5 tokens/s |
| Mean steady-step throughput | 946.7 tokens/s |
| Peak training allocation | 30.32 GiB |
| Model-free full-loader peak RSS | 16.84 GiB |

Throughput values immediately after validation include validation wall time in
the step timer and are not representative. The robust median and steady-step
mean exclude that distortion.

The 31.35 million processed tokens represent only a fraction of the complete
prepared data. Loading the complete dataset made all records eligible for
sampling; it did not mean the model completed one epoch.

## Runtime analysis

| Component | Result |
|---|---:|
| Total W&B runtime | 24.23 hours |
| Mean full-validation duration | 89.63 minutes |
| Number of full-validation passes | 10 |
| Total full-validation time | 14.94 hours |
| Typical optimizer step | Approximately 33 seconds |
| Total checkpoint payload | Approximately 1.9 GiB |

Full validation consumed approximately 62% of total wall-clock time. This was
the dominant runtime cost, not model training. The data confirms that future
runs should not evaluate all 87,640 validation records every 100 steps unless
that cost is explicitly desired.

## Checkpoints and artifacts

Checkpoints were written after steps:

```text
99, 199, 299, 399, 499, 599, 699, 799, 899, 999
```

Each checkpoint occupies approximately 195 MiB. Every checkpoint includes:

- LoRA adapter and tokenizer/processor assets
- Optimizer state
- Stateful dataloader state
- RNG state
- Step scheduler state
- Loss summary
- Configuration snapshot

Final checkpoint component sizes:

| Component | Size |
|---|---:|
| Model directory | 82 MiB |
| Optimizer state | 113 MiB |
| Dataloader state | 8 KiB |
| RNG state | 20 KiB |

The final model directory contains:

- `adapter_model.safetensors` (52,749,000 bytes)
- `adapter_config.json`
- `automodel_peft_config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `processor_config.json`
- `chat_template.jinja`

Both `LATEST` and `LOWEST_VAL` resolve to `epoch_0_step_999`.

The artifact is a LoRA adapter plus resumable training state. It is not a
merged, standalone full-model export. `checkpoint.save_consolidated` was
disabled. Deployment or external evaluation that requires a standalone model
must explicitly load the adapter with the base model or perform a reviewed
LoRA merge/export step.

## Run warnings and known limitations

The completed console contained three notable nonfatal warnings:

1. Transformers reported that `torch_dtype` is deprecated in favor of
   `dtype`.
2. NeMo AutoModel force-disabled `rope_fusion` globally, referencing issue
   `#3027`.
3. The run used `Gemma4Processor` with the default collate function.

No warning caused training or validation failure.

Additional limitations:

- No pretrained, step-zero validation baseline was recorded.
- Validation was in-distribution and not source-disjoint.
- Dataset and validation metrics are dominated by xP3/Multitask examples.
- Cross-entropy improvement does not establish factuality, generation quality,
  instruction following, or general-capability retention.
- The full run did not need to resume after interruption. Resume correctness
  was established separately in the 20-to-25-step qualification.
- Runtime CLI overrides are accurately stored in W&B, but the checkpoint's
  `config.yaml` preserves base recipe values. Reproduction requires this report
  or W&B config, not that checkpoint YAML alone.
- The exact training worktree included uncommitted production fixes. Those
  fixes should be reviewed and committed before the run is treated as fully
  reproducible from Git.

## Interpretation

The run provides strong evidence that LoRA optimization was successful for
held-out examples drawn from the same AfriInstruct distribution:

- Training was numerically stable.
- Validation improved monotonically.
- The latest checkpoint was also the best checkpoint.
- The model reached diminishing returns by the end of the schedule.
- The complete data loader and model fit comfortably on the GB10.

The magnitude of validation improvement is meaningful for in-distribution
next-token prediction. The final increments are small enough that extending
training without behavioral evaluation would be speculative.

## Recommended next steps

1. Preserve `epoch_0_step_999` as the primary candidate adapter.
2. Compare the pretrained base model and adapters from steps 599, 799, and 999
   on fixed generation prompts.
3. Build a source-disjoint evaluation set and report results by language,
   task, and source.
4. Measure task-appropriate metrics in addition to aggregate loss.
5. Inspect generation for empty output, repetition, missing EOS, truncation,
   unexpected language switching, and instruction-following failures.
6. Test retention on general instruction-following prompts outside
   AfriInstruct.
7. For future training, use a stratified validation subset for frequent checks
   and run full validation at the end or at sparse milestones.
8. Do not extend directly to 10,000 steps based only on the recipe default.
   If evaluation justifies more training, start a separately tracked phase
   from the best adapter with an explicit low learning rate and bounded
   500-1,000-step scheduler.
9. Review and commit the qualification-driven framework fixes and their tests.
10. Review all upstream dataset licenses before publishing or redistributing
    the adapter.

## Final assessment

The first full-data AfriInstruct LoRA run completed successfully and produced
a stable adapter with a substantial in-distribution validation-loss reduction.
The infrastructure, complete data preparation, token metadata, packed VLM
training, checkpoint cadence, resume mechanism, and team-visible W&B logging
are all operational.

Checkpoint `epoch_0_step_999` is the correct candidate for behavioral
evaluation. Additional training is not yet justified; the next decision should
be based on source-disjoint task and generation quality rather than further
cross-entropy optimization alone.