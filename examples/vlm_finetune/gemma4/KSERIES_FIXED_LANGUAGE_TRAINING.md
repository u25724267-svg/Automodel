# K-series fixed-language training reproduction

Status: K10 and K14 are running from clean launch commit `cd8acc9e`. On
2026-08-27 the user explicitly accepted the pre-shift token metric and
authorized concurrent launch despite active GPU workloads owned by another
user. No existing process was stopped by this workflow.

## Run matrix

| Field | K10 | K14 |
|---|---|---|
| Run name | `gemma4-e2b-k10-p4-5m-per-language-r16-1ep-v1` | `gemma4-e2b-k14-p4-5m-per-language-r16-1ep-v1` |
| Recipe | `gemma4_e2b_k10_p4_5m_r16_1ep_nvidia_lr.yaml` | `gemma4_e2b_k14_p4_5m_r16_1ep_nvidia_lr.yaml` |
| Dataset | `/data/gemma4-k10/mixture-p4-5m-per-language-v2` | `/data/gemma4-k14/mixture-p4-5m-per-language-v1` |
| Languages | 10 | 14 |
| Materialized pre-shift tokens | 50,000,000 | 70,000,000 |
| Post-shift model-input tokens | 49,553,273 | 69,395,884 |
| Epochs | 1 | 1 |
| Packed batches | 13,446 | 18,831 |
| Optimizer steps | 1,681 | 2,354 |
| LR warmup steps | 168 | 235 |
| LR decay steps | 1,681 | 2,354 |
| Validation/checkpoint cadence | 200 steps | 200 steps |
| Default GPU | 0 | 1 |
| Checkpoint root | `/checkpoints/gemma4-e2b-k10/p4-5m-per-language-r16-1ep-v1` | `/checkpoints/gemma4-e2b-k14/p4-5m-per-language-r16-1ep-v1` |

Both arms start independently from `google/gemma-4-E2B-it` revision
`3e22461f65e89153144f8adb70e3b8c2cc9845a7`. Neither initializes from an
existing adapter or another arm.

## Active launch

| Field | K10 | K14 |
|---|---|---|
| Launch commit | `cd8acc9e69283208e8e2832176905f23ba3e215b` | `cd8acc9e69283208e8e2832176905f23ba3e215b` |
| Container start | `2026-08-27T09:17:59Z` | `2026-08-27T09:18:28Z` |
| W&B ID | `97hpwt0r` | `eugmkhkk` |
| W&B URL | `https://wandb.ai/dsfsi/gemma4-african-instruction/runs/97hpwt0r` | `https://wandb.ai/dsfsi/gemma4-african-instruction/runs/eugmkhkk` |
| Step 0 loss / PPL | 4.4937 / 89.4492 | 4.5273 / 92.5123 |
| Step 0 gradient norm | 34.6458 | 40.5018 |
| Step 0 LR | `2.11e-5` | `2.08e-5` |
| Step 0 throughput | 1,134.21 tokens/s | 1,145.79 tokens/s |
| Step 0 GPU allocation | 30.27 GiB | 30.27 GiB |
| Existing GPU PID at start | `1639574` | `1640844` |

Both step-zero records are finite. Resolved schedules match preflight exactly:
K10 uses 1,681 optimizer steps with 168 warmup steps; K14 uses 2,354 optimizer
steps with 235 warmup steps.

## Saved artifacts

- [K10 recipe](gemma4_e2b_k10_p4_5m_r16_1ep_nvidia_lr.yaml)
- [K14 recipe](gemma4_e2b_k14_p4_5m_r16_1ep_nvidia_lr.yaml)
- [Detached launcher](launch_p4_fixed_language_training.sh)
- [Data reproduction and audits](KSERIES_FIXED_LANGUAGE_DATA_REPRODUCTION.md)

| Artifact | SHA-256 |
|---|---|
| K10 recipe | `10988900d83a81bfaee6dd5753caf6a74f968ada26a396f0223ed036564b895c` |
| K14 recipe | `71769f694a0dc3edc54094503a297858de6dac79697f0c380358c05c746eddf7` |
| Detached launcher | `e08eaa5eb7aa89179051238e3d8edd053aa7ef112edea507a65e13f14ed20ad5` |
| K10 data audit | `27e81495c51619920bbbd1ac58219d90949543e7e0aadb788cce8d56dca1cb3b` |
| K14 data audit | `a7a70645b094560f530f48b3db8f4bfce105fac3bea357109324b95fb16e7594` |

The launcher verifies recipe and data-audit hashes before creating a container.
It also snapshots the recipe, launcher, data audit, data-reproduction document,
Git HEAD/status, binary source diff, and per-file hashes under
`<checkpoint-root>/launch-artifacts/`. These snapshots are the copyable launch
record for exact reruns and post-hoc review.

## Training policy

The new recipes preserve the qualified r16 NVIDIA-policy configuration from the
completed K10/K14 two-epoch ablations:

- LoRA rank 16, alpha 32, dropout 0.0;
- AdamW peak LR `2e-4`, weight decay 0.01, betas 0.9/0.95, epsilon `1e-8`;
- derived cosine schedule from initial `2e-5` to minimum `2e-6`;
- global/local batch 8/1;
- sequence length and pack size 4,096, packing ratio 0.9;
- BF16 and SDPA;
- FSDP2 with TP/CP/PP 1/1/1;
- answer-only fused linear cross-entropy;
- four persistent training workers;
- in-process validation with zero workers;
- seed 42 with ranked RNG;
- checkpoints and validation every 200 optimizer steps.

The only intentional changes are accepted P4 data paths, one epoch, run names,
W&B metadata, and isolated checkpoint roots. One epoch consumes each selected
record once. The 5M target is the pre-shift processor-token metric used during
planning and materialization; the causal-shift totals in the run matrix were
reviewed and accepted before launch.

## W&B contract

Both runs use:

| Field | Value |
|---|---|
| Entity | `${WANDB_ENTITY}` from `.env` (`dsfsi`) |
| Project | `${WANDB_PROJECT}` from `.env` (`gemma4-african-instruction`) |
| Group | `gemma4-e2b-p4-5m-per-language-r16-1ep` |
| Mode | `${WANDB_MODE}` from `.env` (`online`) |
| Persistent directory | `/logs/wandb` mapped to the experiment W&B root |

Repository policy requires example YAMLs to ship with `wandb.enable: false`.
The saved launcher applies the fixed `--wandb.enable true` override. This is the
only training-behavior override and is preserved in the launcher snapshot and
Docker container command.

## Immutable runtime

| Item | Value |
|---|---|
| Image | `nvcr.io/nvidia/nemo-automodel@sha256:7213eab8055a2029ce1ef9022384a780f9095b9577f32edca4c48d303907421f` |
| Training executable | `/opt/venv/bin/automodel` |
| Processes | One process and one GPU per arm |
| Development base | `77394a463c9d177a35be09737cdab5c388e32025` |
| Exact launch commit | `cd8acc9e69283208e8e2832176905f23ba3e215b` |
| Host experiment root | `/ext_data/casper_neo/Casper/kseries-next-run` |

Do not launch with `uv run` in the bind-mounted container. Do not change the
image digest for an exact comparison.

## Preflight evidence

Completed before container creation:

- both accepted data audits pass;
- K10 production dataset builder: 446,727 train and 30,306 validation records;
- K14 production dataset builder: 604,116 train and 23,440 validation records;
- production packed loader resolves K10 to 13,446 batches and 1,681 steps;
- production packed loader resolves K14 to 18,831 batches and 2,354 steps;
- derived warmup is 10%: 168 K10 steps and 235 K14 steps;
- 11 focused Gemma recipe tests pass;
- both recipes pass `tools/lint_example_yamls.py`;
- YAML/editor diagnostics and launcher shell syntax pass;
- W&B/HF environment fields are present without exposing values;
- checkpoint roots and container identities are unused.

## Detached launch procedure

The launcher uses `docker create` followed by `docker start`. Containers retain
their logs and exit/OOM state after completion.

```bash
# Create both containers and copy launch artifacts, but do not start training.
sg docker -c \
  'examples/vlm_finetune/gemma4/launch_p4_fixed_language_training.sh create all'

# Start only when both assigned GPUs report no active compute PIDs.
sg docker -c \
  'examples/vlm_finetune/gemma4/launch_p4_fixed_language_training.sh start all'

# Authorized 2026-08-27 start while other-user workloads remain active.
sg docker -c \
  'ALLOW_BUSY_GPU=1 examples/vlm_finetune/gemma4/launch_p4_fixed_language_training.sh start all'

# Equivalent create+start operation when GPUs are already idle.
sg docker -c \
  'examples/vlm_finetune/gemma4/launch_p4_fixed_language_training.sh launch all'
```

The `start` action fails closed if another compute PID is visible on the
assigned GPU unless `ALLOW_BUSY_GPU=1` is explicitly set. This run was
authorized to use that override. GPU assignments can be changed with `K10_GPU`
and `K14_GPU`; either override is recorded in the launch snapshot and run
report.

## Monitoring

```bash
sg docker -c \
  'examples/vlm_finetune/gemma4/launch_p4_fixed_language_training.sh status all'

sg docker -c \
  'examples/vlm_finetune/gemma4/launch_p4_fixed_language_training.sh logs k10'

sg docker -c \
  'examples/vlm_finetune/gemma4/launch_p4_fixed_language_training.sh logs k14'
```

After launch, require for each arm:

1. a W&B run under `dsfsi/gemma4-african-instruction` with the exact name/group;
2. resolved optimizer steps and warmup matching the table above;
3. finite step-0 loss, perplexity, gradient norm, and throughput;
4. stable GPU allocation without CUDA OOM;
5. persistent `training.jsonl`, `validation.jsonl`, W&B files, and checkpoint artifacts;
6. no host OOM or validation-worker process duplication.

Do not call a run accepted from container state alone. After completion, record
W&B IDs/URLs, start and finish times, final/best validation metrics, final and
best checkpoints, runtime, adapter hash, container exit/OOM state, and the
launch-artifact hashes in this document.