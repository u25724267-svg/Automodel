# Gemma 4 K-series two-epoch run report

This is the canonical record for the two-epoch K-series experiments run on
`pitori`. It records accepted data, exact recipes, runtime environment,
completed outcomes, failures and resumes, artifact hashes, and the next
approved run. These are hyperparameter ablations and are not additional
language-breadth staircase arms.

## Artifact authority

For a completed run, use this order:

1. W&B history and resolved config.
2. Final checkpoint config and state.
3. Version-controlled recipe.
4. This report and the accepted mixture summary.

Secrets are stored only in the ignored repository-root `.env` file. They must
never be committed or copied into this report.

## Runtime environment

| Field | Value |
|---|---|
| Host | `pitori` |
| Architecture | `x86_64` |
| GPU | NVIDIA RTX A6000, 49,140 MiB |
| NVIDIA driver | `580.159.03` |
| Image | `nvcr.io/nvidia/nemo-automodel@sha256:7213eab8055a2029ce1ef9022384a780f9095b9577f32edca4c48d303907421f` |
| Local image ID | `sha256:9e664a080b3663095427b3cff014141117c3ec564642e74e216bfa5963979d52` |
| Image architecture / size | AMD64 / 28,002,920,062 bytes |
| Base model | `google/gemma-4-E2B-it` |
| Base model revision | `3e22461f65e89153144f8adb70e3b8c2cc9845a7` |
| Framework path | NeMo AutoModel VLM text-only SFT |
| Precision / attention | BF16 / SDPA |
| Processes | One process and one GPU per run |
| W&B project | `dsfsi/gemma4-african-instruction` |

All launches use `/opt/venv/bin/automodel`; never use `uv run` in the
bind-mounted container.

## Accepted data

Both mixtures use the deterministic `p2_quality_constrained` policy, exact
Gemma token counts, answer-only labels, benchmark blocking, and the accepted
K-series train/validation split.

| Field | K6 P2 v2 | K10 P2 v1 |
|---|---:|---:|
| Languages | 6 | 10 |
| Train records | 153,273 | 143,579 |
| Train tokens | 31,348,990 | 31,349,312 |
| Planned train tokens | 31,353,533 | 31,353,533 |
| Token deviation | -4,543 | -4,221 |
| Validation records | 23,470 | 30,306 |
| Validation tokens | 8,198,968 | 9,784,477 |
| Container root | `/data/gemma4-k6/mixture-p2-v2` | `/data/gemma4-k10/mixture-p2-v1` |
| Host root | `/ext_data/casper_neo/Casper/kseries-next-run/data/k6-sft/mixture-p2-v2` | `/ext_data/casper_neo/Casper/kseries-next-run/data/k10-sft/mixture-p2-v1` |

K6 languages are Hausa, Igbo, Kinyarwanda, Swahili, Yoruba, and isiZulu. K10
adds Amharic, isiXhosa, Shona, and Wolof. Both target instruction, QA,
translation, classification, and NER.

### Data hashes

| Artifact | SHA-256 |
|---|---|
| K6 `summary.json` | `ac39fc5d90a2f53869fb790334472b2f9ce36482a0a82dbce226afd6aad4331c` |
| K6 `train_meta.json` | `37f2abcb40d91700d2969cb962f99423eaba00e72519b7e1cc2ed2a57e516a02` |
| K6 `validation_meta.json` | `09142fdc9c39b05bb05b3de680433db85e477a688bcf269b9224f3256bc40252` |
| K10 `summary.json` | `775da5fc0d6481b50ed328e060933f39e31319f539df87a55696711c09d6cf95` |
| K10 `train_meta.json` | `24c36637225b8ecf474c15301a6cbc5c916e92abe9eef237aea21cada41e9bb2` |
| K10 `validation_meta.json` | `3906353425ab0da96d81ac7c08a9ec8ca75eed1d3bd2efc1107d0730e3f1c2f8` |

## Experiment matrix

| Run | Data | LoRA r/a/dropout | Peak / minimum LR | Warmup | Epochs / steps | Validation | W&B |
|---|---|---|---|---:|---|---|---|
| K6 r16 NVIDIA LR | K6 P2 v2 | 16 / 32 / 0.0 | `2e-4` / `2e-6` derived | 211 derived | 2 / 2,116 | Every 100 | `m8r79yjp` |
| K10 r32 original LR | K10 P2 v1 | 32 / 32 / 0.05 | `5e-5` / `5e-6` | 50 | 2 / 2,116 | Every 100 initially; every 200 after resume | `w9knou92` |
| K10 r16 NVIDIA LR | K10 P2 v1 | 16 / 32 / 0.0 | `2e-4` / `2e-6` derived | 211 derived | 2 / 2,116 expected | Every 200 | Assigned at launch |

Every arm starts independently from the pinned Gemma base. No arm initializes
from another adapter.

## Completed K6 r16 NVIDIA-LR run

| Field | Value |
|---|---|
| Run name | `gemma4-e2b-k6-p2-r16-2ep-nvidia-lr-v1` |
| Recipe | `gemma4_e2b_k6_p2_r16_2ep.yaml` |
| Launch source commit | `cb8c03f0ac3962625946c241e27239a607dc1dd6` |
| Recipe SHA-256 | `9f5d43818da2dede6d52625fda7e65605fd0ce49afde327f3656cb4b43c112fe` |
| W&B | `https://wandb.ai/dsfsi/gemma4-african-instruction/runs/m8r79yjp` |
| Container start / finish | `2026-08-17T21:29:13Z` / `2026-08-18T21:19:02Z` |
| Active runtime | 23.8301 hours |
| Checkpoint root | `/ext_data/casper_neo/Casper/kseries-next-run/checkpoints/gemma4-e2b-k6/p2-r16-2ep-nvidia-lr-v1` |
| Final / best checkpoint | `epoch_1_step_2115` / `epoch_1_step_2115` |
| Final adapter SHA-256 | `55fde2417930a4288c0a8d5403f28d39333903eacb324776d37c81683cba7074` |

### K6 outcome

| Metric | Value |
|---|---:|
| Optimizer steps | 2,116 |
| First / final train loss | 3.923018 / 1.738887 |
| First / last 100-step mean loss | 2.383240 / 1.418930 |
| Best / final validation step | 2,115 / 2,115 |
| Best / final validation loss | 2.007643 / 2.007643 |
| Best / final validation PPL | 7.445746 / 7.445746 |
| Median throughput | 1,408.42 tokens/s |
| Peak logged GPU allocation | 30.365 GiB |

This run completed normally. Its four persistent train workers and two
persistent validation workers consumed about 66 GiB of host RAM. Do not reuse
that validation-worker policy for later runs.

## Completed K10 r32 original-LR run

| Field | Value |
|---|---|
| Run name | `gemma4-e2b-k10-p2-r32-2ep-v1` |
| Fresh recipe | `gemma4_e2b_k10_p2_r32_2ep.yaml` |
| Fresh launch source | HEAD `cb8c03f0`; recipe later committed in `77be8d577fbf8a995e4422d56d66ba694c80caa1` |
| Fresh recipe SHA-256 | `46e5592e6bbc5bd0af5c51409ecad55b7bcffde2bcb035efe26f0a7ef5fff305` |
| Resume recipe | `gemma4_e2b_k10_p2_r32_2ep_resume_step1199.yaml` |
| Resume source commit | `77be8d577fbf8a995e4422d56d66ba694c80caa1` plus the hashed resume recipe |
| Resume recipe SHA-256 | `611938d58c5ea773e52ec22cd9df07018f5bcfabcb7b303e1b2392d339a13e5a` |
| W&B | `https://wandb.ai/dsfsi/gemma4-african-instruction/runs/w9knou92` |
| Initial segment | `2026-08-17T21:54:45Z` to `2026-08-18T13:28:17Z` |
| Resume segment | `2026-08-18T15:07:43Z` to `2026-08-19T00:13:53Z` |
| Total active runtime | 24.6617 hours |
| Checkpoint root | `/ext_data/casper_neo/Casper/kseries-next-run/checkpoints/gemma4-e2b-k10/p2-r32-2ep-v1` |
| Final / best checkpoint | `epoch_1_step_2115` / `epoch_1_step_1799` |
| Final adapter SHA-256 | `eeccc4509a269673208fcf0896f06db30b67a0afeacd4ace2dceefb932652bc7` |

### K10 failure and resume

The initial container was host-OOM-killed after step 1,249. GPU allocation was
stable at 34.19 GiB; this was not a CUDA OOM. The last complete checkpoint was
step 1,199. Persistent validation workers had inherited the full process state,
raising host RAM to about 65.8 GiB. Concurrent K6 and external jobs exhausted
the 125 GiB host.

The run resumed from `LATEST` at step 1,199 with the same model, optimizer, LR
scheduler, RNG, and four-worker train-dataloader state. It kept the same W&B ID,
set validation and checkpoints to every 200 steps, and changed only the
validation dataloader to `num_workers: 0` and `persistent_workers: false`.
Steady K10 host memory fell to about 13.7 GiB. The 50 uncheckpointed steps were
replayed from restored RNG and train-dataloader state.

### K10 outcome

| Metric | Value |
|---|---:|
| Optimizer steps | 2,116 |
| First / final train loss | 3.429548 / 1.183821 |
| First / last 100-step mean loss | 2.433578 / 1.667239 |
| Best validation step | 1,799 |
| Best validation loss / PPL | 2.524178 / 12.480634 |
| Final validation step | 2,115 |
| Final validation loss / PPL | 2.525325 / 12.494957 |
| Median throughput | 1,342.42 tokens/s |
| Peak logged GPU allocation | 34.189 GiB |

W&B contains all 17 validation records. The local `validation.jsonl` contains
only the five post-resume records because the initial process died before its
buffered validation logger closed. W&B is authoritative for the complete
validation series.

## Approved K10 r16 NVIDIA-LR run

Status: configured and validated; not yet launched.

| Field | Value |
|---|---|
| Planned run name | `gemma4-e2b-k10-p2-r16-2ep-nvidia-lr-v1` |
| Recipe | `gemma4_e2b_k10_p2_r16_2ep_nvidia_lr.yaml` |
| Recipe SHA-256 before launch | `02fa6f8d94313a0c9cca990ac2dbe50ec2673b05ab22a03dadd313d5bf5f9ae6` |
| Data | Accepted K10 P2 v1, unchanged |
| LoRA | rank 16, alpha 32, dropout 0.0 |
| Optimizer | AdamW, peak LR `2e-4`, weight decay 0.01, betas 0.9/0.95, epsilon `1e-8` |
| LR schedule | NVIDIA-derived cosine: initial `2e-5`, 10% warmup, minimum `2e-6` |
| Duration | Two epochs; expected 2,116 optimizer steps |
| Batch / packing | Global 8, local 1, 4,096 tokens, packing ratio 0.9 |
| Validation / checkpoint | Every 200 optimizer steps |
| Train / validation workers | 4 persistent / 0 non-persistent |
| W&B name | `gemma4-e2b-k10-p2-r16-2ep-nvidia-lr-v1` |
| Checkpoint root | `/checkpoints/gemma4-e2b-k10/p2-r16-2ep-nvidia-lr-v1` |

The launch must start from the unchanged pinned Gemma base, use a clean source
commit containing this recipe and report, and pass the production loader plus
step-0 gates. After launch, append the source commit, W&B ID, container times,
and resolved schedule to this section. After completion, append final/best
metrics and adapter hashes.

## Launch and monitoring contract

For every future run:

1. Commit the recipe, tests, and pre-launch report before launch.
2. Record the Git commit and recipe SHA-256.
3. Verify data summary and manifest hashes.
4. Run focused config tests and the example-YAML linter.
5. Load train and validation manifests through the production container.
6. Launch with the exact image digest and `/opt/venv/bin/automodel`.
7. Mount data read-only and persist HF cache, checkpoints, and W&B under
   `/ext_data/casper_neo/Casper/kseries-next-run`.
8. Confirm the intended W&B project/name, finite step 0, GPU allocation, host
   memory, and persistent metric files.
9. Record every interruption, checkpoint used for resume, config change, and
   replayed step range.
10. On completion, record final and best checkpoints, W&B state, train and
    validation metrics, runtime, and SHA-256 hashes.
