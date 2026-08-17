# Gemma 4 experiment reproduction index

Use this document to reproduce the six completed full Gemma 4 African-language
experiments. Smoke, qualification, interrupted, and resume-test runs are not
part of the primary comparison.

## Source checkout

```bash
git clone https://github.com/u25724267-svg/Automodel.git
cd Automodel
git checkout main
git pull --ff-only
git merge-base --is-ancestor 91b2ad08ae706fce6f9b085958823536ba2c086f HEAD
```

Read these detailed documents when reproducing a specific phase:

- `AFRIINSTRUCT_LORA_FULL_1K_RUN_REPORT.md`
- `README_afriinstruct.md`
- `README_inkuba_afriinstruct_v2.md`
- `KSERIES_NEXT_RUN_HANDOFF.md`
- `K6_K10_BENCHMARK_HANDOFF.md`

## Completed full runs

| Run | Steps | LoRA | Host data root | Host checkpoint root | W&B ID |
|---|---:|---:|---|---|---|
| AfriInstruct full | 1,000 | r16 | `/home/casper/afriinstruct/e2b-full-v1` | `/home/casper/checkpoints/gemma4-e2b-afriinstruct/lora-full-1k-v1` | `k5sc78hr` |
| Inkuba-only v1 ablation | 1,000 | r16 | `/home/casper/afriinstruct/gemma4-african-v2/inkuba-clean-v2` | `/home/casper/checkpoints/gemma4-e2b-inkuba-v1-ablation/full-1k-v1` | `qfa8rk2r` |
| Inkuba + AfriInstruct v2 | 3,000 | r32 | `/home/casper/afriinstruct/gemma4-african-v2/mixture-r32-v2` | `/home/casper/checkpoints/gemma4-e2b-african-v2/full-3k-v1` | `y33vdowm` |
| K6 P2 | 1,000 | r32 | `/home/casper/k6-sft/mixture-p2-v2` | `/home/casper/checkpoints/gemma4-e2b-k6/p2-r32-1k-v1` | `fw1uvbfb` |
| K10 P2 | 1,000 | r32 | `/home/casper/k10-sft/mixture-p2-v1` | `/home/casper/checkpoints/gemma4-e2b-k10/p2-r32-1k-v1` | `kje5g1e7` |
| K14 P3 | 1,000 | r32 | `/home/casper/k14-sft/mixture-p3-final-v2` | `/home/casper/checkpoints/gemma4-e2b-k14/p3-r32-1k-v1` | `nvjz0y4j` |

All checkpoint roots contain `training.jsonl`, `validation.jsonl`, and full
resumable checkpoints with model, optimizer, RNG, dataloader, scheduler, and
config state. Use the `LATEST` symlink for the final step. K14's
`LOWEST_VAL` points to step 899; its `LATEST` points to step 999.

## Configuration authority

For each run, inspect configuration in this order:

1. **Resolved W&B config:**
   `/home/casper/wandb/wandb/run-<timestamp>-<id>/files/config.yaml`
2. **Final checkpoint config:** `<checkpoint-root>/<LATEST>/config.yaml`
3. **Version-controlled recipe:** listed below.
4. **Run report/data audit:** documents source revisions and accepted risks.

W&B config is authoritative for runtime CLI overrides and concrete data and
checkpoint paths. Checkpoint YAML may preserve environment placeholders or
recipe defaults. In particular, the AfriInstruct checkpoint YAML says 10,000
steps, while its resolved W&B config and run report record the actual 1,000-step
run.

Recipes:

| Run | Recipe |
|---|---|
| AfriInstruct | `gemma4_e2b_afriinstruct_peft.yaml` |
| Inkuba-only v1 | `gemma4_e2b_inkuba_v1_ablation_peft.yaml` |
| Inkuba + AfriInstruct v2 | `gemma4_e2b_inkuba_afriinstruct_peft.yaml` |
| K6/K10/K14 | `gemma4_e2b_kseries_p2_peft.yaml` |

All recipe paths are under `examples/vlm_finetune/gemma4/`.

## Exact environment

Use the pullable image digest:

```text
nvcr.io/nvidia/nemo-automodel@sha256:7213eab8055a2029ce1ef9022384a780f9095b9577f32edca4c48d303907421f
```

Its local image ID was:

```text
sha256:c7111bca2dbc213aac84119b194a509e0b5dd7b0b6c37fe0d1bb2070ea24948a
```

Base model revisions:

| Run | `google/gemma-4-E2B-it` revision |
|---|---|
| AfriInstruct full | `9dbdf8a839e4e9e0eb56ed80cc8886661d3817cf` |
| Inkuba v1, combined v2, K6, K10, K14 | `3e22461f65e89153144f8adb70e3b8c2cc9845a7` |

Each W&B run preserves `requirements.txt`, `config.yaml`, `output.log`,
`wandb-metadata.json`, and `wandb-summary.json`.

Do not use `uv run` to launch training in a bind-mounted 26.04 container. It can
resolve and replace NVIDIA Torch with a public wheel. Launch with:

```text
/opt/venv/bin/automodel
```

## Transfer artifacts

Run transfers from the destination machine. The original host is
`casper@spark-dsfsi.up.ac.za`.

Example for one data root and one final checkpoint:

```bash
mkdir -p "$HOME/reproduction/data" "$HOME/reproduction/checkpoints"

rsync -a --info=progress2 --partial --append-verify \
  casper@spark-dsfsi.up.ac.za:/home/casper/k14-sft/mixture-p3-final-v2/ \
  "$HOME/reproduction/data/k14/"

rsync -a --info=progress2 --partial --append-verify \
  casper@spark-dsfsi.up.ac.za:/home/casper/checkpoints/gemma4-e2b-k14/p3-r32-1k-v1/epoch_0_step_999/ \
  "$HOME/reproduction/checkpoints/k14/epoch_0_step_999/"
```

Transfer the complete parent data roots when rebuilding data rather than merely
rerunning training:

```text
/home/casper/afriinstruct/e2b-full-v1
/home/casper/afriinstruct/gemma4-african-v2
/home/casper/k6-sft
/home/casper/k10-sft
/home/casper/k14-sft
```

K14 source profiling references pools under all three K6/K10/K14 roots.

## Data configuration and audit locations

AfriInstruct:

```text
examples/vlm_finetune/gemma4/AFRIINSTRUCT_LORA_FULL_1K_RUN_REPORT.md
examples/vlm_finetune/gemma4/README_afriinstruct.md
/home/casper/afriinstruct/e2b-full-v1/summary.json
```

Inkuba v1 and combined v2:

```text
examples/vlm_finetune/gemma4/README_inkuba_afriinstruct_v2.md
/home/casper/afriinstruct/gemma4-african-v2/inkuba-profile-v2.json
/home/casper/afriinstruct/gemma4-african-v2/inkuba-clean-v2/summary.json
/home/casper/afriinstruct/gemma4-african-v2/mixture-r32-v2/summary.json
/home/casper/afriinstruct/gemma4-african-v2/benchmarks.summary.json
```

K-series:

```text
/home/casper/k6-sft/k6_sft_profile.yaml
/home/casper/k6-sft/coverage_review.md
/home/casper/k10-sft/k10_sft_profile.yaml
/home/casper/k10-sft/coverage_review.md
/home/casper/k14-sft/k14_sft_profile.yaml
/home/casper/k14-sft/final_audit.json
/home/casper/k14-sft/coverage_review.md
```

Each K-series root also contains exact `profile-*`, `plans-*`, accepted mixture
summaries, manifests, and normalized source pools.

## Reproduction procedure

1. Clone and verify the repository baseline.
2. Pull the exact container digest.
3. Authenticate interactively for the gated Gemma model and W&B.
4. Transfer the selected run's complete data and final checkpoint/config.
5. Verify adapter and manifest hashes recorded in the detailed handoffs/reports.
6. Mount the data at the concrete container path from the resolved W&B config.
7. Load both manifests through the production VLM dataset builder.
8. Use the resolved W&B config to recreate runtime overrides.
9. Start from the pinned Gemma revision for a fresh reproduction, or use the
   complete checkpoint state for an exact resume.
10. Launch through `/opt/venv/bin/automodel`, monitor step 0, and compare loss,
    gradient norm, throughput, and validation against the preserved logs.

## Provenance limitations

- AfriInstruct ran from commit `6c5290dd` plus documented uncommitted
  qualification fixes.
- Inkuba v1, combined v2, K6, and K10 also predate clean committed execution
  baselines. Their resolved configs, environments, data, and complete checkpoint
  states are preserved, but a bit-for-bit fresh source checkout cannot be
  proven.
- K14 has a recorded clean launch baseline at `7db2a455`.
- The experiments are operationally reproducible and exactly resumable while
  the external data/checkpoint artifacts remain intact. Preserve them in durable
  storage with per-file SHA-256 manifests for archival-grade reproducibility.