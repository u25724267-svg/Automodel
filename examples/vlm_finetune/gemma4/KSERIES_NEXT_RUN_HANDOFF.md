# Gemma 4 K-series next-run handoff

This is the canonical operational handoff for continuing the fixed-compute
Gemma 4 African-language staircase on another machine. It records what has been
completed, what is still open, which artifacts are authoritative, how to move
them, and the constraints that the next run must preserve.

For reproduction of the complete lineage beginning with AfriInstruct and
Inkuba, read `GEMMA4_EXPERIMENT_REPRODUCTION.md` first.

The raw Copilot chat can also be transferred as supporting evidence, but it is
not a substitute for this document. A copied chat transcript does not
automatically become an active chat session on another machine.

## Immediate continuation point

As of 2026-08-17:

- K6, K10, and K14 data preparation and 1,000-step training are complete.
- All three runs start from the same `google/gemma-4-E2B-it` base model.
- K6 and K10 author-suite raw results and provenance are committed in Git.
- K14 has not yet been run through the controlled benchmark matrix.
- The next training tranche has not been defined. Do not assume it is K18 or
  choose four languages without explicit approval.
- The next agent's first training-design task is to agree on the next language
  set and then produce a source-quality matrix for all five tasks.

The next agent must not resume from a K6, K10, or K14 adapter. Each staircase
arm is an independent LoRA run from the unchanged Gemma base so that language
breadth is the controlled variable.

## Repository state

- Repository: `https://github.com/u25724267-svg/Automodel`
- Branch: `main`
- Minimum required source baseline:
  `d872244e650a4739e2ac451e1e6a7182dd325ea8`
- Baseline commit title:
  `test(benchmarks): add K6 and K10 author-suite results`
- Training image: `nvcr.io/nvidia/nemo-automodel:26.04`
- Base model: `google/gemma-4-E2B-it`
- Reusable recipe:
  `examples/vlm_finetune/gemma4/gemma4_e2b_kseries_p2_peft.yaml`

Clone `main` and require the baseline to be an ancestor rather than checking out
the baseline directly. This allows the commit containing this handoff to be
newer while still proving that all required implementation and results exist.

```bash
git clone https://github.com/u25724267-svg/Automodel.git
cd Automodel
git checkout main
git pull --ff-only
git merge-base --is-ancestor \
  d872244e650a4739e2ac451e1e6a7182dd325ea8 HEAD
git log -1 --oneline
```

If the ancestry check fails, stop. Do not recreate missing tools from memory.

## Research objective

The experiment is a fixed-compute language-token staircase. The model,
materialized token budget, optimizer, LoRA capacity, update count, sequence
length, and runtime policy stay fixed while African-language breadth grows.

The intended comparison is:

1. Does adding languages improve performance on newly represented languages?
2. Does multilingual transfer improve or preserve older languages?
3. At fixed tokens and updates, when does language dilution outweigh transfer?
4. Which task/source-quality gaps explain observed regressions?

Do not change multiple independent variables in the next arm. In particular,
do not increase token budget, steps, LoRA rank, batch size, or change the base
model merely because the new machine is faster.

## Language staircase

K6:

- Hausa (`hau`)
- Igbo (`ibo`)
- Kinyarwanda (`kin`)
- Swahili (`swh`; benchmark code commonly uses `swa`)
- Yoruba (`yor`)
- isiZulu (`zul`)

K10 adds:

- Amharic (`amh`)
- isiXhosa (`xho`)
- Shona (`sna`)
- Wolof (`wol`)

K14 adds:

- Twi (`twi`)
- Luganda (`lug`)
- Sesotho (`sot`)
- Somali (`som`)

Every arm targets instruction, QA, translation, classification, and NER.

## Controlled training contract

| Setting | Required value |
|---|---|
| Base model | `google/gemma-4-E2B-it` |
| Framework route | NeMo AutoModel VLM, text-only SFT |
| Precision | bfloat16 |
| Attention | SDPA |
| Materialized token budget | 31,353,533 |
| Sequence / pack length | 4,096 |
| LoRA rank / alpha / dropout | 32 / 32 / 0.05 |
| Optimizer | AdamW |
| Learning rate | `5e-5` |
| Minimum learning rate | `5e-6` |
| Warmup | 50 steps |
| Weight decay | 0.01 |
| Global / local batch | 8 / 1 |
| Updates | 1,000 |
| RNG seed | 42, ranked |
| Validation / checkpoint interval | 100 steps |
| Distributed strategy | FSDP2 |
| TP / CP / PP | 1 / 1 / 1 |
| Vision tower / audio tower / embeddings | frozen |
| Language model | trainable through LoRA |
| Loss | answer-only fused linear cross-entropy |

Gemma 4 E2B remains on the VLM path even though examples are text-only. Do not
switch to `NeMoAutoModelForCausalLM`, an LLM recipe, or a different chat
template.

A more powerful single GPU may reduce wall time without changing the contract.
Using multiple GPUs or changing DP topology can alter shuffling, numerical
behavior, and throughput accounting. Treat any topology change as a separately
qualified experiment, not a transparent acceleration.

## Completed runs

| Metric | K6 | K10 | K14 |
|---|---:|---:|---:|
| Languages | 6 | 10 | 14 |
| Train records | 153,273 | 143,579 | 268,251 |
| Materialized tokens | 31,348,990 | 31,349,312 | 31,353,562 |
| Populated language-task cells | 29/30 | 47/50 | 70/70 |
| Validation records | 23,470 | 30,306 | 23,440 |
| Maximum record repetitions | 4 | 4 | 4 |
| Benchmark fingerprint matches | 0 | 0 | 0 |
| Train-validation overlap | 0 | 0 | 0 |
| Final validation PPL | 12.3081 | 14.7193 | 18.1070 |
| Best validation step | 999 | 999 | 899 |
| Best validation PPL | 12.3081 | 14.7193 | 18.0862 |
| Logged non-padding train tokens | 29,625,932 | 29,626,426 | 29,727,513 |
| Token-equivalent epochs | 0.9450 | 0.9450 | 0.9481 |

Validation perplexities are not directly comparable as a language-quality
ranking because each arm's monitor set has a different language/task/source
composition. Use held-out benchmarks for controlled conclusions.

All local and W&B rows report integer `epoch = 0`. This is expected under the
current scheduler: epoch is a zero-based dataloader-pass index and increments
only after exhausting the dataloader. The fixed-step runs stopped near 95% of
their first pass. Use `step` as the W&B x-axis. For future runs, prefer an
additional metric:

```text
effective_epoch = cumulative_non_padding_tokens / materialized_mixture_tokens
```

Do not reinterpret integer epoch 0 as zero training.

## Completed checkpoint inventory

Original-machine roots:

```text
/home/casper/checkpoints/gemma4-e2b-k6/p2-r32-1k-v1
/home/casper/checkpoints/gemma4-e2b-k10/p2-r32-1k-v1
/home/casper/checkpoints/gemma4-e2b-k14/p3-r32-1k-v1
```

All `LATEST` links resolve to `epoch_0_step_999`.

Final adapter hashes:

| Run | `adapter_model.safetensors` SHA-256 |
|---|---|
| K6 | `12e2c1012d74872c7978cf3d853028fbf1997ba32328e2845b079314627d2e87` |
| K10 | `5d34c9e1f97b48c61e4079cd34fe0c5b359ce592911300812a674461ffbe31bb` |
| K14 | `da196a1ef094f1708f6a8d2254ac12b0449f9c969e5d3c74ba74320df5fe085f` |

W&B runs:

- K6: `https://wandb.ai/dsfsi/gemma4-african-instruction/runs/fw1uvbfb`
- K10: `https://wandb.ai/dsfsi/gemma4-african-instruction/runs/kje5g1e7`
- K14: `https://wandb.ai/dsfsi/gemma4-african-instruction/runs/nvjz0y4j`

## K14 accepted data state

Authoritative artifacts:

```text
/home/casper/k14-sft/k14_sft_profile.yaml
/home/casper/k14-sft/profile-v2/
/home/casper/k14-sft/plans-v2/
/home/casper/k14-sft/mixture-p3-final-v2/
/home/casper/k14-sft/final_audit.json
/home/casper/k14-sft/coverage_review.md
```

K14 final acceptance:

- 31,353,562 packed tokens: +29 tokens, or +0.000092%, from target.
- 12,669,914 supervised tokens: 40.41% of packed tokens.
- 268,251 train records and 219,196 unique train records.
- 70/70 language-task cells.
- 23,440 validation records.
- Zero train-validation overlap.
- Zero published benchmark fingerprint matches.
- Maximum source-record repetition count 4.
- Human 47.11%, curated 7.34%, mixed 41.26%, silver 4.29% by packed tokens.

Important K14 quality constraints:

- Kinyarwanda QA and isiXhosa QA were the lowest-capacity cells.
- Wolof classification uses a GlotLID-filtered silver source.
- Amharic and Somali NER use fiNERweb silver annotations.
- Sesotho, Twi, and Luganda NER use human corpora.
- AfriInstruct remains mixed-review-required and includes known xP3 language
  noise accepted consistently with K6/K10.

K14's profile references pools across all three data roots:

- `/data/gemma4-k6`
- `/data/gemma4-k10`
- `/data/gemma4-k14`

Transfer all three roots if the next agent must re-profile or materialize from
existing sources. Transferring only `/data/gemma4-k14` is insufficient for that
operation.

## Benchmark state

At source baseline `d872244e`, the repository contains:

```text
kseries-benchmark/results/author-suite/k6-step999-20260814/
kseries-benchmark/results/author-suite/k10-step999-20260815/
kseries-benchmark/K6_BENCHMARK_MATRIX.md
```

The K6 and K10 raw author-suite result JSON and run provenance are committed.
Do not rerun or overwrite them merely to reorganize outputs. Analyze them
in-place or write derived tables to a new path.

The local file
`/home/casper/afriinstruct/benchmark-results/final-v2/benchmark-sequence.log`
belongs to the earlier Inkuba/AfriInstruct v2 benchmark sequence. It is not a
K6/K10/K14 result and must not be mislabeled as one.

Open benchmark work:

1. Produce a controlled base/K6/K10 synthesis from the committed author-suite
   outputs if one does not yet exist.
2. Run K14 through the same pinned author-suite matrix.
3. Complete or explicitly defer the AfriInstruct 4,946-record longitudinal
   control for K6/K10/K14.
4. The MasakhaNER evaluator and overlap audit described in
   `kseries-benchmark/K6_BENCHMARK_MATRIX.md` remain separate work unless a
   newer committed result proves completion.

Do not use benchmark results to choose data filters, checkpoints, prompts, or
generation settings for the next training arm.

## Next run is not yet specified

The next agent must obtain an explicit decision for:

1. The next run name and total language count.
2. The exact languages being added.
3. Whether the staircase still adds four languages per arm.
4. Any benchmark coverage required for those languages.

Do not infer these choices from geographic balance, dataset availability, or
the phrase "next run." Until approved, use `<KNEXT>` and `<new languages>` in
notes and commands.

Once approved, research a matrix with one row per new language and one column
for instruction, QA, translation, classification, and NER. For each candidate
source, record:

- immutable revision or release identifier;
- license and noncommercial restrictions;
- source split and split safety;
- human, curated, mixed, synthetic, or silver creation method;
- target-language validation method;
- raw and accepted capacity;
- benchmark overlap risk;
- task-specific label/direction/entity coverage;
- fallback source and quality tier when no human source exists.

Prefer human or curated data. Use the best defensible synthetic/silver source
only where a cell otherwise remains empty, and preserve that quality tier in
metadata and reporting.

## Next-run data contract

Unless explicitly changed as a new experiment, retain K14's P3 planning policy:

```yaml
planning:
  packed_token_budget: 31353533
  temperature: 2.0
  max_epochs: 4.0
  default_source_family_cap: 0.5
  fallback_source_families: [afriinstruct]
  fallback_base_share: 0.25
  task_token_shares:
    instruction: 0.33
    qa: 0.15
    translation: 0.10
    classification: 0.17
    ner: 0.25
  minimum_label_tokens: 10400000
  sampling_seed: 42
  minimum_cell_token_share: 0.6
```

Source/task selection must be deterministic and stratified:

- classification by label;
- translation by direction;
- QA by source family and available domain/subtype metadata;
- NER by entity-type density;
- instruction by source family and available domain/subtype metadata.

Use exact Gemma processor token counts and answer-only labels. Record-balanced
sampling is not sufficient because record lengths differ sharply by source and
task.

## Required data workflow

1. Approve the next languages and benchmark plan.
2. Research and review the source-quality matrix.
3. Add only required source adapters to
   `tools/prepare_k6_sft_sources.py`; the legacy filename is the generic K-series
   preparer.
4. Pin revisions, normalize records, and materialize versioned source pools.
5. Extend the benchmark blocklist before profiling.
6. Run exact Gemma token profiling twice and compare outputs byte-for-byte.
7. Build a `p3_token_stratified` plan with fixed task-token shares.
8. Materialize with seeded token-aware stratum allocation.
9. Reconcile integer-record token drift only with validation-disjoint donor
   records and while preserving the four-appearance cap.
10. Audit every train and validation row.
11. Produce `coverage_review.md` and a machine-readable `final_audit.json`.
12. Run focused tests, Ruff check, and Ruff format check.
13. Perform a read-only production-container loader preflight.
14. Launch only after all gates pass.

Do not overwrite accepted K6/K10/K14 artifacts. Use a new root and versioned
output directories for every next-run attempt.

## Materialization acceptance gates

The next mixture is accepted only when:

- packed-token drift is negligible and documented;
- every expected language-task cell is populated or an approved gap is stated;
- supervised tokens are at least 10.4M;
- maximum source-record appearances are at most four;
- train-validation overlap is zero;
- benchmark fingerprint matches are zero;
- source licenses and revisions are recorded;
- task shares and per-language shares are reported;
- quality-tier shares are reported;
- low-capacity and synthetic/silver cells are explicitly identified;
- both manifests load through the production VLM dataset builder;
- focused tests and lint pass.

## Artifact transfer tiers

Use direct resumable `rsync` from the new machine. The original host is
`spark-dsfsi.up.ac.za`, user `casper`, SSH port 22.

### Tier A: minimum context

Git already contains the preparer, profiler, materializer, recipe, K6/K10 raw
author-suite results, and existing documentation. Pull only the external K14
audit and chat when the next agent is initially researching the next languages.

```bash
ROOT="$HOME/kseries-next-run"
mkdir -p "$ROOT/context" "$ROOT/k14-summary"

rsync -a --info=progress2 --partial --append-verify \
  casper@spark-dsfsi.up.ac.za:/home/casper/k14-sft/coverage_review.md \
  casper@spark-dsfsi.up.ac.za:/home/casper/k14-sft/final_audit.json \
  casper@spark-dsfsi.up.ac.za:/home/casper/k14-sft/k14_sft_profile.yaml \
  "$ROOT/k14-summary/"
```

### Tier B: source pools and materialization state

Required to re-profile, extend, or rematerialize from the existing staircase
sources. Current sizes are approximately K6 4.8 GB, K10 3.7 GB, and K14 0.85
GB.

```bash
ROOT="$HOME/kseries-next-run"
mkdir -p "$ROOT/data"/{k6-sft,k10-sft,k14-sft}

rsync -a --info=progress2 --partial --append-verify \
  casper@spark-dsfsi.up.ac.za:/home/casper/k6-sft/ \
  "$ROOT/data/k6-sft/"
rsync -a --info=progress2 --partial --append-verify \
  casper@spark-dsfsi.up.ac.za:/home/casper/k10-sft/ \
  "$ROOT/data/k10-sft/"
rsync -a --info=progress2 --partial --append-verify \
  casper@spark-dsfsi.up.ac.za:/home/casper/k14-sft/ \
  "$ROOT/data/k14-sft/"
```

Mount them at their historical container paths:

```bash
-v "$ROOT/data/k6-sft:/data/gemma4-k6" \
-v "$ROOT/data/k10-sft:/data/gemma4-k10" \
-v "$ROOT/data/k14-sft:/data/gemma4-k14"
```

### Tier C: final adapters

Not required to train the next arm from base. Transfer them for benchmarking,
parity checks, or consolidated analysis. Copying only each final `model/`
directory is about 132 MB per run; copying each complete final step is about 345
MB per run and preserves config/loss metadata.

```bash
ROOT="$HOME/kseries-next-run"
mkdir -p "$ROOT/checkpoints"/{k6,k10,k14}

rsync -a --info=progress2 --partial --append-verify \
  casper@spark-dsfsi.up.ac.za:/home/casper/checkpoints/gemma4-e2b-k6/p2-r32-1k-v1/epoch_0_step_999/ \
  "$ROOT/checkpoints/k6/epoch_0_step_999/"
rsync -a --info=progress2 --partial --append-verify \
  casper@spark-dsfsi.up.ac.za:/home/casper/checkpoints/gemma4-e2b-k10/p2-r32-1k-v1/epoch_0_step_999/ \
  "$ROOT/checkpoints/k10/epoch_0_step_999/"
rsync -a --info=progress2 --partial --append-verify \
  casper@spark-dsfsi.up.ac.za:/home/casper/checkpoints/gemma4-e2b-k14/p3-r32-1k-v1/epoch_0_step_999/ \
  "$ROOT/checkpoints/k14/epoch_0_step_999/"
```

Verify the three adapter hashes against the checkpoint table above before use.

## Chat export

The raw chat transcript on the original machine is:

```text
/home/casper/.vscode-server/data/User/workspaceStorage/
a180fdf4f6a25c7f6d7341ebc9121d74/GitHub.copilot-chat/transcripts/
f4ec334f-e2c2-4059-9f3f-82b8801ca5db.jsonl
```

It is approximately 3.5 MB. The transcript changes while this chat continues,
so no fixed hash in this document can remain authoritative. Finish the chat,
then compute the source hash immediately before transfer and compare it with the
destination hash.

```bash
ROOT="$HOME/kseries-next-run"
mkdir -p "$ROOT/context"

ssh casper@spark-dsfsi.up.ac.za \
  'sha256sum /home/casper/.vscode-server/data/User/workspaceStorage/a180fdf4f6a25c7f6d7341ebc9121d74/GitHub.copilot-chat/transcripts/f4ec334f-e2c2-4059-9f3f-82b8801ca5db.jsonl'

rsync -a --info=progress2 --partial --append-verify \
  casper@spark-dsfsi.up.ac.za:/home/casper/.vscode-server/data/User/workspaceStorage/a180fdf4f6a25c7f6d7341ebc9121d74/GitHub.copilot-chat/transcripts/f4ec334f-e2c2-4059-9f3f-82b8801ca5db.jsonl \
  "$ROOT/context/original-chat.jsonl"

sha256sum "$ROOT/context/original-chat.jsonl"
```

On the new machine, attach `original-chat.jsonl` to the new chat or instruct the
agent to search/read it for historical detail. Do not ask the agent to ingest
all lines blindly; the canonical handoff and current repository should be read
first, and the transcript should be used only to resolve missing history.

Some VS Code builds also expose an Export Chat action in the chat overflow menu
or Command Palette. That UI export is optional and version-dependent. The raw
JSONL path above is the verified portable fallback.

Never transfer shell history, `.netrc`, Hugging Face tokens, W&B keys, or other
credentials as chat context.

## New-machine environment

Create persistent host roots:

```bash
mkdir -p \
  "$HOME/kseries-next-run" \
  "$HOME/hf-cache" \
  "$HOME/checkpoints" \
  "$HOME/wandb"
```

Use the 26.04 image. Authenticate interactively on the new machine; a human must
enter secrets directly into the terminal. Never paste credentials into an agent
chat, Markdown, YAML, Dockerfile, or Git.

The successful K14 launch uncovered an important container rule:

> Do not launch training with `uv run` in a bind-mounted 26.04 checkout.

`uv run` attempted to resolve the project and download a public Torch wheel,
which would replace the NVIDIA stack. The unsafe container was stopped before
training and discarded. Launch with the image's installed entry point:

```text
/opt/venv/bin/automodel
```

Preflight the image and mounted source:

```bash
docker run --rm --gpus all \
  -v "$PWD:/opt/Automodel:ro" \
  -w /opt/Automodel \
  nvcr.io/nvidia/nemo-automodel:26.04 \
  /opt/venv/bin/python -c \
  "import nemo_automodel, torch; \
print(torch.__version__); \
print(nemo_automodel.__file__); \
print(torch.cuda.get_device_name())"
```

The imported `nemo_automodel` path must point into the mounted checkout and
Torch must remain the NVIDIA build.

## Next-run launch template

Do not execute this template until `<KNEXT>`, the approved language set, data
root, audit, and exact final mixture exist.

```bash
KNEXT="${KNEXT:?Set KNEXT to the approved lowercase run name, for example k18}"
name="gemma4-e2b-${KNEXT}-p3-r32-1k-v1"

docker create \
  --name "$name" \
  --gpus all \
  --network host \
  --ipc host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --log-opt max-size=100m \
  --log-opt max-file=5 \
  -e KSERIES_DATA_DIR="/data/gemma4-${KNEXT}/mixture-p3-final-v1" \
  -e KSERIES_CHECKPOINT_DIR="/checkpoints/gemma4-e2b-${KNEXT}/p3-r32-1k-v1" \
  -e KSERIES_WANDB_DIR=/logs/wandb \
  -e KSERIES_WANDB_NAME="gemma4-e2b-${KNEXT}-p3-r32-1k-v1" \
  -e KSERIES_WANDB_GROUP="gemma4-e2b-${KNEXT}-p3-r32" \
  -v "$PWD:/opt/Automodel" \
  -v "$HOME/kseries-next-run/data/${KNEXT}-sft:/data/gemma4-${KNEXT}:ro" \
  -v "$HOME/hf-cache:/root/.cache/huggingface" \
  -v "$HOME/checkpoints:/checkpoints" \
  -v "$HOME/wandb:/logs/wandb" \
  -w /opt/Automodel \
  nvcr.io/nvidia/nemo-automodel:26.04 \
  /opt/venv/bin/automodel \
    examples/vlm_finetune/gemma4/gemma4_e2b_kseries_p2_peft.yaml \
    --nproc-per-node 1 \
    --wandb.enable true
```

W&B authentication must already exist inside the container through an
interactive login or a securely mounted credential file. Do not embed the key
in the create command.

Launch acceptance requires:

- the container remains running;
- NVIDIA Torch is unchanged;
- the production loader reports the expected train/validation record counts;
- W&B creates the intended run identity;
- step 0 has finite loss, perplexity, gradient norm, and throughput;
- GPU allocation and utilization are stable;
- no CUDA OOM, traceback, or dependency installation appears;
- persistent metric files are created under the intended checkpoint root.

## Known lessons

- Equal record counts do not imply equal tokens; materialization must be
  token-aware.
- Exact deterministic profiling does not eliminate integer-record realization
  drift.
- Source-specific lengths can make record-balanced strata miss task-token
  targets materially.
- A valid final reconciliation must remain deterministic, validation-disjoint,
  benchmark-safe, and within the repetition cap.
- Validation PPL rises as the validation distribution broadens; do not treat it
  as the sole cross-run quality metric.
- The current integer epoch metric is correct but too coarse for sub-epoch
  fixed-step runs.
- More powerful hardware is useful only if the experimental contract remains
  unchanged.

## New-agent checklist

1. Read `AGENTS.md` and the relevant repository skills.
2. Read this handoff before the raw chat transcript.
3. Confirm the repository contains baseline `d872244e`.
4. Confirm whether benchmark synthesis or K14 evaluation has advanced since
   this document was written.
5. Obtain explicit approval for the next language tranche.
6. Build and review the language-task-source-quality matrix.
7. Transfer Tier B artifacts only when source reuse is required.
8. Keep the fixed-compute training contract unchanged.
9. Use versioned data roots and never overwrite accepted artifacts.
10. Satisfy all materialization acceptance gates.
11. Validate through the production container.
12. Launch with `/opt/venv/bin/automodel`, never `uv run`.
13. Monitor step 0, W&B identity, GPU state, and persistent logs.
14. Preserve raw outputs and report all deviations from this handoff.