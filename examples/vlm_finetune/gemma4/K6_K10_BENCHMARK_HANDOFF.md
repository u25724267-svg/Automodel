# Gemma 4 K6/K10 benchmark handoff

This document is the operational handoff for an agent benchmarking the completed
K6 and K10 Gemma 4 E2B adapters on another machine. It records the research
question, controlled training setup, data differences, exact artifacts, tested
environment setup, benchmark commands, reporting requirements, and known
limitations.

The benchmark agent should not rebuild the training mixtures or resume training.
The immediate job is to transfer the two final adapters, run one base-model
control and both adapters through the same evaluation environment, and report
base, K6, K10, and K10-minus-K6 results without tuning on benchmark data.

## Current status

Status as of 2026-08-13:

- Repository: `https://github.com/u25724267-svg/Automodel`
- Branch: `main`
- Required commit: `065c68005da7100b217a5910058b822a584aaba6`
- Base model: `google/gemma-4-E2B-it`
- K6 training: complete through step 999
- K10 training: complete through step 999
- For both runs, `LATEST` and `LOWEST_VAL` resolve to `epoch_0_step_999`
- Primary benchmark implementation: `tools/run_author_benchmarks.py`
- Longitudinal control implementation: `tools/afriinstruct_benchmark.py`
- No K6/K10 benchmark run has been completed yet

The source commit is published on the fork's `main` branch. Do not silently use
a newer commit. This handoff document is newer than that commit and must be sent
with the transfer bundle unless it is committed separately. If a source change
is required, record the new commit and explain why.

## Research objective

This project tests a fixed-compute language-token staircase. We keep the model,
training token budget, LoRA capacity, optimizer, and number of updates fixed
while increasing African-language breadth:

1. The historical AfriInstruct run established the initial baseline.
2. K6 introduced a six-language, multisource, quality-constrained mixture.
3. K10 retained K6 and added four languages without increasing the token or
   optimization budget.
4. K14 has been sampled separately, but it is not part of this benchmark task.

The K6/K10 comparison asks whether broader language coverage under fixed compute
improves transfer and newly added languages, and what it costs on the original
K6 languages and general capability. It is not a comparison of two runs with
equal examples per language: K10 spreads approximately the same number of
tokens across ten languages instead of six.

## Controlled training setup

Both adapters used the same reusable recipe:

`examples/vlm_finetune/gemma4/gemma4_e2b_kseries_p2_peft.yaml`

Important invariants:

| Setting | K6 and K10 |
|---|---|
| Base model | `google/gemma-4-E2B-it` |
| Framework path | NeMo AutoModel VLM, text-only SFT |
| Precision | bfloat16 |
| Attention | SDPA |
| Sequence and pack length | 4,096 |
| LoRA rank / alpha / dropout | 32 / 32 / 0.05 |
| Learning rate | `5e-5` |
| Minimum learning rate | `5e-6` |
| Warmup | 50 steps |
| Optimizer | AdamW, weight decay 0.01 |
| Global / local batch | 8 / 1 |
| Updates | 1,000 |
| Distributed strategy | single-GPU FSDP2 |
| RNG seed | 42 |
| Checkpoint and validation interval | 100 steps |
| Vision and audio towers | frozen |
| Embeddings | frozen |
| Loss | answer-only fused linear cross-entropy |

Gemma 4 E2B is a multimodal architecture even in this text-only experiment.
The benchmark must load it through the repository's Gemma-compatible
multimodal path. Do not substitute a causal-LM loader or a different chat
template.

## Run comparison

| Property | K6 | K10 |
|---|---:|---:|
| Languages | 6 | 10 |
| Training records | 153,273 | 143,579 |
| Unique training records | 128,527 | 122,312 |
| Realized packed tokens | 31,348,990 | 31,349,312 |
| Target packed tokens | 31,353,533 | 31,353,533 |
| Validation records | 23,470 | 30,306 |
| Populated language-task cells | 29/30 | 47/50 |
| Maximum source-record repetitions | 4 | 4 |
| Train-validation overlap | 0 | 0 |
| Benchmark fingerprint matches | 0 | 0 |
| Final validation loss | 2.5102548 | 2.6891565 |
| Final validation perplexity | 12.3081 | 14.7193 |

K6 languages:

- Hausa (`hau`)
- Igbo (`ibo`)
- Kinyarwanda (`kin`)
- Swahili (`swh`, represented as `swa` in the author benchmark)
- Yoruba (`yor`)
- isiZulu (`zul`)

K10 adds:

- Amharic (`amh`)
- isiXhosa (`xho`)
- Shona (`sna`)
- Wolof (`wol`)

Both mixtures cover instruction, QA, translation, classification, and NER.
K6 leaves Kinyarwanda QA empty. K10 leaves Kinyarwanda QA, Amharic NER, and
Wolof classification empty. These were intentional quality decisions rather
than accidental sampling omissions.

Do not rank the adapters from validation perplexity alone. Their monitor sets
have different language/task compositions and different supervised-token
counts. The held-out benchmarks are the controlled comparison.

The detailed data reviews on the original machine are:

- `/home/casper/k6-sft/coverage_review.md`
- `/home/casper/k10-sft/coverage_review.md`

K10's recorded W&B run is
`https://wandb.ai/dsfsi/gemma4-african-instruction/runs/kje5g1e7`.

## Final checkpoint inventory

Original-machine checkpoint roots:

```text
/home/casper/checkpoints/gemma4-e2b-k6/p2-r32-1k-v1
/home/casper/checkpoints/gemma4-e2b-k10/p2-r32-1k-v1
```

Final checkpoint directories:

```text
/home/casper/checkpoints/gemma4-e2b-k6/p2-r32-1k-v1/epoch_0_step_999
/home/casper/checkpoints/gemma4-e2b-k10/p2-r32-1k-v1/epoch_0_step_999
```

Each full final step occupies approximately 345 MB. Each `model/` directory is
approximately 132 MB and contains:

```text
adapter_config.json
adapter_model.safetensors
automodel_peft_config.json
chat_template.jinja
processor_config.json
tokenizer_config.json
tokenizer.json
```

The author-aligned harness requires at least `adapter_config.json` and
`adapter_model.safetensors`. Copy the complete `model/` directory. Copy the
complete final step directory when the additional 213 MB per run is acceptable;
that preserves the exact recipe and loss metadata and is the safer payload for
the longitudinal NeMo checkpoint loader.

Expected final-adapter hashes:

| Run | File | SHA-256 |
|---|---|---|
| K6 | `adapter_config.json` | `0619d3e5996921d951cff6bef15fb6a2b140a75112420449f897bb523773c4cf` |
| K6 | `adapter_model.safetensors` | `12e2c1012d74872c7978cf3d853028fbf1997ba32328e2845b079314627d2e87` |
| K10 | `adapter_config.json` | `0619d3e5996921d951cff6bef15fb6a2b140a75112420449f897bb523773c4cf` |
| K10 | `adapter_model.safetensors` | `5d34c9e1f97b48c61e4079cd34fe0c5b359ce592911300812a674461ffbe31bb` |

The identical config hashes are expected because both runs use the same LoRA
topology. The adapter weight hashes must differ.

## Build the transfer bundle on the original machine

The following creates a self-verifying bundle with both full final checkpoints,
run logs, data reports, and the optional AfriInstruct benchmark inputs. It does
not include the training mixtures or the nine earlier checkpoints.

```bash
set -euo pipefail

TRANSFER_ROOT="$HOME/kseries-benchmark-transfer"
if [[ -e "$TRANSFER_ROOT" ]]; then
  echo "Refusing to overwrite $TRANSFER_ROOT" >&2
  exit 1
fi

mkdir -p \
  "$TRANSFER_ROOT/checkpoints/k6" \
  "$TRANSFER_ROOT/checkpoints/k10" \
  "$TRANSFER_ROOT/run-info/k6" \
  "$TRANSFER_ROOT/run-info/k10" \
  "$TRANSFER_ROOT/benchmarks"

cp "$HOME/AfricaLLM/Automodel/examples/vlm_finetune/gemma4/K6_K10_BENCHMARK_HANDOFF.md" \
  "$TRANSFER_ROOT/"

cp -a \
  "$HOME/checkpoints/gemma4-e2b-k6/p2-r32-1k-v1/epoch_0_step_999" \
  "$TRANSFER_ROOT/checkpoints/k6/"
cp -a \
  "$HOME/checkpoints/gemma4-e2b-k10/p2-r32-1k-v1/epoch_0_step_999" \
  "$TRANSFER_ROOT/checkpoints/k10/"

cp "$HOME/checkpoints/gemma4-e2b-k6/p2-r32-1k-v1/training.jsonl" \
  "$HOME/checkpoints/gemma4-e2b-k6/p2-r32-1k-v1/validation.jsonl" \
  "$HOME/k6-sft/coverage_review.md" \
  "$TRANSFER_ROOT/run-info/k6/"
cp "$HOME/checkpoints/gemma4-e2b-k10/p2-r32-1k-v1/training.jsonl" \
  "$HOME/checkpoints/gemma4-e2b-k10/p2-r32-1k-v1/validation.jsonl" \
  "$HOME/k10-sft/coverage_review.md" \
  "$TRANSFER_ROOT/run-info/k10/"

cp -a "$HOME/afriinstruct/benchmarks/afriinstruct-paper" \
  "$TRANSFER_ROOT/benchmarks/"

(
  cd "$TRANSFER_ROOT"
  find . -type f ! -name SHA256SUMS -print0 \
    | sort -z \
    | xargs -0 sha256sum > SHA256SUMS
)

tar -C "$TRANSFER_ROOT" -czf "$HOME/kseries-benchmark-transfer.tar.gz" .
sha256sum "$HOME/kseries-benchmark-transfer.tar.gz" \
  > "$HOME/kseries-benchmark-transfer.tar.gz.sha256"
```

Transfer both the archive and its `.sha256` file with `rsync -avP`, `scp`, or
the site's approved object store. Do not put either adapter in Git.

## New-machine prerequisites

Recommended:

- Linux with a working NVIDIA driver and Docker GPU support
- One CUDA GPU; keep benchmark batch size at 1
- At least 24 GB GPU memory is preferred
- Enough storage for the container, Gemma weights, adapters, task datasets,
  sample logs, and result files
- Hugging Face access to gated `google/gemma-4-E2B-it`
- Network access for the first model and benchmark dataset download

Check the machine before starting:

```bash
nvidia-smi
docker --version
docker run --rm --gpus all nvcr.io/nvidia/nemo-automodel:26.04 \
  /opt/venv/bin/python -c \
  "import torch; print(torch.__version__); print(torch.cuda.get_device_name())"
df -h
```

## Restore source and artifacts

Clone and pin the exact source revision:

```bash
git clone https://github.com/u25724267-svg/Automodel.git
cd Automodel
git checkout 065c68005da7100b217a5910058b822a584aaba6
test "$(git rev-parse HEAD)" = \
  "065c68005da7100b217a5910058b822a584aaba6"
```

Restore the transferred bundle outside the repository:

```bash
mkdir -p "$HOME/kseries-benchmark"
sha256sum -c "$HOME/kseries-benchmark-transfer.tar.gz.sha256"
tar -C "$HOME/kseries-benchmark" \
  -xzf "$HOME/kseries-benchmark-transfer.tar.gz"
(
  cd "$HOME/kseries-benchmark"
  sha256sum -c SHA256SUMS
)
```

Verify the two critical weights again:

```bash
sha256sum \
  "$HOME/kseries-benchmark/checkpoints/k6/epoch_0_step_999/model/adapter_model.safetensors" \
  "$HOME/kseries-benchmark/checkpoints/k10/epoch_0_step_999/model/adapter_model.safetensors"
```

Expected hashes are the K6 and K10 values in the checkpoint table above.

## Start the benchmark container

From the `Automodel` checkout on the new machine:

```bash
mkdir -p "$HOME/hf-cache" "$HOME/kseries-benchmark/results"

docker run --gpus all --network=host -it \
  --name gemma4-kseries-bench \
  --shm-size=32g \
  -v "$PWD:/opt/Automodel:ro" \
  -v "$HOME/kseries-benchmark:/workspace" \
  -v "$HOME/hf-cache:/root/.cache/huggingface" \
  -w /opt/Automodel \
  nvcr.io/nvidia/nemo-automodel:26.04 /bin/bash
```

The named container can be re-entered after a disconnect:

```bash
docker start -ai gemma4-kseries-bench
```

All remaining commands run inside this container unless stated otherwise.

## Install the pinned benchmark harness

The current repository README mentions a `benchmark` optional extra, but no
such extra exists in the current or historical `pyproject.toml`. Use this tested
runtime installation instead:

```bash
TORCH_BEFORE=$(/opt/venv/bin/python -c 'import torch; print(torch.__version__)')

uv pip install --python /opt/venv/bin/python \
  "lm_eval @ git+https://github.com/EleutherAI/lm-evaluation-harness.git@f4d4b3de3ee6741a7151a9fe74945ee515262f4c" \
  "sacrebleu==2.6.0" \
  "fuzzywuzzy==0.18.0" \
  "lxml<7"

TORCH_AFTER=$(/opt/venv/bin/python -c 'import torch; print(torch.__version__)')
test "$TORCH_BEFORE" = "$TORCH_AFTER"

/opt/venv/bin/python -c \
  "import fuzzywuzzy, lm_eval, lxml, peft, sacrebleu, torch, transformers; \
print('torch', torch.__version__); \
print('transformers', transformers.__version__); \
print('peft', peft.__version__); \
print('fuzzywuzzy', fuzzywuzzy.__version__); \
print('lxml', lxml.__version__); \
print('sacrebleu', sacrebleu.__version__)"
```

Do not add the harness's `[hf]` extra. In the 26.04 container that extra resolves
a new public PyTorch wheel and replaces the NVIDIA PyTorch build. The command
above was tested in a disposable 26.04 container: it preserved NVIDIA PyTorch,
Transformers, and PEFT, installed LM Evaluation Harness `0.4.13.dev0` from the
pinned commit, and configured 120 tasks for each adapter.

Authenticate interactively with Hugging Face. The human operator must enter any
token directly into the terminal; never place it in this document, a command,
YAML, a result file, or an agent chat message.

```bash
/opt/venv/bin/hf auth login
```

Confirm access:

```bash
/opt/venv/bin/python -c \
  "from transformers import AutoProcessor; \
AutoProcessor.from_pretrained('google/gemma-4-E2B-it'); \
print('Gemma 4 access OK')"
```

## Validate the local benchmark code

Run the focused CPU tests:

```bash
/opt/venv/bin/python -m pytest \
  tests/unit_tests/tools/test_run_author_benchmarks.py \
  tests/unit_tests/tools/test_afriinstruct_benchmark.py \
  -q
```

Dry-run all three model identities before loading model weights:

```bash
/opt/venv/bin/python tools/run_author_benchmarks.py \
  --suite all \
  --output-dir /workspace/results/author-dry-run/base \
  --dry-run

/opt/venv/bin/python tools/run_author_benchmarks.py \
  --suite all \
  --checkpoint-path /workspace/checkpoints/k6/epoch_0_step_999 \
  --output-dir /workspace/results/author-dry-run/k6 \
  --dry-run

/opt/venv/bin/python tools/run_author_benchmarks.py \
  --suite all \
  --checkpoint-path /workspace/checkpoints/k10/epoch_0_step_999 \
  --output-dir /workspace/results/author-dry-run/k10 \
  --dry-run
```

Each command must report `Configured 120 pinned upstream tasks`. Inspect each
`run_provenance.json`; it should record the exact checkpoint path, task list,
command, and these revisions:

| Upstream | Revision |
|---|---|
| LM Evaluation Harness | `f4d4b3de3ee6741a7151a9fe74945ee515262f4c` |
| IrokoBench / Masakhane NLU | `1f8be590da6699aee3dc23de6f63e801e2352eff` |
| Belebele | `918890beb2290a8d3ef2d7a90369925959e1bacf` |

## Primary benchmark: author-aligned suite

This is the primary K6/K10 evaluation. It delegates prompt construction,
log-likelihood multiple-choice scoring, AfriMGSM generation and filters,
exact-match scoring, and aggregation to the benchmark authors' pinned LM
Evaluation Harness tasks.

The default matrix has 120 task variants:

- Suites: AfriMMLU, AfriXNLI, AfriMGSM, and Belebele
- Languages: English, Hausa, Swahili, isiXhosa, Yoruba, and isiZulu
- Prompt variants: all five author prompts
- Few-shot setting: zero-shot
- Chat template: enabled
- Batch size: 1
- Predictions/sample logs: retained

The repository compatibility bootstrap only exposes the harness's existing
weighted-F1 function to the pinned AfriMMLU/AfriXNLI YAML. It does not replace
prompts, model requests, predictions, filters, or aggregators. Always launch
through `tools/run_author_benchmarks.py`; do not invoke `lm-eval` directly.

### GPU smoke gate

Run a small Hausa/isiXhosa subset for base, K6, and K10. Use separate output
directories from the full run.

```bash
/opt/venv/bin/python tools/run_author_benchmarks.py \
  --suite all --languages hau,xho --prompt-ids 1 --limit 2 \
  --output-dir /workspace/results/smoke/base

/opt/venv/bin/python tools/run_author_benchmarks.py \
  --suite all --languages hau,xho --prompt-ids 1 --limit 2 \
  --checkpoint-path /workspace/checkpoints/k6/epoch_0_step_999 \
  --output-dir /workspace/results/smoke/k6

/opt/venv/bin/python tools/run_author_benchmarks.py \
  --suite all --languages hau,xho --prompt-ids 1 --limit 2 \
  --checkpoint-path /workspace/checkpoints/k10/epoch_0_step_999 \
  --output-dir /workspace/results/smoke/k10
```

Proceed only if all three models load, sample logs are nonempty, metrics are
finite, the K6 and K10 provenance points to the intended adapter, and GPU memory
stays stable. This gate is plumbing validation, not a result to report or tune
against.

### Full runs

Run the base control in the same container and software environment as both
adapters:

```bash
/opt/venv/bin/python tools/run_author_benchmarks.py \
  --suite all \
  --output-dir /workspace/results/author-suite/base

/opt/venv/bin/python tools/run_author_benchmarks.py \
  --suite all \
  --checkpoint-path /workspace/checkpoints/k6/epoch_0_step_999 \
  --output-dir /workspace/results/author-suite/k6-step999

/opt/venv/bin/python tools/run_author_benchmarks.py \
  --suite all \
  --checkpoint-path /workspace/checkpoints/k10/epoch_0_step_999 \
  --output-dir /workspace/results/author-suite/k10-step999
```

Do not change language lists, prompt IDs, chat-template behavior, precision,
attention backend, batch size, or task revisions between models. Do not choose
the best prompt variant after seeing results.

LM Evaluation Harness output is not guaranteed to resume an interrupted full
invocation. Preserve completed output directories and use new, clearly named
directories if a run must be repeated.

## Secondary benchmark: AfriInstruct longitudinal control

This suite reproduces the historical AfriInstruct paper software artifact. It
contains 4,946 unique scorable records after stable deduplication:

- A 1,000-record mixed-language benchmark
- Hausa, Igbo, Kinyarwanda, Swahili, Yoruba, and isiZulu language files
- QA, translation, and topic-classification metrics

It is useful for continuity with the historical AfriInstruct LoRA result but is
secondary to the author-aligned suite. Its generation cache is resumable by
stable record ID and flushes each prediction immediately.

Set the ordered benchmark arguments once:

```bash
BENCH_ROOT=/workspace/benchmarks/afriinstruct-paper
BENCH_ARGS=(
  --benchmark-file "$BENCH_ROOT/benchmark.json"
  --benchmark-file "$BENCH_ROOT/benchmark_hau_806.json"
  --benchmark-file "$BENCH_ROOT/benchmark_ibo_806.json"
  --benchmark-file "$BENCH_ROOT/benchmark_kin_806.json"
  --benchmark-file "$BENCH_ROOT/benchmark_swa_602.json"
  --benchmark-file "$BENCH_ROOT/benchmark_yor_602.json"
  --benchmark-file "$BENCH_ROOT/benchmark_zul_602.json"
)
mkdir -p /workspace/results/afriinstruct-paper
```

Generate base, K6, and K10 predictions:

```bash
/opt/venv/bin/python -m tools.afriinstruct_benchmark generate \
  "${BENCH_ARGS[@]}" \
  --base-model google/gemma-4-E2B-it \
  --model-label base \
  --output /workspace/results/afriinstruct-paper/base.jsonl \
  --max-new-tokens 128

/opt/venv/bin/python -m tools.afriinstruct_benchmark generate \
  "${BENCH_ARGS[@]}" \
  --base-model google/gemma-4-E2B-it \
  --checkpoint-path /workspace/checkpoints/k6/epoch_0_step_999 \
  --model-label k6-step999 \
  --output /workspace/results/afriinstruct-paper/k6-step999.jsonl \
  --max-new-tokens 128

/opt/venv/bin/python -m tools.afriinstruct_benchmark generate \
  "${BENCH_ARGS[@]}" \
  --base-model google/gemma-4-E2B-it \
  --checkpoint-path /workspace/checkpoints/k10/epoch_0_step_999 \
  --model-label k10-step999 \
  --output /workspace/results/afriinstruct-paper/k10-step999.jsonl \
  --max-new-tokens 128
```

Score each cache with the same ordered inputs:

```bash
for MODEL in base k6-step999 k10-step999; do
  /opt/venv/bin/python -m tools.afriinstruct_benchmark score \
    "${BENCH_ARGS[@]}" \
    --predictions "/workspace/results/afriinstruct-paper/${MODEL}.jsonl" \
    --output "/workspace/results/afriinstruct-paper/${MODEL}-metrics.json"
done
```

The default canonical topic matcher treats unmatched model text as incorrect.
If reproducing the paper's fuzzy topic omission, run a second explicitly named
score with `--topic-matching paper-fuzzy`; never mix the two policies in one
comparison.

Expected source benchmark hashes:

| File | SHA-256 |
|---|---|
| `benchmark.json` | `c4831933c60cf1009c0d3d9112f0cc9516f294b685e0754c0930f4ee320812b5` |
| `benchmark_hau_806.json` | `0f735e51716d6393b8cc95c103ea71b86bf17b2540586cbcdaae6b234300dd00` |
| `benchmark_ibo_806.json` | `2e97d74766c2224a4f143bbdc786825345c23087615539f43e94e533a936a153` |
| `benchmark_kin_806.json` | `6063f3a38d31931232d090e60da743074c2b75d07e9a59123f6643d1b65993c0` |
| `benchmark_swa_602.json` | `950675ac5fd19984633fb16406ae1ceb5f69f29b5c4483383647deeda2511b81` |
| `benchmark_yor_602.json` | `9b08304cc5e35c12a62283e3de9df562bb87e5efb66d3c18412caccd76e0f07d` |
| `benchmark_zul_602.json` | `b7ff77608998a6fc9a0c590895aff8116fce2d2b4f989ffbe7e7a63283f05dfb` |

## Required analysis and report

Keep raw outputs immutable. Build any tables in a separate analysis directory.
The report must include:

1. Exact repository commit, container image ID, GPU model, driver, CUDA-visible
   device, package versions, adapter hashes, benchmark revisions, and commands.
2. Base, K6, and K10 absolute scores.
3. K6-minus-base, K10-minus-base, and K10-minus-K6 deltas.
4. Author-suite results by benchmark, language, and prompt variant.
5. Mean and variability across all five prompt variants. Do not report only the
   best prompt.
6. A K6-language comparison for Hausa, Swahili, Yoruba, and isiZulu.
7. An isiXhosa comparison, highlighting that it is newly trained in K10 and was
   unseen by K6.
8. English results as a general-capability retention check.
9. AfriInstruct QA token F1, translation ChrF++, topic accuracy, topic macro F1,
   and topic match rate, overall and by language/task where available.
10. Counts of missing predictions, generation token-limit hits, empty outputs,
    repeated-output loops, and run failures.
11. Wall time and peak GPU memory for each model.
12. A conclusion that separates measured evidence from hypotheses about
    language-count scaling or transfer.

Use macro views alongside aggregate views so large tasks or languages do not
hide regressions. A useful final table has rows for benchmark-language pairs and
columns for base, K6, K10, K6-base, K10-base, and K10-K6.

## Interpretation boundaries

- The author-aligned suite evaluates English, Hausa, Swahili, isiXhosa, Yoruba,
  and isiZulu. It does not evaluate Igbo, Kinyarwanda, Amharic, Shona, or Wolof.
- Of the four K10 additions, only isiXhosa is covered by the current primary
  harness. Do not generalize its result to Amharic, Shona, or Wolof.
- The AfriInstruct longitudinal suite covers the six K6 languages but none of
  the other K10 additions.
- K6 and K10 validation perplexities are not directly comparable because their
  monitor sets differ.
- These are single training runs with no training-seed replication. Benchmark
  prompt variation is not a substitute for training-run uncertainty.
- Benchmark data must not be used to alter checkpoints, choose a checkpoint,
  tune generation, select a prompt, or filter training data retroactively.
- K6/K10 training mixtures were benchmark-blocklisted against pinned AfriMMLU,
  AfriXNLI, AfriGSM, and Belebele text and 12-token fragments, with zero final
  fingerprint matches. This substantially reduces exact leakage risk but is not
  proof against semantic memorization from upstream pretraining.

## Common failures

### `Checkpoint is missing standard PEFT files`

`--checkpoint-path` must point to the step directory whose child is `model/`,
not directly to `model/`:

```text
/workspace/checkpoints/k6/epoch_0_step_999
  model/
    adapter_config.json
    adapter_model.safetensors
```

### Torch changes during harness installation

The wrong command included `lm_eval[hf]`. Discard and recreate the container,
then use the exact harness-only command above. Do not repair the environment by
installing another Torch wheel.

### Weighted-F1 import failure

Launch through `tools/run_author_benchmarks.py`. It invokes
`tools/lm_eval_compat.py`, which supplies the narrow compatibility alias expected
by the pinned Iroko task definitions.

### Hugging Face 401/403

The account has not accepted Gemma's terms, authentication is missing, or the
cache belongs to a different user. Authenticate interactively and rerun the
processor access check.

### CUDA out of memory

Keep `--batch-size 1`, close other GPU processes, and confirm the expected GPU
is visible. Do not change dtype, quantize one model only, shorten tasks, or use a
different backend for only one arm. If the machine cannot run all three models
under one policy, stop and report the comparability blocker.

### Interrupted AfriInstruct generation

Rerun the exact same command with the same output JSONL. Completed stable record
IDs are loaded and skipped. Do not truncate a valid cache.

## Completion checklist

- Source checkout is exactly `065c6800...`
- K6 and K10 adapter hashes match this handoff
- NVIDIA Torch version is unchanged after harness installation
- Focused benchmark tests pass
- Base, K6, and K10 dry runs each configure 120 tasks
- GPU smoke gate passes for all three models
- Full author-aligned suite completes for base, K6, and K10
- Optional AfriInstruct control completes and scores all three models
- Raw outputs, samples, provenance, environment details, and hashes are retained
- Final report includes absolute scores, all three deltas, per-language/task
  breakdowns, prompt variability, diagnostics, and stated coverage limitations