# Gemma 4 12B Top-40 gold migration runbook

Date: 2026-09-22

## Purpose

This runbook transfers the proven Top-40 gold instruction-tuning workflow to
another machine and adapts it from `google/gemma-4-E2B-it` to
`google/gemma-4-12B-it`.

The 12B target is not a model-ID-only replacement:

- E2B uses `FinetuneRecipeForVLM`, `NeMoAutoModelForImageTextToText`, and the
  VLM meta-dataset loader.
- Gemma 4 12B uses `TrainFinetuneRecipeForNextTokenPrediction`,
  `NeMoAutoModelForCausalLM`, and the LLM `ChatDataset` path.
- The prepared OpenAI-format conversations can be reused, but they must be
  exported from the VLM meta shards to a format that `ChatDataset` loads.
- Every 12B machine must pass a new 20-step qualification and step-25 resume
  gate before a long run begins.

The repository contains an official dense 12B example at
`examples/llm_finetune/gemma/gemma_4_12b_hellaswag.yaml`. Both
`google/gemma-4-12B` and `google/gemma-4-12B-it` currently resolve to
`Gemma4UnifiedForConditionalGeneration`; the `-it` tokenizer has a chat
template and is therefore the appropriate starting point for this instruction
dataset. The exact `-it` LoRA path described here still requires qualification
on the destination machine.

## Source reference

The source machine was captured at:

| Field | Value |
|---|---|
| Branch | `u25724267-svg/chore/afrigemma-sft-casper` |
| Commit | `749dc9776350d86a27df6663b08946474d3523b6` |
| `pyproject.toml` SHA-256 | `59982f08c7f45bfddd9912d121c87e71b35d8043e52adef685183d520590e1f9` |
| Data readiness SHA-256 | `c63c865a04a861b489967b7ce2f4c735913b42be77497e4fc087f331da709d7d` |
| Qualified E2B recipe SHA-256 | `49f179d214e65121aa4a53b6ff1cb79e0fea6d2f928a0ffe5f011b43b6406375` |

The reference E2B run is:

- W&B name: `gemma4-e2b-top40-gold-lora-r16-10k-v1`
- W&B group: `gemma4-e2b-top40-gold-lora`
- W&B project: `dsfsi/gemma4-african-instruction`
- W&B URL: <https://wandb.ai/dsfsi/gemma4-african-instruction/runs/5y70c2ki>
- Train records: 407,567
- Full validation records: 70,053
- Training monitor: deterministic 1% sample, 696 records
- Sequence length: 4,096
- LoRA rank / alpha: 16 / 32
- Global / local batch: 8 / 1
- Checkpoint and validation interval: 500 optimizer steps
- Long-run target: 10,000 optimizer steps

The E2B qualification passed on one A100 40 GB. It reached 30.83 GiB peak
allocated memory, completed steps 0-19, restored from step 19, and continued
through step 24 with finite losses and gradient norms. These memory numbers do
not apply to the dense 12B target.

## Destination hardware

Recommended starting configurations for 12B LoRA are:

- one 80 GB GPU; or
- at least two 40 GB GPUs with FSDP2 data parallelism.

The official 12B full-finetuning example documents an eight-GPU launch. LoRA
reduces optimizer and gradient memory, but no single-40-GB claim should be made
until the destination qualification proves it. Keep `tp_size`, `cp_size`, and
`pp_size` at one initially; additional processes provide FSDP2 data
parallelism. If the 4,096-token qualification is out of memory, first enable
activation checkpointing, then test a 2,048-token qualification. Record any
such deviation as a new run version.

## Clone and environment

Clone the exact branch and verify the commit:

```bash
git clone <repository-url> Automodel
cd Automodel
git checkout u25724267-svg/chore/afrigemma-sft-casper
test "$(git rev-parse HEAD)" = "749dc9776350d86a27df6663b08946474d3523b6"
```

Install the locked environment. The `dev` group is required by
`FusedLinearCrossEntropy` because it supplies `cut-cross-entropy`.

```bash
uv sync --locked --group dev
```

For an NVIDIA container, follow the repository build skill instead: mount the
checkout, run `docker/common/update_pyproject_pytorch.sh /opt/Automodel`, and
then run the locked `uv sync` inside the container. Do not install a separate
Torch/Torchvision pair over the container stack.

Create `.env` locally. Never commit it.

```dotenv
WANDB_API_KEY=<secret>
HF_TOKEN=<secret>
WANDB_MODE=online
WANDB_ENTITY=dsfsi
WANDB_PROJECT=gemma4-african-instruction
WANDB_DIR=/logs/wandb
```

Validate without printing secrets:

```bash
set -a
source .env
set +a
test -n "$WANDB_API_KEY"
test -n "$HF_TOKEN"
test "$WANDB_MODE" = online
test "$WANDB_ENTITY" = dsfsi
test "$WANDB_PROJECT" = gemma4-african-instruction
mkdir -p "$WANDB_DIR"
```

## Transfer the ignored dataset

The materialized data is intentionally ignored by Git, so cloning the
repository does not transfer it. Copy it separately while excluding source-run
checkpoints:

```bash
mkdir -p examples/vlm_finetune/gemma4/data/top40_gold_v1
rsync -a --info=progress2 \
  --exclude 'qualification/' \
  --exclude 'runs/' \
  <source-host>:/path/to/Automodel/examples/vlm_finetune/gemma4/data/top40_gold_v1/ \
  examples/vlm_finetune/gemma4/data/top40_gold_v1/
```

Verify the copied corpus:

```bash
DATA_ROOT="$PWD/examples/vlm_finetune/gemma4/data/top40_gold_v1"
test "$(sha256sum "$DATA_ROOT/readiness_report.json" | cut -d' ' -f1)" = \
  "c63c865a04a861b489967b7ce2f4c735913b42be77497e4fc087f331da709d7d"
```

The corpus has 407,567 training records, 70,053 validation records, zero exact
train-validation overlap, and zero remaining matches against the available
benchmark blocklist. Exact FLORES+ coverage and licensing remain separately
documented limitations.

## Export conversations for `ChatDataset`

Do not point the 12B LLM recipe at `train_meta.json`. That manifest belongs to
the VLM loader. Also do not concatenate the JSONL shards with a naive
`splitlines()` parser: legal Unicode separators exist inside some strings.

Export normalized fields to Parquet, which `ChatDataset` loads directly and
which preserves those Unicode strings safely:

```bash
export DATA_ROOT="$PWD/examples/vlm_finetune/gemma4/data/top40_gold_v1"
export EXPORT_ROOT="$DATA_ROOT/gemma4_12b_chat"

.venv/bin/python - <<'PY'
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

root = Path(os.environ["DATA_ROOT"])
export_root = Path(os.environ["EXPORT_ROOT"])
if export_root.exists():
    raise FileExistsError(f"Refusing to overwrite {export_root}")

def export_split(split: str) -> None:
    manifest = json.loads((root / f"{split}_meta.json").read_text(encoding="utf-8"))
    destination = export_root / split / "part-00000.parquet"
    destination.parent.mkdir(parents=True)
    writer = None
    rows = []

    def flush() -> None:
        nonlocal writer, rows
        if not rows:
            return
        table = pa.Table.from_pylist(rows)
        if writer is None:
            writer = pq.ParquetWriter(destination, table.schema, compression="zstd")
        writer.write_table(table)
        rows = []

    for entry in manifest.values():
        with (root / entry["file_name"]).open(encoding="utf-8") as source_file:
            for line in source_file:
                record = json.loads(line)
                rows.append(
                    {
                        "messages": record["messages"],
                        "language": str(record.get("language", "unknown")),
                        "task": str(record.get("task", "unknown")),
                        "source_dataset": str(record.get("source_dataset", "unknown")),
                    }
                )
                if len(rows) >= 5_000:
                    flush()
    flush()
    if writer is not None:
        writer.close()
    print(split, pq.ParquetFile(destination).metadata.num_rows)

for split in ("train", "validation"):
    export_split(split)
PY
```

Expected counts are 407,567 train and 70,053 validation. For the in-training
monitor, `ChatDataset` can deterministically shuffle and slice validation to
1,024 examples. Retain the complete validation Parquet for milestone and final
evaluation.

## Draft 12B LoRA recipe

Create
`examples/llm_finetune/gemma/gemma_4_12b_top40_gold_lora.yaml` with the
following starting configuration. This is a qualification recipe until the
20-step and resume gates pass on the destination hardware.

```yaml
recipe: TrainFinetuneRecipeForNextTokenPrediction

step_scheduler:
  global_batch_size: 8
  local_batch_size: 1
  ckpt_every_steps: 500
  val_every_steps: 500
  log_remote_every_steps: 10
  max_steps: 10000
  num_epochs: null

dist_env:
  backend: nccl
  timeout_minutes: 30

rng:
  _target_: nemo_automodel.components.training.rng.StatefulRNG
  seed: 42
  ranked: true

model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: google/gemma-4-12B-it
  torch_dtype: bf16
  attn_implementation: sdpa
  output_hidden_states: true

peft:
  _target_: nemo_automodel.components._peft.lora.PeftConfig
  target_modules: "*_proj"
  dim: 16
  alpha: 32
  dropout: 0.0
  use_triton: true

checkpoint:
  enabled: true
  checkpoint_dir: ${TOP40_12B_CHECKPOINT_DIR,checkpoints/gemma4-12b-top40-gold-lora}
  model_save_format: safetensors
  save_consolidated: false

distributed:
  strategy: fsdp2
  dp_size: none
  tp_size: 1
  cp_size: 1
  pp_size: 1
  sequence_parallel: false
  activation_checkpointing: true

loss_fn:
  _target_: nemo_automodel.components.loss.linear_ce.FusedLinearCrossEntropy
  reduction: sum

dataset:
  _target_: nemo_automodel.components.datasets.llm.chat_dataset.ChatDataset
  path_or_dataset_id: ${TOP40_12B_DATA_DIR}/train
  seq_length: 4096
  truncation: true

packed_sequence:
  packed_sequence_size: 0

dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  collate_fn: nemo_automodel.components.datasets.utils.default_collater
  shuffle: true

validation_dataset:
  _target_: nemo_automodel.components.datasets.llm.chat_dataset.ChatDataset
  path_or_dataset_id: ${TOP40_12B_DATA_DIR}/validation
  split: "train[:1024]"
  shuffle_seed: 42
  seq_length: 4096
  truncation: true

validation_dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  collate_fn: nemo_automodel.components.datasets.utils.default_collater

optimizer:
  _target_: torch.optim.AdamW
  lr: 1.0e-4
  weight_decay: 0.01
  betas: [0.9, 0.95]
  eps: 1.0e-8

lr_scheduler:
  lr_decay_style: cosine
  lr_warmup_steps: 500
  min_lr: 1.0e-5

clip_grad_norm:
  max_norm: 1.0

wandb:
  enable: false
  project: ${WANDB_PROJECT}
  entity: ${WANDB_ENTITY}
  name: gemma4-12b-top40-gold-lora-r16-10k-v1
  group: gemma4-12b-top40-gold-lora
  mode: ${WANDB_MODE,online}
  dir: ${WANDB_DIR,./wandb}
```

The 12B repository example uses eager attention. SDPA above is a proposed
memory-efficient starting point and must be covered by the qualification. If it
fails model parity or execution, revert to `attn_implementation: eager` and
record a new run version.

## Validate data and recipe

```bash
set -a
source .env
set +a

export TOP40_12B_DATA_DIR="$PWD/examples/vlm_finetune/gemma4/data/top40_gold_v1/gemma4_12b_chat"

.venv/bin/python - <<'PY'
import torch
from transformers import AutoConfig, AutoTokenizer
from nemo_automodel.components.datasets.llm.chat_dataset import ChatDataset

model_id = "google/gemma-4-12B-it"
config = AutoConfig.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
assert config.model_type == "gemma4_unified"
assert tokenizer.chat_template
dataset = ChatDataset(
    "examples/vlm_finetune/gemma4/data/top40_gold_v1/gemma4_12b_chat/train",
    tokenizer,
    seq_length=4096,
    truncation=True,
)
sample = dataset[0]
assert len(sample["input_ids"]) <= 4096
assert any(value != -100 for value in sample["labels"])
assert torch.cuda.is_available()
print(type(config).__name__, len(dataset), len(sample["input_ids"]))
PY

.venv/bin/python tools/lint_example_yamls.py \
  examples/llm_finetune/gemma/gemma_4_12b_top40_gold_lora.yaml
```

## Qualification gate

Use a fresh run identity. Do not reuse the E2B adapter or optimizer state; the
12B architecture and scheduler state are different.

```bash
export TOP40_12B_CHECKPOINT_DIR="$PWD/examples/vlm_finetune/gemma4/data/top40_gold_v1/runs/gemma4-12b-qual20-v1/checkpoints"
export RUN_NAME=gemma4-12b-top40-gold-lora-r16-qual20-v1
export NPROC_PER_NODE=2  # use 1 only on a proven 80 GB target

.venv/bin/automodel \
  examples/llm_finetune/gemma/gemma_4_12b_top40_gold_lora.yaml \
  --nproc-per-node "$NPROC_PER_NODE" \
  --step_scheduler.max_steps 20 \
  --step_scheduler.ckpt_every_steps 10 \
  --step_scheduler.val_every_steps 10 \
  --lr_scheduler.lr_warmup_steps 2 \
  --wandb.enable true \
  --wandb.name "$RUN_NAME"
```

The gate passes only when:

- model/tokenizer loading completes;
- every rank advances through 20 optimizer steps;
- training loss, validation loss, and gradient norms are finite;
- peak memory leaves operational headroom;
- checkpoints contain adapter, optimizer, RNG, scheduler, and dataloader state;
- W&B receives the resolved configuration and metrics.

Resume the same checkpoint root to step 25:

```bash
.venv/bin/automodel \
  examples/llm_finetune/gemma/gemma_4_12b_top40_gold_lora.yaml \
  --nproc-per-node "$NPROC_PER_NODE" \
  --checkpoint.restore_from LATEST \
  --step_scheduler.max_steps 25 \
  --step_scheduler.ckpt_every_steps 5 \
  --step_scheduler.val_every_steps 5 \
  --lr_scheduler.lr_warmup_steps 2 \
  --wandb.enable true \
  --wandb.name "${RUN_NAME}-resume25"
```

Verify the resumed log begins at 20/25 and does not restart from zero.

## Long 12B LoRA run

After qualification, choose a new checkpoint directory and run name:

```bash
export TOP40_12B_CHECKPOINT_DIR="$PWD/examples/vlm_finetune/gemma4/data/top40_gold_v1/runs/gemma4-12b-top40-gold-lora-r16-10k-v1/checkpoints"

.venv/bin/automodel \
  examples/llm_finetune/gemma/gemma_4_12b_top40_gold_lora.yaml \
  --nproc-per-node "$NPROC_PER_NODE" \
  --wandb.enable true \
  --wandb.name gemma4-12b-top40-gold-lora-r16-10k-v1
```

Do not resume the qualification checkpoint into the long run: its scheduler is
configured for 20 steps. Run complete validation at major checkpoints and at
the end, not only the 1,024-record in-training monitor.

## Evidence to retain

For each qualification and long run preserve:

- the recipe YAML and its SHA-256;
- `git rev-parse HEAD` and `git status --short`;
- `uv` environment and GPU/driver versions;
- data readiness and Parquet hashes;
- W&B entity, project, group, run name, run ID, and URL;
- `training.jsonl`, `validation.jsonl`, and full console log;
- resolved checkpoint `config.yaml`;
- adapter, optimizer, RNG, scheduler, and dataloader state;
- peak GPU memory, throughput, train loss, validation loss, and gradient norms;
- the step-25 resume proof.

## Known boundaries

- The current E2B run proves the data and general AutoModel LoRA workflow, not
  12B memory fit or numerical quality.
- The official repository example proves dense 12B model support on HellaSwag,
  not this `-it` chat-LoRA configuration.
- The 12B tokenizer may produce different sequence lengths; reprofile and
  reject overlong rows after the Parquet export.
- `swa`/`swh` alias normalization and `unknown` language labels remain data
  analysis limitations.
- Exact FLORES+ contamination coverage and licensing remain unresolved gates.
- Do not promote to 12B full SFT until the long 12B LoRA run demonstrates
  measurable held-out gains.
