# K6 multisource SFT preprocessing

This workflow profiles the initial Hausa, Igbo, Kinyarwanda, Swahili, Yoruba,
and isiZulu mixture before any training data is materialized. The target tasks
are instruction following, question answering, translation, classification,
and named entity recognition.

## Allocation rationale

The planner does not assign equal packed-token budgets to tasks.

- T5 found equal task mixing substantially worse than capped
  examples-proportional sampling and found temperature `2` strongest for most
  tasks.
- The FLAN Collection found task balancing and prompt-format diversity critical
  to instruction tuning.
- MzansiLM found that a general multitask mixture diluted task-specific signal,
  particularly for sequence labelling and generation.
- UniMax motivates the four-epoch capacity limit and redistribution of genuine
  shortfalls rather than unrestricted repetition.

The primary policy samples tasks according to:

```text
p(task) proportional to min(unique_examples, task_cap) ** (1 / temperature)
```

Packed input tokens measure compute, supervised assistant tokens measure loss
exposure, and unique records measure task exposure. All three are reported.

## Input contract

Prepare one manifest per source pool. Each JSONL record must contain:

```json
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "lang": "yor",
  "task": "translation",
  "source": "afridocmt",
  "direction": "eng-yor"
}
```

The optional `direction` and `entity_types` fields enrich translation and NER
coverage reports. Dataset-specific adapters must normalize their upstream task
and language names to the values in the profile configuration.

Prepare the independent pools with `tools/prepare_k6_sft_sources.py`. To reuse
an existing AfriInstruct preparation as the fallback pool, canonicalize it
without downloading or duplicating unrelated languages:

```bash
uv run python tools/prepare_k6_sft_sources.py \
  --output-root /data/gemma4-k6/pools \
  --sources afriinstruct \
  --afriinstruct-manifest /data/afriinstruct/e2b-full-v1/train_meta.json
```

The adapter maps `swa` to `swh`; maps explicit QA, translation,
classification, and NER labels to their K6 task; maps `Multitask` and
summarization to instruction following; and excludes POS and unknown tasks.
It does not infer QA or other task labels from generic xP3 prompts.

Copy
`examples/vlm_finetune/gemma4/data/k6_sft_profile.example.yaml` to a run-specific
configuration outside the repository. Pin every source revision and update the
manifest paths. Do not resolve missing QA or NER cells by adding an evaluation
dataset.

## Profile

```bash
uv run python tools/profile_sft_mixture.py profile \
  --config /data/gemma4-k6/k6_sft_profile.yaml \
  --benchmark-blocklist /data/gemma4-k6/benchmarks.sqlite3 \
  --output-dir /data/gemma4-k6/profile-v1
```

Review these outputs:

- `profile.md`: language-task capacities and blocking issues.
- `cells.csv`: source-level records, prompt tokens, label tokens, and packed
  tokens.
- `profile.json`: exact machine-readable statistics, length histograms, class
  labels, translation directions, and entity types.
- `review_samples.jsonl`: deterministic examples for human inspection.

The profiler rejects malformed records, out-of-scope languages and tasks,
benchmark matches, exact cross-pool duplicates, records without supervised
assistant tokens, and records longer than the configured sequence length.

## Simulate allocations

```bash
uv run python tools/profile_sft_mixture.py plan \
  --config /data/gemma4-k6/k6_sft_profile.yaml \
  --profile /data/gemma4-k6/profile-v1/profile.json \
  --output-dir /data/gemma4-k6/plans-v1
```

The planner emits:

- `p0_equal_packed_tokens`: control policy only.
- `p1_capped_examples_t2`: T5-style capped example sampling with temperature
  smoothing.
- `p2_quality_constrained`: the primary candidate, adding quality weighting,
  a 50% default source-family cap, and a soft AfriInstruct fallback. P2 uses a
  25% AfriInstruct anchor when independent capacity is sufficient, increases
  that share to fill measured shortfalls, and can use 100% AfriInstruct for a
  cell with no independent source.

## Materialize P2

After accepting the review findings, materialize the selected records with the
same tokenizer, benchmark blocklist, quality ordering, and maximum-epoch policy
used by profiling:

```bash
uv run python tools/build_k6_sft_mixture.py \
  --config /data/gemma4-k6/k6_sft_profile.yaml \
  --plan /data/gemma4-k6/plans-v4/plans.json \
  --policy p2_quality_constrained \
  --benchmark-blocklist /data/gemma4-k6/benchmarks.sqlite3 \
  --output-dir /data/gemma4-k6/mixture-p2-v2 \
  --validation-records-per-pool 2500
```

The builder rounds pool allocations by largest remainder within each
language-task cell, selects records by stable digest order, repeats records only
up to the configured `max_epochs`, and balances AfriHate labels subject to that
same repetition cap. Training is selected first. Validation then uses only
unused unique records, so small pools consumed at full capacity can contribute
zero validation records rather than leak training examples.

## Train P2

The rank-32 LoRA recipe runs 1,000 optimizer steps, which closely matches one
pass over the materialized 31.35M-token mixture. It validates and checkpoints
every 100 steps.

```bash
export KSERIES_DATA_DIR=/data/gemma4-k6/mixture-p2-v2
export KSERIES_CHECKPOINT_DIR=/checkpoints/gemma4-e2b-k6/p2-r32-1k-v1
export KSERIES_WANDB_DIR=/logs/wandb
export KSERIES_WANDB_NAME=gemma4-e2b-k6-p2-r32-1k-v1
export KSERIES_WANDB_GROUP=gemma4-e2b-k6-p2-r32

uv run automodel \
  examples/vlm_finetune/gemma4/gemma4_e2b_kseries_p2_peft.yaml \
  --nproc-per-node 1 \
  --wandb.enable true
```

## Review gate

Do not materialize a training mixture until:

1. Every unexpected empty language-task cell is resolved or explicitly accepted.
2. Every source has an immutable revision and resolved license.
3. Benchmark contamination is zero after filtering.
4. Cross-source duplicate rates and sequence-length rejections are understood.
5. Classification labels, translation directions, and NER entity coverage are
   plausible for every represented language.
6. Human review confirms language and response quality in each populated cell.
7. P0, P1, and P2 differences are documented before choosing a materialization
   policy.

Hausa, Igbo, Kinyarwanda, Yoruba, and isiZulu lack dedicated independent QA
corpora in the primary registry. Swahili QA is covered by KenSwQuAD. Human Aya
prompt-completion records provide provisional open-ended QA candidates for all
K6 languages except Kinyarwanda, but Aya has no task labels. An adapter must
classify each Aya record into exactly one task before profiling so the same
record cannot be counted as both general instruction and QA.

## QA source decisions

KenSwQuAD is included as the independent Swahili QA source. It contains 7,506
extractive QA pairs from 1,441 Swahili contexts. Native speakers manually wrote
the questions and answers, a 12.5% quality-assurance sample was reviewed, and
the release is CC-BY-4.0 at revision
`a7f211970d5929c438c7cf3e971d8d34a7292d6e`.

Other discovered QA resources are not primary K6 training pools:

- TyDi-QA is high-quality native Swahili QA, but xP3 already contains TyDi-QA
  training prompts. It therefore overlaps the AfriInstruct anchor and must not
  be counted as an independent source.
- HausaVQA and Afri-MCQA are image-grounded multimodal QA datasets. Adding them
  would change the text-only experiment and Afri-MCQA exposes only benchmark
  development and test splits.
- MultiWikiQA, AfriQA, Uhura, BLEnD, and Belebele remain evaluation sources.
- `vaghawan/hausa-qa-1k` lacks documented human QA creation and language
  validation beyond synthetic question audio.
- `Remithefirst/yoruba-academic-wikipedia-qa` lacks a declared license,
  generation method, and human validation.

A pinned Aya train-split audit found the following clearly interrogative
candidate counts. These are heuristic upper bounds, not accepted QA counts:

| Language | Aya train records | Interrogative candidates |
|---|---:|---:|
| Hausa | 3,512 | 3,117 |
| Igbo | 1,534 | 1,066 |
| Swahili | 366 | 214 |
| Yoruba | 11,758 | 1,144 |
| isiZulu | 1,833 | 1,583 |
| Kinyarwanda | 0 | 0 |

AfriQA and Uhura-ARC-Easy expose genuine train splits and may be used in a
separate benchmark-supervised ablation. They are excluded from the independent
source arm because AfriQA is a planned evaluation source and Uhura translates
ARC-Easy content already represented in xP3/AfriInstruct. Belebele, AfriMMLU,
NaijaRC test subsets, and Uhura-TruthfulQA remain evaluation-only.

## Real-data profiler smoke result

The profiler was exercised against the complete 8,708,033-record prepared
AfriInstruct training manifest while retaining only the Hausa NER cell. It
scanned the full manifest, rejected 8,702,408 out-of-scope records and one
cross-record duplicate, and accepted 5,624 MasakhaNER 2.0 records.

| Metric | Value |
|---|---:|
| Prompt tokens | 1,135,184 |
| Supervised label tokens | 1,094,584 |
| Packed tokens | 2,229,768 |
| Mean prompt tokens per record | 201.85 |
| Mean label tokens per record | 194.63 |
| Mean packed tokens per record | 396.47 |
| Maximum packed length | 978 |

This result demonstrates why task names alone cannot determine token budgets.
The generative NER representation emits a tagged sequence rather than a short
class label, so its supervised-token requirement resembles generation more than
single-label classification. Full K6 allocation decisions must use the measured
record format for each source.