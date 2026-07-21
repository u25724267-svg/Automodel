# Gemma 4 E2B AfriInstruct LoRA full benchmark report

## Executive summary

This report compares the unchanged `google/gemma-4-E2B-it` model with the
step-999 LoRA adapter produced by the full-data 1,000-step AfriInstruct run.
Both models generated deterministic responses for the same 4,946 unique prompts
drawn from the benchmark files attached to the AfriInstruct Findings of EMNLP
2024 paper.

The adapter improved every primary aggregate metric:

| Metric | Base | Adapter | Absolute change |
|---|---:|---:|---:|
| QA normalized token F1 | 7.08 | 39.86 | +32.78 |
| Translation corpus ChrF++ | 25.44 | 30.54 | +5.10 |
| Topic accuracy | 46.40% | 71.85% | +25.44 points |
| Topic macro F1 | 50.52 | 70.08 | +19.55 |

Paired bootstrap confidence intervals excluded zero for QA, translation, and
topic accuracy. All six focal languages improved on all three task families.

The adapter also generated substantially shorter, more direct responses:

- EOS rate increased from 47.43% to 89.99%.
- Token-limit rate decreased from 52.83% to 10.01%.
- Mean generated tokens decreased from 95.05 to 30.11.
- Mean per-prompt latency decreased from 3.71 seconds to 1.32 seconds.

The latency reduction primarily reflects earlier termination, not a faster
model kernel.

The adapter is clearly better than base Gemma on this source-disjoint African
QA, translation, and classification suite. It is not failure-free. Some
translations still enter repetition loops, Yoruba QA remains weak, and 495
adapter responses hit the 128-token limit. General-capability retention was not
measured by this benchmark.

## Benchmark identity

The benchmark source is the software artifact attached to:

> Kosei Uemura, Mahe Chen, Alex Pejovic, Chika Maduabuchi, Yifei Sun, and
> En-Shiun Annie Lee. 2024. AfriInstruct: Instruction Tuning of African
> Languages for Diverse Tasks. Findings of EMNLP 2024.

Paper: <https://aclanthology.org/2024.findings-emnlp.793/>

Software artifact:
<https://aclanthology.org/attachments/2024.findings-emnlp.793.software.zip>

The source files were copied without modification into:

```text
/data/afriinstruct/benchmarks/afriinstruct-paper/
```

The benchmark comprises:

| File group | Raw records |
|---|---:|
| General mixed-language benchmark | 1,000 |
| Hausa | 806 |
| Igbo | 806 |
| Kinyarwanda | 806 |
| Swahili | 602 |
| Yoruba | 602 |
| Zulu | 602 |

Records duplicated exactly across the general and language-specific files were
removed by stable content hash. One additional duplicate exists across language
files. The authoritative suite therefore contains 4,946 unique records, not
4,947. Existing prediction filenames retain `4947` for continuity, but all
scoring and completion checks use 4,946.

Task composition after exact deduplication:

| Task | Records | Share |
|---|---:|---:|
| QA | 1,326 | 26.81% |
| Translation | 2,661 | 53.80% |
| Topic classification | 959 | 19.39% |
| Total | 4,946 | 100% |

The source datasets are AfriQA, NTREX, and SIB-200. These source names were not
present in the AfriInstruct training mixture used by this project, so the suite
is source-disjoint from training. Near-duplicate semantic content was not
audited exhaustively.

## Models

### Base

```text
google/gemma-4-E2B-it
```

### Adapter

```text
Base: google/gemma-4-E2B-it
Checkpoint: /checkpoints/gemma4-e2b-afriinstruct/lora-full-1k-v1/epoch_0_step_999
Adapter: model/adapter_model.safetensors
LoRA rank: 16
LoRA alpha: 32
```

The adapter loader restored the AutoModel PEFT configuration and checkpoint
through the existing VLM generation/checkpoint path. Base and adapter were
loaded and evaluated sequentially on the same NVIDIA GB10.

## Generation protocol

The benchmark harness is:

```text
tools/afriinstruct_benchmark.py
```

Generation settings were identical for base and adapter:

| Setting | Value |
|---|---|
| Prompt format | Native Gemma 4 chat template |
| Semantic prefix | Paper-compatible African-language proficiency prefix |
| Sampling | Disabled |
| Decoding | Greedy |
| Maximum new tokens | 128 |
| Attention backend | SDPA |
| Batch size | 1 |
| Image inputs | None, text-only |

The semantic prefix was:

```text
You are very proficient in African languages, and you are very good at responding in those languages.
```

The benchmark instruction was appended after the prefix and passed as one user
turn through Gemma's native chat template.

SDPA was required because Gemma 4 E2B uses attention head dimension 512, while
the container's FlashAttention path supports at most 256 for this model. The
same SDPA backend was used during qualified training.

Each result was appended and flushed immediately to JSONL. A stable SHA-256 ID
over instruction, reference, language, source, and task made generation
resumable without repeating completed prompts.

Each prediction record contains:

- Stable record ID
- Instruction, reference, language, source, and task
- Model label and checkpoint path
- Prediction
- Input and generated token counts
- EOS reached flag
- Token-limit flag
- Per-prompt generation latency

## Metrics

### QA

QA uses normalized token F1 in the range 0-100:

- Unicode-preserving case folding
- Punctuation removal
- English article removal
- Whitespace normalization
- Multiset token overlap
- Best score across accepted answer alternatives

One paper benchmark record contains an empty answer list (`[]`). It is scored
as zero, matching the paper evaluator's no-candidate behavior.

### Translation

Translation uses corpus ChrF++ from `sacrebleu 2.6.0` with `word_order=2`.

The report also computes paired sentence-level ChrF++ differences for
bootstrap confidence intervals. The paired mean sentence delta and corpus-level
delta differ because corpus ChrF++ aggregates statistics globally.

### Topic classification

Generated text is normalized and searched for one of the seven canonical
labels:

```text
science/technology, travel, politics, sports, health, entertainment, geography
```

The report includes exact normalized accuracy and macro F1.

### Difference from the paper scorer

The paper software uses set-based QA overlap, fuzzy partial matching for topic
labels, and a deprecated metric loader. This harness uses standardized
multiset QA F1, exact canonical topic extraction, and sacreBLEU ChrF++.
Consequently, the values in this report should not be compared numerically to
the paper's tables as if they came from an identical scorer.

## Staged validation

### Seven-prompt smoke

The first gate used `benchmark_sample.json`.

Results established that:

- Base and adapter checkpoints load under SDPA.
- Text-only Gemma chat generation works.
- Prediction JSONL is resumable.
- Translation scoring works.
- Adapter generation reaches EOS more reliably.

On the seven prompts, adapter ChrF++ improved from 47.78 to 58.47. Adapter EOS
was 7/7 versus 3/7 for base. Both models failed two context-inconsistent QA
examples in the smoke file.

### General 1,000-prompt pilot

| Metric | Base | Adapter | Delta |
|---|---:|---:|---:|
| QA token F1 | 5.65 | 28.21 | +22.56 |
| Translation ChrF++ | 24.31 | 29.74 | +5.43 |
| Topic accuracy | 51.50% | 79.40% | +27.90 points |
| Topic macro F1 | 54.80 | 77.43 | +22.63 |

The pilot showed statistically and practically meaningful gains and justified
the full language-specific extension.

## Full aggregate results

| Metric | Base | Adapter | Absolute delta | Relative change |
|---|---:|---:|---:|---:|
| QA token F1 | 7.08 | 39.86 | +32.78 | +463.0% |
| Translation corpus ChrF++ | 25.44 | 30.54 | +5.10 | +20.0% |
| Topic accuracy | 46.40% | 71.85% | +25.44 points | +54.8% |
| Topic macro F1 | 50.52 | 70.08 | +19.55 | +38.7% |

No benchmark record was missing from either prediction cache.

## Paired statistical comparison

Paired bootstrap resampling used predictions from the exact same prompts.

| Metric | Mean paired delta | 95% confidence interval | Adapter wins | Base wins | Ties |
|---|---:|---:|---:|---:|---:|
| QA token F1 | +32.78 | [30.52, 35.01] | 597 | 224 | 505 |
| Translation sentence ChrF++ | +6.49 | [6.00, 6.99] | 1,916 | 744 | 1 |
| Topic accuracy | +25.44 points | [22.00, 28.78] | 290 | 46 | 623 |

All intervals exclude zero. The improvement is not explained by a small number
of outliers.

## Focal-language results

The table reports adapter minus base absolute change.

| Language | Records | QA F1 delta | Translation ChrF++ delta | Topic accuracy delta |
|---|---:|---:|---:|---:|
| Hausa | 826 | +37.10 | +2.93 | +10.46 points |
| Igbo | 834 | +53.30 | +6.82 | +40.28 points |
| Kinyarwanda | 827 | +31.49 | +0.36 | +28.75 points |
| Swahili | 637 | +17.25 | +12.68 | +24.53 points |
| Yoruba | 629 | +4.38 | +3.41 | +30.58 points |
| Zulu | 635 | +45.94 | +7.66 | +21.19 points |

All 18 focal-language/task deltas are positive. Gains are uneven:

- Igbo and Zulu show the largest QA improvements.
- Swahili shows the largest translation improvement.
- Kinyarwanda translation improves only slightly.
- Yoruba QA remains low in absolute terms despite a positive delta.

Absolute adapter scores for focal slices are preserved in
`full-comparison.json`.

## Generation quality and efficiency

| Metric | Base | Adapter | Change |
|---|---:|---:|---:|
| EOS count | 2,346 | 4,451 | +2,105 |
| EOS rate | 47.43% | 89.99% | +42.56 points |
| Token-limit count | 2,613 | 495 | -2,118 |
| Token-limit rate | 52.83% | 10.01% | -42.82 points |
| Empty responses | 0 | 0 | No change |
| Mean generated tokens | 95.05 | 30.11 | -64.94 |
| Median generated tokens | 128 | 13 | -115 |
| P90 generated tokens | 128 | 127 | -1 |
| Mean latency | 3.706 s | 1.319 s | -64.4% |
| Median latency | 4.926 s | 0.566 s | -88.5% |
| Total generation time | 5.092 h | 1.812 h | -3.280 h |

The adapter is more task-direct and terminates much more often. This is a
behavioral efficiency gain, not evidence that its matrix operations are
intrinsically faster.

P90 output length remains near the 128-token cap because a minority of adapter
responses still overgenerate or repeat.

## Qualitative findings

### Improvements

The adapter frequently changed long refusals or meta-explanations into concise
answers grounded in the supplied context.

Examples include:

- Igbo QA: returned `2002` where base claimed the context was insufficient.
- Kinyarwanda QA: returned `Lee Hsien Loong` where base answered the wrong
  office.
- Igbo QA: returned `1502` for Saint Helena where base rejected the premise.
- Swahili translation: produced `Mtu mwingine mweusi?!` exactly, while base
  generated a long explanation and an incorrect form.
- Afrikaans translation: produced the exact reference where base supplied
  several alternatives and commentary.

### Regressions

The adapter is not uniformly better at the example level.

Observed failures include:

- Some QA responses became too short and selected a nearby entity, date, or
  country instead of the requested answer.
- Hausa, Kinyarwanda, and Yoruba translations sometimes entered repeated-word
  loops and hit the 128-token cap.
- One Hausa translation reproduced the benchmark proficiency prefix instead of
  translating the source.
- Some base responses contained the correct answer in a longer sentence while
  the adapter emitted an incorrect short span.

The largest improvements and regressions are preserved in
`full-examples.json` for human review.

## Interpretation

The benchmark provides strong evidence that AfriInstruct LoRA training improved
Gemma 4 E2B's behavior on source-disjoint African-language tasks:

- QA extraction improved substantially.
- Translation quality improved overall and in every focal language.
- Topic-label following improved substantially.
- The model became more concise and much more likely to terminate.
- Improvements are statistically robust under paired resampling.

The results align with the training run's monotonically improving validation
loss, but they are more decision-relevant than loss alone because they measure
generated task outputs.

The adapter should be retained as the primary African-task candidate. It should
not yet replace base Gemma universally because this evaluation does not measure
general knowledge, English reasoning, safety, or broad instruction retention.

## Limitations

1. The paper's exact scorer was not reproduced; standardized metrics were used.
2. Gemma's native chat template differs from the paper model's Alpaca template.
3. The fixed paper proficiency prefix may influence verbosity and task
   behavior.
4. Generation was greedy and capped at 128 new tokens.
5. Evaluation was sequential with batch size one.
6. One QA record has no accepted reference and is scored zero.
7. Exact source separation is established by dataset name, but semantic
   near-duplicate auditing was not performed.
8. The published AfriInstruct model was not run as a third reference model.
9. General-capability and safety benchmarks were not run.
10. Confidence intervals cover paired benchmark sampling, not model-training
    seed variance; only one training seed was evaluated.

## Reproducibility

### Source revision

```text
6c5290ddd0d1585903c6610f0712ba8254f26ba6
```

The worktree also contains the qualification-driven framework fixes documented
in `AFRIINSTRUCT_LORA_FULL_1K_RUN_REPORT.md` and the uncommitted benchmark
harness.

### Scoring dependencies

```text
sacrebleu==2.6.0
lxml<7
```

The `lxml` constraint prevents the repository's global prerelease policy from
selecting an alpha release through sacreBLEU's transitive dependency.

### Generate command pattern

```bash
python -m tools.afriinstruct_benchmark generate \
  --benchmark-file /data/afriinstruct/benchmarks/afriinstruct-paper/benchmark.json \
  --benchmark-file /data/afriinstruct/benchmarks/afriinstruct-paper/benchmark_hau_806.json \
  --benchmark-file /data/afriinstruct/benchmarks/afriinstruct-paper/benchmark_ibo_806.json \
  --benchmark-file /data/afriinstruct/benchmarks/afriinstruct-paper/benchmark_kin_806.json \
  --benchmark-file /data/afriinstruct/benchmarks/afriinstruct-paper/benchmark_swa_602.json \
  --benchmark-file /data/afriinstruct/benchmarks/afriinstruct-paper/benchmark_yor_602.json \
  --benchmark-file /data/afriinstruct/benchmarks/afriinstruct-paper/benchmark_zul_602.json \
  --base-model google/gemma-4-E2B-it \
  --model-label MODEL_LABEL \
  --output OUTPUT.jsonl \
  --max-new-tokens 128
```

For adapter generation, add:

```text
--checkpoint-path /checkpoints/gemma4-e2b-afriinstruct/lora-full-1k-v1/epoch_0_step_999
```

### Score command pattern

Use the same ordered `--benchmark-file` arguments:

```bash
python -m tools.afriinstruct_benchmark score \
  --benchmark-file ... \
  --predictions OUTPUT.jsonl \
  --output METRICS.json
```

### Cache behavior

The generator loads existing JSONL records by stable ID, skips completed
records, and flushes every new prediction immediately. Connection loss does not
invalidate the cache. The completed base and adapter caches each contain 4,946
unique valid JSON records.

## Artifact inventory and hashes

Result root:

```text
/data/afriinstruct/benchmark-results/afriinstruct-paper/
```

| Artifact | SHA-256 |
|---|---|
| `base-full-4947.jsonl` | `5884f9af7307672e65a26c1337d94401eeec88c55835f71283d642b906335749` |
| `adapter-step999-full-4947.jsonl` | `1dc962b96dc163670e2570c5406c61a911b237e134533d26bb4fe416e230554e` |
| `base-full-metrics.json` | `ba2b965d9edf6e21dd1567a1c291f2d117f0a00e0beb490ddb8d868433cffda1` |
| `adapter-step999-full-metrics.json` | `a9e150cdcec7da8efeae1183a7ca9de9fde3c1938baa26c67296067edd137ee6` |
| `full-comparison.json` | `6b618e598c0bcfa86d42667be61ff831a66dcc92da977cc0e30aa072eac698df` |

Source benchmark hashes are available from the persistent benchmark directory
and should be recorded with any copied result bundle.

## Validation evidence

The benchmark harness unit tests cover:

- Stable deduplication IDs
- Paper-prefix prompt construction
- Standard normalized QA token F1
- Empty QA reference handling
- Topic extraction and macro F1
- SDPA configuration
- ChrF++ translation scoring

The harness was validated with Ruff and focused pytest runs before GPU
generation. Final clean-environment validation is part of the repository change
verification and uses the declared test dependencies above.

## Recommended next steps

1. Review `full-examples.json`, especially all adapter token-limit outputs.
2. Add automated repetition-loop and language-ID diagnostics.
3. Evaluate base and adapter on general capability retention, including English
   instruction following, reasoning, and MMLU-like tasks.
4. Evaluate safety and refusal behavior separately.
5. Run the published AfriInstruct model through the same standardized harness
   if a direct reference is required.
6. Use task/language-balanced sampling, rank 32 LoRA, and a lower learning rate
   for the proposed v2 training experiment.
7. Preserve the current adapter as the baseline for every v2 comparison.
8. Do not infer universal model superiority from this African-task suite alone.

## Final assessment

Checkpoint `epoch_0_step_999` materially and consistently improves Gemma 4 E2B
on the AfriInstruct paper's source-disjoint task suite. The aggregate and paired
results justify using the adapter for further African-language evaluation and
as the baseline for improved training experiments.

The primary residual quality issue is a 10.01% token-limit rate with visible
repetition failures in a subset of translations. The next engineering effort
should target generation robustness and general-capability retention while
preserving the large QA, translation, and topic-classification gains measured
here.