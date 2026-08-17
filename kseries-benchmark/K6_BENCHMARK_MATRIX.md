# K6 benchmark matrix

Status: finalized for execution on 2026-08-13.

This plan evaluates the base `google/gemma-4-E2B-it` model and the K6
`epoch_0_step_999` adapter under identical runtime settings. K10 must later use
the same matrix for a controlled comparison.

## Execution tiers

### Tier 1: author-aligned core

Run the complete official test sets for every cell, zero-shot, with all five
author prompt variants. Report each prompt separately plus its mean and
variability. Do not select the best prompt.

| Language | Role | AfriMMLU | AfriXNLI | AfriMGSM CoT | Belebele |
|---|---|---:|---:|---:|---:|
| English (`eng`) | General-capability control | Yes | Yes | Yes | Yes |
| Hausa (`hau`) | K6 trained | Yes | Yes | Yes | Yes |
| Igbo (`ibo`) | K6 trained | Yes | Yes | Yes | Yes |
| Kinyarwanda (`kin`) | K6 trained; QA transfer cell | Yes | Yes | Yes | Yes |
| Swahili (`swa`) | K6 trained | Yes | Yes | Yes | Yes |
| isiXhosa (`xho`) | Unseen-language and future K10 control | Yes | Yes | Yes | Yes |
| Yoruba (`yor`) | K6 trained | Yes | Yes | Yes | Yes |
| isiZulu (`zul`) | K6 trained | Yes | Yes | Yes | Yes |

This is `8 languages x 4 benchmarks x 5 prompts = 160` task configurations
per model. The base-plus-K6 phase therefore contains 320 task configurations.

### Tier 2: K6 task-aligned longitudinal control

Run the complete 4,946-record AfriInstruct paper artifact after restoring and
hash-verifying its seven JSON files. This tier covers all six K6 languages.

| Language | AfriQA | NTREX | SIB-200 |
|---|---:|---:|---:|
| Hausa (`hau`) | QA token F1 | Translation ChrF++ | Topic accuracy and macro F1 |
| Igbo (`ibo`) | QA token F1 | Translation ChrF++ | Topic accuracy and macro F1 |
| Kinyarwanda (`kin`) | QA token F1 | Translation ChrF++ | Topic accuracy and macro F1 |
| Swahili (`swa`) | QA token F1 | Translation ChrF++ | Topic accuracy and macro F1 |
| Yoruba (`yor`) | QA token F1 | Translation ChrF++ | Topic accuracy and macro F1 |
| isiZulu (`zul`) | QA token F1 | Translation ChrF++ | Topic accuracy and macro F1 |

Kinyarwanda QA is a transfer evaluation because the K6 training mixture left
that language-task cell empty.

### Tier 2b: MasakhaNER 2.0

Run the complete official test split for all six K6 languages. K6 preparation
used MasakhaNER-derived training records, so the official test split is the
correct held-out evaluation surface; it must not be replaced with train or
validation records.

| Language | Official test sentences |
|---|---:|
| Hausa (`hau`) | 1,633 |
| Igbo (`ibo`) | 2,181 |
| Kinyarwanda (`kin`) | 2,235 |
| Swahili (`swa`) | 1,883 |
| Yoruba (`yor`) | 1,964 |
| isiZulu (`zul`) | 1,670 |

Use source revision `ba5843cd08aa491d5f96a5e809e71eb9ec461391`. Prompt the
model with the same JSON `type`/`text` entity representation used by K6
preparation. Report strict entity-level precision, recall, and micro F1, plus
per-entity-type F1, malformed-output count, and hallucinated-span count. Before
result-producing runs, compare the official test records against the actual
materialized K6 training manifest for exact and fragment overlap.

## Final capability coverage

| Capability | Benchmark evidence | Languages | Status |
|---|---|---|---|
| Knowledge QA | AfriMMLU | Core eight | Ready |
| Natural-language inference | AfriXNLI | Core eight | Ready |
| Mathematical reasoning | AfriMGSM CoT | Core eight | Ready |
| Reading comprehension | Belebele | Core eight | Ready |
| Open QA | AfriQA longitudinal control | Six K6 | Ready; source hash verified |
| Machine translation | NTREX longitudinal control | Six K6 | Ready; source hash verified |
| Topic classification | SIB-200 longitudinal control | Six K6 | Ready; source hash verified |
| General instruction following | None | None | Explicit gap |
| Named entity recognition | MasakhaNER 2.0 official test splits | Six K6 | Evaluator and overlap audit required |

## Deferred supplements

These are not blockers for the base-plus-K6 run and must not be added without a
new pinned evaluator, contamination audit, and recorded source revision.

| Candidate | Decision | Reason |
|---|---|---|
| IrokoBench translate-test | Defer | Diagnostic reuse of the same underlying questions, not independent evidence |
| IrokoBench few-shot | Defer | Measures in-context learning and changes the input-token budget |
| Injongo intent detection | Defer | Valuable classification coverage but no qualified repository harness |
| Injongo slot filling | Defer | Useful domain-specific extraction supplement, but MasakhaNER is the primary NER benchmark |
| AfriDocMT test | Defer | K6 used AfriDocMT train data; official split and harness still require qualification |
| FLORES MT | Exclude pending audit | FLORES appears in an upstream K6 training source; exact test contamination is unresolved |

NTREX is the only finalized machine-translation benchmark for the immediate K6
campaign. Any additional isiZulu MT benchmarks must be run as complete,
separate test sets after the deferred checks; their examples must not be mixed
with NTREX into a custom test set.

## Reporting policy

1. Preserve raw outputs and provenance files unchanged.
2. Report base, K6, and K6-minus-base for every benchmark-language-prompt cell.
3. Keep translation directions separate.
4. Report per-benchmark results before any macro summary.
5. Give each benchmark equal weight within a language-task macro; do not weight
   by record count.
6. Do not average unlike metrics into one score.
7. Use paired confidence intervals or significance tests on identical examples.
8. Record missing outputs, token-limit hits, empty outputs, repeated generation,
   wall time, and peak GPU memory.

## Readiness

- K6 checkpoint path: `kseries-benchmark/checkpoints/k6/epoch_0_step_999`
- The author runner dry-run validates 160 task configurations and the complete
  eight-language list.
- All seven AfriInstruct benchmark JSON files are present under
  `kseries-benchmark/benchmarks/afriinstruct-paper` and match the handoff
  SHA-256 hashes.
- The MasakhaNER test files and generative evaluator are not present locally
  yet; the actual K6 training manifest is also needed for the final overlap
  audit.
- Host `uv` and `pytest` are unavailable; focused tests must run in the pinned
  benchmark container.
- The current checkout is `7db2a4553f5c53d34c9bc9faa4ac9c4847f07a3e`,
  not the handoff commit. The matrix expansion is an intentional source change
  and needs a recorded commit before result-producing runs.