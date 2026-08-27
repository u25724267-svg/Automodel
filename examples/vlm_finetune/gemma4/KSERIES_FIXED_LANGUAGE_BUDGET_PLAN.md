# K-series fixed per-language token-budget plan

Status: P4 planning, deterministic profiling, capacity review, exact
materialization, and exhaustive audits are complete for K10 and K14. See
[KSERIES_FIXED_LANGUAGE_DATA_REPRODUCTION.md](KSERIES_FIXED_LANGUAGE_DATA_REPRODUCTION.md)
for the saved wrapper, accepted artifacts, hashes, and full reproduction
procedure. No training run in this phase is approved for launch.

## Research question

The original staircase fixed total materialized compute near 31.35M tokens, so
per-language exposure fell as breadth increased. This phase fixes materialized
pre-shift processor tokens at exactly 5,000,000 per African language while
allowing total compute to scale with breadth. One epoch consumes every selected
record once. The VLM causal shift removes one model-input token per record; on
2026-08-27 the resulting 4.9175M-4.9694M per-language model-input range was
explicitly accepted without rematerialization.

| Arm | Languages | Materialized pre-shift target | Epochs |
|---|---:|---:|---:|
| K10 | 10 | 50,000,000 packed tokens | 1 |
| K14 | 14 | 70,000,000 packed tokens | 1 |

One epoch is required: two epochs would expose 10M tokens per language and
would not test the stated 5M consumed-token budget. Both arms start
independently from `google/gemma-4-E2B-it` revision
`3e22461f65e89153144f8adb70e3b8c2cc9845a7`.

## Proposed training control

Unless changed before materialization, both arms use the same r16 NVIDIA Gemma
4 E2B policy used in the completed K6/K10/K14 ablations:

- LoRA rank 16, alpha 32, dropout 0.0;
- AdamW peak LR `2e-4`, weight decay 0.01, betas 0.9/0.95;
- NVIDIA-derived cosine schedule over one epoch;
- global/local batch 8/1, sequence and pack length 4,096;
- BF16, SDPA, FSDP2 with TP/CP/PP 1/1/1;
- answer-only fused linear cross-entropy;
- validation and checkpoints every 200 optimizer steps;
- four persistent train workers and in-process validation (`num_workers: 0`);
- seed 42, ranked;
- W&B project `dsfsi/gemma4-african-instruction`.

The exact optimizer-step and warmup counts must be resolved from each final
packed mixture and recorded before launch.

## Task-share prior

The accepted K14 P3 prior is retained as a preregistered starting point, not as
a literature-proven optimum.

| Task | Share | Target per language |
|---|---:|---:|
| Instruction | 33% | 1,650,000 |
| QA | 15% | 750,000 |
| Translation | 10% | 500,000 |
| Classification | 17% | 850,000 |
| NER | 25% | 1,250,000 |

The minimum supervised-token ratio scales from the accepted 10.4M/31.353533M
gate (33.1701056%): at least 16,585,053 supervised tokens for K10 and
23,219,074 for K14.

## Current K10 composition

The accepted K10 P2 mixture is not language-balanced and is not suitable for
simple proportional expansion.

| Language | Current tokens | Gap to 5M |
|---|---:|---:|
| Amharic | 6,047,898 | -1,047,898 |
| Hausa | 2,984,093 | 2,015,907 |
| Igbo | 2,572,342 | 2,427,658 |
| Kinyarwanda | 2,790,252 | 2,209,748 |
| Shona | 3,454,080 | 1,545,920 |
| Swahili | 4,122,944 | 877,056 |
| Wolof | 1,036,618 | 3,963,382 |
| isiXhosa | 2,303,622 | 2,696,378 |
| Yoruba | 3,478,559 | 1,521,441 |
| isiZulu | 2,558,904 | 2,441,096 |

Current K10 task shares are instruction 34.18%, QA 11.09%, translation 7.82%,
classification 24.38%, and NER 22.54%. Supervised tokens are 30.34% of packed
tokens, below the proposed 33.17% minimum. Its largest source shares are Aya
20.77%, MasakhaNER2.0 17.59%, XL-Sum 17.25%, and MasakhaNEWS 13.04%.

The new K10 profile must use the unified K14-era source registry for the ten
represented languages. In particular, it must include the accepted Amharic NER
and Wolof classification sources that were absent from the old K10 registry.

## Current K14 composition

K14 P3 is already approximately language-balanced but has only about 2.0-2.39M
tokens per language, requiring 2.233x total growth.

| Language | Current tokens | Gap to 5M |
|---|---:|---:|
| Amharic | 2,163,836 | 2,836,164 |
| Hausa | 2,315,328 | 2,684,672 |
| Igbo | 2,348,008 | 2,651,992 |
| Kinyarwanda | 1,996,091 | 3,003,909 |
| Luganda | 2,293,922 | 2,706,078 |
| Shona | 2,261,214 | 2,738,786 |
| Somali | 2,272,502 | 2,727,498 |
| Sesotho | 2,391,856 | 2,608,144 |
| Swahili | 2,271,078 | 2,728,922 |
| Twi | 2,312,133 | 2,687,867 |
| Wolof | 2,043,843 | 2,956,157 |
| isiXhosa | 2,117,604 | 2,882,396 |
| Yoruba | 2,254,819 | 2,745,181 |
| isiZulu | 2,311,328 | 2,688,672 |

Current K14 task shares are instruction 32.86%, QA 15.45%, translation 9.99%,
classification 17.13%, and NER 24.57%. Supervised tokens are 40.41% of packed
tokens. Its largest source shares are Aya 19.56%, xP3 13.20%, MasakhaNER2.0
9.98%, and MasakhaNER 7.09%.

## Preliminary capacity risks

The accepted summaries expose filtered unique-source capacity. Multiplying it
by the four-appearance cap gives the following preliminary shortfalls against
the exact task targets. These values must be recomputed from transferred source
pools with exact profiling before planning.

| Language/task | Target | Four-appearance capacity | Shortfall |
|---|---:|---:|---:|
| Amharic QA | 750,000 | 392,368 | 357,632 |
| Kinyarwanda QA | 750,000 | 35,136 | 714,864 |
| Sesotho classification | 850,000 | 354,364 | 495,636 |
| Wolof instruction | 1,650,000 | 493,348 | 1,156,652 |
| Wolof QA | 750,000 | 621,732 | 128,268 |
| isiXhosa QA | 750,000 | 175,072 | 574,928 |
| Yoruba QA | 750,000 | 679,424 | 70,576 |
| isiZulu classification | 850,000 | 506,660 | 343,340 |
| Shona QA | 750,000 | 737,588 | 12,412 |
| Somali instruction | 1,650,000 | 1,627,480 | 22,520 |

The largest risks are Kinyarwanda/isiXhosa/Amharic QA, Wolof instruction, and
Sesotho/isiZulu classification. No plan may silently fill these gaps by
exceeding four appearances or increasing mixed/silver fallback.

## Required allocation behavior

The existing `p3_token_stratified` planner divides every task budget equally
across languages only when all cells have sufficient capacity. With the
shortfalls above, its global task-level waterfill moves missing tokens to other
languages, violating the fixed 5M/language contract.

The new policy must:

1. Set a hard 5,000,000 packed-token budget for every language.
2. Apply the task-share targets within each language.
3. Cap each language/task cell by benchmark-safe four-appearance capacity.
4. Report cells below 60% of their task target as explicit quality/coverage
   gaps.
5. Redistribute unavoidable cell shortfalls only among other tasks in the same
   language, using residual high-quality capacity.
6. Preserve source-family caps, quality weighting, and deterministic stratified
   selection.
7. Fail planning if a language cannot reach 5M without violating a hard gate.
8. Report target versus realized task shares globally and per language.

This policy is implemented as `p4_fixed_language_tokens`. It wraps the existing
P3 allocator once per language using capacity-adjusted task shares, preserving
P3 quality weighting, source-family caps, fallback behavior, and pool
allocation. Focused tests verify:

- existing P0-P3 output remains unchanged when the field is absent;
- tokens cannot move between languages;
- task shortfalls redistribute only within their language;
- a language below 5M reports a hard coverage shortfall;
- global budget must equal language budget times language count;
- the existing materializer accepts P4 plans and rejects coverage shortfalls;
- deterministic strata selection and four-appearance enforcement remain active.

## Planning configuration

The new optional field is `planning.language_token_budget`. K10 and K14 use the
same P3 controls and differ only in language list and global budget.

### K10 planning block

```yaml
planning:
  packed_token_budget: 50000000
  language_token_budget: 5000000
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
  minimum_label_tokens: 16585053
  sampling_seed: 42
  minimum_cell_token_share: 0.6
```

### K14 planning block

```yaml
planning:
  packed_token_budget: 70000000
  language_token_budget: 5000000
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
  minimum_label_tokens: 23219074
  sampling_seed: 42
  minimum_cell_token_share: 0.6
```

## Frozen validation contract

The accepted K10 and K14 validation monitor sets must remain unchanged so the
old and new training arms use the same in-distribution measurement. Before
training selection:

1. Hash every existing validation record using the canonical record digest.
2. Reserve those digests so none can enter the new training mixture.
3. Copy the accepted validation shards and manifests unchanged into each new
   versioned mixture.
4. Verify train-validation overlap remains zero and validation manifest/shard
   hashes match the accepted monitor.

`fixed_validation_manifest` now enforces this contract in both profiling and
materialization. Profiling excludes frozen validation digests before capacity
accounting. Candidate indexing independently excludes them again. The
materializer requires `validation_records_per_pool: 0`, copies the accepted
manifest and shards byte-for-byte, validates relative paths and token metadata,
and reports the fixed monitor in `summary.json`. Tests verify byte identity and
zero train-validation overlap.

## Required source artifacts

The complete K6, K10, and K14 roots are present because K14 profiles reference
pools across all three historical container paths.

| Root | Local size | Pool manifests |
|---|---:|---:|
| K6 | 4.8 GB | 12 |
| K10 | 3.7 GB | 9 |
| K14 | 1.2 GB | 19 |

| Artifact | SHA-256 |
|---|---|
| K6 profile config | `fe5418c57443edd845649c23cf57a1326bcaa203b8807aacac51c78547b6cfc1` |
| K6 benchmark blocklist | `b318c8f7e7392e7e905ba88f6bba9680d47006e1fc258f4acfd08952e4c75808` |
| K6 coverage review | `ff0e1060ef2095cdad15b8bd3390dad347f76442f568cf346e286f74935c4731` |
| K10 profile config | `0cf1a5705c60729ed54bbd63d6baa4f0d2ce2cd3dc28a28e9ec1c22c0ac8477d` |
| K10 benchmark blocklist | `e0d71cc1190a374a40f9ab5e030c3f28f98e1b640279c4e5f9afe0d2164c85fe` |
| K10 coverage review | `36a42cd478fa72a99cb75712cb2fd0060c69adf48706994d9a9e05fb685ddd8b` |
| K14 profile config | `53ffeeb4ee7bde1a91e96cbff02ba1029f3bf57c32400440abb1222daf76c672` |
| K14 coverage review | `d521bd16355d3d017112b6f3290f77a45b008ac64dbba69430378543399ba6f8` |
| K14 final audit | `f9aaaf605976f156005ed08f30da210017f71f274ce42386ac2f5636ede615e3` |

```bash
ROOT="/ext_data/casper_neo/Casper/kseries-next-run/data"

rsync -a --no-owner --no-group --info=progress2 --partial --append-verify \
  casper@spark-dsfsi.up.ac.za:/home/casper/k6-sft/ \
  "$ROOT/k6-sft/"
rsync -a --no-owner --no-group --info=progress2 --partial --append-verify \
  casper@spark-dsfsi.up.ac.za:/home/casper/k10-sft/ \
  "$ROOT/k10-sft/"
rsync -a --no-owner --no-group --info=progress2 --partial --append-verify \
  casper@spark-dsfsi.up.ac.za:/home/casper/k14-sft/ \
  "$ROOT/k14-sft/"
```

The transfer is complete. The K10 blocklist is the latest available database
and is the baseline for both new profiles; frozen validation digests provide an
additional independent exclusion layer.

## Deterministic preprocessing sequence

After complete source roots are present, preprocessing must use the exact image
digest and mounted historical paths from the two-epoch run report.

1. Copy the accepted K14 profile into separate K10/K14 P4 profile files.
2. Restrict K10 languages to its ten-language subset while retaining all
   relevant K14-era sources.
3. Update only global/language budgets, supervised-token minimum, and versioned
   output paths.
4. Extend the benchmark blocklist with the frozen validation digests.
5. Run exact profiling twice into independent directories:

```bash
/opt/venv/bin/python tools/profile_sft_mixture.py profile \
  --config /data/gemma4-k10/k10_p4_5m_per_language.yaml \
  --benchmark-blocklist /data/gemma4-k10/benchmarks.sqlite3 \
  --output-dir /data/gemma4-k10/profile-p4-5m-a

/opt/venv/bin/python tools/profile_sft_mixture.py profile \
  --config /data/gemma4-k10/k10_p4_5m_per_language.yaml \
  --benchmark-blocklist /data/gemma4-k10/benchmarks.sqlite3 \
  --output-dir /data/gemma4-k10/profile-p4-5m-b
```

6. Compare every generated artifact byte-for-byte.
7. Build plans and select only `p4_fixed_language_tokens`:

```bash
/opt/venv/bin/python tools/profile_sft_mixture.py plan \
  --config /data/gemma4-k10/k10_p4_5m_per_language.yaml \
  --profile /data/gemma4-k10/profile-p4-5m-a/profile.json \
  --output-dir /data/gemma4-k10/plans-p4-5m
```

8. Review `profile.md`, `cells.csv`, `plans.json`, per-language targets,
   below-minimum cells, source quality/family shares, label-token estimate, and
   all shortfalls before materialization.
9. Materialize into a new versioned root using
  `--policy p4_fixed_language_tokens`, `--validation-records-per-pool 0`, and
  the configured frozen validation manifest.
10. Produce composition CSV/Markdown, `summary.json`, `final_audit.json`, and
    `coverage_review.md`; hash all outputs.
11. Load both manifests through the production VLM builder.
12. Review and approve final composition before creating training recipes.

## Accepted artifacts and planned run names

Use new versioned roots; never overwrite accepted P2/P3 mixtures.

| Arm | Accepted mixture root | Planned run name |
|---|---|---|
| K10 | `/data/gemma4-k10/mixture-p4-5m-per-language-v2` | `gemma4-e2b-k10-p4-5m-per-language-r16-1ep-v1` |
| K14 | `/data/gemma4-k14/mixture-p4-5m-per-language-v1` | `gemma4-e2b-k14-p4-5m-per-language-r16-1ep-v1` |

Both runs use `dsfsi/gemma4-african-instruction`, with distinct names, groups,
checkpoint roots, configs, hashes, and W&B IDs.

## Review gates before implementation

1. Confirm that 5M means consumed tokens and therefore one epoch.
2. Approve within-language task-shortfall redistribution.
3. Verify the recorded source/profile/audit hashes before profiling.
4. Re-profile twice and require byte-identical outputs.
5. Review per-language/task capacity and quality before materialization.
6. Review and commit planner/materializer tests plus this plan before building
  mixtures.
7. Materialize, audit, hash, production-load, and document both mixtures.
8. Review final composition before creating or launching training recipes.
