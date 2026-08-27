# K-series fixed-language data reproduction

Status: K10 and K14 P4 mixtures materialized and accepted on 2026-08-26. This
document reproduces preprocessing only. It does not authorize or launch a
training run.

## Final datasets

| Arm | Host output | Container output | Languages | Train tokens |
|---|---|---|---:|---:|
| K10 | `/ext_data/casper_neo/Casper/kseries-next-run/data/k10-sft/mixture-p4-5m-per-language-v2` | `/data/gemma4-k10/mixture-p4-5m-per-language-v2` | 10 | 50,000,000 |
| K14 | `/ext_data/casper_neo/Casper/kseries-next-run/data/k14-sft/mixture-p4-5m-per-language-v1` | `/data/gemma4-k14/mixture-p4-5m-per-language-v1` | 14 | 70,000,000 |

K10 v1 is retained as a rejected provenance artifact. Its record-count
realization produced 49,488,239 tokens and did not satisfy the exact 5M per
language contract. K10 v2 uses deterministic integer-token reconciliation.
K14 v1 was built with the same reconciled materializer. These targets count
pre-shift Gemma processor tokens. The VLM packer removes one token per record
for causal training, yielding 49,553,273 K10 and 69,395,884 K14 model-input
tokens. The 4.9175M-4.9694M per-language post-shift range was explicitly
accepted on 2026-08-27.

A post-shift K10 v3 rebuild was started and then intentionally stopped when the
pre-shift mixtures were accepted. Its partial root contains only a temporary
`.mixture.sqlite3` and no `summary.json`; it is not a training dataset.

## Saved entry point

Use
[materialize_p4_fixed_language_mixtures.sh](materialize_p4_fixed_language_mixtures.sh)
for all accepted materialization and audit commands. The script:

- pins the immutable NeMo AutoModel image digest;
- mounts K6, K10, and K14 at their canonical container paths;
- selects only `p4_fixed_language_tokens`;
- requires frozen validation with `--validation-records-per-pool 0`;
- refuses to overwrite a nonempty output directory;
- materializes K10 and K14 sequentially;
- independently scans every emitted row;
- writes `final_audit.json` with source provenance and every output file hash.

The script accepts these commands:

```bash
# Audit the accepted outputs without rebuilding them.
sg docker -c \
  'examples/vlm_finetune/gemma4/materialize_p4_fixed_language_mixtures.sh audit all'

# Rebuild both arms sequentially into new names.
sg docker -c '
  K10_OUTPUT_NAME=mixture-p4-5m-per-language-repro \
  K14_OUTPUT_NAME=mixture-p4-5m-per-language-repro \
  examples/vlm_finetune/gemma4/materialize_p4_fixed_language_mixtures.sh materialize all
'

# Build or audit one arm.
sg docker -c \
  'examples/vlm_finetune/gemma4/materialize_p4_fixed_language_mixtures.sh materialize k10'
sg docker -c \
  'examples/vlm_finetune/gemma4/materialize_p4_fixed_language_mixtures.sh audit k14'
```

Set `EXPERIMENT_ROOT` if the source roots are not under
`/ext_data/casper_neo/Casper/kseries-next-run`. Set `REPO_ROOT` if the script is
invoked outside this checkout. Do not override `IMAGE` for an exact
reproduction.

## Immutable environment

| Item | Value |
|---|---|
| Container | `nvcr.io/nvidia/nemo-automodel@sha256:7213eab8055a2029ce1ef9022384a780f9095b9577f32edca4c48d303907421f` |
| Python in container | `/opt/venv/bin/python` |
| Model | `google/gemma-4-E2B-it` |
| Model revision | `3e22461f65e89153144f8adb70e3b8c2cc9845a7` |
| Sequence/pack length | 4,096 |
| Sampling seed | 42 |
| Maximum appearances | 4 |
| Repository base commit | `77394a463c9d177a35be09737cdab5c388e32025` |
| Exact-reconciliation patch SHA-256 | `6b7ad2bac4eed9a3c91d96c7f9fa0023daab1b87bcb718e3fdc2e85da5042476` |

The repository base commit predates the exact integer-token reconciliation.
For exact reproduction, the checkout must also contain the saved versions of
[build_k6_sft_mixture.py](../../../tools/build_k6_sft_mixture.py) and
[test_build_k6_sft_mixture.py](../../../tests/unit_tests/tools/test_build_k6_sft_mixture.py)
whose hashes are listed below. Once these changes are committed, the resulting
commit supersedes the base-plus-patch description.

Materialization reads the repository `.env` but passes only `HF_TOKEN` into the
container for gated Hugging Face access. Never place credentials in this
document. W&B credentials are not exposed to preprocessing.

## Required filesystem

The default experiment root must contain:

```text
kseries-next-run/
  hf-cache/
  data/
    k6-sft/
    k10-sft/
      benchmarks.sqlite3
      mixture-p2-v1/
      profile-p4-5m-a/
      profile-p4-5m-b/
      plans-p4-5m/
    k14-sft/
      mixture-p3-final-v2/
      profile-p4-5m-a/
      profile-p4-5m-b/
      plans-p4-5m/
```

The profile configs are the source registry. Each pool entry records its
canonical manifest path, source family, pinned revision, license, and quality
tier:

- [K10 profile config](data/k10_p4_5m_per_language_profile.yaml)
- [K14 profile config](data/k14_p4_5m_per_language_profile.yaml)

The complete K6/K10/K14 source roots are required because both configs draw
from all three historical roots. Transfer instructions and the rationale for
the source choices are in
[KSERIES_FIXED_LANGUAGE_BUDGET_PLAN.md](KSERIES_FIXED_LANGUAGE_BUDGET_PLAN.md).

## Policy contract

Both arms use `p4_fixed_language_tokens` with these controls:

| Control | K10 | K14 |
|---|---:|---:|
| Global packed-token budget | 50,000,000 | 70,000,000 |
| Per-language budget | 5,000,000 | 5,000,000 |
| Minimum supervised tokens | 16,585,053 | 23,219,074 |
| Languages | 10 | 14 |
| Epochs when consumed | 1 | 1 |

The task prior per language is instruction 33%, QA 15%, translation 10%,
classification 17%, and NER 25%. Capacity shortfalls may be redistributed only
among tasks in the same language. The planner retains P3 quality weights,
source-family caps, fallback behavior, and deterministic stratified selection.

The materializer first stages planned record appearances in SQLite. For each
language, it compares exact Gemma processor token counts with the 5M target.
Under-budget languages add only validation-disjoint, benchmark-safe, previously
unselected donor records. Over-budget languages remove staged appearances.
An exact bounded subset solver chooses additions or removals whose integer
token lengths equal the discrepancy. Candidate cells are ordered by deviation
from their approved pool targets. The build fails if an exact subset does not
exist. Final shards are written only after every language reaches exactly 5M
pre-shift processor tokens.

## Frozen validation

K10 reuses `/data/gemma4-k10/mixture-p2-v1/validation_meta.json`; K14 reuses
`/data/gemma4-k14/mixture-p3-final-v2/validation_meta.json`. Candidate indexing
reserves every canonical validation digest before training selection. The
accepted validation manifests and all shards are copied byte-for-byte:

| Arm | Validation records | Validation tokens | Shards |
|---|---:|---:|---:|
| K10 | 30,306 | 9,784,477 | 13 |
| K14 | 23,440 | 6,379,444 | 24 |

## Full regeneration sequence

### 1. Verify source and code inputs

```bash
sha256sum \
  tools/build_k6_sft_mixture.py \
  examples/vlm_finetune/gemma4/data/k10_p4_5m_per_language_profile.yaml \
  examples/vlm_finetune/gemma4/data/k14_p4_5m_per_language_profile.yaml \
  /ext_data/casper_neo/Casper/kseries-next-run/data/k10-sft/benchmarks.sqlite3
```

Expected hashes:

| Artifact | SHA-256 |
|---|---|
| Materializer | `5231dba08350eb24673a26334224404f3540d6992af88eaeb459a4d1d9eeb055` |
| Materializer tests | `96d421ed45541c8a57382d8250ce621dd6120f46bb2dd406e615e48c8e28bf18` |
| Wrapper | `d5a05ad35ff67e8cc768820cd3f709eba88105f01c0e08b659acef3c44480c79` |
| K10 config | `be1560f057644776a93e7cd698de74f1e2457a2a0166a1c94970b3bc9511a6ef` |
| K14 config | `d046661b9ad34b6a2ce758c43fbf5c73210f7fffbb3f145c25fb35e9f98c7344` |
| Benchmark blocklist | `e0d71cc1190a374a40f9ab5e030c3f28f98e1b640279c4e5f9afe0d2164c85fe` |

### 2. Profile twice

Run these commands inside the pinned container with the canonical source mounts
used by the wrapper:

```bash
/opt/venv/bin/python tools/profile_sft_mixture.py profile \
  --config examples/vlm_finetune/gemma4/data/k10_p4_5m_per_language_profile.yaml \
  --benchmark-blocklist /data/gemma4-k10/benchmarks.sqlite3 \
  --output-dir /data/gemma4-k10/profile-p4-5m-a

/opt/venv/bin/python tools/profile_sft_mixture.py profile \
  --config examples/vlm_finetune/gemma4/data/k10_p4_5m_per_language_profile.yaml \
  --benchmark-blocklist /data/gemma4-k10/benchmarks.sqlite3 \
  --output-dir /data/gemma4-k10/profile-p4-5m-b
```

Repeat with the K14 config and K14 output root. Compare all four artifacts:

```bash
for arm in k10 k14; do
  root="/ext_data/casper_neo/Casper/kseries-next-run/data/${arm}-sft"
  for file in profile.json cells.csv profile.md review_samples.jsonl; do
    cmp "$root/profile-p4-5m-a/$file" "$root/profile-p4-5m-b/$file"
  done
done
```

Expected A/B-identical hashes:

| Arm | `profile.json` | `cells.csv` | `profile.md` | `review_samples.jsonl` |
|---|---|---|---|---|
| K10 | `3d91c8ed631a3658f565858de966bd50453786c5df0fe227d05b88843a394713` | `5ddb1c179b223af99375db57dd518e7aad5c966669b2f85062094cf47774e75a` | `b93d579ea9e755c5258b18ce7406dc1c6a178ddb3dcfbd0af6bb91e3f00c53ae` | `cb49dfacf84248791cc8793f9900495683cbfdee8c14af5677f170ac407d9fc6` |
| K14 | `7a5ae45009dd91937b6fb4b4f98401f2aab168c46779e89337ee431205005cbb` | `a0d00657e87b477d1a008fb2e704132455036644a8a9591193e54c4693cd1e36` | `350dfb118afb6781d096657533518f701f20333e27441e7cae9a8c67af599b69` | `9b2a01c718899725d6998849f933fd9a26b21f760d1e78ab11d735a30d4e330e` |

### 3. Generate and review plans

Inside the same container:

```bash
/opt/venv/bin/python tools/profile_sft_mixture.py plan \
  --config examples/vlm_finetune/gemma4/data/k10_p4_5m_per_language_profile.yaml \
  --profile /data/gemma4-k10/profile-p4-5m-a/profile.json \
  --output-dir /data/gemma4-k10/plans-p4-5m

/opt/venv/bin/python tools/profile_sft_mixture.py plan \
  --config examples/vlm_finetune/gemma4/data/k14_p4_5m_per_language_profile.yaml \
  --profile /data/gemma4-k14/profile-p4-5m-a/profile.json \
  --output-dir /data/gemma4-k14/plans-p4-5m
```

| Arm | `plans.json` | `plans.md` |
|---|---|---|
| K10 | `4f0441526ae222c8d11bd35b214ab7831a3943ec62913fd5cb41ffd5a510bb1c` | `ef521201e79318f6fb6db620fe3fb3b7f7b0a4dc6894a84ee0f4bb7f556584bb` |
| K14 | `1555db81080568ee14a6a56da93e8e9ace18ff737edc99cb200bd7f2411bdf89` | `9dd950bd9d0139385708e5bd505b1f362388521f50c171280d20ba9e83d3c69b` |

Before materialization, require zero `coverage_shortfall_tokens`, zero
`label_token_shortfall`, exactly 5M planned tokens for every language, and
explicit approval of the source/quality concentrations and below-60% cells.

### 4. Materialize and audit

Choose unused output names and run:

```bash
sg docker -c '
  K10_OUTPUT_NAME=mixture-p4-5m-per-language-repro \
  K14_OUTPUT_NAME=mixture-p4-5m-per-language-repro \
  examples/vlm_finetune/gemma4/materialize_p4_fixed_language_mixtures.sh materialize all
'
```

The command is intentionally sequential. Candidate indexing scans about eight
million source records and spends most of its time on AfriInstruct. Quiet log
intervals are expected. During a build, `.mixture.sqlite3` is temporary; a
successful build removes it and writes `summary.json`, `train_meta.json`,
`validation_meta.json`, train/validation shards, and `final_audit.json`.

Do not infer failure while the container is still active. Do not delete a
partial directory until the process has exited and its failure is understood.

## Accepted results

| Metric | K10 | K14 |
|---|---:|---:|
| Train records | 446,727 | 604,116 |
| Unique train records | 342,106 | 444,095 |
| Pre-shift processor tokens | 50,000,000 | 70,000,000 |
| Post-shift model-input tokens | 49,553,273 | 69,395,884 |
| Supervised tokens | 17,479,939 | 27,699,644 |
| Maximum appearances | 4 | 4 |
| Train-validation overlap | 0 | 0 |
| Benchmark matches | 0 | 0 |

Every K10 and K14 language has exactly 5,000,000 pre-shift processor tokens.
Post-shift model-input exposure is within 1.65% of 5M for every language.
Realized pre-shift task shares:

| Arm | Instruction | QA | Translation | Classification | NER |
|---|---:|---:|---:|---:|---:|
| K10 | 32.2088% | 11.4300% | 11.0782% | 18.0164% | 27.2666% |
| K14 | 32.7332% | 12.5824% | 10.8617% | 17.1568% | 26.6659% |

Realized quality shares:

| Arm | Human | Curated | Mixed | Silver |
|---|---:|---:|---:|---:|
| K10 | 36.7405% | 5.7036% | 52.0031% | 5.5529% |
| K14 | 41.3067% | 7.1989% | 45.7211% | 5.7733% |

AfriInstruct contributes 51.9328% of K10 and 36.9135% of K14. These
concentrations and the capacity-driven task deviations were explicitly
approved before materialization.

## Accepted artifact hashes

| Arm | Artifact | SHA-256 |
|---|---|---|
| K10 | `summary.json` | `2b334e9050d0a6fec0e898d6bb13f792fe16bce467123349fe446d006333f668` |
| K10 | `train_meta.json` | `a6781ce99a87189500c8fc31d323f5091e642bb5f2d57207e66b5db57d6853d2` |
| K10 | `validation_meta.json` | `3906353425ab0da96d81ac7c08a9ec8ca75eed1d3bd2efc1107d0730e3f1c2f8` |
| K10 | `final_audit.json` | `27e81495c51619920bbbd1ac58219d90949543e7e0aadb788cce8d56dca1cb3b` |
| K14 | `summary.json` | `59f5d5bbbec279ab855431ff5ba0ba5fd9dce1c73e6cadc8a663ac51170a0d15` |
| K14 | `train_meta.json` | `f756e8f883cacbe08fd89b15ea3b77016e29d639ec8bc6ec94194c96d15004d8` |
| K14 | `validation_meta.json` | `560549273ef3ee460d7501aa69fbc939ede3e7871ce992b4381199b5997c02ad` |
| K14 | `final_audit.json` | `a7a70645b094560f530f48b3db8f4bfce105fac3bea357109324b95fb16e7594` |

Each `final_audit.json` contains hashes for every train and validation shard,
the three manifests/summary files, every source manifest, the approved plan,
the config, the benchmark blocklist, the frozen validation manifest, and the
materializer. Use those files as the complete machine-readable integrity
record rather than duplicating dozens of shard hashes here.

## Validation commands

The exact-reconciliation implementation is covered by CPU tests for donor
addition, selected-record removal, impossible exact targets, frozen validation,
shortfall rejection, deterministic strata, and repetition limits:

```bash
sg docker -c 'docker run --rm \
  -v /ext_data/casper_neo/Casper/Automodel:/workspace \
  -w /workspace \
  nvcr.io/nvidia/nemo-automodel@sha256:7213eab8055a2029ce1ef9022384a780f9095b9577f32edca4c48d303907421f \
  /opt/venv/bin/python -m pytest \
  tests/unit_tests/tools/test_profile_sft_mixture.py \
  tests/unit_tests/tools/test_build_k6_sft_mixture.py -q'
```

Expected result: 22 tests pass. Also require:

```bash
bash -n examples/vlm_finetune/gemma4/materialize_p4_fixed_language_mixtures.sh
sg docker -c 'docker run --rm \
  -v /ext_data/casper_neo/Casper/Automodel:/workspace \
  -w /workspace \
  nvcr.io/nvidia/nemo-automodel@sha256:7213eab8055a2029ce1ef9022384a780f9095b9577f32edca4c48d303907421f \
  /opt/venv/bin/ruff check tools/build_k6_sft_mixture.py'
```

Before training, separately run the production VLM dataset-loader preflight,
create the amended two-epoch run configs pointing to the accepted manifests, resolve exact
optimizer/warmup steps from the packed dataset, and complete final execution
review. The completed recipes, schedules, detached launcher, and launch gates
are recorded in
[KSERIES_FIXED_LANGUAGE_TRAINING.md](KSERIES_FIXED_LANGUAGE_TRAINING.md).
Materialization acceptance does not imply training approval.