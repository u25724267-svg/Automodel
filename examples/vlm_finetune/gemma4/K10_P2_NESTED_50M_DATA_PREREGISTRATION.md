# K10 P2 nested 50M data preregistration

Date: 2026-09-06

## Question

Does adding high-quality data improve the K10 P2 result when the accepted 30M
mixture remains intact and the extension stays as close as capacity permits to
the original language, task, and source composition?

Benchmark results must not be used to choose sources, filters, allocations, or
fallbacks. All decisions below are fixed before profiling and materialization.

## Locked base and extension

The accepted K10 P2 v1 train mixture is copied byte-for-byte into the new
mixture. It contains 143,579 appearances, 122,312 unique records, and
31,349,312 prepared text tokens. Its validation manifest is also copied
byte-for-byte and remains excluded from all extension candidates.

The extension target is exactly 18,650,688 prepared text tokens, producing a
combined target of exactly 50,000,000. Every extension record must be unique
within the extension and disjoint from the locked base. The extension therefore
uses `max_epochs: 1.0`; it does not increase nominal volume through repetition.

## Composition policy

The accepted P2 `summary.json` is the allocation authority. Reference
language-task-source token allocations are multiplied by:

```text
18,650,688 / 31,349,312 = 0.594931...
```

For each language-task cell, allocation proceeds in this fixed order:

1. unused unique records from the same P2 source pool;
2. additional unused capacity from another source already used by P2 in that cell;
3. unused capacity from another original P2 registry source in that cell, ordered by quality and capacity;
4. deterministic within-language redistribution when the complete original registry cannot satisfy the cell.

No source introduced for P4 is eligible. The source registry is exactly the 15
pools used to construct P2. Human and curated sources precede mixed
AfriInstruct fallback through the declared quality tiers. Mixed data is capped
at its scaled P2 source allocation and cannot absorb another source or cell's
shortfall; only unique human or curated records may exceed their scaled source
targets.

Exact proportional expansion is impossible for Wolof instruction and isiXhosa
QA because the original registry has insufficient residual capacity. Their
shortfalls are redistributed only among other nonempty P2 tasks in the same
language, weighted by the accepted P2 task allocation. If a language then has
insufficient total unique capacity, only its irreducible remainder moves to
other languages, prioritizing the same task and weighting recipients by their
accepted P2 allocation. Every language deviation is reported. Kinyarwanda QA,
Amharic NER, and Wolof classification remain empty because they were empty in
P2; the experiment does not introduce new task coverage.

After capacity redistribution, a bounded label-density reconciliation may move
only allocation above scaled reference-source targets from a lower-density to a
higher-density human or curated cell. It may cross a language or task boundary
only by the minimum token amount required to restore the preregistered
supervised-token floor. The moved token count and before/after label estimates
are recorded in the plan. Exact integer-token materialization may add only an
unselected human or curated record from a nonempty planned cell, or remove an
appearance from an allocation already above its plan target.

## Quality and safety gates

- Extension prepared tokens: exactly 18,650,688.
- Combined prepared tokens: exactly 50,000,000.
- Extension supervised tokens: at least 5,658,730, preserving P2's observed label-token ratio.
- Planning supervised-token target: 5,758,730, adding a 100,000-token margin for integer-record rounding.
- Locked-base shard bytes and record multiplicities unchanged.
- Extension/base normalized-message digest overlap: zero.
- Train/frozen-validation overlap: zero.
- Benchmark fingerprint matches: zero.
- Extension record appearances: exactly one.
- All source revisions, licenses, quality tiers, and rejection counts reported.
- Profile outputs from two independent passes must be byte-identical.
- Every target-versus-realized language, task, source, and quality deviation reported.

## Materialization artifacts

The checked-in profile is
[`data/k10_p2_nested_50m_profile.yaml`](data/k10_p2_nested_50m_profile.yaml).
Generated artifacts use versioned directories under `/data/gemma4-k10/`:

- `profile-p2-nested-50m-a/`
- `profile-p2-nested-50m-b/`
- `plans-p2-nested-50m/`
- `mixture-p2-nested-50m-v1/`

The final audit must include hashes for the config, accepted P2 manifests and
shards, profile, plan, benchmark blocklist, combined manifests and shards, and
the materializer source.

## Accepted profile and plan

Two complete profiles of roughly 7.2 million source records were byte-identical.

| Artifact | SHA-256 |
|---|---|
| Profile config | `a92a6c594f1199b6f9890a3a88696e10b3232fd07aeafdaf85db62c545500ac2` |
| `profile.json` | `cb1a673fc6cf9f3d5047de450b6734fa81961b402de8f57923e50605c861efa5` |
| `cells.csv` | `eea01027c74611c0b4afd76050dd7bd8b905dbb0d6457b93b6235d2490e38f10` |
| `profile.md` | `1a9782a51161fc0f3cf583f7ba6a34b3a72be218b5f10e8d8d29868614f71c24` |
| `review_samples.jsonl` | `a7f76779f5f949395702f9e01b6f8ecbb7c8caf2a9f81cc5bd83a0025d21351d` |
| `plans.json` | `f0575c188956192358de8a4ee20ae38b70df969fb42364015d0448537844968c` |
| `plans.md` | `66bbd7314b1fcb4eed6cbb49f58c2a4ec68b0a4e10e84ea03fe6def202106112` |

The accepted plan contains exactly 18,650,688 prepared tokens and 5,758,730
estimated supervised tokens, including 100,000 tokens of record-rounding
headroom above the 5,658,730 acceptance floor. It has zero token and label
shortfall. Extension quality shares are 46.40% human, 9.85% curated, and 43.75%
mixed, with no silver or synthetic data. Label reconciliation moves 194,223
prepared tokens above scaled reference targets to higher-density human data.

Unique-capacity limits move 347,474 Wolof tokens to other languages. The final
language deviations from proportional P2 extension targets are: Igbo -372,194,
Kinyarwanda +531,630, Swahili +1,870,860, Yoruba +194,223, isiXhosa -386,233,
Shona -1,490,812, and Wolof -347,474 tokens. Hausa, Amharic, and isiZulu remain
at their proportional targets within rounding.

## Accepted materialization

Status: accepted by independent retokenizing audit on 2026-09-07.

| Metric | Value |
|---|---:|
| Base records / unique | 143,579 / 122,312 |
| Extension records / unique | 111,375 / 111,375 |
| Combined records / unique | 254,954 / 233,687 |
| Base prepared tokens | 31,349,312 |
| Extension prepared tokens | 18,650,688 |
| Combined prepared tokens | 50,000,000 |
| Extension supervised tokens | 5,700,057 |
| Combined supervised tokens | 15,211,624 |
| Maximum combined repetitions | 4 |
| Base-extension overlap | 0 |
| Train-validation overlap | 0 |
| Benchmark matches | 0 |

The audited extension contains 8,694,190 human, 1,918,657 curated, and
8,037,841 mixed tokens: 46.62%, 10.29%, and 43.10% respectively. It contains no
silver or synthetic data. Every extension record appears exactly once.

| Artifact | SHA-256 |
|---|---|
| `summary.json` | `7c4986dc03d2aaef07d6887641475665a29e0de6755db2b13128cbfd91d6de22` |
| `train_meta.json` | `8e491592f58bc704990ed66238c90c79fe1678eb3582610d96178cba00244a07` |
| `validation_meta.json` | `3906353425ab0da96d81ac7c08a9ec8ca75eed1d3bd2efc1107d0730e3f1c2f8` |
| `final_audit.json` | `12ebd9e7ce63f7c14da8db473dae49fc9f338b06f83e617cc61b6fe62c681acd` |
| Materializer | `5439694e5ace608b40a86b2303be9b20e3894eeaf152cf4339ad011aee464ebb` |
| Auditor | `8e92ead7c8bee43f059de0d53881202f9a2c3e249b1a4ee82362a91b3d9a5b73` |

## C2 training recipe

The C2 recipe is
[`gemma4_e2b_k10_p2_nested_50m_r16_2ep_nvidia_lr.yaml`](gemma4_e2b_k10_p2_nested_50m_r16_2ep_nvidia_lr.yaml),
SHA-256 `bcde66563ef4f8ac8e92e14d9b90bde41ee30c2c884fdbda154dabbe77f747c5`.

It is a strict clone of the completed C0
`gemma4_e2b_k10_p2_r16_2ep_nvidia_lr.yaml` training policy. The only
behavioral change is the training manifest:
`/data/gemma4-k10/mixture-p2-nested-50m-v1/train_meta.json`. The original P2
validation path remains unchanged. Model and processor revision, rank-16 LoRA
targets, freezing, optimizer, cosine scheduling policy, two epochs, seed 42,
packing, dataloaders, loss, distributed strategy, validation cadence, and
checkpoint cadence are identical.

Operational identities use checkpoint root
`/checkpoints/gemma4-e2b-k10/p2-nested-50m-r16-2ep-nvidia-lr-v1` and W&B name
`gemma4-e2b-k10-p2-nested-50m-r16-2ep-nvidia-lr-v1`.

## C2 production-loader preflight

The pinned production container loaded the audited manifest through the exact
VLM dataset and packed-dataloader builders under a single-rank process group.

| Metric | Value |
|---|---:|
| Physical train examples | 254,954 |
| Packed batches per epoch | 13,491 |
| Gradient accumulation steps | 8 |
| Optimizer steps per epoch | 1,687 |
| Epochs | 2 |
| Total optimizer steps | 3,374 |
| LR warmup steps | 337 |
| LR decay steps | 3,374 |
| Initial / peak / minimum LR | `2e-5` / `2e-4` / `2e-6` |
| Decay style | Cosine |

Checkpoint steps 1,999 and 2,199 bracket C0's 62,698,624 prepared-token
exposure and must both be evaluated before any benchmark result is inspected.
The final comparison checkpoint is step 3,373.

## Active C2 run

Status: running. The detached container started on GPU 0 at
`2026-09-07T09:29:20Z` after the launcher confirmed no active compute process.

| Field | Value |
|---|---|
| Container | `gemma4-e2b-k10-p2-nested-50m-r16-2ep-nvidia-lr-v1` |
| W&B ID | `363t39xb` |
| W&B URL | `https://wandb.ai/dsfsi/gemma4-african-instruction/runs/363t39xb` |
| Checkpoint root | `/checkpoints/gemma4-e2b-k10/p2-nested-50m-r16-2ep-nvidia-lr-v1` |
| Launch Git HEAD | `3286d2285b4e0c7585938216c8fdea4abb2429e0` with captured dirty status and source patch |
| Image | `nvcr.io/nvidia/nemo-automodel@sha256:7213eab8055a2029ce1ef9022384a780f9095b9577f32edca4c48d303907421f` |
| Resolved schedule | 3,374 optimizer steps; 337 warmup; cosine to step 3,374 |

Step 0 completed with loss 3.9829, perplexity 53.6716, gradient norm 36.1325,
LR `2.05e-5`, 30.27 GiB GPU allocation, 1,195.61 tokens/s, and 8,405
supervised tokens. Steps 1-3 also completed with finite loss and gradients. The
container reported no OOM, and four persistent training workers stabilized near
12.7 GiB aggregate host memory during initialization.