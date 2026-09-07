# K10 30M versus 50M data-composition report

Date: 2026-09-06

## Executive conclusion

The K10 50M run is not a scaled-up version of the K10 30M run. It is a
substantially different mixture. Only 54.41% of the 30M mixture's unique
records occur in the 50M mixture, and only 27.64% of the 50M token exposure is
the same retained record exposure. The 50M mixture simultaneously changes
language balance, task balance, source and quality shares, sequence lengths,
and repetition.

Performance degradation is confirmed on the shared K10 validation monitor:

| Metric | K10 30M P2 | K10 50M P4 | Change |
|---|---:|---:|---:|
| Best validation loss | 2.190011 | 2.296752 | +0.106741 (+4.87%) |
| Final validation loss | 2.190011 | 2.296923 | +0.106912 (+4.88%) |
| Best validation perplexity | 8.935315 | 9.941840 | +11.26% |
| Final validation perplexity | 8.935315 | 9.943538 | +11.28% |

Data composition is therefore a credible and likely important contributor to
the higher validation loss. The strongest evidence is that the frozen monitor
is heavily concentrated in sources and languages that P4 deliberately reduced,
while P4 is measurably farther from that monitor across language, task, and
source distributions.

This does **not** establish that P4 is globally worse. The monitor is inherited
from P2 and is highly unbalanced. No directly comparable P2-r16 versus P4-r16
downstream benchmark outputs are preserved in the repository. A broad
performance-degradation claim remains provisional until both final adapters
are evaluated on the same balanced benchmark suite.

## Runs compared

"30M" refers to the accepted K10 P2 v1 mixture, which contains 31,349,312
prepared text tokens per epoch. "50M" refers to K10 P4 fixed-language v2,
which contains exactly 50,000,000 prepared text tokens per epoch.

| Property | K10 30M P2 | K10 50M P4 |
|---|---:|---:|
| Mixture policy | `p2_quality_constrained` | `p4_fixed_language_tokens` |
| W&B run | `vebv31eh` | `dedr4hqq` |
| Epochs | 2 | 2 |
| Optimizer steps | 2,116 | 3,362 |
| Prepared tokens per epoch | 31,349,312 | 50,000,000 |
| Prepared token exposure | 62,698,624 | 100,000,000 |
| Model-input tokens per epoch | 31,205,733 | 49,553,273 |
| Train records per epoch | 143,579 | 446,727 |
| Unique train records | 122,312 | 342,106 |
| Base model and revision | Gemma 4 E2B IT, `3e22461...` | Same |
| LoRA rank / alpha / dropout | 16 / 32 / 0.0 | Same |
| Peak / minimum LR | 2e-4 / 2e-6 | Same |
| Global batch / pack size | 8 / 4,096 | Same |
| Seed | 42 | Same |
| Validation data | Frozen K10 P2 monitor | Same monitor, byte-for-byte |

The recipes control most model and optimization variables. P4 necessarily has
a longer cosine schedule because it contains 59.49% more tokens per epoch and
59% more optimizer steps. The P4 run was interrupted by a host reboot and
resumed from complete optimizer, RNG, dataloader, and scheduler state; there is
no evidence that the interruption caused the quality gap.

## Top-level data makeup

| Metric | K10 30M P2 | K10 50M P4 | Change |
|---|---:|---:|---:|
| Materialized records | 143,579 | 446,727 | +211.12% |
| Unique records | 122,312 | 342,106 | +179.70% |
| Duplicate excess | 21,267 (14.81%) | 104,621 (23.42%) | +8.61 pp |
| Prepared text tokens | 31,349,312 | 50,000,000 | +59.49% |
| Supervised label tokens | 9,511,567 | 17,479,939 | +83.77% |
| Supervised-token share | 30.34% | 34.96% | +4.62 pp |
| Mean text tokens / record | 218.34 | 111.93 | -48.74% |
| Median text tokens / record | 85 | 65 | -23.53% |
| 90th percentile length | 595 | 193 | -67.56% |
| 99th percentile length | 1,793 | 830 | -53.71% |
| Records at most 128 tokens | 65.79% | 81.92% | +16.13 pp |
| Records above 512 tokens | 12.33% | 3.06% | -9.27 pp |

P4 contains more unique records and more supervised tokens, so the degradation
cannot be explained by a simple shortage of data or labels. The important
change is what those tokens represent: P4 is much more dominated by short
examples and has a higher repeated-record share.

## Language composition

P2 allocates languages according to available capacity and quality weighting.
P4 forces every language to exactly 5M prepared tokens.

| Language | P2 tokens | P2 share | P4 tokens | P4 share | Token change |
|---|---:|---:|---:|---:|---:|
| Amharic | 6,047,898 | 19.29% | 5,000,000 | 10.00% | -17.33% |
| Hausa | 2,984,093 | 9.52% | 5,000,000 | 10.00% | +67.56% |
| Igbo | 2,572,342 | 8.21% | 5,000,000 | 10.00% | +94.38% |
| Kinyarwanda | 2,790,252 | 8.90% | 5,000,000 | 10.00% | +79.20% |
| Shona | 3,454,080 | 11.02% | 5,000,000 | 10.00% | +44.76% |
| Swahili | 4,122,944 | 13.15% | 5,000,000 | 10.00% | +21.27% |
| Wolof | 1,036,618 | 3.31% | 5,000,000 | 10.00% | +382.34% |
| isiXhosa | 2,303,622 | 7.35% | 5,000,000 | 10.00% | +117.05% |
| Yoruba | 3,478,559 | 11.10% | 5,000,000 | 10.00% | +43.74% |
| isiZulu | 2,558,904 | 8.16% | 5,000,000 | 10.00% | +95.40% |

Consequences:

- Amharic is the only language with fewer absolute tokens under P4. Its
  exposure falls by 1.05M tokens per epoch and its share nearly halves.
- Wolof grows 4.82x. Wolof also has the highest P4 record count (82,489), and
  a meaningful part of its QA, NER, and translation exposure is repetition.
- Equal token allocation does not mean equal information. Tokenizer fertility,
  average record length, source diversity, and repetition differ by language.
- The shared validation set contains no Wolof records, so the largest P4
  allocation increase cannot improve the aggregate validation metric.

## Task composition

| Task | P2 tokens | P2 share | P4 tokens | P4 share | Token change | Share change |
|---|---:|---:|---:|---:|---:|---:|
| Instruction | 10,713,778 | 34.18% | 16,104,382 | 32.21% | +50.32% | -1.97 pp |
| QA | 3,475,531 | 11.09% | 5,715,011 | 11.43% | +64.44% | +0.34 pp |
| Translation | 2,451,718 | 7.82% | 5,539,111 | 11.08% | +125.93% | +3.26 pp |
| Classification | 7,643,186 | 24.38% | 9,008,220 | 18.02% | +17.86% | -6.36 pp |
| NER | 7,065,099 | 22.54% | 13,633,276 | 27.27% | +92.97% | +4.73 pp |

P4 sharply increases translation and NER exposure while classification grows
only 17.86% against a 59.49% larger total budget. If degradation is strongest
on classification or broad instruction following, this dilution is a plausible
mechanism. Globally, label density improves in classification, instruction, and
QA but decreases slightly in NER and translation, so aggregate lack of
loss-bearing tokens is not the issue.

## Source and quality composition

The largest source movements are:

| Source pool | P2 share | P4 share | Direction |
|---|---:|---:|---|
| AfriInstruct | 44.49% | 51.93% | +7.44 pp; tokens +86.17% |
| Aya | 20.78% | 15.55% | -5.22 pp; tokens +19.40% |
| MasakhaNEWS | 13.04% | 2.47% | -10.57 pp; tokens -69.82% |
| MasakhaNER | 2.47% | 6.05% | +3.58 pp; tokens +291.29% |
| AfriHate | 3.21% | 5.43% | +2.22 pp; tokens +169.85% |
| Pontoon | 2.49% | 4.12% | +1.63 pp; tokens +163.67% |
| Swahili News | 3.57% | 0.78% | -2.79 pp; tokens -65.14% |
| KinNews | 2.88% | 1.00% | -1.89 pp; tokens -44.91% |

P4 also introduces accepted sources that were absent from P2, notably
FineWeb-derived Amharic NER, filtered Wolof sentiment, and Kinyarwanda QA.

| Quality tier | P2 share | P4 share | Change |
|---|---:|---:|---:|
| Human | 49.94% | 36.74% | -13.20 pp |
| Mixed | 44.49% | 52.00% | +7.51 pp |
| Curated | 5.57% | 5.70% | +0.13 pp |
| Silver | 0.00% | 5.55% | +5.55 pp |

The lower human-data share and the replacement of long news examples by short
NER/classification records are credible generalization risks. This does not
prove that the new sources are low quality, but it changes the type and density
of supervision enough that equal token totals are not comparable.

## Exact overlap and repetition

Records were matched using their normalized message payload and metadata,
excluding the stored token-count fields.

| P4 exposure category | Records | P4 record share | Tokens | P4 token share |
|---|---:|---:|---:|---:|
| Same retained copies as P2 | 85,633 | 19.17% | 13,821,055 | 27.64% |
| Extra repetitions of P2 records | 60,048 | 13.44% | 5,180,676 | 10.36% |
| Records novel relative to P2 | 301,046 | 67.39% | 30,998,269 | 62.00% |

Only 66,550 of P2's 122,312 unique records occur in P4, a retention rate of
54.41%. P4 is therefore a replacement mixture, not an augmentation mixture.
It adds substantial genuinely novel content, but also removes 45.59% of P2's
unique examples and increases repeated exposure.

The most conspicuous repetition pressure is in capacity-constrained cells:

- Wolof adds about 892K extra repeated NER tokens, 415K QA tokens, and 278K
  translation tokens relative to P2 multiplicities.
- Shona adds about 420K repeated QA tokens and 216K repeated translation tokens.
- Igbo adds about 392K repeated QA tokens and 181K repeated classification tokens.
- isiZulu adds about 322K repeated classification tokens while adding no novel
  classification payloads relative to P2.

These cells can improve memorization of their represented tasks while reducing
the marginal information gained per nominal token.

## Validation-monitor composition

Both runs use exactly the same validation set: 30,306 records and 9,784,477
prepared tokens. This is excellent for detecting movement on the old P2
distribution, but the monitor is not balanced.

Language concentration:

| Language | Validation token share |
|---|---:|
| Swahili | 46.95% |
| Kinyarwanda | 20.42% |
| Yoruba | 16.75% |
| Amharic | 6.85% |
| All remaining represented languages | 9.03% |
| Wolof | 0.00% |

Task concentration:

| Task | Validation token share |
|---|---:|
| Classification | 47.72% |
| QA | 24.40% |
| Instruction | 17.37% |
| NER | 5.32% |
| Translation | 5.19% |

Source concentration is similarly strong: KenSwQuAD, Swahili News, KinNews,
and Aya contribute 79.10% of validation tokens. P4 reduces the training share
of all four pools; Aya grows modestly in absolute tokens while the other three
also fall in absolute tokens.

Distribution distance from the shared monitor was measured using token shares:

| Dimension | P2 total variation | P4 total variation | P2 JS divergence | P4 JS divergence |
|---|---:|---:|---:|---:|
| Language | 0.5097 | 0.5412 | 0.2293 bits | 0.2826 bits |
| Task | 0.3665 | 0.4268 | 0.1096 bits | 0.1500 bits |
| Source pool | 0.6082 | 0.6697 | 0.3795 bits | 0.5105 bits |

P4 is farther from the monitor on every measured composition axis. The higher
P4 validation loss is therefore consistent with distribution mismatch. It may
partly be an expected cost of balancing ten languages rather than evidence of
uniformly worse capability.

## Training-curve evidence

P2's final checkpoint is reached after 62.70M prepared-token exposure. The
nearest P4 validations bracket comparable exposure:

| Run / step | Approximate prepared-token exposure | Validation loss | Perplexity |
|---|---:|---:|---:|
| P2 step 2,115 | 62.70M | 2.1900 | 8.9353 |
| P4 step 1,999 | 59.49M | 2.3396 | 10.3773 |
| P4 step 2,199 | 65.44M | 2.3250 | 10.2271 |
| P4 best, step 3,199 | 95.18M | 2.2968 | 9.9418 |
| P4 final, step 3,361 | 100.00M | 2.2969 | 9.9435 |

At approximately matched token exposure, P4 is already worse on the shared
monitor. An additional roughly 37M tokens improves P4 but does not close the
gap. This argues against "P4 merely needed more steps" as the explanation.
The comparison is not perfectly controlled because P4 is at a different point
in its longer cosine schedule at matched token exposure.

## Causal assessment

### High-confidence findings

1. The validation degradation is real on the frozen P2 monitor.
2. P4 is compositionally much farther from that monitor.
3. P4 is a replacement and reweighting of P2, not a proportional expansion.
4. More P4 tokens and supervised labels do not compensate for the changed
   distribution on this monitor.

### Most likely data mechanisms

1. **Monitor-alignment loss.** P4 reduces Swahili, classification, KenSwQuAD,
   Swahili News, KinNews, and Aya shares relative to what the validation set
   emphasizes. This is the strongest explanation for aggregate validation loss.
2. **Short-example dominance.** Halving mean record length and raising the
   at-most-128-token share to 81.92% can shift optimization toward local tagging
   and classification behavior at the expense of long-context instruction and
   comprehension behavior.
3. **Quality/source dilution.** Human-tier share falls 13.20 points while mixed
   and silver data increase. Several long-form news sources shrink sharply.
4. **Capacity-driven repetition.** Duplicate excess rises to 23.42%, with
   concentrated repetition in low-capacity language-task cells.
5. **Language trade-offs.** Equal language tokens remove Amharic exposure and
   substantially dilute previously overrepresented languages to fund Wolof,
   isiXhosa, isiZulu, Igbo, and others. Aggregate metrics can fall even while
   low-resource-language performance improves.

### Confidence in the user's hypothesis

- **Higher loss on the shared K10 monitor:** high confidence that composition
  is a major contributor.
- **Broad downstream performance degradation:** moderate-to-low confidence
  until matched benchmark results are available. The current monitor is too
  skewed to support that stronger conclusion.
- **Optimization or infrastructure as the primary cause:** low confidence.
  The controlled recipes match closely, training remained finite, and the P4
  resume restored all relevant state.

## Recommended experiments

Run these in order:

1. **Evaluate both final adapters on one frozen balanced suite.** Report every
   metric by language and task, with identical prompts and decoding settings.
   Include Wolof, which is absent from the current validation monitor.
2. **Add grouped validation.** Compute loss separately for each language,
   task, source, and sequence-length band. Aggregate loss alone cannot identify
   whether P4 trades high-resource performance for low-resource gains.
3. **Use a 2x2 composition-budget ablation.** Train P2 and P4 composition at
   both 31.35M and 50M tokens, using a token-based LR schedule shared across
   arms. This separates composition from token budget and schedule length.
4. **Test P4 without excess repetition.** Cap each unique record at one
   appearance, then redistribute only to genuinely novel records. Compare
   low-capacity cells against the accepted P4 run.
5. **Restore source-quality constraints.** Hold the P4 language balance fixed
   while increasing human-tier and long-form source shares toward P2 levels.
6. **Use a balanced validation monitor alongside the frozen P2 monitor.** Keep
   the old monitor for continuity, but do not use it as the sole model-selection
   criterion for a fixed-language training policy.

The minimum decisive experiment is item 1. If degradation is concentrated in
Swahili, Amharic, classification, and long-context tasks while Wolof and other
expanded languages improve, the result is a composition trade-off. If nearly
all language-task cells degrade, prioritize the short-example, repetition, and
quality hypotheses and run items 3 through 5.

## Evidence and provenance

Repository references:

- `data/data_mixture_registry.yaml`: canonical mixture identities and totals.
- `data/k10_p4_5m_per_language_profile.yaml`: P4 source and allocation policy.
- `gemma4_e2b_k10_p2_r16_2ep_nvidia_lr.yaml`: controlled P2 recipe.
- `gemma4_e2b_k10_p4_5m_r16_2ep_nvidia_lr.yaml`: controlled P4 recipe.
- `KSERIES_FIXED_LANGUAGE_DATA_REPRODUCTION.md`: P4 materialization and audit.
- `KSERIES_FIXED_LANGUAGE_TRAINING.md`: P4 launch and resume provenance.

Local signed/materialized artifacts:

- `/home/casper_neo/ext_data/Casper/kseries-next-run/data/k10-sft/mixture-p2-v1/summary.json`
- `/home/casper_neo/ext_data/Casper/kseries-next-run/data/k10-sft/mixture-p4-5m-per-language-v2/summary.json`
- `/home/casper_neo/ext_data/Casper/kseries-next-run/data/k10-sft/mixture-p4-5m-per-language-v2/final_audit.json`
- `/home/casper_neo/ext_data/Casper/kseries-next-run/checkpoints/gemma4-e2b-k10/p2-r16-2ep-nvidia-lr-v1/validation.jsonl`
- `/home/casper_neo/ext_data/Casper/kseries-next-run/checkpoints/gemma4-e2b-k10/p4-5m-per-language-r16-2ep-v1/validation.jsonl`

All composition calculations use prepared pre-shift Gemma processor tokens.
Record-level scans reconcile exactly to 31,349,312 P2 tokens and 50,000,000 P4
tokens. Exact overlap matching excludes only the stored `_prompt_tokens`,
`_label_tokens`, and `_text_tokens` fields from otherwise normalized records.