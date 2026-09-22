# K6 P2 complete data report

Date: 2026-09-08

## Executive summary

This report describes the accepted `mixture-p2-v2` K6 training data and its completed rank-16 NVIDIA-policy NeMo AutoModel run. This is a P2 quality-constrained mixture, **not** a fixed 5M-per-language P4 arm; no K6 P4 run exists.

| Metric | Value |
|---|---:|
| Languages / tasks | 6 / 5 |
| Physical examples / appearances | 153,273 |
| Unique normalized examples | 128,527 |
| Duplicate excess | 24,746 (16.15%) |
| Prepared text tokens | 31,348,990 |
| Model-input tokens per epoch | 31,195,717 |
| Prepared assistant-label tokens | 11,703,943 |
| Runtime-supervised tokens per epoch | 11,090,851 |
| Maximum appearances per normalized record | 4 |

The native coverage review accepted the mixture with 29/30 populated language-task cells, zero reported benchmark matches, and zero train-validation overlap. This report-time scan independently reconciles all rows, normalized identities, prepared tokens, prepared label metadata, and train-validation identity separation.

The rank-16 run completed two epochs and 2,116 optimizer steps without interruption. It processed 62,697,980 prepared tokens and finished at validation loss 2.007643 and perplexity 7.445746 on the K6 monitor.

## Scope and terminology

An **example** is one physical JSONL row; repeated rows count as separate appearances. A **unique example** uses NFKC normalization, case folding, whitespace collapse, ordered role/content pairs, and a 20-byte BLAKE2b digest. Provenance and token-count fields do not create a new identity.

Prepared text tokens are Gemma processor `_text_tokens` before causal shift. Model-input tokens exclude one position per physical row. Prepared assistant-label tokens are `_label_tokens` before runtime masking removes four assistant-format marker positions per row. The mixture has 11,703,943 prepared assistant-label tokens and 11,090,851 runtime-supervised positions per epoch; two epochs expose 22,181,702 runtime-supervised positions.

Languages are Hausa (`hau`), Igbo (`ibo`), Kinyarwanda (`kin`), Swahili (`swh`), Yoruba (`yor`), and isiZulu (`zul`). Tasks are instruction following, question answering, translation, classification, and named entity recognition.

## Examples by language and task

| Language | Instruction | QA | Translation | Classification | NER | Total |
|---|---:|---:|---:|---:|---:|---:|
| Hausa | 6,189 | 3,003 | 6,189 | 6,189 | 4,476 | 26,046 |
| Igbo | 6,189 | 3,003 | 6,189 | 6,189 | 4,476 | 26,046 |
| Kinyarwanda | 6,189 | 0 | 6,189 | 6,189 | 4,476 | 23,043 |
| Swahili | 6,189 | 3,003 | 6,189 | 6,189 | 4,476 | 26,046 |
| Yoruba | 6,189 | 3,003 | 6,189 | 6,189 | 4,476 | 26,046 |
| isiZulu | 6,189 | 3,003 | 6,189 | 6,189 | 4,476 | 26,046 |
| **Overall** | **37,134** | **15,015** | **37,134** | **37,134** | **26,856** | **153,273** |

Kinyarwanda QA is intentionally empty because no accepted independent source was available. All other 29 cells are populated.

## Prepared tokens by language and task

| Language | Instruction | QA | Translation | Classification | NER | Total |
|---|---:|---:|---:|---:|---:|---:|
| Hausa | 3,788,873 | 336,889 | 503,249 | 342,554 | 782,391 | 5,753,956 |
| Igbo | 647,806 | 608,583 | 383,825 | 353,610 | 2,019,587 | 4,013,411 |
| Kinyarwanda | 533,637 | 0 | 427,103 | 1,816,150 | 1,633,604 | 4,410,494 |
| Swahili | 1,008,475 | 1,525,487 | 484,433 | 2,136,532 | 1,815,900 | 6,970,827 |
| Yoruba | 3,641,428 | 522,634 | 610,235 | 445,801 | 901,058 | 6,121,156 |
| isiZulu | 774,218 | 1,771,104 | 525,023 | 311,453 | 697,348 | 4,079,146 |
| **Overall** | **10,394,437** | **4,764,697** | **2,933,868** | **5,406,100** | **7,849,888** | **31,348,990** |

## Language totals

| Language | Examples | Unique | Prepared tokens | Prepared label tokens |
|---|---:|---:|---:|---:|
| Hausa | 26,046 | 21,703 | 5,753,956 | 1,478,036 |
| Igbo | 26,046 | 19,744 | 4,013,411 | 1,932,668 |
| Kinyarwanda | 23,043 | 21,653 | 4,410,494 | 1,208,462 |
| Swahili | 26,046 | 24,969 | 6,970,827 | 1,604,949 |
| Yoruba | 26,046 | 20,262 | 6,121,156 | 4,378,118 |
| isiZulu | 26,046 | 20,196 | 4,079,146 | 1,101,710 |
| **Overall** | **153,273** | **128,527** | **31,348,990** | **11,703,943** |

## Task totals

| Task | Examples | Unique | Prepared tokens | Prepared label tokens |
|---|---:|---:|---:|---:|
| Instruction | 37,134 | 33,582 | 10,394,437 | 5,149,895 |
| QA | 15,015 | 8,776 | 4,764,697 | 1,225,744 |
| Translation | 37,134 | 34,487 | 2,933,868 | 1,503,611 |
| Classification | 37,134 | 29,621 | 5,406,100 | 239,982 |
| NER | 26,856 | 22,061 | 7,849,888 | 3,584,711 |
| **Overall** | **153,273** | **128,527** | **31,348,990** | **11,703,943** |

## Source datasets and creation methods

Quality tier is a pool-level allocation classification, not a guarantee that every row has the same creation method or quality.

| Pool | Upstream dataset/source | Method and evidence | Revision | License | Tier |
|---|---:|---:|---:|---:|---:|
| `afriinstruct` | llama-lang-adapt/AfriInstruct-Data | **Mixed.** xP3-led aggregation of human, translated, templated, and synthetic-style datasets; no reliable row-level method labels or proportions. | `0a6325e6806ffcd7b3baefcc3c6b2de435ac6e1e` | mixed-review-required | mixed |
| `aya` | CohereLabs/aya_dataset | **Human tier.** Human-authored annotations and human edits of generated material; QA assignment is heuristic because Aya has no task labels. | `f9ea04583f02a8f86404ff6c58bf75fe637df8a2` | Apache-2.0 | human |
| `kenswquad` | Kencorpus/KenSwQuAD | **Human tier.** Native speakers authored extractive Swahili QA; documented quality review covered a 12.5% sample. | `a7f211970d5929c438c7cf3e971d8d34a7292d6e` | CC-BY-4.0 | human |
| `afrihate` | afrihate/afrihate | **Human tier.** Native-speaker hate, abuse, and normal annotation; code mixing is intentionally retained. | `bb341d2234e404c1efa355ad2d7973f74ca4c140` | Apache-2.0 | human |
| `kinnews` | saradhix/kinnews_kirnews, KINNEWS | **Curated tier.** Cleaned, labeled Kinyarwanda news with 14 topics; local evidence does not establish the label-production protocol. | `97dce354a363f853fa94c439dd42135fc31618f0` | MIT | curated |
| `swahili_news` | Mollel/SwahiliNewsClassification | **Human tier, mixed provenance.** Card narrative says human annotation while metadata also lists machine-generated and crowdsourced annotations; rows lack origin flags. | `24fcf066e6b96f9e0d743e8b79184e0c599f73c3` | Apache-2.0 | human |
| `afridocmt` | masakhane/AfriDocMT | **Human tier.** Technology and health documents described as human-translated; translator protocol is sparse and licenses differ by domain. | `8ae06a3f73aa9194a0d29b32e6c1037b3f9420fb` | CC-BY-NC-SA-3.0-and-CC-BY-NC-4.0 | human |
| `pontoon` | ayymen/Pontoon-Translations | **Curated tier.** Crowdsourced Mozilla software localization; detailed contributor and review protocol is unavailable locally. | `7318c7cac7cdd5e2e7efc7e86d84256aaf54c8dc` | MPL-2.0 | curated |
| `digital_umuganda` | DigitalUmuganda/kinyarwanda-english-machine-translation-dataset | **Curated tier.** English-Kinyarwanda parallel text; the dataset name does not establish row-level human-versus-machine generation. | `a69005be8d20e2982fedbe0474c5a653adf290f8` | CC-BY-4.0 | curated |
| `hausa_voa_ner` | uds-lsv/transfer-distant-transformer-african, Hausa VOA NER | **Human tier.** Existing BIO-tagged Hausa news entities converted to JSON spans; original annotation protocol is not reproduced locally. | `a6cbe619a7a18309bbd7a3813033b180a043c1a8` | CC-BY-4.0 | human |
| `yoruba_gv_ner` | ajesujoba/YorubaTwi-Embedding, Yoruba NER | **Human tier.** Existing BIO tags converted to JSON spans; original annotation and QA details are sparse locally. | `5f5c76d9ea6a6a7205e53c842c61af719aa8b8c5` | CC-BY-3.0 | human |
| `nchlt_zulu_ner` | SADiLaR NCHLT isiZulu Named Entity Annotated Corpus | **Human tier.** Government-domain BIO annotation converted to JSON spans; records with known mojibake are removed. | `4f9fee744c6d5ca9fcc921c621386a81ccfa2837` | CC-BY-2.5-ZA | human |

No pool is supportably classified as wholly synthetic. AfriInstruct is heterogeneous and Swahili News has mixed annotation provenance. Reliable row-level human/synthetic labels are unavailable, so the report does not fabricate synthetic-example counts.

## Source totals and quality coverage

| Pool | Examples | Unique | Prepared tokens | Prepared label tokens | Languages | Tasks | Quality |
|---|---:|---:|---:|---:|---:|---:|---:|
| `afriinstruct` | 62,410 | 62,410 | 14,291,532 | 5,008,247 | hau, ibo, kin, swh, yor, zul | instruction, qa, translation, classification, ner | mixed |
| `aya` | 22,234 | 12,443 | 7,861,843 | 5,127,026 | hau, ibo, swh, yor, zul | instruction, qa | human |
| `kenswquad` | 1,408 | 1,408 | 1,301,878 | 14,168 | swh | qa | human |
| `afrihate` | 25,015 | 17,502 | 1,459,091 | 164,018 | hau, ibo, kin, swh, yor, zul | classification | human |
| `kinnews` | 2,063 | 2,063 | 1,519,773 | 12,378 | kin | classification | curated |
| `swahili_news` | 2,321 | 2,321 | 1,923,014 | 17,176 | swh | classification | human |
| `afridocmt` | 10,316 | 10,316 | 1,045,857 | 630,754 | hau, swh, yor, zul | translation | human |
| `pontoon` | 14,666 | 12,019 | 757,140 | 396,827 | hau, ibo, kin, swh, yor, zul | translation | curated |
| `digital_umuganda` | 2,870 | 2,870 | 139,223 | 59,767 | kin | translation | curated |
| `hausa_voa_ner` | 3,357 | 1,004 | 335,512 | 96,629 | hau | ner | human |
| `yoruba_gv_ner` | 3,256 | 814 | 346,640 | 68,736 | yor | ner | human |
| `nchlt_zulu_ner` | 3,357 | 3,357 | 367,487 | 108,217 | zul | ner | human |
| **Overall** | **153,273** | **128,527** | **31,348,990** | **11,703,943** | **6** | **5** | — |

Quality-tier rollup:

| Quality | Examples | Unique | Prepared tokens | Prepared label tokens | Token share |
|---|---:|---:|---:|---:|---:|
| Curated | 19,599 | 16,952 | 2,416,136 | 468,972 | 7.71% |
| Human | 71,264 | 49,165 | 14,641,322 | 6,226,724 | 46.70% |
| Mixed | 62,410 | 62,410 | 14,291,532 | 5,008,247 | 45.59% |
| **Overall** | **153,273** | **128,527** | **31,348,990** | **11,703,943** | **100.00%** |

## Selection and preprocessing

P2 uses capped examples-proportional sampling with temperature 2, source-quality weights, a default 50% source-family cap, and a soft 25% AfriInstruct anchor. Capacity shortfalls may increase fallback share, but no source record may appear more than four times. Pool allocations use largest-remainder rounding and stable digest ordering.

Source adapters normalize records to user/assistant conversations with language, task, and source provenance. Translation direction, classification label, QA subtype, and NER entity metadata are retained where available. NER BIO tags are converted to JSON spans; known NCHLT mojibake is removed.

Profiling rejects malformed or out-of-scope rows, missing supervised assistant content, invalid token counts, sequences over 4,096 tokens, benchmark matches, and normalized cross-pool duplicates. Training is selected first; validation uses only unused unique records.

## Filtering and materialization results

The accepted mixture is 4,543 prepared tokens below the 31,353,533-token plan, a -0.014% deviation. Realized total is 31,348,990.

The accepted profile reported 6,333 duplicates and 259 overlength rows; materialization reported 6,332 duplicates and 260 overlength rows because check ordering differs. The profile also rejected 673,631 Belebele matches and two AfriMMLU/AfriXNLI matches. These are source-occurrence counters, not distinct benchmark-document counts.

The mixture contains 24,746 repeated appearances above 128,527 unique examples (16.15% duplicate excess), with maximum multiplicity four. Deterministic review found wrong-language and mixed-language xP3 outputs. The accepted decision was to retain them, along with code-mixed AfriHate inputs, while keeping AfriInstruct marked `mixed-review-required`.

## Validation and contamination

Validation contains 23,470 unique examples and 8,198,968 prepared tokens. Its 1,226,055 prepared assistant-label tokens become 1,132,175 runtime-supervised positions after marker masking. The coverage review reports zero train-validation overlap and zero accepted benchmark matches; the report-time normalized-message scan independently confirms zero train-validation identity overlap.

Validation by language:

| Language | Examples | Prepared tokens | Prepared label tokens | Token share |
|---|---:|---:|---:|---:|
| Hausa | 682 | 65,190 | 37,595 | 0.80% |
| Igbo | 411 | 41,405 | 18,031 | 0.51% |
| Kinyarwanda | 5,000 | 1,978,202 | 66,629 | 24.13% |
| Swahili | 9,821 | 4,684,126 | 135,784 | 57.13% |
| Yoruba | 4,404 | 1,094,582 | 851,783 | 13.35% |
| isiZulu | 3,152 | 335,463 | 116,233 | 4.09% |
| **Overall** | **23,470** | **8,198,968** | **1,226,055** | **100.00%** |

Validation by task:

| Task | Examples | Prepared tokens | Prepared label tokens | Token share |
|---|---:|---:|---:|---:|
| Instruction | 3,458 | 1,014,134 | 789,984 | 12.37% |
| QA | 2,512 | 2,335,925 | 25,712 | 28.49% |
| Translation | 7,500 | 516,023 | 279,928 | 6.29% |
| Classification | 7,500 | 4,058,195 | 49,603 | 49.50% |
| NER | 2,500 | 274,691 | 80,828 | 3.35% |
| **Overall** | **23,470** | **8,198,968** | **1,226,055** | **100.00%** |

This monitor is suitable for comparisons that reuse it, but it is not equally balanced across languages, tasks, sources, or sequence lengths.

## Completed rank-16 NVIDIA NeMo training

The completed run used NeMo AutoModel `FinetuneRecipeForVLM` and started independently from the pinned Gemma base.

| Field | Value |
|---|---:|
| Run / W&B | `gemma4-e2b-k6-p2-r16-2ep-nvidia-lr-v1` / `m8r79yjp` |
| Base model / revision | `google/gemma-4-E2B-it` / `3e22461f65e89153144f8adb70e3b8c2cc9845a7` |
| Precision / attention | BF16 / SDPA |
| Distributed strategy | FSDP2, one process, one NVIDIA RTX A6000 |
| LoRA | Rank 16, alpha 32, dropout 0.0 |
| Trainable parameters | 26,333,184 of 5,150,690,848 (0.51%) |
| Optimizer | AdamW, peak LR `2e-4`, weight decay 0.01, betas 0.9/0.95 |
| Schedule | Cosine from `2e-5` to `2e-6`; 211 warmup steps |
| Batch / packing | Global 8, local 1, pack length 4,096 |
| Seed | 42 |
| Epochs / optimizer steps | 2 / 2,116 |
| Prepared-token exposure | 62,697,980 |
| Model-input-token exposure | 62,391,434 |
| Derived runtime-supervised exposure | 22,181,702 |
| Final / best checkpoint | `epoch_1_step_2115` |

| Metric | Value |
|---|---:|
| First / final train loss | 3.923018 / 1.738887 |
| First / final train perplexity | 50.552782 / 5.691006 |
| Best / final validation step | 2,115 / 2,115 |
| Best / final validation loss | 2.007643 / 2.007643 |
| Best / final validation perplexity | 7.445746 / 7.445746 |
| Median throughput | 1,408.42 tokens/s |
| Peak logged GPU allocation | 30.365 GiB |
| Active runtime | 23.8301 hours |

The run completed normally without interruption, OOM, resume, or replay. Four persistent train workers and two persistent validation workers consumed approximately 66 GiB host RAM; this worker policy was not reused for later experiments.

## Integrity and reproducibility

| Artifact | SHA-256 |
|---|---:|
| Profile config | `fe5418c57443edd845649c23cf57a1326bcaa203b8807aacac51c78547b6cfc1` |
| Accepted profile | `fc371653d7e0f33121d526a9d09d2f8825f6081c8e1a9f7a514a1e53a9687b93` |
| Accepted plan | `266d571a30196501542556f86fce4233a9ce67fa06427ee7914adda829fba158` |
| Benchmark blocklist | `b318c8f7e7392e7e905ba88f6bba9680d47006e1fc258f4acfd08952e4c75808` |
| Materialized summary | `ac39fc5d90a2f53869fb790334472b2f9ce36482a0a82dbce226afd6aad4331c` |
| Train manifest | `37f2abcb40d91700d2969cb962f99423eaba00e72519b7e1cc2ed2a57e516a02` |
| Validation manifest | `09142fdc9c39b05bb05b3de680433db85e477a688bcf269b9224f3256bc40252` |
| Training recipe | `9f5d43818da2dede6d52625fda7e65605fd0ce49afde327f3656cb4b43c112fe` |
| Final adapter | `55fde2417930a4288c0a8d5403f28d39333903eacb324776d37c81683cba7074` |
| Pinned NeMo image | `7213eab8055a2029ce1ef9022384a780f9095b9577f32edca4c48d303907421f` |

K6 P2 predates the later standalone `final_audit.json` format. Its native authority is `coverage_review.md` plus the materialized summary/manifests; this report adds a read-only row-level reconciliation but does not relabel it as the original acceptance audit.

### Train shard hashes

| Shard | SHA-256 |
|---|---|
| `processed/train/afridocmt-00000.jsonl` | `14868caa3676e0432fb90bbbdfdf137833a211c9fbcc05826a8ba3306f52e442` |
| `processed/train/afrihate-00000.jsonl` | `6a6f2c0b8791f890077fabc784effd6293bf46a2dea5f37fd2a528552e5ca2c9` |
| `processed/train/afriinstruct-00000.jsonl` | `ec58856ad47984f80d947423877fe544a7eafc18286720d1e90b2c21d86cc90a` |
| `processed/train/aya-00000.jsonl` | `db7907435f3c6c432127022d67688833c7af2e83ef1f2ae6390e644a1d7e3191` |
| `processed/train/digital_umuganda-00000.jsonl` | `8ed93b8c81a4a241839e8ef525ef5a7b12d305286f1e7b57b925685b81b1ef85` |
| `processed/train/hausa_voa_ner-00000.jsonl` | `ddb26ed1cf81328df45e3e012bf36b3967894d2c9ff02bc69ab4e3042116bd0b` |
| `processed/train/kenswquad-00000.jsonl` | `279ec2aad3f8fc9e3bf0f8bfd49ba5d526a0687b942695164a3b84197cae9628` |
| `processed/train/kinnews-00000.jsonl` | `06bfd8580913b67269ff6d15c1688bb86b1781c0fb21f11a0ee36551457d64ec` |
| `processed/train/nchlt_zulu_ner-00000.jsonl` | `e961420f16e17fc73fda020795ec81de0e7dc805f85d0a3ef916cb132d870ff0` |
| `processed/train/pontoon-00000.jsonl` | `09f8fcf5ecda3d85102be8f656de16e2ea673d15c0c0d5945cb36b8d57e02f4a` |
| `processed/train/swahili_news-00000.jsonl` | `9f7bc79b3baa1adc71a42230777e2674b105f6237a96f2350fb508e3e4c64daa` |
| `processed/train/yoruba_gv_ner-00000.jsonl` | `fbe12a01123ce4f4930755832aff1d8b84e2d907ad302239caacd1f7042c026b` |

### Validation shard hashes

| Shard | SHA-256 |
|---|---|
| `processed/validation/afridocmt-00000.jsonl` | `311657a8d7eb5bba189c5ee627b5c4fb2cd58476c3cc7c0da9c1e5ecd42bf20f` |
| `processed/validation/afrihate-00000.jsonl` | `aff78809153cba8799b69339f25916a63b885e6730a20cb9e40a57bbcc0ca124` |
| `processed/validation/afriinstruct-00000.jsonl` | `8c018f6396fa8686c46905f645b05899e5a32f72dc23177a4d0d19c5a7df2c31` |
| `processed/validation/aya-00000.jsonl` | `4f21ec1096fdf376c0cf9ecda8a3181299b60295c5c788e5f90be9d8ec92766d` |
| `processed/validation/digital_umuganda-00000.jsonl` | `0615360c9e6994b8e30bb8683788ba2e515ba7c0d7e7d7634d4b8b4c9161bab5` |
| `processed/validation/kenswquad-00000.jsonl` | `8a7155f43f485d10425f11e67255f330a60f03fb51d43a22c2190dfb8450cb0b` |
| `processed/validation/kinnews-00000.jsonl` | `aab424a24e1aba1c0e45d8f6ead894d26980f61603623ee8fe848546cdfae241` |
| `processed/validation/nchlt_zulu_ner-00000.jsonl` | `fd5ea4a46b721aa63f51a40d9b10f7f92d51c055a94df641ec0e9cbd3d0faba8` |
| `processed/validation/pontoon-00000.jsonl` | `5313f364bda4aff034b92098be8827d01bf5af91e611ccd17de34cbf51137fc8` |
| `processed/validation/swahili_news-00000.jsonl` | `10b7494a23c440a6b7f59e8749137ef2f9d6b0be23cdd5cf0889eabc40880607` |

## Known limitations

- Kinyarwanda QA is absent; several other QA cells depend heavily on repeated Aya records.
- AfriInstruct has heterogeneous creation methods and unresolved mixed licensing; exact human/synthetic proportions are unknown.
- Accepted xP3 rows include audited wrong-language, mixed-language, and related-language outputs.
- AfriHate includes intentional code mixing and substantial language-ID disagreement in some languages.
- Quality tiers are pool-level planning labels, not row-level guarantees.
- Repeated examples consume compute without equivalent new information.
- Some configured revisions are recorded but not enforced by static upstream download URLs.
- The validation monitor is not balanced across languages, tasks, sources, or sequence lengths.
- High persistent-worker host memory is an operational defect of this historical recipe, not a data-quality property.
- Validation loss does not establish balanced downstream quality or deployment suitability.

## Appendix: source pool by language and task

| Pool | Language | Task | Records | Unique | Prepared tokens | Prepared label tokens | Quality | Max repetitions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `afriinstruct` | hau | instruction | 4,657 | 4,657 | 3,261,145 | 256,754 | mixed | 1 |
| `afriinstruct` | hau | translation | 1,547 | 1,547 | 167,644 | 70,660 | mixed | 1 |
| `afriinstruct` | hau | classification | 1,547 | 1,547 | 83,620 | 9,282 | mixed | 1 |
| `afriinstruct` | hau | ner | 1,119 | 1,119 | 446,879 | 219,821 | mixed | 1 |
| `afriinstruct` | ibo | instruction | 4,401 | 4,401 | 499,174 | 208,993 | mixed | 1 |
| `afriinstruct` | ibo | translation | 1,547 | 1,547 | 150,510 | 63,100 | mixed | 1 |
| `afriinstruct` | ibo | classification | 1,547 | 1,547 | 87,355 | 9,282 | mixed | 1 |
| `afriinstruct` | ibo | ner | 4,476 | 4,476 | 2,019,587 | 1,025,413 | mixed | 1 |
| `afriinstruct` | kin | instruction | 6,189 | 6,189 | 533,637 | 218,228 | mixed | 1 |
| `afriinstruct` | kin | translation | 1,547 | 1,547 | 185,852 | 77,003 | mixed | 1 |
| `afriinstruct` | kin | classification | 1,547 | 1,547 | 104,460 | 9,282 | mixed | 1 |
| `afriinstruct` | kin | ner | 4,476 | 4,476 | 1,633,604 | 764,482 | mixed | 1 |
| `afriinstruct` | swh | instruction | 5,597 | 5,597 | 941,307 | 275,516 | mixed | 1 |
| `afriinstruct` | swh | qa | 751 | 751 | 107,501 | 6,789 | mixed | 1 |
| `afriinstruct` | swh | translation | 1,547 | 1,547 | 141,553 | 57,944 | mixed | 1 |
| `afriinstruct` | swh | classification | 1,547 | 1,547 | 96,971 | 9,282 | mixed | 1 |
| `afriinstruct` | swh | ner | 4,476 | 4,476 | 1,815,900 | 890,181 | mixed | 1 |
| `afriinstruct` | yor | instruction | 1,547 | 1,547 | 210,171 | 81,175 | mixed | 1 |
| `afriinstruct` | yor | translation | 1,547 | 1,547 | 177,600 | 75,429 | mixed | 1 |
| `afriinstruct` | yor | classification | 1,547 | 1,547 | 131,816 | 9,282 | mixed | 1 |
| `afriinstruct` | yor | ner | 1,220 | 1,220 | 554,418 | 271,944 | mixed | 1 |
| `afriinstruct` | zul | instruction | 5,365 | 5,365 | 442,478 | 186,990 | mixed | 1 |
| `afriinstruct` | zul | translation | 1,547 | 1,547 | 168,489 | 72,127 | mixed | 1 |
| `afriinstruct` | zul | ner | 1,119 | 1,119 | 329,861 | 139,288 | mixed | 1 |
| `aya` | hau | instruction | 1,532 | 383 | 527,728 | 450,476 | human | 4 |
| `aya` | hau | qa | 3,003 | 3,003 | 336,889 | 160,871 | human | 1 |
| `aya` | ibo | instruction | 1,788 | 447 | 148,632 | 107,132 | human | 4 |
| `aya` | ibo | qa | 3,003 | 969 | 608,583 | 359,529 | human | 4 |
| `aya` | swh | instruction | 592 | 148 | 67,168 | 52,660 | human | 4 |
| `aya` | swh | qa | 844 | 211 | 116,108 | 84,624 | human | 4 |
| `aya` | yor | instruction | 4,642 | 4,642 | 3,431,257 | 3,246,071 | human | 1 |
| `aya` | yor | qa | 3,003 | 978 | 522,634 | 311,058 | human | 4 |
| `aya` | zul | instruction | 824 | 206 | 331,740 | 65,900 | human | 4 |
| `aya` | zul | qa | 3,003 | 1,456 | 1,771,104 | 288,705 | human | 3 |
| `kenswquad` | swh | qa | 1,408 | 1,408 | 1,301,878 | 14,168 | human | 1 |
| `afrihate` | hau | classification | 4,642 | 3,801 | 258,934 | 30,483 | human | 4 |
| `afrihate` | ibo | classification | 4,642 | 3,033 | 266,255 | 30,359 | human | 4 |
| `afrihate` | kin | classification | 2,579 | 2,518 | 191,917 | 17,194 | human | 2 |
| `afrihate` | swh | classification | 2,321 | 2,321 | 116,547 | 15,474 | human | 1 |
| `afrihate` | yor | classification | 4,642 | 3,325 | 313,985 | 30,227 | human | 4 |
| `afrihate` | zul | classification | 6,189 | 2,504 | 311,453 | 40,281 | human | 4 |
| `kinnews` | kin | classification | 2,063 | 2,063 | 1,519,773 | 12,378 | curated | 1 |
| `swahili_news` | swh | classification | 2,321 | 2,321 | 1,923,014 | 17,176 | human | 1 |
| `afridocmt` | hau | translation | 2,579 | 2,579 | 245,241 | 141,291 | human | 1 |
| `afridocmt` | swh | translation | 2,579 | 2,579 | 231,142 | 126,271 | human | 1 |
| `afridocmt` | yor | translation | 2,579 | 2,579 | 314,401 | 213,142 | human | 1 |
| `afridocmt` | zul | translation | 2,579 | 2,579 | 255,073 | 150,050 | human | 1 |
| `pontoon` | hau | translation | 2,063 | 2,063 | 90,364 | 41,769 | curated | 1 |
| `pontoon` | ibo | translation | 4,642 | 3,324 | 233,315 | 128,860 | curated | 2 |
| `pontoon` | kin | translation | 1,772 | 443 | 102,028 | 50,128 | curated | 4 |
| `pontoon` | swh | translation | 2,063 | 2,063 | 111,738 | 54,864 | curated | 1 |
| `pontoon` | yor | translation | 2,063 | 2,063 | 118,234 | 71,054 | curated | 1 |
| `pontoon` | zul | translation | 2,063 | 2,063 | 101,461 | 50,152 | curated | 1 |
| `digital_umuganda` | kin | translation | 2,870 | 2,870 | 139,223 | 59,767 | curated | 1 |
| `hausa_voa_ner` | hau | ner | 3,357 | 1,004 | 335,512 | 96,629 | human | 4 |
| `yoruba_gv_ner` | yor | ner | 3,256 | 814 | 346,640 | 68,736 | human | 4 |
| `nchlt_zulu_ner` | zul | ner | 3,357 | 3,357 | 367,487 | 108,217 | human | 1 |

## Authorities

- `examples/vlm_finetune/gemma4/data/data_mixture_registry.yaml`
- `examples/vlm_finetune/gemma4/README_k6_sft_preprocessing.md`
- `examples/vlm_finetune/gemma4/KSERIES_TWO_EPOCH_RUN_REPORT.md`
- `examples/vlm_finetune/gemma4/gemma4_e2b_k6_p2_r16_2ep.yaml`
- `/home/casper_neo/ext_data/Casper/kseries-next-run/data/k6-sft/k6_sft_profile.yaml`
- `/home/casper_neo/ext_data/Casper/kseries-next-run/data/k6-sft/profile-v2/profile.json`
- `/home/casper_neo/ext_data/Casper/kseries-next-run/data/k6-sft/plans-v4/plans.json`
- `/home/casper_neo/ext_data/Casper/kseries-next-run/data/k6-sft/coverage_review.md`
- `/home/casper_neo/ext_data/Casper/kseries-next-run/data/k6-sft/mixture-p2-v2/summary.json`
- `/home/casper_neo/ext_data/Casper/kseries-next-run/data/k6-sft/mixture-p2-v2/train_meta.json`
- `/home/casper_neo/ext_data/Casper/kseries-next-run/data/k6-sft/mixture-p2-v2/validation_meta.json`
- `/home/casper_neo/ext_data/Casper/kseries-next-run/checkpoints/gemma4-e2b-k6/p2-r16-2ep-nvidia-lr-v1/training.jsonl`
- `/home/casper_neo/ext_data/Casper/kseries-next-run/checkpoints/gemma4-e2b-k6/p2-r16-2ep-nvidia-lr-v1/validation.jsonl`
- `tools/prepare_k6_sft_sources.py`
- `tools/profile_sft_mixture.py`
- `tools/build_k6_sft_mixture.py`
