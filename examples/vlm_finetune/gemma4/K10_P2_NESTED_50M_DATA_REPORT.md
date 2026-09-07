# K10 P2 nested 50M complete data report

Date: 2026-09-07

## Executive summary

This report describes the accepted `mixture-p2-nested-50m-v1` training data.
The mixture keeps the complete accepted K10 P2 v1 dataset byte-for-byte and
adds a disjoint extension from the same 15-source registry.

| Metric | Locked base | Unique extension | Combined |
|---|---:|---:|---:|
| Physical examples / appearances | 143,579 | 111,375 | 254,954 |
| Unique normalized examples | 122,312 | 111,375 | 233,687 |
| Prepared text tokens | 31,349,312 | 18,650,688 | 50,000,000 |
| Supervised label tokens | 9,511,567 | 5,700,057 | 15,211,624 |

The extension contains 8,694,190 human-tier tokens (46.62%), 1,918,657
curated-tier tokens (10.29%), and 8,037,841 mixed-tier tokens (43.10%). It
contains no silver or synthetic-tier pool. The combined mixture contains
24,349,617 human-tier tokens (48.70%), 3,664,783 curated-tier tokens (7.33%),
and 21,985,600 mixed-tier tokens (43.97%).

The independent audit accepted the dataset with zero base-extension overlap,
zero train-validation overlap, zero benchmark matches, one appearance per
extension record, and at most four combined appearances per normalized record.

## Scope and terminology

An **example** is one physical JSONL row. Repeated base rows count as multiple
examples because they create multiple training appearances. A **unique
example** is identified by the ordered roles and canonicalized message text;
language, task, source, and token metadata are not part of that identity.

Prepared text tokens are exact Gemma processor `_text_tokens` before the VLM
causal shift. Supervised label tokens are positions contributing to the
answer-only loss. Packed model-input exposure during training is a different
metric and is not yet available because this dataset has not been trained.

The configured languages are Amharic (`amh`), Hausa (`hau`), Igbo (`ibo`),
Kinyarwanda (`kin`), Shona (`sna`), Swahili (`swh`), Wolof (`wol`), isiXhosa
(`xho`), Yoruba (`yor`), and isiZulu (`zul`). Tasks are instruction following,
question answering, translation, classification, and named entity recognition.

## Examples by language and task

These are combined physical examples, including accepted base appearances and
the unique extension.

| Language | Instruction | QA | Translation | Classification | NER | Total |
|---|---:|---:|---:|---:|---:|---:|
| Amharic | 5,148 | 1,773 | 10,569 | 9,164 | 0 | 26,654 |
| Hausa | 4,322 | 2,426 | 10,160 | 6,466 | 3,777 | 27,151 |
| Igbo | 4,147 | 1,773 | 4,611 | 5,206 | 5,210 | 20,947 |
| Kinyarwanda | 5,201 | 0 | 5,404 | 7,318 | 5,232 | 23,155 |
| Shona | 4,623 | 1,773 | 3,300 | 4,218 | 6,011 | 19,925 |
| Swahili | 4,805 | 3,617 | 5,313 | 15,734 | 5,242 | 34,711 |
| Wolof | 3,300 | 2,004 | 6,598 | 0 | 4,593 | 16,495 |
| isiXhosa | 4,886 | 836 | 12,206 | 5,408 | 5,823 | 29,159 |
| Yoruba | 5,045 | 1,773 | 11,610 | 4,872 | 3,776 | 27,076 |
| isiZulu | 4,769 | 1,773 | 9,859 | 3,667 | 9,613 | 29,681 |
| **Overall** | **46,246** | **17,748** | **79,630** | **62,053** | **49,277** | **254,954** |

The three intentional empty cells inherited from P2 are Amharic NER,
Kinyarwanda QA, and Wolof classification.

## Prepared tokens by language and task

| Language | Instruction | QA | Translation | Classification | NER | Total |
|---|---:|---:|---:|---:|---:|---:|
| Amharic | 6,701,386 | 154,456 | 982,603 | 1,797,443 | 0 | 9,635,888 |
| Hausa | 2,459,042 | 275,196 | 886,951 | 364,079 | 763,876 | 4,749,144 |
| Igbo | 418,252 | 358,655 | 295,408 | 293,010 | 2,365,191 | 3,730,516 |
| Kinyarwanda | 446,594 | 0 | 366,990 | 2,335,455 | 1,914,221 | 5,063,260 |
| Shona | 470,899 | 317,614 | 277,839 | 1,951,879 | 986,762 | 4,004,993 |
| Swahili | 784,406 | 2,305,321 | 414,277 | 2,846,368 | 2,067,550 | 8,417,922 |
| Wolof | 466,376 | 155,433 | 269,552 | 0 | 414,500 | 1,305,861 |
| isiXhosa | 444,651 | 175,072 | 646,428 | 1,242,614 | 778,347 | 3,287,112 |
| Yoruba | 2,923,989 | 313,386 | 1,296,893 | 347,600 | 843,306 | 5,725,174 |
| isiZulu | 659,134 | 1,044,460 | 896,939 | 184,525 | 1,295,072 | 4,080,130 |
| **Overall** | **15,774,729** | **5,099,593** | **6,333,880** | **11,362,973** | **11,428,825** | **50,000,000** |

## Language totals and origin

| Language | Base examples | Extension examples | Combined examples | Unique combined | Prepared tokens | Label tokens |
|---|---:|---:|---:|---:|---:|---:|
| Amharic | 12,040 | 14,614 | 26,654 | 25,820 | 9,635,888 | 1,053,211 |
| Hausa | 15,324 | 11,827 | 27,151 | 24,078 | 4,749,144 | 1,575,972 |
| Igbo | 15,324 | 5,623 | 20,947 | 18,139 | 3,730,516 | 1,819,285 |
| Kinyarwanda | 13,551 | 9,604 | 23,155 | 22,361 | 5,063,260 | 1,283,808 |
| Shona | 15,324 | 4,601 | 19,925 | 15,776 | 4,004,993 | 924,112 |
| Swahili | 15,324 | 19,387 | 34,711 | 33,813 | 8,417,922 | 1,671,767 |
| Wolof | 11,657 | 4,838 | 16,495 | 14,070 | 1,305,861 | 636,591 |
| isiXhosa | 14,387 | 14,772 | 29,159 | 27,496 | 3,287,112 | 870,296 |
| Yoruba | 15,324 | 11,752 | 27,076 | 24,551 | 5,725,174 | 4,045,673 |
| isiZulu | 15,324 | 14,357 | 29,681 | 27,583 | 4,080,130 | 1,330,909 |
| **Overall** | **143,579** | **111,375** | **254,954** | **233,687** | **50,000,000** | **15,211,624** |

## Task totals

| Task | Base examples | Extension examples | Combined examples | Prepared tokens | Label tokens |
|---|---:|---:|---:|---:|---:|
| Instruction | 33,000 | 13,246 | 46,246 | 15,774,729 | 5,305,371 |
| QA | 15,020 | 2,728 | 17,748 | 5,099,593 | 1,191,236 |
| Translation | 33,000 | 46,630 | 79,630 | 6,333,880 | 3,400,349 |
| Classification | 33,003 | 29,050 | 62,053 | 11,362,973 | 396,843 |
| NER | 29,556 | 19,721 | 49,277 | 11,428,825 | 4,917,825 |
| **Overall** | **143,579** | **111,375** | **254,954** | **50,000,000** | **15,211,624** |

## Source datasets and creation methods

Quality tier is the reviewed registry classification used for allocation. It
does not imply that every row in a pool has the same creation method.

| Pool | Upstream dataset | Method / quality evidence | Revision | License |
|---|---|---|---|---|
| `afriinstruct` | `llama-lang-adapt/AfriInstruct-Data` | **Mixed.** Aggregates predominantly xP3 plus AfriSenti, MasakhaNEWS, XL-Sum, FLORES, MAFAND, MasakhaNER2.0, MENYO, NollySenti, and others. It includes heterogeneous human, translated, templated, and synthetic-style instruction data. Exact human/synthetic proportions are unavailable. | `0a6325e6806ffcd7b3baefcc3c6b2de435ac6e1e` | Mixed; review required |
| `aya` | `CohereLabs/aya_dataset` | **Human tier.** Annotated multilingual prompts and completions. Heuristics separate open-ended QA from general instruction rows while retaining annotation metadata. | `f9ea04583f02a8f86404ff6c58bf75fe637df8a2` | Apache-2.0 |
| `kenswquad` | `Kencorpus/KenSwQuAD` | **Human tier.** Manually authored Swahili extractive questions and answers; a documented 12.5% sample received quality review. | `a7f211970d5929c438c7cf3e971d8d34a7292d6e` | CC-BY-4.0 |
| `afrihate` | `afrihate/afrihate` | **Human tier.** Annotated hate, abuse, and normal text classification data. | `bb341d2234e404c1efa355ad2d7973f74ca4c140` | Apache-2.0 |
| `kinnews` | `saradhix/kinnews_kirnews`, KINNEWS | **Curated tier.** Cleaned Kinyarwanda news articles with 14 topic labels. | `97dce354a363f853fa94c439dd42135fc31618f0` | MIT |
| `swahili_news` | `Mollel/SwahiliNewsClassification` | **Human tier.** Swahili news content with topic labels. | `24fcf066e6b96f9e0d743e8b79184e0c599f73c3` | Apache-2.0 |
| `afridocmt` | `masakhane/AfriDocMT` | **Human tier.** English-to-target document translation in technology and health domains. Exact translator protocol is not fully documented locally. | `8ae06a3f73aa9194a0d29b32e6c1037b3f9420fb` | CC-BY-NC-SA-3.0 and CC-BY-NC-4.0 |
| `pontoon` | `ayymen/Pontoon-Translations` | **Curated tier.** English-to-target software-localization translations. | `7318c7cac7cdd5e2e7efc7e86d84256aaf54c8dc` | MPL-2.0 |
| `digital_umuganda` | `DigitalUmuganda/kinyarwanda-english-machine-translation-dataset` | **Curated tier.** Prepared English-Kinyarwanda parallel text. The local evidence does not establish a row-level human versus machine-generation split. | `a69005be8d20e2982fedbe0474c5a653adf290f8` | CC-BY-4.0 |
| `hausa_voa_ner` | `uds-lsv/transfer-distant-transformer-african`, Hausa VOA NER | **Human tier.** BIO-tagged Hausa news entities converted to JSON spans. Used only in the locked base. | `a6cbe619a7a18309bbd7a3813033b180a043c1a8` | CC-BY-4.0 |
| `yoruba_gv_ner` | `ajesujoba/YorubaTwi-Embedding`, Yoruba NER | **Human tier.** BIO-tagged entities converted to JSON spans. Used only in the locked base. | `5f5c76d9ea6a6a7205e53c842c61af719aa8b8c5` | CC-BY-3.0 |
| `nchlt_zulu_ner` | SADiLaR NCHLT isiZulu Named Entity Annotated Corpus | **Human tier.** Annotated NER corpus converted from BIO tags to JSON spans. | `4f9fee744c6d5ca9fcc921c621386a81ccfa2837` | CC-BY-2.5-ZA |
| `afrisenti` | `masakhane/afrisenti` | **Human tier.** Annotated Amharic positive, negative, and neutral sentiment data. | `eb42667d2e83d0081864767b47681fbaf00144fb` | CC-BY-NC-SA-2.0 |
| `masakhanews` | `masakhane/masakhanews` | **Human tier.** News headline and article topic classification. Used only in the locked base. | `fa3b5fff8a91d187bf0c5900a39c4271d08cf7fe` | AFL-3.0 |
| `masakhaner` | `masakhane-io/masakhane-ner`, MasakhaNER2.0 | **Human tier.** Annotated BIO entities converted to JSON spans. | `ba5843cd08aa491d5f96a5e809e71eb9ec461391` | AFL-3.0 |

No pool is classified as wholly synthetic. AfriInstruct is the only mixed
pool and contains synthetic or machine-translated components, but the selected
rows do not carry a reliable per-example human/synthetic flag. The report
therefore does not fabricate a synthetic example count.

## Source totals and coverage

| Pool | Base examples | Extension examples | Combined examples | Unique combined | Prepared tokens | Label tokens | Languages | Tasks |
|---|---:|---:|---:|---:|---:|---:|---|---|
| `afridocmt` | 6,875 | 25,367 | 32,242 | 32,242 | 3,355,611 | 2,049,521 | amh, hau, swh, yor, zul | Translation |
| `afrihate` | 17,112 | 19,511 | 36,623 | 34,047 | 2,128,629 | 240,384 | amh, hau, ibo, kin, swh, xho, yor, zul | Classification |
| `afriinstruct` | 51,065 | 28,734 | 79,799 | 79,474 | 21,985,600 | 6,294,359 | amh, hau, ibo, kin, sna, swh, xho, yor, zul | Instruction, QA, translation, classification, NER |
| `afrisenti` | 917 | 2,565 | 3,482 | 3,482 | 263,314 | 20,892 | amh | Classification |
| `aya` | 26,423 | 2,098 | 28,521 | 16,657 | 7,515,310 | 4,991,880 | amh, hau, ibo, sna, swh, wol, xho, yor, zul | Instruction, QA |
| `digital_umuganda` | 1,238 | 1,622 | 2,860 | 2,860 | 138,417 | 59,541 | kin | Translation |
| `hausa_voa_ner` | 2,463 | 0 | 2,463 | 1,004 | 246,103 | 70,830 | hau | NER |
| `kenswquad` | 665 | 1,594 | 2,259 | 2,259 | 2,116,151 | 22,701 | swh | QA |
| `kinnews` | 1,222 | 1,366 | 2,588 | 2,588 | 1,994,888 | 15,528 | kin | Classification |
| `masakhaner` | 8,210 | 5,592 | 13,802 | 13,802 | 1,305,566 | 329,647 | sna, wol, xho | NER |
| `masakhanews` | 5,041 | 0 | 5,041 | 3,236 | 4,086,572 | 30,246 | amh, sna, xho | Classification |
| `nchlt_zulu_ner` | 2,463 | 5,851 | 8,314 | 8,314 | 907,392 | 268,778 | zul | NER |
| `pontoon` | 16,047 | 15,769 | 31,816 | 30,227 | 1,531,478 | 745,413 | all 10 | Translation |
| `swahili_news` | 1,375 | 1,306 | 2,681 | 2,681 | 2,163,056 | 19,965 | swh | Classification |
| `yoruba_gv_ner` | 2,463 | 0 | 2,463 | 814 | 261,913 | 51,939 | yor | NER |
| **Overall** | **143,579** | **111,375** | **254,954** | **233,687** | **50,000,000** | **15,211,624** | **10** | **5** |

The extension uses 12 of the 15 pools. Hausa VOA NER, MasakhaNEWS, and Yoruba
GV NER occur only in the locked base because no eligible unused extension rows
were selected from those pools.

## Selection and preprocessing

All source-specific adapters normalize upstream records to a user/assistant
conversation with `lang`, `task`, and `source` provenance. Translation rows may
carry direction metadata, classification rows retain labels, QA rows may retain
domain or subtype metadata, and NER rows carry entity-type metadata.

Candidate records must:

1. belong to one of the ten configured languages and five configured tasks;
2. contain nonempty user content and supervised assistant content;
3. have positive exact Gemma prompt, label, and text token counts;
4. contain no more than 4,096 prepared text tokens;
5. be disjoint from the locked P2 train data and frozen validation monitor;
6. be unique across candidate pools by canonicalized message digest; and
7. have no full-text or 12-token-fragment match in the benchmark blocklist.

The extension target scales accepted P2 language-task-source allocations by
`18,650,688 / 31,349,312`. It first uses unused rows from the same source and
cell. Human and curated pools may absorb capacity shortfalls; mixed
AfriInstruct cannot exceed its scaled reference target. All extension rows are
unique. Capacity shortfalls are redistributed deterministically, and a bounded
label-density reconciliation moves only allocation above reference targets to
meet the supervised-token floor.

Source-specific preprocessing includes exact normalized deduplication; equal
source/target rejection for translation; BIO-to-JSON conversion for NER;
label-stratified classification; direction-stratified translation;
source/subtype-aware QA; and entity-density-stratified NER. KINNEWS preparation
removed 7,815 normalized duplicates, Swahili News removed 4,231, and NCHLT
isiZulu rows with known mojibake markers were dropped.

## Filtering and materialization results

The two complete profiles were byte-identical. The profile reported 8,220
cross-pool duplicates and 456 overlength records; independent materialization
reported 8,219 duplicates and 457 overlength records. The one-record difference
comes from ordering around independently excluded locked/frozen records and is
retained here rather than silently normalized.

Lower-quality trimming affected 16 AfriInstruct language-task allocations.
Exact integer-token reconciliation added 47 unique human/curated records. The
result contains exactly 18,650,688 extension tokens and exactly 50,000,000
combined prepared tokens.

## Validation and contamination

The validation monitor is copied byte-for-byte from K10 P2 v1: 30,306 unique
examples and 9,784,477 prepared tokens. It is strongly concentrated in
Swahili, Kinyarwanda, Yoruba, classification, and QA, and contains no Wolof
examples. It is suitable for comparison with P2 but is not a balanced measure
of all ten languages.

The blocklist contains official AfriMMLU, AfriXNLI, AfriMGSM, and Belebele
content plus the registered AfriInstruct-paper benchmarks. Both complete fields
and normalized 12-token fragments are blocked. The final independent audit
found zero benchmark matches and zero train-validation overlaps.

## Integrity and reproducibility

| Artifact | SHA-256 |
|---|---|
| Profile config | `a92a6c594f1199b6f9890a3a88696e10b3232fd07aeafdaf85db62c545500ac2` |
| Profile A/B | `cb1a673fc6cf9f3d5047de450b6734fa81961b402de8f57923e50605c861efa5` |
| Plan | `f0575c188956192358de8a4ee20ae38b70df969fb42364015d0448537844968c` |
| Benchmark blocklist | `e0d71cc1190a374a40f9ab5e030c3f28f98e1b640279c4e5f9afe0d2164c85fe` |
| Locked train manifest | `24c36637225b8ecf474c15301a6cbc5c916e92abe9eef237aea21cada41e9bb2` |
| Frozen validation manifest | `3906353425ab0da96d81ac7c08a9ec8ca75eed1d3bd2efc1107d0730e3f1c2f8` |
| Materialized summary | `7c4986dc03d2aaef07d6887641475665a29e0de6755db2b13128cbfd91d6de22` |
| Combined train manifest | `8e491592f58bc704990ed66238c90c79fe1678eb3582610d96178cba00244a07` |
| Final audit | `12ebd9e7ce63f7c14da8db473dae49fc9f338b06f83e617cc61b6fe62c681acd` |

The complete per-shard hashes and target-versus-realized allocation deviations
are stored in `/data/gemma4-k10/mixture-p2-nested-50m-v1/final_audit.json`.

## Known limitations

- AfriInstruct has mixed licensing and heterogeneous creation methods. Exact
  per-row human, translated, templated, and synthetic proportions are unknown.
- Several source datasets do not have complete local documentation of annotator
  protocol, target-language validation, or raw-versus-accepted counts.
- Many selected rows lack stable upstream IDs, limiting row-level traceability
  back to the original release.
- AfriDocMT and AfriSenti carry noncommercial/share-alike restrictions.
- The frozen validation monitor is intentionally inherited from P2 and is not
  language-balanced.
- Dataset acceptance establishes integrity and provenance, not model quality.
  No training result exists for this mixture yet.

## Appendix: source pool by language and task

| Pool | Language | Task | Examples | Unique | Prepared tokens | Label tokens |
|---|---|---|---:|---:|---:|---:|
| `afridocmt` | amh | translation | 7,512 | 7,512 | 752,514 | 441,397 |
| `afridocmt` | hau | translation | 7,145 | 7,145 | 677,632 | 388,974 |
| `afridocmt` | swh | translation | 2,249 | 2,249 | 202,468 | 110,869 |
| `afridocmt` | yor | translation | 8,552 | 8,552 | 1,050,123 | 711,906 |
| `afridocmt` | zul | translation | 6,784 | 6,784 | 672,874 | 396,375 |
| `afrihate` | amh | classification | 3,312 | 3,312 | 259,129 | 22,234 |
| `afrihate` | hau | classification | 5,030 | 4,565 | 284,334 | 32,435 |
| `afrihate` | ibo | classification | 3,776 | 3,113 | 211,126 | 25,243 |
| `afrihate` | kin | classification | 3,269 | 3,269 | 242,341 | 21,283 |
| `afrihate` | swh | classification | 11,592 | 11,592 | 591,877 | 76,220 |
| `afrihate` | xho | classification | 2,571 | 2,367 | 128,544 | 16,687 |
| `afrihate` | yor | classification | 3,406 | 3,325 | 226,753 | 22,394 |
| `afrihate` | zul | classification | 3,667 | 2,504 | 184,525 | 23,888 |
| `afriinstruct` | amh | instruction | 4,896 | 4,896 | 6,631,890 | 351,062 |
| `afriinstruct` | amh | translation | 1,314 | 1,314 | 120,108 | 49,051 |
| `afriinstruct` | amh | classification | 1,454 | 1,454 | 100,070 | 8,724 |
| `afriinstruct` | hau | instruction | 2,790 | 2,790 | 1,931,314 | 154,449 |
| `afriinstruct` | hau | translation | 1,302 | 1,302 | 135,694 | 56,695 |
| `afriinstruct` | hau | classification | 1,436 | 1,436 | 79,745 | 8,616 |
| `afriinstruct` | hau | ner | 1,314 | 1,314 | 517,773 | 253,383 |
| `afriinstruct` | ibo | instruction | 2,359 | 2,359 | 269,620 | 112,754 |
| `afriinstruct` | ibo | translation | 1,287 | 1,287 | 128,324 | 54,003 |
| `afriinstruct` | ibo | classification | 1,430 | 1,430 | 81,884 | 8,580 |
| `afriinstruct` | ibo | ner | 5,210 | 5,210 | 2,365,191 | 1,203,054 |
| `afriinstruct` | kin | instruction | 5,201 | 5,201 | 446,594 | 181,949 |
| `afriinstruct` | kin | translation | 1,307 | 1,307 | 156,894 | 64,930 |
| `afriinstruct` | kin | classification | 1,461 | 1,461 | 98,226 | 8,766 |
| `afriinstruct` | kin | ner | 5,232 | 5,232 | 1,914,221 | 896,504 |
| `afriinstruct` | sna | instruction | 3,527 | 3,527 | 293,227 | 117,518 |
| `afriinstruct` | sna | translation | 2,240 | 1,915 | 233,319 | 98,624 |
| `afriinstruct` | sna | classification | 1,468 | 1,468 | 80,160 | 8,808 |
| `afriinstruct` | sna | ner | 1,312 | 1,312 | 482,012 | 225,070 |
| `afriinstruct` | swh | instruction | 4,213 | 4,213 | 717,238 | 211,207 |
| `afriinstruct` | swh | qa | 693 | 693 | 97,703 | 6,237 |
| `afriinstruct` | swh | translation | 1,322 | 1,322 | 119,450 | 48,784 |
| `afriinstruct` | swh | classification | 1,461 | 1,461 | 91,435 | 8,766 |
| `afriinstruct` | swh | ner | 5,242 | 5,242 | 2,067,550 | 1,003,047 |
| `afriinstruct` | xho | instruction | 4,234 | 4,234 | 342,711 | 139,378 |
| `afriinstruct` | xho | translation | 1,308 | 1,308 | 127,638 | 51,733 |
| `afriinstruct` | xho | classification | 1,462 | 1,462 | 74,147 | 8,772 |
| `afriinstruct` | xho | ner | 1,313 | 1,313 | 392,031 | 167,458 |
| `afriinstruct` | yor | instruction | 1,356 | 1,356 | 185,170 | 69,104 |
| `afriinstruct` | yor | translation | 1,314 | 1,314 | 148,428 | 62,889 |
| `afriinstruct` | yor | classification | 1,466 | 1,466 | 120,847 | 8,796 |
| `afriinstruct` | yor | ner | 1,313 | 1,313 | 581,393 | 283,128 |
| `afriinstruct` | zul | instruction | 3,945 | 3,945 | 327,394 | 138,368 |
| `afriinstruct` | zul | translation | 1,318 | 1,318 | 138,519 | 59,165 |
| `afriinstruct` | zul | ner | 1,299 | 1,299 | 387,680 | 164,987 |
| `afrisenti` | amh | classification | 3,482 | 3,482 | 263,314 | 20,892 |
| `aya` | amh | instruction | 252 | 63 | 69,496 | 47,224 |
| `aya` | amh | qa | 1,773 | 1,128 | 154,456 | 51,167 |
| `aya` | hau | instruction | 1,532 | 383 | 527,728 | 450,476 |
| `aya` | hau | qa | 2,426 | 2,426 | 275,196 | 126,355 |
| `aya` | ibo | instruction | 1,788 | 447 | 148,632 | 107,132 |
| `aya` | ibo | qa | 1,773 | 969 | 358,655 | 216,251 |
| `aya` | sna | instruction | 1,096 | 274 | 177,672 | 116,956 |
| `aya` | sna | qa | 1,773 | 1,028 | 317,614 | 175,043 |
| `aya` | swh | instruction | 592 | 148 | 67,168 | 52,660 |
| `aya` | swh | qa | 665 | 211 | 91,467 | 66,490 |
| `aya` | wol | instruction | 3,300 | 875 | 466,376 | 335,123 |
| `aya` | wol | qa | 2,004 | 2,004 | 155,433 | 90,423 |
| `aya` | xho | instruction | 652 | 163 | 101,940 | 65,704 |
| `aya` | xho | qa | 836 | 209 | 175,072 | 77,136 |
| `aya` | yor | instruction | 3,689 | 3,689 | 2,738,819 | 2,588,407 |
| `aya` | yor | qa | 1,773 | 978 | 313,386 | 188,095 |
| `aya` | zul | instruction | 824 | 206 | 331,740 | 65,900 |
| `aya` | zul | qa | 1,773 | 1,456 | 1,044,460 | 171,338 |
| `digital_umuganda` | kin | translation | 2,860 | 2,860 | 138,417 | 59,541 |
| `hausa_voa_ner` | hau | ner | 2,463 | 1,004 | 246,103 | 70,830 |
| `kenswquad` | swh | qa | 2,259 | 2,259 | 2,116,151 | 22,701 |
| `kinnews` | kin | classification | 2,588 | 2,588 | 1,994,888 | 15,528 |
| `masakhaner` | sna | ner | 4,699 | 4,699 | 504,750 | 144,717 |
| `masakhaner` | wol | ner | 4,593 | 4,593 | 414,500 | 90,633 |
| `masakhaner` | xho | ner | 4,510 | 4,510 | 386,316 | 94,297 |
| `masakhanews` | amh | classification | 916 | 916 | 1,174,930 | 5,496 |
| `masakhanews` | sna | classification | 2,750 | 1,288 | 1,871,719 | 16,500 |
| `masakhanews` | xho | classification | 1,375 | 1,032 | 1,039,923 | 8,250 |
| `nchlt_zulu_ner` | zul | ner | 8,314 | 8,314 | 907,392 | 268,778 |
| `pontoon` | amh | translation | 1,743 | 1,743 | 109,981 | 55,964 |
| `pontoon` | hau | translation | 1,713 | 1,713 | 73,625 | 33,759 |
| `pontoon` | ibo | translation | 3,324 | 3,324 | 167,084 | 92,268 |
| `pontoon` | kin | translation | 1,237 | 443 | 71,679 | 35,307 |
| `pontoon` | sna | translation | 1,060 | 265 | 44,520 | 20,876 |
| `pontoon` | swh | translation | 1,742 | 1,742 | 92,359 | 44,821 |
| `pontoon` | wol | translation | 6,598 | 6,598 | 269,552 | 120,412 |
| `pontoon` | xho | translation | 10,898 | 10,898 | 518,790 | 240,881 |
| `pontoon` | yor | translation | 1,744 | 1,744 | 98,342 | 59,015 |
| `pontoon` | zul | translation | 1,757 | 1,757 | 85,546 | 42,110 |
| `swahili_news` | swh | classification | 2,681 | 2,681 | 2,163,056 | 19,965 |
| `yoruba_gv_ner` | yor | ner | 2,463 | 814 | 261,913 | 51,939 |

## Authorities

- [K10 P2 nested 50M preregistration](K10_P2_NESTED_50M_DATA_PREREGISTRATION.md)
- [Nested profile configuration](data/k10_p2_nested_50m_profile.yaml)
- [Master machine-readable registry](data/data_mixture_registry.yaml)
- `/data/gemma4-k10/mixture-p2-nested-50m-v1/final_audit.json`
- `tools/prepare_k6_sft_sources.py`
- `tools/profile_sft_mixture.py`
- `tools/build_k6_sft_mixture.py`
- `tools/audit_nested_sft_mixture.py`