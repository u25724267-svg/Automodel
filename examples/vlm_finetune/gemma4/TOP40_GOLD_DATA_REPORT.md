# Top-40 gold dataset candidate report

Date: 2026-09-21

## Executive summary

This report records the gold-data review of the 40-language shortlist in `docs/Misc/afrigemma_datasets_2026-09-15_enriched.xlsx`. It is an inventory review, not a prepared training mixture or completed Gemma 4 run.

The accepted quality threshold is documented human collection or annotation, including model-assisted material
that was subsequently corrected by people. Fully synthetic data is excluded. Evaluation datasets remain reserved
even when their construction meets the gold threshold.

| Metric | Value |
|---|---:|
| Top languages reviewed | 40 |
| Potential training / benchmark-only / no qualifying source | 24 / 8 / 8 |
| Qualifying dataset IDs | 17 |
| Language-dataset inventory associations | 192 |
| Candidate dataset IDs / associations | 9 / 93 |
| Candidate language-row sum | 513,085 |
| Candidate dataset-level rows kept | 832,922 |
| Reserved dataset IDs / associations | 8 / 99 |
| Reserved language-row sum | 1,722,112 |
| Reserved dataset-level rows kept | 2,177,649 |

The language-row sum adds the workbook's per-language counts. The dataset-level rows-kept sum adds each dataset's
`Rows kept (dataset)` value once. They are different measures: bilingual records can be associated with both
languages, and full dataset totals can include languages outside the top-40 shortlist. Neither measure is
deduplicated or Gemma-tokenized capacity.

## Scope and terminology

A **qualifying gold source** has workbook evidence of human collection, human annotation, human translation, or
human correction of model-assisted material. A dataset is not considered gold merely because it is curated,
widely used, or multilingual.

A **potential training source** passes the workbook-level provenance screen but is not automatically approved for
training. License compatibility, exact revisions, split boundaries, language aliases, and row-level provenance
still require verification.

A **benchmark-only source** is held out from training by this report even where the workbook inventory labels its
purpose as `train`. This prevents evaluation leakage. FLORES+ is benchmark-only and explicitly prohibits training
use.

The 192 retained rows are language-dataset inventory associations. They are not physical examples, unique examples,
prepared tokens, assistant-label tokens, or usable training capacity.

## Top-40 language and dataset matrix

Counts in parentheses are workbook `Rows for language` values. The candidate and reserved totals sum those
language-level associations; they do not represent unique records.

| Rank | Language | Code | Candidate datasets (rows) | Candidate rows | Reserved datasets (rows) | Reserved rows | Disposition |
|---:|---|---|---|---:|---|---:|---|
| 1 | English | `eng` | `masakhanews` (4,729); `menyo20k_mt` (20,100) | 24,829 | `afri_mgsm` (8); `afrixnli` (1,050); `uhura_arc_easy` (1,318); `uhura_eval` (817) | 3,193 | Candidate |
| 2 | Twi | `twi` | `afrisenti` (4,818); `mafand_mt` (6,121); `masakhaner2` (6,058); `masakhapos` (1,570) | 18,567 | `afri_mgsm` (8); `afri_mmlu` (583); `afriqa` (1,221); `afriqa_gold_passages` (1,220); `afrixnli` (1,050); `flores_plus` (64,288) | 68,370 | Candidate |
| 3 | Xhosa | `xho` | `aya_dataset` (377); `mafand_mt` (1,488); `masakhaner2` (8,168); `masakhanews` (1,476); `masakhapos` (1,503) | 13,012 | `afri_mgsm` (8); `afri_mmlu` (583); `afrixnli` (1,050); `flores_plus` (64,288) | 65,929 | Candidate |
| 4 | Amharic | `amh` | `afrisenti` (9,480); `aya_dataset` (1,207); `mafand_mt` (1,936); `masakhaner1` (2,500); `masakhanews` (1,875) | 16,998 | `afri_mgsm` (8); `afri_mmlu` (583); `afrixnli` (1,050); `flores_plus` (64,288); `uhura_arc_easy` (1,239); `uhura_eval` (815) | 67,983 | Candidate |
| 5 | Igbo | `ibo` | `afrisenti` (15,715); `aya_dataset` (1,534); `mafand_mt` (9,998); `masakhaner1` (3,194); `masakhaner2` (10,908); `masakhanews` (1,940); `masakhapos` (1,605); `naijasenti` (15,715) | 60,609 | `afri_mgsm` (8); `afri_mmlu` (583); `afriqa` (1,022); `afriqa_gold_passages` (1,022); `afrixnli` (1,050); `flores_plus` (64,288) | 67,973 | Candidate |
| 6 | Yoruba | `yor` | `afrisenti` (15,127); `aya_dataset` (12,008); `mafand_mt` (9,746); `masakhaner1` (3,190); `masakhaner2` (9,829); `masakhanews` (2,050); `masakhapos` (1,785); `menyo20k_mt` (20,100); `naijasenti` (15,127) | 88,962 | `afri_mgsm` (8); `afri_mmlu` (583); `afriqa` (502); `afriqa_gold_passages` (502); `afrixnli` (1,050); `flores_plus` (64,288); `uhura_arc_easy` (1,246); `uhura_eval` (815) | 68,994 | Candidate |
| 7 | Zulu | `zul` | `aya_dataset` (1,833); `mafand_mt` (5,737); `masakhaner2` (8,362); `masakhapos` (1,504) | 17,436 | `afri_mgsm` (8); `afri_mmlu` (583); `afriqa` (812); `afriqa_gold_passages` (812); `afrixnli` (1,050); `flores_plus` (64,288); `uhura_arc_easy` (909); `uhura_eval` (761) | 69,223 | Candidate |
| 8 | Shona | `sna` | `aya_dataset` (1,368); `mafand_mt` (1,561); `masakhaner2` (8,870); `masakhanews` (1,842); `masakhapos` (1,492) | 15,133 | `afri_mgsm` (8); `afri_mmlu` (583); `afrixnli` (1,050); `flores_plus` (64,288) | 65,929 | Candidate |
| 9 | Hausa | `hau` | `afrisenti` (22,152); `aya_dataset` (3,512); `mafand_mt` (8,665); `masakhaner1` (2,741); `masakhaner2` (8,198); `masakhanews` (3,173); `masakhapos` (1,504); `naijasenti` (22,152) | 72,097 | `afri_mgsm` (8); `afri_mmlu` (583); `afriqa` (750); `afriqa_gold_passages` (750); `afrixnli` (1,050); `flores_plus` (64,288); `uhura_arc_easy` (1,200); `uhura_eval` (799) | 69,428 | Candidate |
| 10 | Swahili | `swh` | `aya_dataset` (366) | 366 | `flores_plus` (64,288); `uhura_arc_easy` (1,231); `uhura_eval` (812) | 66,331 | Candidate |
| 11 | Somali | `som` | `aya_dataset` (7,704); `masakhanews` (1,463) | 9,167 | `flores_plus` (64,288) | 64,288 | Candidate |
| 12 | Afrikaans | `afr` | — | 0 | `flores_plus` (64,288) | 64,288 | Benchmark-only |
| 13 | Arabic | `ara` | — | 0 | — | 0 | No qualifying source |
| 14 | Portuguese | `por` | — | 0 | — | 0 | No qualifying source |
| 15 | Spanish | `spa` | — | 0 | — | 0 | No qualifying source |
| 16 | French | `fra` | `masakhanews` (2,109) | 2,109 | `afri_mgsm` (8); `afri_mmlu` (583); `afrixnli` (1,050) | 1,641 | Candidate |
| 17 | Fulfulde | `fuj` | — | 0 | `flores_plus` (64,288) | 64,288 | Benchmark-only |
| 18 | Kinyarwanda | `kin` | `afrisenti` (5,155); `mafand_mt` (1,466); `masakhaner1` (3,033); `masakhaner2` (11,195); `masakhapos` (1,512) | 22,361 | `afri_mgsm` (8); `afri_mmlu` (583); `afriqa` (864); `afriqa_gold_passages` (864); `afrixnli` (1,050); `flores_plus` (64,288) | 67,657 | Candidate |
| 19 | Nyanja | `nya` | `aya_dataset` (688); `mafand_mt` (1,487); `masakhaner2` (8,929); `masakhapos` (1,455) | 12,559 | `flores_plus` (64,288) | 64,288 | Candidate |
| 20 | Luganda | `lug` | `mafand_mt` (7,075); `masakhaner1` (2,035); `masakhaner2` (7,060); `masakhanews` (1,104); `masakhapos` (1,465) | 18,739 | `afri_mgsm` (8); `afri_mmlu` (583); `afrixnli` (1,050); `flores_plus` (64,288) | 65,929 | Candidate |
| 21 | Tswana | `tsn` | `mafand_mt` (4,942); `masakhaner2` (4,990); `masakhapos` (1,505) | 11,437 | `flores_plus` (64,288) | 64,288 | Candidate |
| 22 | Northern Sotho | `nso` | — | 0 | `uhura_arc_easy` (952); `uhura_eval` (817) | 1,769 | Benchmark-only |
| 23 | Tsonga | `tso` | `afrisenti` (1,261) | 1,261 | — | 0 | Candidate |
| 24 | German | `deu` | — | 0 | — | 0 | No qualifying source |
| 25 | Plateau Malagasy | `plt` | `aya_dataset` (14,597) | 14,597 | `flores_plus` (64,288) | 64,288 | Candidate |
| 26 | Wolof | `wol` | `aya_dataset` (2,914); `mafand_mt` (6,366); `masakhaner1` (2,677); `masakhaner2` (6,564); `masakhapos` (1,564) | 20,085 | `afri_mgsm` (8); `afri_mmlu` (583); `afrixnli` (1,050); `flores_plus` (64,288) | 65,929 | Candidate |
| 27 | Lingala | `lin` | `masakhanews` (870) | 870 | `afri_mgsm` (8); `afri_mmlu` (583); `afrixnli` (1,050); `flores_plus` (64,288) | 65,929 | Candidate |
| 28 | Ga | `gaa` | — | 0 | — | 0 | No qualifying source |
| 29 | Rundi | `run` | `masakhanews` (1,598) | 1,598 | `flores_plus` (64,288) | 64,288 | Candidate |
| 30 | Fon | `fon` | `mafand_mt` (5,443); `masakhaner2` (6,205); `masakhapos` (1,624) | 13,272 | `afriqa` (964) | 964 | Candidate |
| 31 | Bambara | `bam` | `mafand_mt` (6,013); `masakhaner2` (6,376); `masakhapos` (1,548) | 13,937 | `flores_plus` (64,288) | 64,288 | Candidate |
| 32 | Tumbuka | `tum` | — | 0 | — | 0 | No qualifying source |
| 33 | Sotho | `sot` | — | 0 | `afri_mgsm` (8); `afri_mmlu` (583); `afrixnli` (1,050); `flores_plus` (64,288) | 65,929 | Benchmark-only |
| 34 | Kikuyu | `kik` | — | 0 | `flores_plus` (64,288) | 64,288 | Benchmark-only |
| 35 | Bemba | `bem` | — | 0 | `afriqa` (778); `afriqa_gold_passages` (778); `flores_plus` (64,288) | 65,844 | Benchmark-only |
| 36 | Tigre | `tig` | — | 0 | — | 0 | No qualifying source |
| 37 | Dinka | `din` | — | 0 | — | 0 | No qualifying source |
| 38 | Kabyle | `kab` | — | 0 | `flores_plus` (64,288) | 64,288 | Benchmark-only |
| 39 | Kamba | `kam` | — | 0 | `flores_plus` (64,288) | 64,288 | Benchmark-only |
| 40 | Nigerian Pidgin | `pcm` | `afrisenti` (10,556); `mafand_mt` (7,838); `masakhaner1` (3,039); `masakhaner2` (8,074); `masakhanews` (1,517); `masakhapos` (1,504); `naijasenti` (10,556) | 43,084 | — | 0 | Candidate |

## Qualifying dataset families

### Potential training families

`Language rows` is the sum of the dataset's top-40 language associations. `Rows kept` is the workbook's full
dataset-level count and can include other languages.

| Dataset ID | Task | Languages | Language rows | Rows kept | Gold evidence | Primary-source license |
|---|---|---:|---:|---:|---|---|
| `aya_dataset` | Instruction | 12 | 48,108 | 204,112 | Human generated and human annotated | Apache-2.0 |
| `afrisenti` | Classification | 8 | 84,264 | 92,165 | Human collected, annotated, and validated | CC-BY-4.0 |
| `mafand_mt` | Translation | 16 | 85,882 | 142,909 | Human translated with human quality control | CC-BY-NC-4.0; source texts vary |
| `masakhaner1` | NER | 8 | 22,409 | 95,590 | Human collected, annotated, and validated | CC-BY-4.0 |
| `masakhaner2` | NER | 15 | 119,786 | 152,904 | Human collected, annotated, and validated | CC-BY-4.0 |
| `masakhanews` | Topic classification | 13 | 25,746 | 31,088 | Model-assisted human labels with human validation | AFL-3.0 |
| `masakhapos` | POS | 15 | 23,140 | 30,504 | Human collected, annotated, and validated | MIT |
| `menyo20k_mt` | Translation | 2 | 40,200 | 20,100 | Human translated with human quality control | CC-BY-NC-4.0 |
| `naijasenti` | Classification | 4 | 63,550 | 63,550 | Human collected, annotated, and validated | CC-BY-NC-SA-4.0 |
| **Overall** | **6 task labels** | **24 covered** | **513,085** | **832,922** | — | — |

Task-level association totals are 48,108 instruction, 147,814 classification, 126,082 translation, 142,195 NER,
25,746 topic-classification, and 23,140 POS rows. These categories follow workbook task labels; classification and
topic classification are not merged in the total.

### Benchmark-reserved families

| Dataset ID | Evaluation role | Languages | Language rows | Rows kept | License | Training decision |
|---|---|---:|---:|---:|---|---|
| `afri_mgsm` | Mathematical reasoning | 15 | 120 | 152 | Apache-2.0 | Reserve |
| `afri_mmlu` | Multitask knowledge and reasoning | 14 | 8,162 | 9,911 | Apache-2.0 | Reserve |
| `afrixnli` | Natural-language inference | 15 | 15,750 | 18,900 | Apache-2.0 | Reserve |
| `afriqa` | Question answering | 8 | 6,913 | 7,208 | CC-BY-4.0 | Reserve |
| `afriqa_gold_passages` | Question-answering gold passages | 7 | 5,948 | 6,243 | CC-BY-4.0 | Reserve |
| `flores_plus` | Machine translation | 26 | 1,671,488 | 2,121,504 | CC-BY-SA-4.0 | Reserve; training prohibited |
| `uhura_arc_easy` | Multiple-choice reasoning | 7 | 8,095 | 8,095 | MIT | Reserve |
| `uhura_eval` | Question answering / truthfulness | 7 | 5,636 | 5,636 | MIT | Reserve |
| **Overall** | — | **16 covered** | **1,722,112** | **2,177,649** | — | **Exclude from training** |

## Quality and provenance gates

Before any candidate becomes training data, verify all of the following against primary dataset documentation and a
pinned revision:

1. The retained split was collected, authored, translated, annotated, or corrected by people.
2. Model-assisted rows have documented human correction rather than acceptance without review.
3. Synthetic-only subsets and rows are identifiable and excluded.
4. Training, validation, and test splits remain separate.
5. Benchmark and evaluation rows are excluded from every training source.
6. Dataset and component licenses permit the intended research and redistribution workflow.
7. Language identifiers and aliases resolve correctly, including `swa`/`swh` and the `fuj`/`fuv` distinction.
8. Source revisions, configurations, and split names are pinned before download or profiling.

## License and usage review

Workbook inclusion is not final license approval. The enriched workbook records primary-source licenses for all nine candidate families, but MAFAND-MT notes that source-text licenses vary. Its components must be reviewed individually. MasakhaNER, MasakhaPOS, and MasakhaNEWS should retain their pinned primary-source evidence because earlier inventory wording conflicted with the verified licenses recorded in the enriched sheet.

FLORES+ must not enter training because its recorded terms prohibit training use. All benchmark-reserved families
must remain blocked regardless of whether their licenses would otherwise allow training.

## Exclusions

The review excludes:

- fully synthetic datasets and synthetic-only subsets;
- sources whose workbook evidence does not establish human creation, annotation, translation, or correction;
- all benchmark-reserved rows, including gold-standard evaluation data;
- sources with unresolved training restrictions until those restrictions are cleared;
- the eight shortlist languages for which no qualifying gold source was identified.

The absence of a qualifying source is an inventory conclusion, not a claim that no gold data exists anywhere for the
language. It means the reviewed workbook did not provide enough evidence for inclusion under this threshold.

## Statistics not yet measured

Unlike the K6 P2 report, this inventory report cannot state:

- physical or unique example counts;
- duplicate rates or maximum repetition;
- Gemma 4 prepared, input, or assistant-label token counts;
- sequence-length rejection counts;
- language-task cell counts;
- train-validation overlap;
- benchmark contamination counts;
- materialized shard hashes; or
- training or validation metrics.

Those values require a separately approved profiling and materialization pass. They must not be inferred from the
192 inventory rows.

## Implementation boundary

No training registry, YAML recipe, CSV export, preprocessing script, manifest, or additional report was created as
part of this review. The workbook and all underlying datasets were left unchanged.

If implementation is later approved, the next evidence-producing step is to export the qualifying workbook rows,
resolve aliases and licenses, pin revisions, isolate benchmarks, and only then profile eligible training splits with
the Gemma 4 E2B processor. Any resulting mixture should receive an independent contamination and train-validation
overlap audit before use.

## Known limitations

- The workbook is an inventory and may contain stale upstream metadata.
- Gold quality is established at dataset or subset level, not guaranteed for every row.
- Human-corrected model-assisted data can vary substantially in correction depth.
- Per-language counts are workbook inventory associations, not independently recounted source rows.
- Some benchmark datasets are marked `train` in the inventory purpose column; this report overrides that field and
  reserves all eight benchmark IDs to prevent evaluation leakage.
- Language aliases can overstate or understate coverage until normalized.
- Inventory counts do not represent deduplicated or Gemma-tokenized capacity.
- A qualifying provenance label does not establish license compatibility, balanced coverage, or downstream quality.

## Authority

- `docs/Misc/afrigemma_datasets_2026-09-15_enriched.xlsx`
- `examples/vlm_finetune/gemma4/K6_P2_DATA_REPORT (1).md` (report structure only)
