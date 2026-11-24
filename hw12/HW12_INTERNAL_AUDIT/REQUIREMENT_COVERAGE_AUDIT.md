# DATA 266 HW12: Requirement Coverage Audit

*Internal document. Do not submit it as part of the report.*

## Source-of-truth caveat (read first)

The professor's original HW12 instruction PDF and rubric are **not on this machine**. I searched `~/Desktop/hw12`, `~/Downloads`, `~/Documents` and ran a Spotlight full-text search for "GPTQ" and "knowledge distillation". The requirements below were therefore **reconstructed from secondary sources**:

| Source | What it states about the assignment |
|---|---|
| `DATA266_HW12_GPTQ (2).ipynb`, cell 0 | "Part 1: LLM Optimization with GPTQ — Speed and Accuracy Comparison **(5 Points)**"; requires an "instructor-approved" LLM |
| `Downloads/DATA266_HW12_Distillation_Final_Completed (2).ipynb` | Part 2 (KD) steps; "Part 3, question 2" = T sweep {1,2,5,10} at α=0.5; "question 3" = α {0.25,0.5,0.75} at best T; "Question 4" = per-class performance, the most-degraded class, five teacher-correct/student-wrong examples and what they share |
| `Downloads/DATA266_HW12_Part2_Unscaled_KD_SEED1598_Colab.ipynb` (never executed) | Claims the assignment writes the loss as **L = αCE + (1−α)KL (no T²)**, requires teacher logits for **all 120,000** training rows, and requires deliverables `RUN_LOG.txt`, `METRICS.md`, `AI_USE.md`, an IEEE report, and a private GitHub repo with an `hw12` tag |
| Standing course rule (all notebooks) | SEED must equal SID4 (1598) |

**Marks are known only for Part 1 (5 points).** Mark allocations for Parts 2 and 3 cannot be verified. Before submitting, compare this table against the real assignment PDF.

## Requirement-by-requirement mapping

Status key: **FC** = fully completed · **PC** = partially completed · **IU** = implemented but not executed · **NC** = not completed · **CV** = cannot be verified

| # | Requirement | Marks | Status | Evidence (exact source) | Gaps and grading implications |
|---|---|---|---|---|---|
| P1.1 | Select and verify an (instructor-approved) LLM | of 5 | **PC / CV** | GPTQ nb cells 5, 7; `outputs/part1_gptq/original_model_info.json` (OPT-125M, rev 27dcfa7, 125,239,296 params) | Model verified, but `model_metadata.json` records `"instructor_approval": "Not yet confirmed"`. If a specific model was mandated, this item is at risk |
| P1.2 | WikiText-2 calibration/evaluation data prep | of 5 | **FC** | GPTQ cell 9; `dataset/dataset_metadata.json` (128 train samples, seed 1598, separate test split) | Below GPTQModel's recommended ≥256 calibration samples (library warning; disclosed) |
| P1.3 | FP16 baseline quality (NLL/PPL) | of 5 | **FC** | cell 11 (PPL 26.1036); cell 31 same-environment re-check (PPL 26.1023) | — |
| P1.4 | GPTQ 4-bit quantization | of 5 | **FC** | cell 21 (44.44 s, 12 layers, 72 modules, g=128); cell 23 reload + generation check | — |
| P1.5 | Quantized model quality | of 5 | **FC** | cell 25 (NLL 3.387793, PPL 29.6005, +13.40%) | — |
| P1.6 | Speed / memory / size comparison | of 5 | **FC** | cell 27 (p50 7.914 vs 16.356 ms; p95 10.295 vs 18.907; peak 0.265 vs 0.148 GiB; 238.896 vs 119.318 MiB); cell 29 (5 figures) | Latency used random tokens, seq 128, batch 1 (disclosed). No generation-throughput metric |
| P1.7 | Trade-off analysis | of 5 | **FC** | cell 28 markdown; report §III-C, §VI | — |
| P2.1 | Load instructor teacher locally | ? | **FC** | Submission nb cells 7–9 (BERT, 109,485,316 params, 4 labels) | — |
| P2.2 | Stratified split with seed 1598 | ? | **FC** | cell 11 (108,000/12,000, balanced); manifest SHA b6cbd82e… | — |
| P2.3 | Cache teacher logits (training rows) | ? | **FC** | cell 13 onward; `teacher_train_logits.npy` (108,000×4) | — |
| P2.3b | Logits for **all 120,000** training rows (if required) | ? | **FC** | cell 46; `teacher_all_120000_logits.npy` + metadata | Produced *after* training (validation logits unused by the student; disclosed) |
| P2.4 | Distillation loss as specified | ? | **PC** | All 6 executed runs use **αCE + (1−α)T²KL** (`train_t2_kl_loss` keys; 0.5·0.3231+0.5·0.7414 = 0.5322 ✓) | **If the PDF specifies the unscaled loss, this deviates for T>1.** At T=1 the two are identical. The unscaled notebook is **IU** (no outputs). The report states this openly (§IV-E, Eq. 5). Possible deduction |
| P2.5 | Train DistilBERT student (3 epochs, fresh init) | ? | **FC** | cell 22 (T=2 baseline); histories JSON | — |
| P2.6 | Undistilled (CE-only) student baseline | ? | **NC / CV** | none | Unknown whether required. Without it, the KD gain over plain fine-tuning is unmeasured (disclosed) |
| P3.Q1 | Teacher vs student: accuracy, F1, latency, memory, size | ? | **FC** | cells 38, 43; `final_test_evaluation/benchmark_config.json`, `teacher_student_benchmark.csv` | Benchmark ran on a T4 while training ran on an L4 (same GPU for both models in the benchmark; disclosed) |
| P3.Q2 | Temperature sweep T∈{1,2,5,10}, α=0.5 | ? | **FC** | cells 26–29; `selected_temperature.json`; T1/T2/T10 histories | T=5 history file not exported (values from notebook output) |
| P3.Q3 | α sweep {0.25,0.5,0.75} at best T | ? | **FC** | cells 32–34; `alpha_T5_alpha025/experiment_summary.json` | α=0.75 history not exported (notebook output only) |
| P3.Q4a | Per-class teacher/student performance | ? | **PC** | cell 40 (per-class accuracy both models); teacher confusion matrix from old nb cell 23 (verified identical counts) | **Student confusion matrix and per-class precision/F1 were never printed** and the CSVs are only on Google Drive |
| P3.Q4b | Most-degraded class + 5 real examples + common factor | ? | **FC** | cell 40 (Business −0.1579 pp; indices 110, 399, 408, 467, 1124; 92 total) | — |
| D1 | IEEE report | ? | **FC** | `DATA266_HW12_Final_Report.pdf` | — |
| D2 | RUN_LOG.txt / METRICS.md / AI_USE.md | ? | **PC / CV** | Drafts exist: `RUN_LOG_DISTILLATION.txt`, `RUN_LOG_GPTQ_DRAFT.txt`, `METRICS_DISTILLATION.md`, `Downloads/AI_USE_DISTILLATION_DRAFT.md`. `METRICS_GPTQ.md` was generated on Drive but is **not** in local files | The student must consolidate these and author the AI-use statement. I did not create them |
| D3 | Private GitHub repo + `hw12` tag | ? | **CV** | not on this machine | Student action |
| S1 | Seed = SID4 everywhere | — | **FC** | All final runs use 1598. The seed-42 pilot was superseded and excluded from the report | — |
| S2 | Use of SLICE / HP_ID / CLS_A / CLS_B | ? | **CV** | Values listed; no HW12 code consumes them | If HW12 maps HP_ID to a hyperparameter choice, that is unverified |
