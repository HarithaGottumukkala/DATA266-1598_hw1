# DATA 266 — Homework 12: Knowledge Distillation Metrics

**Scope:** Distillation only (Part 2 and associated measurement/analysis). GPTQ evidence remains separately in `METRICS_GPTQ.md`. Data source: preserved outputs in the uploaded seed-1598 notebook and final successful Colab execution records.

## Data, models and settings

- Teacher: instructor-provided local fine-tuned BERT, `DATA266_HW12/teacher_model`.
- Student: `distilbert/distilbert-base-uncased`; classifier output dimension = 4.
- Dataset: AG News (`fancyzhx/ag_news`), original 120,000 training records and 7,600 official test records.
- Stratified training/validation split: 108,000/12,000, **SEED = 1598**. Test split never used for hyperparameter selection.
- Student runs: 3 epochs, batch size 16, learning rate 2e-5, weight decay 0.01, warmup ratio 0.06, max sequence length 256, gradient clip norm 1.0. Separate initialized students for each distinct sweep run.
- Loss: `alpha × CE(student, labels) + (1−alpha) × T² × KL(teacher_soft(T) || student_soft(T))`. `alpha` weights the cross-entropy term.
- Saved full teacher-logit coverage: **120,000 × 4** (`float32`); the actual student training used **108,000 × 4** original training logits. The remaining 12,000 were added later only to complete extraction coverage.

## Temperature sweep — fixed alpha = 0.50 (validation)

| Temperature | Accuracy | Best epoch |
|---:|---:|---:|
| 1 | 94.7167% | 3 |
| 2 | 94.7000% | 3 |
| **5** | **94.7750%** | **3** |
| 10 | 94.7083% | 3 |

Winner by validation accuracy: **T = 5**.

## Loss-weight sweep — fixed T = 5 (validation)

| Alpha (CE weight) | Accuracy | Macro-F1 | Best epoch |
|---:|---:|---:|---:|
| **0.25** | **94.7917%** | **0.947965** | **3** |
| 0.50 | 94.7750% | 0.947808 | 3 |
| 0.75 | 94.6583% | 0.946652 | 3 |

Final choice by validation accuracy: **T=5, alpha=0.25**, checkpoint `outputs/part2_distillation_seed1598/alpha_T5_alpha025/checkpoint_epoch_3`.

## Table HW11.2 — separate AG News teacher–student comparison

| Metric | BERT teacher | Best distilled student |
|---|---:|---:|
| AG News test accuracy | 94.4605% | **94.6579%** |
| Test macro-F1 | 0.944652 | **0.946626** |
| p50 latency (ms) | 16.7124 | **8.5814** |
| p95 latency (ms) | 17.1056 | **8.9077** |
| Peak GPU allocated memory (GiB) | 0.4264 | **0.2680** |
| Checkpoint directory size (MiB) | 418.5823 | **255.4317** |

**Benchmark protocol:** Tesla T4, batch size 1, sequence length padded to 256, identical 100 AG News inputs, 20 warmups and 100 timed CUDA-synchronized forward passes. Tokenization and model loading were excluded. Reported memory is PyTorch CUDA peak allocated, not total device occupancy. File sizes are binary MiB; the original CSV header says `checkpoint_size_mb`.

**Measured trade-off:** student test accuracy +0.1974 percentage points, p50 ≈1.95× faster (≈48.65% lower latency), peak allocated GPU memory ≈37.15% lower, checkpoint-directory size ≈38.98% lower. These findings apply to this specific evaluation and benchmark.

## Per-class test accuracy

| AG News class | Teacher | Student | Student minus teacher (percentage points) |
|---|---:|---:|---:|
| World | 94.9474% | 95.4211% | +0.4737 |
| Sports | 98.8421% | 98.8947% | +0.0526 |
| **Business** | **91.3684%** | **91.2105%** | **−0.1579** |
| Sci/Tech | 92.6842% | 93.1053% | +0.4211 |

**Largest degradation:** Business (−0.1579 pp). There were 92 total teacher-correct/student-wrong records. The five requested examples, taken from Business, are:

| Test index | True class | Student prediction | Article identifying text |
|---:|---|---|---|
| 110 | Business | Sci/Tech | “Yahoo! Ups Ante for Small Businesses” |
| 399 | Business | Sci/Tech | “Nortel Downsizes Again” |
| 408 | Business | World | “Chavez rejects CD as opposition” |
| 467 | Business | Sci/Tech | “Deal has S2io champing at the gigabit” |
| 1124 | Business | Sci/Tech | “Judge: Geico can sue Google, Overture over ads” |

Full, unmodified examples are preserved in the notebook output and `final_test_evaluation/five_failure_examples.csv`. Four of these five contain business stories with substantial technology context that the student predicted as Sci/Tech; the fifth had political context and was predicted as World. This supports a possible topic-overlap explanation, not a definitive causal claim.

## Training-loss evidence

Three-epoch training histories and per-epoch checkpoints were saved in each experiment directory. The submitted notebook retains the original executed training logs. **Do not substitute validation accuracy for training loss.** For loss curves, use the saved `training_history.json` files and the exact keys recorded therein; no missing epoch-loss numbers have been invented in this report.

## Evidence files

`outputs/part2_distillation_seed1598/` contains the split manifest, tokenizer metadata, training configuration, two sweep-comparison CSVs, selection JSONs, training checkpoints and histories, and the teacher-logit caches. `final_test_evaluation/` contains `final_test_metrics.json`, `paired_test_predictions.csv`, `per_class_accuracy.csv`, `five_failure_examples.csv`, `all_teacher_correct_student_wrong.csv`, `teacher_student_benchmark.csv` and `benchmark_config.json`.

**Do not combine this accuracy table with GPTQ WikiText-2 perplexity.** The model families, datasets and metrics differ; GPTQ results belong in their own table.
