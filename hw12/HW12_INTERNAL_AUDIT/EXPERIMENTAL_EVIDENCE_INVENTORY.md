# DATA 266 HW12: Experimental Evidence Inventory

*Internal document.* Every number in the final report traces to one of the sources below. Paths are relative to `~/Desktop/hw12/` unless stated.

## 1. Files inspected

| File | Role | Executed? | Used in report? |
|---|---|---|---|
| `DATA266_HW12_GPTQ (2).ipynb` (md5 71a77d18…, identical to `Downloads/… (1)`) | **Final** Part 1 notebook | Yes, all outputs present | **Yes** |
| `Downloads/DATA266_HW12_GPTQ.ipynb` (older, md5 d6bf1061…) | Earlier GPTQ copy | Yes | No (superseded) |
| `DATA266_HW12_Distillation_SEED1598_Submission (1).ipynb` (md5 616e7a03…) | **Final** Part 2/3 notebook (seed 1598) | Yes, outputs preserved | **Yes** |
| `Downloads/DATA266_HW12_Distillation_Final_Completed (2).ipynb` | Master working notebook: seed-42 pilot plus seed-1598 recovery | Yes | Only for the **teacher confusion matrix** (cell 23; the teacher is deterministic, and its counts exactly reproduce the seed-1598 teacher accuracy and per-class values) |
| `Downloads/…Final_Completed.ipynb`, `(1)` | Earlier seed-42 versions | Yes | No |
| `Downloads/DATA266_HW12_Part2_Unscaled_KD_SEED1598_Colab.ipynb` | Corrected unscaled-KL objective | **No (no outputs)** | Mentioned only as "prepared, not executed" |
| `Downloads/DATA266_HW12_Part2_STUDENT_STYLE_SEED1598.ipynb` | Restyled variant | No outputs | No |
| `METRICS_DISTILLATION.md`, `RUN_LOG_DISTILLATION.txt`, `DATA266_HW12/RUN_LOG_GPTQ_DRAFT.txt` | Prior summaries | — | Cross-checked; all values agree with notebook outputs |
| `Downloads/DATA266_HW12-20261121T092834Z-1-001.zip` (1.7 GB) | Drive export | — | Listed only; contents identical to `DATA266_HW12/` |
| `DATA266_HW12/outputs/figures/*.png` | Old figures | — | **Not reused**: `selected_student_confusion_matrix.png` is the **seed-42** student (matches `part3_analysis/alpha_sweep/kd_T1_alpha025/final_test_metrics.json`) |
| `DATA266_HW12/outputs/part1_gptq/figures/*.png` | GPTQ figures | — | Superseded by the regenerated Fig. 6(b) |

## 2. Verified results recovered

### Part 1: GPTQ (NVIDIA L4, seed 1598)
| Quantity | FP16 | GPTQ 4-bit | Source |
|---|---|---|---|
| Avg NLL | 3.262023 (final env.; 3.262074 pre-upgrade) | 3.387793 | GPTQ cells 31, 11, 25; RUN_LOG_GPTQ_DRAFT |
| Perplexity | 26.1023 (26.1036 pre-upgrade) | 29.6005 | same |
| p50 / p95 / mean (ms) | 7.914 / 10.295 / 8.366 | 16.356 / 18.907 / 16.582 | cell 27 |
| Peak allocated GPU (GiB) | 0.2654 | 0.1481 | cell 27 |
| Saved weights (MiB) | 238.896 | 119.318 | cell 27 |
| Quantization time | — | 44.44 s; 12 layers; 72 modules; peak VRAM 0.38 GiB; 4.2875 bpw | cell 21 |
| Eval tokens | 287,644 total / 287,643 scored / 562 windows (1024/512) | same | cells 11, 25 |

### Part 2/3: Distillation (seed 1598; T²-scaled loss for all runs)
| Run | Val acc (%) | Val macro-F1 | Per-epoch history source |
|---|---|---|---|
| T=1, α=.5 | 94.7167 | 0.947245 | `temperature_T1_alpha05/training_history.json` |
| T=2, α=.5 | 94.7000 | 0.947056 | `kd_baseline_T2_alpha05/training_history.json` |
| T=5, α=.5 | 94.7750 | 0.947808 | `selected_temperature.json`; per-epoch values from notebook printout only (2 d.p.) |
| T=10, α=.5 | 94.7083 | 0.947146 | `temperature_T10_alpha05/experiment_summary.json` |
| **T=5, α=.25 (selected)** | **94.7917** | **0.947965** | `alpha_T5_alpha025/experiment_summary.json` |
| T=5, α=.75 | 94.6583 | 0.946652 | notebook printout only |

Final test (7,600): teacher 94.4605% / 0.944652; student 94.6579% / 0.946626; 92 teacher-correct/student-wrong. Per-class accuracy teacher/student: World 94.9474/95.4211, Sports 98.8421/98.8947, Business 91.3684/91.2105, Sci/Tech 92.6842/93.1053. Benchmark (Tesla T4, batch 1, 256 tokens): `final_test_evaluation/benchmark_config.json`. Parameter counts: teacher 109,485,316, student 66,956,548 (cells 9, 23).

## 3. Derived on CPU for this report (no training or inference)

Computed by `latex_source/other_required_files/derive_metrics.py` and written to `derived_metrics.json`:
- Teacher soft-target entropy and non-argmax mass at T ∈ {1,2,5,10}, from `teacher_all_120000_logits.npy`.
- Teacher per-class P/R/F1 from its confusion matrix. The recomputed macro-F1, 0.944652, equals the recorded value exactly.
- Paired contingency (7,087 both correct / 92 teacher only / 107 student only / 314 both wrong) and exact McNemar test, p = 0.321.
- 95% CI half-width for a 12,000-example validation accuracy: ±0.40 pp.
- Equivalent unscaled weight α̃ = α / (α + (1−α)T²).
- Analytical GPTQ size estimate: 67.8% of parameters quantized; ≈119–120 MiB predicted vs 119.3 MiB measured.

## 4. Evidence that is not recoverable without Google Drive or a GPU
- Student confusion matrix and per-class precision/F1 for the seed-1598 student (`paired_test_predictions.csv` and `per_class_accuracy.csv` exist only on Drive).
- Weights of the selected checkpoint (`alpha_T5_alpha025/checkpoint_epoch_3/model.safetensors` was not in the local export).
- `METRICS_GPTQ.md`, `fp16_wikitext_metrics.json`, `gptq_quantization_result.json` (Drive only; values are present in the notebook and the run log).

> If you can still open Google Drive, downloading `outputs/part2_distillation_seed1598/final_test_evaluation/paired_test_predictions.csv` would let the student's confusion matrix and per-class P/R/F1 be added **on CPU, without a GPU**.

## 5. Excluded (incompatible) results
- **Seed-42 pilot** (`outputs/part2_distillation/`, `outputs/part3_analysis/`; T=1 selected; test 94.70%; L4 benchmarks): this pilot violated the seed = SID4 rule and used BF16 autocast. It is not mixed into any table.
- GPTQ and KD results are never placed in the same table. Fig. 6 shows them in separate panels with an explicit warning that they are not comparable.
