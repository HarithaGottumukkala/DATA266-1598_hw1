# DATA 266 – Homework 12: Metrics

Every value below comes from an executed notebook output or a saved artifact in this repository (source given per table). Values derived later on CPU are marked **(derived)** and come from [`src/derive_metrics.py`](src/derive_metrics.py) → [`results/knowledge_distillation/derived_metrics.json`](results/knowledge_distillation/derived_metrics.json). Seed = SID4 = 1598 throughout. MiB/GiB are binary units.

---

## Part 1 – GPTQ (facebook/opt-125m, NVIDIA L4)

Source: [`notebooks/DATA266_HW12_Part1_GPTQ_executed.ipynb`](notebooks/DATA266_HW12_Part1_GPTQ_executed.ipynb) (cells 7, 11, 21, 25, 27, 31), transcribed in [`results/gptq/gptq_metrics_from_notebook_outputs.json`](results/gptq/gptq_metrics_from_notebook_outputs.json).

### Original model

| Property | Value |
|---|---|
| Model / revision | `facebook/opt-125m` / `27dcfa74d334bc871f3234de431e71c6eeba5dd6` |
| Architecture | OPTForCausalLM, 12 layers, vocab 50,272 |
| Parameters | 125,239,296 |
| Precision | FP16 |

### Quantization

| Setting / outcome | Value |
|---|---|
| Method / library | GPTQ / GPTQModel 7.12.0 |
| Bits / group size / desc_act / lm_head | 4 / 128 / False / not quantized |
| Calibration | 128 WikiText-2 train texts, ≤ 256 tokens, 18,595 tokens, batch 1 |
| Layers / modules quantized | 12 / 72 |
| Quantization time / peak VRAM | 44.44 s / 0.38 GiB |
| Estimated bits per weight (GPTQModel) | 4.2875 |

### Quality: WikiText-2 test (287,644 tokens; 287,643 scored; 562 windows; window 1024, stride 512)

| Model | Avg NLL | Perplexity |
|---|---:|---:|
| FP16 (initial environment, torch 2.14.1) | 3.262074 | 26.1036 |
| FP16 (final environment, torch 2.15.0), **used for comparison** | 3.262023 | **26.1023** |
| GPTQ 4-bit | 3.387793 | **29.6005** |
| Change (GPTQ vs final FP16) | +0.125770 | **+3.4983 (+13.40 %)** |

### Efficiency (batch 1, 128 seeded random tokens, 15 warm-up + 80 timed forward passes)

| Metric | FP16 | GPTQ 4-bit | Change |
|---|---:|---:|---:|
| p50 latency (ms) | 7.914 | 16.356 | 2.07× slower |
| p95 latency (ms) | 10.295 | 18.907 | 1.84× slower |
| Mean latency (ms) | 8.366 | 16.582 | 1.98× slower |
| Peak allocated GPU memory (GiB) | 0.2654 | 0.1481 | −44.21 % |
| Saved weight size (MiB) | 238.896 | 119.318 | −50.05 % |

GPTQModel's own log reports 478.03 MB → 119.27 MB (−75.05 %), measured against the **FP32** pre-quantization size. Against FP16 the reduction is 50.05 %. **(derived)** 67.8 % of parameters are in quantized linear layers; the embeddings (40.18 M parameters) stay FP16. The analytical size estimate is ≈ 119–120 MiB, consistent with the measured 119.3 MiB.

---

## Parts 2–3 – Knowledge distillation (AG News, seed 1598)

### Models

| | Teacher | Student |
|---|---|---|
| Model | instructor-provided BERT-base (`BertForSequenceClassification`) | `distilbert/distilbert-base-uncased` |
| Layers / hidden | 12 / 768 | 6 / 768 |
| Parameters | 109,485,316 | 66,956,548 (−38.84 %) |

### Data split

| Split | Size | Per class |
|---|---:|---:|
| Train (stratified, seed 1598) | 108,000 | 27,000 |
| Validation | 12,000 | 3,000 |
| Official test | 7,600 | 1,900 |

### Hyperparameters (all runs)

3 epochs · batch 16 · AdamW lr 2e-5 · weight decay 0.01 · warm-up 6 % (1,215 / 20,250 steps) · grad clip 1.0 · max length 256 · FP32 · loss `α·CE + (1−α)·T²·KL` (α = hard-label weight). Source: [`results/knowledge_distillation/training_config_seed1598.json`](results/knowledge_distillation/training_config_seed1598.json).

### Validation results: all six runs (best epoch = 3 for every run)

| Sweep | T | α | Val accuracy (%) | Val macro-F1 | Equivalent unscaled α̃ **(derived)** | Source |
|---|---:|---:|---:|---:|---:|---|
| Q2 | 1 | 0.50 | 94.7167 | 0.947245 | 0.500 | `temperature_T1_alpha05/` |
| Q2 | 2 | 0.50 | 94.7000 | 0.947056 | 0.200 | `kd_baseline_T2_alpha05/` |
| Q2 | **5** | 0.50 | **94.7750** | **0.947808** | 0.038 | `selected_temperature.json` |
| Q2 | 10 | 0.50 | 94.7083 | 0.947146 | 0.010 | `temperature_T10_alpha05/` |
| Q3 | 5 | **0.25** | **94.7917** | **0.947965** | 0.013 | `alpha_T5_alpha025/` |
| Q3 | 5 | 0.50 | 94.7750 | 0.947808 | 0.038 | (shared with Q2) |
| Q3 | 5 | 0.75 | 94.6583 | 0.946652 | 0.107 | notebook cell 34 |

Selected configuration: **T = 5, α = 0.25, epoch 3** (validation accuracy, tie-break macro-F1). α̃ = α / (α + (1−α)T²) is the weight under which the T²-scaled loss is proportional to the unscaled loss `α̃·CE + (1−α̃)·KL`. The two coincide at T = 1.

**(derived)** The 95 % binomial half-width for 12,000 validation examples at ≈ 94.8 % is **± 0.40 pp**. Every sweep difference (≤ 0.13 pp) lies inside it.

### Per-epoch histories

| Run | Epoch | Train total loss | Train CE | Train T²·KL | Val CE | Val acc (%) | Val macro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| T=1, α=.5 | 1 | 0.2603 | 0.2759 | 0.2446 | 0.1947 | 93.6833 | 0.936794 |
| | 2 | 0.1268 | 0.1434 | 0.1101 | 0.1762 | 94.6250 | 0.946267 |
| | 3 | 0.0806 | 0.0963 | 0.0648 | 0.1897 | 94.7167 | 0.947245 |
| T=2, α=.5 | 1 | 0.5322 | 0.3231 | 0.7414 | 0.2451 | 93.5833 | 0.935880 |
| | 2 | 0.2275 | 0.1762 | 0.2789 | 0.2280 | 94.4417 | 0.944444 |
| | 3 | 0.1406 | 0.1189 | 0.1623 | 0.2284 | 94.7000 | 0.947056 |
| T=10, α=.5 | 1 | 0.8231 | 0.3516 | 1.2946 | 0.2513 | 93.9833 | 0.939882 |
| | 2 | 0.2975 | 0.1896 | 0.4053 | 0.2489 | 94.4583 | 0.944611 |
| | 3 | 0.1882 | 0.1296 | 0.2469 | 0.2351 | 94.7083 | 0.947146 |
| T=5, α=.25 | 1 | 1.0646 | 0.3589 | 1.2999 | 0.2598 | 93.9500 | 0.939536 |
| | 2 | 0.3607 | 0.1954 | 0.4158 | 0.2514 | 94.5167 | 0.945159 |
| | 3 | 0.2228 | 0.1375 | 0.2513 | 0.2361 | 94.7917 | 0.947965 |
| T=5, α=.5 † | 1 / 2 / 3 | 0.8305 / 0.3052 / 0.1916 | – | – | – | 93.93 / 94.50 / 94.7750 | 0.9394 / 0.9450 / 0.947808 |
| T=5, α=.75 † | 1 / 2 / 3 | 0.5866 / 0.2427 / 0.1544 | – | – | – | 93.77 / 94.50 / 94.6583 | 0.9377 / 0.9450 / 0.946652 |

† History files not exported; values printed in the notebook ([`notebook_recorded_epoch_values.json`](results/knowledge_distillation/notebook_recorded_epoch_values.json)). Total losses are not comparable across temperatures because of the T² factor.

### Final test results (7,600 articles, Tesla T4)

Source: notebook cells 38, 40, 43; [`final_test_evaluation/`](results/knowledge_distillation/final_test_evaluation/).

| Metric | BERT teacher | DistilBERT student (T=5, α=0.25) | Change |
|---|---:|---:|---:|
| Accuracy | 94.4605 % | 94.6579 % | +0.1974 pp |
| Macro-F1 | 0.944652 | 0.946626 | +0.001974 |
| Errors | 421 | 406 | −15 |
| p50 latency (ms) | 16.7124 | 8.5814 | 1.95× faster (−48.65 %) |
| p95 latency (ms) | 17.1056 | 8.9077 | 1.92× faster |
| Peak allocated GPU memory (GiB) | 0.4264 | 0.2680 | −37.15 % |
| Checkpoint directory (MiB) | 418.5823 | 255.4317 | −38.98 % |
| Parameters | 109,485,316 | 66,956,548 | −38.84 % |

Benchmark: batch 1, padded to 256 tokens, same 100 test texts, 20 warm-up + 100 CUDA-synchronised timed passes; tokenization and model loading excluded.

### Statistical significance **(derived)**

| | Student correct | Student wrong |
|---|---:|---:|
| **Teacher correct** | 7,087 | 92 |
| **Teacher wrong** | 107 | 314 |

The exact McNemar test on 92 vs 107 discordant pairs gives **p = 0.321** (χ² with continuity correction = 0.985). The student and teacher are statistically indistinguishable in accuracy. (The 92 is recorded; the other cells follow from the recorded correct counts 7,179 and 7,194.)

### Class-wise metrics

| Class | Teacher precision | Teacher recall | Teacher F1 | Student accuracy (= recall) | Student − teacher |
|---|---:|---:|---:|---:|---:|
| World | 0.9626 | 0.9495 | 0.9560 | 95.4211 % (1,813/1,900) | +0.4737 pp |
| Sports | 0.9869 | 0.9884 | 0.9876 | 98.8947 % (1,879/1,900) | +0.0526 pp |
| **Business** | 0.9166 | 0.9137 | 0.9151 | 91.2105 % (1,733/1,900) | **−0.1579 pp** |
| Sci/Tech | 0.9129 | 0.9268 | 0.9198 | 93.1053 % (1,769/1,900) | +0.4211 pp |
| Macro | 0.9448 | 0.9446 | **0.944652** | 94.6579 % | +0.1974 pp |

Teacher precision/recall/F1 **(derived)** come from the teacher's test confusion matrix (below). The recomputed macro-F1 matches the recorded 0.944652 exactly. **Student precision and F1 per class are not available**: the student's paired predictions were saved only to Google Drive and were not printed.

Teacher confusion matrix (rows = true, columns = predicted), from [`notebooks/supporting/DATA266_HW12_Distillation_working_notebook_executed.ipynb`](notebooks/supporting/DATA266_HW12_Distillation_working_notebook_executed.ipynb), cell 23:

| | World | Sports | Business | Sci/Tech |
|---|---:|---:|---:|---:|
| World | 1804 | 10 | 50 | 36 |
| Sports | 9 | 1878 | 6 | 7 |
| Business | 33 | 6 | 1736 | 125 |
| Sci/Tech | 28 | 9 | 102 | 1761 |

Business ↔ Sci/Tech confusion accounts for 227 of the teacher's 421 errors (53.9 %).

### Error analysis: teacher correct, student wrong (92 total; five examples, all true class Business)

| Test index | Headline | Student prediction |
|---:|---|---|
| 110 | Yahoo! Ups Ante for Small Businesses | Sci/Tech |
| 399 | Nortel Downsizes Again | Sci/Tech |
| 408 | Chavez rejects CD as opposition | World |
| 467 | Deal has S2io champing at the gigabit | Sci/Tech |
| 1124 | Judge: Geico can sue Google, Overture over ads | Sci/Tech |

Common factor: business stories about technology companies or political figures. The label depends on how the story is framed (pricing, layoffs, alliances, litigation) rather than on the entities named. This is an interpretation of five examples, not a causal finding.

### Teacher soft targets vs. temperature **(derived from the saved 120,000 × 4 logit cache)**

| T | Mean entropy / log 4 | Mean max prob. | Mean non-argmax mass |
|---:|---:|---:|---:|
| 1 | 0.014 | 0.995 | 0.005 |
| 2 | 0.165 | 0.949 | 0.051 |
| 5 | 0.726 | 0.654 | 0.346 |
| 10 | 0.931 | 0.446 | 0.554 |

Median teacher top-1 − top-2 logit margin: 7.69.

---

## Observations

1. **GPTQ** halves stored weights and cuts peak memory by 44 %. On this small model at batch 1 it costs 13.4 % perplexity and about 2× latency, because dequantization overhead outweighs bandwidth savings.
2. **Distillation** preserves accuracy (p = 0.32) while halving latency and cutting size and memory by about 38 %.
3. **Temperature and α** have effects smaller than validation noise. The teacher is nearly one-hot at T = 1, so a moderate T is needed for the soft targets to carry information.
4. **Business** is the only degraded class; Business/Sci/Tech ambiguity dominates errors in both models.
5. GPTQ and distillation numbers come from different models, tasks and GPUs and must not be compared directly.
