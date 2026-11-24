# DATA 266 – Generative AI and Large Language Models: Homework 12

**Model compression for Transformer inference: GPTQ 4-bit quantization and knowledge distillation**

Haritha Gottumukkala · San José State University · SID4 = 1598

📄 **Final report:** [`DATA266_HW12_Final_Report.pdf`](DATA266_HW12_Final_Report.pdf) (IEEE two-column, 7 pages)
📊 **All metrics:** [`METRICS.md`](METRICS.md) · 🧾 **Execution log:** [`RUN_LOG.txt`](RUN_LOG.txt) · 🤖 **AI-use disclosure:** [`AI_USE.md`](AI_USE.md) · ▶️ **How the runs were executed:** [`TRAINING_COMMAND.txt`](TRAINING_COMMAND.txt)

---

## 1. Overview and objectives

| Part | Task | Notebook |
|---|---|---|
| **Part 1: GPTQ** | Quantize a causal LLM (`facebook/opt-125m`) from FP16 to 4-bit with GPTQ. Compare WikiText-2 perplexity, p50/p95 latency, peak GPU memory and saved weight size | [`notebooks/DATA266_HW12_Part1_GPTQ_executed.ipynb`](notebooks/DATA266_HW12_Part1_GPTQ_executed.ipynb) |
| **Part 2: Knowledge distillation** | Distil the instructor-provided BERT AG News classifier (teacher) into `distilbert-base-uncased` (student) using cached teacher logits and a combined CE + KL objective | [`notebooks/DATA266_HW12_Part2_3_Distillation_SEED1598_executed.ipynb`](notebooks/DATA266_HW12_Part2_3_Distillation_SEED1598_executed.ipynb) |
| **Part 3: Analysis** | Q1 teacher vs. student (accuracy, F1, speed, size); Q2 temperature sweep T ∈ {1, 2, 5, 10}; Q3 α sweep {0.25, 0.5, 0.75}; Q4 per-class degradation and five teacher-correct / student-wrong examples | same notebook |

Personal parameters (course standing rule SEED = SID4): `SID4 = 1598, SEED = 1598, SLICE = 598, HP_ID = 2, CLS_A = 8, CLS_B = 5`. Seed 1598 was used for the data split, calibration sampling, student initialisation, shuffling and benchmark inputs.

## 2. Main results

**Part 1: GPTQ, OPT-125M, NVIDIA L4**

| Metric | FP16 | GPTQ 4-bit (g = 128) | Change |
|---|---:|---:|---:|
| WikiText-2 perplexity ↓ | 26.1023 | 29.6005 | +13.40 % |
| p50 latency, batch 1, 128 tok (ms) ↓ | 7.914 | 16.356 | 2.07× slower |
| p95 latency (ms) ↓ | 10.295 | 18.907 | 1.84× slower |
| Peak allocated GPU memory (GiB) ↓ | 0.2654 | 0.1481 | −44.21 % |
| Saved weight size (MiB) ↓ | 238.896 | 119.318 | −50.05 % |

**Part 2/3: distillation, AG News test set (7,600 articles); efficiency on a Tesla T4**

| Metric | BERT teacher | DistilBERT student (T = 5, α = 0.25) |
|---|---:|---:|
| Parameters | 109,485,316 | 66,956,548 (−38.84 %) |
| Test accuracy | 94.4605 % | 94.6579 % |
| Test macro-F1 | 0.944652 | 0.946626 |
| p50 / p95 latency (ms) | 16.712 / 17.106 | 8.581 / 8.908 (1.95× faster) |
| Peak allocated GPU memory (GiB) | 0.4264 | 0.2680 (−37.15 %) |
| Checkpoint directory (MiB) | 418.58 | 255.43 (−38.98 %) |

The student's +0.20 pp is **not statistically significant** (exact McNemar test, p = 0.32): distillation preserved teacher accuracy at about half the inference cost. Business was the only class to degrade (−0.16 pp), and Business ↔ Sci/Tech confusion dominates the errors. Full tables are in [`METRICS.md`](METRICS.md).

## 3. Data and models

| Item | Details |
|---|---|
| GPTQ model | `facebook/opt-125m`, revision `27dcfa74d334bc871f3234de431e71c6eeba5dd6`, 125,239,296 parameters. Chosen because it fits an L4 comfortably; instructor approval of this specific model is **not documented** in the project records |
| GPTQ data | `Salesforce/wikitext`, `wikitext-2-raw-v1`: 128 train texts (≤ 256 tokens) for calibration; full test split (287,644 tokens) for perplexity |
| Teacher | Instructor-provided local BERT-base checkpoint (config origin `odunola/bert-base-uncased-ag-news-finetuned`), 12 layers, 4 classes. Config: [`configs/teacher_model_config.json`](configs/teacher_model_config.json) |
| Student | `distilbert/distilbert-base-uncased` with a new 4-way head, 6 layers |
| KD data | `fancyzhx/ag_news`: stratified 108,000 / 12,000 train/validation split (seed 1598); official 7,600-article test set used once |

## 4. Methodology summary

**GPTQ.** Layer-wise post-training quantization with GPTQModel 7.12.0: 4-bit weights, group size 128, `desc_act=False`, `lm_head` not quantized. Perplexity uses a 1,024-token sliding window with stride 512. Latency uses 15 warm-ups and 80 CUDA-synchronised timed forward passes on seeded random 128-token inputs. Configuration: [`configs/gptq_quantization_config.json`](configs/gptq_quantization_config.json).

**Distillation.** Teacher logits were cached once (108,000 × 4 training rows; a full 120,000 × 4 cache was produced later, and validation logits were never used for training). Every run trained a fresh student with seed 1598 for 3 epochs: batch 16, AdamW at 2e-5, weight decay 0.01, 6 % warm-up, gradient clip 1.0, max length 256, FP32. Checkpoints were selected on **validation accuracy only**.

**Loss function (important).** All six executed runs used the Hinton-style **temperature-squared** objective

```
L = α · CE(z_s, y) + (1 − α) · T² · KL( softmax(z_t / T) || softmax(z_s / T) )
```

with α as the hard-label weight. The unscaled form `α·CE + (1 − α)·KL` is **identical at T = 1** and differs only by the constant factor T² on the KL term for T > 1 (see report §IV-E). A notebook implementing the unscaled form was prepared but **not executed**, so no unscaled results exist: [`notebooks/supporting/DATA266_HW12_Part2_Unscaled_KD_NOT_EXECUTED.ipynb`](notebooks/supporting/DATA266_HW12_Part2_Unscaled_KD_NOT_EXECUTED.ipynb).

Six seed-1598 runs: T ∈ {1, 2, 5, 10} at α = 0.5, then α ∈ {0.25, 0.75} at the selected T = 5 (the T = 5, α = 0.5 run is shared between the two sweeps).

## 5. Hardware and software

| | Part 1 (GPTQ) | Parts 2–3 (distillation) |
|---|---|---|
| Platform | Google Colab | Google Colab |
| GPU | NVIDIA L4 (22 GiB) | Training: NVIDIA L4 · Final test + benchmark: Tesla T4 |
| Recorded versions | Python 3.13.16, PyTorch 2.15.0+cu132, Transformers 5.23.0, Datasets 5.2.0, GPTQModel 7.12.0 | Transformers 5.22.1 (from saved checkpoint configs); other versions not recorded |
| Files | [`requirements/requirements_gptq_recorded.txt`](requirements/requirements_gptq_recorded.txt) | [`requirements/requirements_distillation_recorded.txt`](requirements/requirements_distillation_recorded.txt) |

## 6. Repository structure

```
hw12/
├── README.md                     this file
├── DATA266_HW12_Final_Report.pdf IEEE report (final submission)
├── METRICS.md                    every reported number, with its source
├── RUN_LOG.txt                   execution log reconstructed from recorded outputs and timestamps
├── AI_USE.md                     AI-assistance disclosure
├── TRAINING_COMMAND.txt          exact cells/functions and settings used to run each experiment
├── notebooks/
│   ├── DATA266_HW12_Part1_GPTQ_executed.ipynb                    Part 1 (executed, outputs preserved)
│   ├── DATA266_HW12_Part2_3_Distillation_SEED1598_executed.ipynb Parts 2–3 (executed, outputs preserved)
│   └── supporting/
│       ├── DATA266_HW12_Distillation_working_notebook_executed.ipynb  full working history incl. teacher evaluation
│       └── DATA266_HW12_Part2_Unscaled_KD_NOT_EXECUTED.ipynb          prepared, never run (no results)
├── results/
│   ├── gptq/                     model/dataset metadata, metrics extracted from notebook outputs, Colab log, Colab figures
│   └── knowledge_distillation/   seed-1598 histories, configs, selection files, benchmark, teacher logits, derived statistics
├── figures/                      report figures (PNG)
├── src/                          CPU-only scripts that regenerate derived statistics and figures
├── configs/                      GPTQ config, teacher model/tokenizer config
├── requirements/                 recorded package versions per environment
└── report_source/                main.tex, references.bib, figures/ (PDF), DATA266_HW12_Overleaf.zip
```

## 7. Reproducing the work

**Notebook execution order (Google Colab GPU runtime, Google Drive mounted at `/content/drive/MyDrive/DATA266_HW12`):**

1. `notebooks/DATA266_HW12_Part1_GPTQ_executed.ipynb`: run top to bottom. Install GPTQModel in cell 15, **restart the runtime**, and if the `AutoProcessor` / TorchAudio CUDA error appears, run the `pip uninstall -y torchaudio` cell (18) before continuing.
2. `notebooks/DATA266_HW12_Part2_3_Distillation_SEED1598_executed.ipynb`: requires the instructor's `teacher_model/` folder at `DATA266_HW12/teacher_model`. Training cells **refuse to overwrite** populated experiment folders, so to reproduce from scratch point the output paths at an empty directory. Do not use “Run all” against existing results.

See [`TRAINING_COMMAND.txt`](TRAINING_COMMAND.txt) for the cell-by-cell settings.

**Regenerate the report's derived statistics and figures (CPU only, no GPU):**

```bash
cd hw12
python3 src/derive_metrics.py   # -> results/knowledge_distillation/derived_metrics.json
python3 src/make_figures.py     # -> figures/*.png and report_source/figures/*.pdf
```

Requires only `numpy` and `matplotlib` ([`requirements/requirements_analysis_cpu.txt`](requirements/requirements_analysis_cpu.txt)).

**Rebuild the report:** upload [`report_source/DATA266_HW12_Overleaf.zip`](report_source/DATA266_HW12_Overleaf.zip) to Overleaf (pdfLaTeX), or locally run `cd report_source && pdflatex main && bibtex main && pdflatex main && pdflatex main`.

## 8. Figures and artifacts

| Figure | File |
|---|---|
| Training dynamics (validation accuracy; train vs. validation CE) | [`figures/fig_training_curves.png`](figures/fig_training_curves.png) |
| Temperature and α sweeps with 95 % CIs | [`figures/fig_sweeps.png`](figures/fig_sweeps.png) |
| Teacher soft-target entropy vs. temperature (from saved logits) | [`figures/fig_soft_targets.png`](figures/fig_soft_targets.png) |
| Per-class accuracy and teacher confusion matrix | [`figures/fig_perclass.png`](figures/fig_perclass.png) |
| Relative efficiency changes: distillation and GPTQ | [`figures/fig_efficiency.png`](figures/fig_efficiency.png) |
| Original Colab GPTQ plots | [`results/gptq/colab_figures/`](results/gptq/colab_figures/) |

**Model checkpoints are not in this repository.** Every checkpoint exceeds GitHub's 100 MB per-file limit (BERT teacher ≈ 438 MB; each DistilBERT checkpoint ≈ 268 MB; FP16 OPT-125M ≈ 251 MB). They are kept in the Google Drive workspace `DATA266_HW12/outputs/` (selected student: `part2_distillation_seed1598/alpha_T5_alpha025/checkpoint_epoch_3`; GPTQ model: `part1_gptq/opt125m_gptq_4bit`) and can be shared on request.

## 9. Known limitations

- All distillation runs use the T²-scaled loss (see §4). No hard-label-only (α = 1) student was trained, so the benefit of distillation over plain fine-tuning is not isolated.
- Single seed. The sweep differences (≤ 0.13 pp) are smaller than the ±0.40 pp validation confidence interval.
- The student's paired test predictions were saved only to Google Drive, so its confusion matrix and per-class precision/F1 are not available. Per-class accuracy (recall) is reported.
- GPTQ covers one small model and one configuration. Calibration used 128 texts (below GPTQModel's recommendation), and latency is batch-1 forward passes, not generation throughput.
- The original assignment PDF was not available when this package was assembled; requirements were taken from the course notebooks.

## 10. References

1. G. Hinton, O. Vinyals, J. Dean, “Distilling the Knowledge in a Neural Network,” arXiv:1503.02531, 2015.
2. E. Frantar, S. Ashkboos, T. Hoefler, D. Alistarh, “GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers,” ICLR 2023.
3. V. Sanh, L. Debut, J. Chaumond, T. Wolf, “DistilBERT, a distilled version of BERT,” arXiv:1910.01108, 2019.
4. J. Devlin et al., “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding,” NAACL-HLT 2019.
5. S. Zhang et al., “OPT: Open Pre-trained Transformer Language Models,” arXiv:2205.01068, 2022.
6. X. Zhang, J. Zhao, Y. LeCun, “Character-level Convolutional Networks for Text Classification” (AG News), NeurIPS 2015.
7. S. Merity et al., “Pointer Sentinel Mixture Models” (WikiText), ICLR 2017.

The complete bibliography (19 entries) is in [`report_source/references.bib`](report_source/references.bib).
