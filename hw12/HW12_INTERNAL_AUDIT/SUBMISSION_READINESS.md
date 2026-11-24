# HW12: Final Readiness Assessment (internal, do not submit)

Prepared 24 Nov 2026. Supersedes the earlier version of this file. `REQUIREMENT_COVERAGE_AUDIT.md` and `EXPERIMENTAL_EVIDENCE_INVENTORY.md` in this folder remain valid, but some file paths in them predate the reorganisation: the report source now lives in `HW12_FINAL_SUBMISSION/report_source/`, the scripts in `src/`, and `derived_metrics.json` in `results/knowledge_distillation/`.

## Source-of-truth status
- **Original assignment PDF and rubric: still not found.** Searched hw12, Downloads, Documents, Desktop and Spotlight; no local Google Drive sync exists. Requirements come from the course notebooks (Part 1 = 5 pts; Part 3 Q2–Q4) and from the HW3 submission convention (private repo `data266-1598`, one folder per homework, tag per homework; required files: executed notebook, report PDF, METRICS.md, AI_USE.md, RUN_LOG.txt, checkpoints, TRAINING_COMMAND.txt).
- **No precise grade is given**, because the rubric is unavailable.

## Completed
- Part 1 GPTQ: every measurement executed and reported (PPL, p50/p95/mean latency, peak memory, weight size, trade-off analysis).
- Part 2 KD: seeded split, teacher-logit caches (108k and 120k), six fresh-student runs, validation-only selection.
- Part 3 Q1 (teacher vs student, including McNemar), Q2 (T sweep), Q3 (α sweep), Q4 (per-class accuracy, most-degraded class, five real examples with shared factor).
- Report (7 pages, IEEE), README, METRICS, RUN_LOG, TRAINING_COMMAND, AI_USE draft, recorded requirements, configs, reproducible CPU scripts.

## Partially completed
- Loss formula: every run uses the T²-scaled form. If the PDF mandates the unscaled form, results comply only at T = 1. This is disclosed and explained (report §IV-E, README §4).
- Q4 class-wise: the student's per-class precision/F1 and confusion matrix are unavailable (predictions exist only on Drive); per-class accuracy is reported.
- Environment record for KD: only transformers 5.22.1 was recorded.

## Unverified
- Instructor approval of OPT-125M (no evidence; not claimed).
- Whether a hard-label-only student baseline was required.
- Whether HP_ID / SLICE / CLS_A / CLS_B have a role in HW12.
- Page limit and exact repository URL/branch (username `HarithaGottumukkala` inferred from commit history; repo name `data266-1598` from the HW3 instructions).

## Missing evidence (needs Google Drive or a GPU)
- `paired_test_predictions.csv` and the other final-evaluation CSVs; all model checkpoints (each > 100 MB, so not uploadable to GitHub in any case); the T=5/α=0.5 and α=0.75 history JSON files (printed values were recovered instead).

## Remaining risks
1. Loss-convention mismatch (largest risk).
2. Checkpoints are not in the repo; the HW3 convention listed checkpoints as required. The README explains the GitHub size limit and the Drive location.
3. AI_USE.md needs the student's confirmation of the ☐ items (GPTQ code authorship, the student's own contributions).

## Issues resolved in this final preparation
- Package reorganised into a GitHub-ready `hw12/` layout; internal documents moved out.
- Report: course line set to “DATA 266 – Generative AI and Large Language Models”; appendix paths updated to the package; overfull lines fixed; still 7 pages, 0 errors/overfull/undefined; Overleaf ZIP recompiled from a fresh extract.
- Scripts made self-contained (`python3 src/derive_metrics.py` reproduces `derived_metrics.json` byte-for-byte).
- Duplicate figures removed (PNG in `figures/`, PDF in `report_source/figures/`); duplicate configs removed.
- Values that existed only in notebook printouts were transcribed into labelled JSON files with provenance.
- Final search for paired predictions: none recoverable locally.
- QA: every Markdown link resolves; no local paths, email addresses or secrets; largest file 2.6 MB; notebook copies are byte-identical to the executed originals; original files unmodified.
