# DATA 266 - Homework 2

**Student:** Haritha Gottumukkala  
**SID4:** 1598  
**SEED:** 1598  
**Course:** DATA 266 - Generative AI and Large Language Models  

## Overview

This homework contains three parts:

1. **Embedding-Based Transfer Learning**
   - Loaded the pretrained `word2vec-google-news-300` model.
   - Compared the top-3 nearest neighbors for:
     `cast`, `score`, `plot`, `screen`, and `review`.
   - Fine-tuned the embeddings using 25,000 IMDB movie reviews.
   - Compared the nearest neighbors before and after fine-tuning.
   - Measured the cosine similarity between the original and fine-tuned vectors.
   - Visualized the embedding shift using t-SNE.

2. **Retrieval-Augmented Generation (RAG)**
   - Loaded Wikipedia pages for 10 movies using LangChain.
   - Used chunk size `500` with overlap `50`.
   - Created embeddings using `sentence-transformers/all-MiniLM-L6-v2`.
   - Stored the chunk embeddings in FAISS.
   - Built the RAG pipeline explicitly using:
     - retriever,
     - prompt,
     - and FLAN-T5-Large.
   - Evaluated 5 questions using top-3 retrieval.
   - Manually checked the first relevant retrieved chunk for each question.
   - Tested an alternate chunking configuration of `300 / 75` for two retrieval cases.
   - Analyzed two retrieval-ranking failures.

3. **Training Optimization**
   - Compared:
     - CPU-to-GPU tensor transfer vs. GPU-resident data,
     - default vs. Kaiming initialization,
     - activation checkpointing,
     - gradient accumulation,
     - mixed precision training.
   - Used the same controlled neural-network setup for all comparisons.
   - Measured:
     - average training time,
     - peak GPU memory,
     - and final training loss.

## Main Results

### Part 1

| Word | Original vs. Fine-Tuned Cosine Similarity |
|---|---:|
| cast | 0.6646 |
| score | 0.5624 |
| plot | 0.6457 |
| screen | 0.6706 |
| review | 0.5391 |

- **Most shifted word:** `review`
- **Least shifted word:** `screen`

### Part 2

- Main chunking: `500 / 50`
- Total chunks: `3,573`
- Retriever: top-3
- Successful top-3 retrievals: `5 / 5`
- **Retrieval success rate: 100%**

Two questions had the direct answer-bearing passage at Rank 2:

- Titanic - Leonardo DiCaprio as Jack Dawson
- The Matrix - Neo takes the red pill

The alternate `300 / 75` chunking configuration did not improve the first relevant rank for either case.

### Part 3

| Experiment | Avg. Time (s) | Peak GPU Memory (MB) | Final Loss |
|---|---:|---:|---:|
| Baseline | 0.1337 | 1975.38 | 2.3033 |
| GPU-resident data | 0.1332 | 1983.41 | 2.3033 |
| Kaiming initialization | 0.1257 | 1975.38 | 2.3266 |
| Activation checkpointing | 0.2380 | 1976.38 | 2.3033 |
| Gradient accumulation x4 | 0.3173 | 1976.38 | 2.3033 |
| Mixed precision | 0.1711 | 1976.38 | 2.3018 |

The optimization results showed different trade-offs. No single technique improved training time, GPU memory, and loss at the same time for this small controlled experiment.

## Files

- `Homework2_1598.ipynb` - executed Google Colab notebook with code, outputs, plots, and written analysis.
- `RUN_LOG.txt` - final execution environment, run details, outputs, warnings, and debugging notes.
- `METRICS.md` - numerical results and comparison tables.
- `AI_USE.md` - required AI-use appendix.
- `data_266_1598_hw2.pdf` - Homework 2 technical report.
- `README.md` - overview of this Homework 2 submission.

## Execution Environment

The final notebook was run in Google Colab.

- GPU: NVIDIA Tesla T4
- Python: 3.13.15
- PyTorch: 2.11.0+cu128
- Gensim: 4.4.0
- Pandas: 2.2.3
- Scikit-learn: 1.6.1
- Seed: 1598

## Reproducibility

The notebook was executed with outputs preserved.

Random seeds were fixed using `SEED = 1598` where applicable.  
The training optimization experiments used one warm-up run followed by three measured runs.

The notebook is intended to run from top to bottom in Google Colab with the required dependencies installed.

## Submission

Homework 2 is stored in the `hw2/` directory of the same DATA 266 repository used for Homework 1.

The submission is tagged as:

`hw2`
