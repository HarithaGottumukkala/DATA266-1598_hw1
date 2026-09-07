# DATA 266 - Homework 2 Metrics

**SID4:** 1598
**SEED:** 1598
**Runtime:** Google Colab
**GPU:** Tesla T4

---

## Part 1 - Embedding-Based Transfer Learning

### Dataset and Model Summary

| Metric                         |                     Result |
| ------------------------------ | -------------------------: |
| Pretrained model               | `word2vec-google-news-300` |
| Pretrained embedding dimension |                        300 |
| Pretrained vocabulary size     |                  3,000,000 |
| IMDB training reviews used     |                     25,000 |
| Trainable IMDB vocabulary size |                     46,149 |
| Words shared with Google News  |                     36,450 |
| Pretrained vocabulary coverage |                     78.98% |
| Fine-tuning epochs             |                          5 |

### Target Word Frequency in IMDB Training Data

| Word   | Count |
| ------ | ----: |
| cast   |  3829 |
| score  |  1030 |
| plot   |  6589 |
| screen |  2492 |
| review |   850 |

### Nearest Neighbors Before and After Fine-Tuning

| Word   | Rank | Before Fine-Tuning | Cosine | After Fine-Tuning | Cosine |
| ------ | ---: | ------------------ | -----: | ----------------- | -----: |
| cast   |    1 | casts              | 0.7219 | supporting        | 0.6074 |
| cast   |    2 | casting            | 0.7188 | actors            | 0.5672 |
| cast   |    3 | Cast               | 0.6638 | ensemble          | 0.5554 |
| score  |    1 | scoring            | 0.7197 | nicolai           | 0.6278 |
| score  |    2 | scores             | 0.6596 | music             | 0.6146 |
| score  |    3 | scored             | 0.6384 | ennio             | 0.5992 |
| plot   |    1 | plots              | 0.7625 | storyline         | 0.6451 |
| plot   |    2 | Plot               | 0.6524 | plots             | 0.6260 |
| plot   |    3 | plotting           | 0.6328 | story             | 0.6125 |
| screen |    1 | screens            | 0.7729 | screens           | 0.5509 |
| screen |    2 | onscreen           | 0.6115 | onscreen          | 0.4936 |
| screen |    3 | LCD_screen         | 0.5599 | piedras           | 0.4413 |
| review |    1 | reviewed           | 0.6630 | comment           | 0.6392 |
| review |    2 | reviewing          | 0.6610 | reviews           | 0.5871 |
| review |    3 | reviews            | 0.6380 | comments          | 0.5545 |

### Original vs. Fine-Tuned Embedding Similarity

| Word   | Cosine Similarity |
| ------ | ----------------: |
| cast   |            0.6646 |
| score  |            0.5624 |
| plot   |            0.6457 |
| screen |            0.6706 |
| review |            0.5391 |

**Most shifted word:** `review` — cosine similarity = **0.5391**

**Least shifted word:** `screen` — cosine similarity = **0.6706**

---

## Part 2 - Retrieval-Augmented Generation

### Main RAG Configuration

| Setting                       | Value                                    |
| ----------------------------- | ---------------------------------------- |
| Number of Wikipedia documents | 10                                       |
| Chunk size                    | 500                                      |
| Chunk overlap                 | 50                                       |
| Total chunks                  | 3,573                                    |
| Embedding model               | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store                  | FAISS                                    |
| Retriever top-k               | 3                                        |
| Generation model              | `google/flan-t5-large`                   |

### Retrieval Evaluation

| Question                                         | Top-3 Success | First Relevant Rank | Generated Answer             |
| ------------------------------------------------ | ------------- | ------------------: | ---------------------------- |
| Who directed Inception?                          | Yes           |                   1 | Christopher Nolan            |
| Who played Jack Dawson in Titanic?               | Yes           |                   2 | Leonardo DiCaprio            |
| Which actor played the Joker in The Dark Knight? | Yes           |                   1 | Heath Ledger                 |
| What color pill does Neo take in The Matrix?     | Yes           |                   2 | red                          |
| Who composed the music for Gladiator?            | Yes           |                   1 | Hans Zimmer and Lisa Gerrard |

### Retrieval Success Rate

| Metric                      |      Result |
| --------------------------- | ----------: |
| Successful top-3 retrievals |           5 |
| Total questions             |           5 |
| Retrieval success rate      | **100.00%** |

### Alternate Chunking Experiment

| Setting         | Main Configuration | Alternate Configuration |
| --------------- | -----------------: | ----------------------: |
| Chunk size      |                500 |                     300 |
| Chunk overlap   |                 50 |                      75 |
| Total chunks    |              3,573 |                   6,416 |
| Retriever top-k |                  3 |                       3 |

### Failure-Case Comparison

| Question                                     | Configuration | First Relevant Rank | Generated Answer  |
| -------------------------------------------- | ------------- | ------------------: | ----------------- |
| Who played Jack Dawson in Titanic?           | 500 / 50      |                   2 | Leonardo DiCaprio |
| Who played Jack Dawson in Titanic?           | 300 / 75      |                   2 | Leonardo DiCaprio |
| What color pill does Neo take in The Matrix? | 500 / 50      |                   2 | red               |
| What color pill does Neo take in The Matrix? | 300 / 75      |                   2 | red               |

The alternate chunking configuration did not improve the first relevant rank for either failure case.

---

## Part 3 - Training Optimization

### Controlled Experimental Settings

| Setting                         | Value             |
| ------------------------------- | ----------------- |
| Model                           | ExperimentNetwork |
| Input dimension                 | 512               |
| Hidden dimension                | 1024              |
| Number of classes               | 10                |
| Dataset size                    | 4096              |
| Effective batch size            | 64                |
| Training steps                  | 50                |
| Optimizer                       | Adam              |
| Learning rate                   | 0.001             |
| Seed                            | 1598              |
| GPU                             | Tesla T4          |
| Warm-up runs                    | 1                 |
| Measured runs per configuration | 3                 |

### Baseline

| Avg Time (s) | Avg Peak GPU Memory (MB) | Avg Final Loss |
| -----------: | -----------------------: | -------------: |
|     0.133675 |              1975.382324 |       2.303293 |

### Tensor Placement: CPU-to-GPU vs. GPU-Resident Data

| Configuration      | Avg Time (s) | Avg Peak GPU Memory (MB) | Avg Final Loss |
| ------------------ | -----------: | -----------------------: | -------------: |
| CPU-to-GPU batches |     0.125763 |              1975.382324 |       2.303293 |
| GPU-resident data  |     0.133244 |              1983.413574 |       2.303293 |

### Weight Initialization

| Configuration          | Avg Time (s) | Avg Peak GPU Memory (MB) | Avg Final Loss |
| ---------------------- | -----------: | -----------------------: | -------------: |
| Default Initialization |     0.128300 |              1975.382324 |       2.303293 |
| Kaiming Initialization |     0.125693 |              1975.382324 |       2.326585 |

### Activation Checkpointing

| Configuration                 | Avg Time (s) | Avg Peak GPU Memory (MB) | Avg Final Loss |
| ----------------------------- | -----------: | -----------------------: | -------------: |
| Without Checkpointing         |     0.127737 |              1975.382324 |       2.303293 |
| With Activation Checkpointing |     0.237984 |              1976.382324 |       2.303293 |

### Gradient Accumulation

| Configuration            | Avg Time (s) | Avg Peak GPU Memory (MB) | Avg Final Loss |
| ------------------------ | -----------: | -----------------------: | -------------: |
| No Gradient Accumulation |     0.138447 |              1976.382324 |       2.303293 |
| Gradient Accumulation x4 |     0.317288 |              1976.380859 |       2.303293 |

### Mixed Precision

| Configuration            | Avg Time (s) | Avg Peak GPU Memory (MB) | Avg Final Loss |
| ------------------------ | -----------: | -----------------------: | -------------: |
| FP32 Training            |     0.126003 |              1976.382324 |       2.303293 |
| Mixed Precision Training |     0.171087 |              1976.382812 |       2.301819 |

### Final Optimization Summary

| Experiment                    | Avg Time (s) | Avg Peak GPU Memory (MB) | Avg Final Loss |
| ----------------------------- | -----------: | -----------------------: | -------------: |
| Baseline                      |       0.1337 |                  1975.38 |         2.3033 |
| GPU-resident data             |       0.1332 |                  1983.41 |         2.3033 |
| Kaiming Initialization        |       0.1257 |                  1975.38 |         2.3266 |
| With Activation Checkpointing |       0.2380 |                  1976.38 |         2.3033 |
| Gradient Accumulation x4      |       0.3173 |                  1976.38 |         2.3033 |
| Mixed Precision Training      |       0.1711 |                  1976.38 |         2.3018 |

### Main Observations

* The five RAG questions achieved a **100% top-3 retrieval success rate**, although two questions required the Rank 2 passage for the direct answer.
* The alternate 300/75 chunking configuration did not improve the Titanic or Matrix retrieval rank.
* Kaiming initialization had the lowest measured time in the final summary, but it also produced a higher final loss than the baseline.
* Keeping all data on the GPU did not improve runtime in this small experiment and used slightly more GPU memory.
* Activation checkpointing increased runtime and did not reduce measured peak GPU memory for this network.
* Gradient accumulation x4 substantially increased runtime while leaving measured peak memory almost unchanged.
* Mixed precision kept the final loss very close to FP32, but it was slower in this particular short Tesla T4 benchmark.
