## Failure Analysis

### 1. Reranking Did Not Improve Every Query

Reranking improved the overall average metrics, but it reduced Precision@5 for Q2 with recursive chunking from 0.80 to 0.60.

This occurred because the cross-encoder promoted passages that were semantically related to kernel methods but were less useful for directly answering the quantile-regression question.

Possible improvements include:

- evaluating a larger reranking model;
- tuning the candidate-pool size;
- combining embedding and cross-encoder scores;
- using a more diverse validation query set.

### 2. Ambiguous Follow-Up Retrieval Failed Without Memory

Without conversational memory, the question “How do they help in high-dimensional problems?” retrieved unrelated passages concerning geometry, image processing, polymers, and physical simulations.

The embedding model could not determine what “they” referred to. Rewriting the query with the remembered topic doubled Precision@5 and pooled Recall@5 from 0.40 to 0.80.

A production system could improve this further using a general coreference-resolution or LLM-based query-rewriting mechanism.

### 3. Recursive Chunking Produced Short Fragments

Recursive chunking generated 36 chunks shorter than 200 characters, including one nine-character fragment. Such fragments may lack enough context for reliable retrieval or reranking.

A possible improvement would merge short final fragments with adjacent chunks or enforce a minimum chunk length.

### 4. Candidate Retrieval Limited Reranking

The reranker can only reorder passages retrieved in the initial top-20 candidate pool. It cannot recover a relevant passage that embedding search failed to include.

Hybrid dense and keyword retrieval could improve candidate coverage before reranking.

### 5. Evaluation Scope

The primary evaluation used two questions derived from one research article. Although this satisfies the minimum query requirement and supports controlled comparisons, it limits generalization.

Future evaluation should use more queries from different documents and research areas.

### 6. Runtime Issues

No code cell produced an execution exception. The only runtime warning was an unauthenticated Hugging Face request warning. It did not affect dataset loading or the results.
