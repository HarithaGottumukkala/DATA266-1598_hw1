# AI-Use Appendix - Haritha Gottumukkala

### 1. Which parts did you use an assistant for, and which did you write yourself?

I used an AI assistant throughout this homework mainly as a learning, implementation, and debugging aid. At the beginning, I used it to break the assignment into smaller steps and understand what was expected in each of the three sections. It helped me understand how pretrained Word2Vec embeddings could be transferred into a trainable model, how to structure the RAG pipeline using chunking, embeddings, FAISS, retrieval, prompting, and generation, and how the different training optimization experiments should be compared fairly.

I also used the assistant to suggest code structure and to troubleshoot errors that appeared while I was working in Google Colab. This was especially useful when library behavior was different from what I expected, when a model-loading method failed, and when the Colab runtime restarted and cleared variables that had already been defined.

I used the assistant again while organizing the final documentation so that the run log and metrics were consistent with the notebook. I did not treat suggested values as my results. I ran the notebook in Colab myself and used the outputs produced by my execution.

For Part 1, I checked that the five required words were present, verified that the pretrained vectors were actually copied into the trainable Word2Vec model, and used the cosine similarities produced by my run.

For Part 2, I inspected the retrieved passages for all five questions instead of assuming that a retrieved result was automatically correct. I manually identified the first rank that contained enough information to answer each question. This is how I found that the Titanic and Matrix questions had their direct answer-bearing passages at Rank 2 rather than Rank 1.

For Part 3, I ran the benchmark experiments in Colab and used the final measured runtime, GPU memory, and loss values from the executed notebook. I also reran parts of the experiment after the runtime state changed, so I made sure that the final written interpretation matched the most recent results rather than an earlier benchmark run.

---

### 2. Give one specific thing it produced that was wrong — a tensor shape error, a deprecated API, a loss function that trained but was wrong, a plausible-looking metric computed incorrectly. Paste the wrong output.

One specific problem came from an earlier RAG setup suggested while I was developing the notebook.

The initial setup used the smaller model:

```text
google/flan-t5-base
```

Before trusting the model for the homework questions, I tested it with a simple factual sanity-check prompt:

```text
Question: What is the capital city of France?
Answer:
```

The generated output was:

```text
london
```

This was clearly incorrect.

There was also an implementation issue with an earlier suggestion to use the Transformers `text2text-generation` pipeline. In my installed Colab environment, that approach produced a `KeyError` instead of creating the generation pipeline successfully.

The incorrect `london` result was the more important problem to me because the code could have appeared to be working while still producing an obviously unreliable factual answer.

---

### 3. How did you find out? What did the failure look like?

I found the problem by testing the generation model separately before connecting it to the final RAG workflow.

I intentionally used a question with an answer that I already knew. The expected answer to the France question was `Paris`, so when the model returned:

```text
london
```

I immediately knew that I should not use that setup for the final experiment without changing and retesting it.

This sanity check was useful because there was no Python exception in that case. The program was able to generate text normally, so simply checking whether the code ran would not have caught the problem. I had to look at the actual model output and judge whether it made sense.

I used the same kind of manual checking in the retrieval experiment. For example, for:

```text
Who played Jack Dawson in Titanic?
```

the first retrieved passage discussed a real grave marked `J. Dawson`. It was related to the query, but it did not answer the question. The second passage directly stated that Leonardo DiCaprio played Jack Dawson.

Similarly, for:

```text
What color pill does Neo take in The Matrix?
```

the first passage only said that Morpheus offered Neo two pills. The second passage explicitly stated that Neo takes the red pill.

Those cases showed me that getting a related passage is not necessarily the same as retrieving the best answer-bearing passage.

---

### 4. What did you change, and why does your version work?

For the generation problem, I stopped using the earlier FLAN-T5-base setup and changed the final generator to:

```text
google/flan-t5-large
```

I also stopped relying on the failing `text2text-generation` pipeline wrapper. Instead, I loaded the sequence-to-sequence model directly using:

```python
AutoTokenizer
AutoModelForSeq2SeqLM
```

After making the change, I repeated the same sanity-check question.

The corrected output was:

```text
Paris
```

Only after this test passed did I use that model in the final RAG pipeline.

The final pipeline first retrieves the top three FAISS chunks, combines them into a context, places the context and question into the prompt, and then sends the completed prompt to FLAN-T5-Large. The prompt also tells the model to use only the retrieved context.

I then manually checked the retrieved passages for all five evaluation questions. Every question had the required answer somewhere in the top three retrieved chunks, giving a final top-3 retrieval success rate of 5/5, or 100%.

The corrected approach works better because I did not depend only on whether the code executed without an exception. I tested the model independently, checked the retrieved evidence manually, and then used the actual outputs from my final notebook run when reporting the results.
