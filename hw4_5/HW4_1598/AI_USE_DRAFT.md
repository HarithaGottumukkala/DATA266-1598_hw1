# AI Use — DRAFT TO REWRITE IN MY OWN WORDS

## Which parts used an assistant, and which parts did I do?

I used an AI assistant to help organize the notebook, draft PyTorch and
vLLM code, diagnose errors, and turn my measured outputs into simple
explanations. I personally ran every cell in Colab, selected the A100
runtime, supplied my student parameters, checked the outputs, shared
the errors, and verified the reported measurements.

## One specific thing the assistant produced that was wrong

The assistant originally tried to load a pinned TweetEval revision using
the `sentiment` builder configuration. The resulting output was:

`ValueError: BuilderConfig 'sentiment' not found. Available: ['default']`

## How did I find out?

I found the problem by running the dataset-loading cell. It failed before
any data was loaded and displayed the builder-configuration error. The
error showed that the pinned revision only exposed a configuration named
`default`.

## What did I change, and why does it work?

I changed the loading code to read the sentiment training Parquet file
directly from the fixed Hugging Face revision. This avoids relying on the
missing builder configuration while keeping the dataset version fixed.
I verified the correction by checking that it loaded 45,615 source rows,
used rows 598 through 5,597, and produced the expected text and label
columns.

IMPORTANT: Rewrite this file naturally in your own words, remove this
instruction, and save the final version as AI_USE.md.
