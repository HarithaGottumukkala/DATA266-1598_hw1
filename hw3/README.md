
# Homework 3: Prompt Engineering and Self-Attention

## Personal Parameters

| Parameter | Value |
|---|---|
| SID4 | 1598 |
| SEED | 1598 |
| SLICE | 598 |
| HP_ID | 2 |
| CLS_A | 8 |
| CLS_B | 5 |

This homework does not use a dataset slice, focus classes, or an HP_ID mapping.

## What This Notebook Does

### Prompt Engineering

I tested six techniques using LangChain and Ollama with `qwen2.5:3b`:

- Zero-Shot
- Few-Shot
- Chain-of-Thought
- Zero-Shot CoT
- Meta-Prompting
- Tree of Thoughts

Each technique has two separate code examples: one math problem and one logic puzzle. I reused the same tasks to compare correctness, response length, model calls, generated tokens, and runtime.

### Self-Attention

I used the exact paragraph provided in the assignment. It contains 46 word tokens and 40 unique words.

I built and trained two single-head attention models:
- An unmasked model that can attend to every position.
- A causal model that blocks future positions before softmax.

Both models use trainable word and position embeddings, learned Q/K/V projections, and next-word prediction.

The attention calculation and causal mask are written manually. No built-in attention or Transformer classes are used.

## Training Settings

Both attention models start with the same weights and use:

- Embedding dimension: 64
- Optimizer: Adam
- Learning rate: 0.01
- Epochs: 400
- Device: CPU
- Seed: 1598

Ollama uses temperature 0 for the usual calls. Tree of Thoughts uses temperature 0.7 and three branch seeds to encourage different plans.

## Main Findings

In the recorded prompt experiments, 5 of 12 final answers passed the checks. Longer explanations and extra reviews did not guarantee correct answers.

The unmasked model's loss decreased from 3.6906 to 0.6325, with 77.78% training accuracy. Its loss increased near the end, showing unstable training.

Both models learned their embeddings and Q/K/V projections. Both heatmaps cover the full 46-word sequence. Every future attention weight in the causal model was exactly zero.

These attention results measure training fit on one paragraph. The unmasked model can see future target words, and neither model was evaluated on unseen text.

## How to Run

1. Open the notebook in Google Colab.
2. Select a T4 GPU runtime for Ollama.
3. Run the cells from top to bottom.
4. Keep the notebook outputs when downloading it.

The initial setup needs internet access to install packages and download the Ollama model. Software versions and model details are recorded in the artifacts.

Fixed seeds help reproducibility, but Ollama outputs can vary across software and hardware.

## Submission Files

- Executed `.ipynb` notebook with outputs intact
- Findings PDF
- `hw3_artifacts/`, including:
  - `RUN_LOG.txt` and `METRICS.md`
  - `AI_USE.md`
  - Both trained model checkpoints
  - `TRAINING_COMMAND.txt`
  - Heatmaps, metrics, prompt outputs, and reproducibility files

Submit through the private `data266-1598` repository, shared with the instructor and grading team. Use the final tag `hw3` and maintain real incremental commits.

## Reference

Vaswani et al. (2017), *Attention Is All You Need*, Section 3.2:
https://arxiv.org/abs/1706.03762
