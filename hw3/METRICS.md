# HW3 Metrics

SID4=1598, SEED=1598, SLICE=598, HP_ID=2, CLS_A=8, CLS_B=5

## Prompt Experiments

| Technique | Task | Correct final | Final words | Calls | Output tokens | Seconds |
| --- | --- | --- | --- | --- | --- | --- |
| Zero-Shot | Math | False | 2 | 1 | 4 | 0.13 |
| Zero-Shot | Logic | False | 160 | 1 | 229 | 2.97 |
| Few-Shot | Math | False | 2 | 1 | 5 | 0.19 |
| Few-Shot | Logic | True | 4 | 1 | 8 | 0.23 |
| Chain-of-Thought | Math | True | 35 | 1 | 61 | 0.9 |
| Chain-of-Thought | Logic | False | 297 | 1 | 384 | 5.06 |
| Zero-Shot CoT | Math | True | 31 | 1 | 67 | 0.98 |
| Zero-Shot CoT | Logic | False | 59 | 1 | 78 | 1.16 |
| Meta-Prompting | Math | True | 75 | 2 | 398 | 5.47 |
| Meta-Prompting | Logic | False | 241 | 2 | 628 | 8.81 |
| Tree of Thoughts | Math | True | 56 | 7 | 977 | 14.73 |
| Tree of Thoughts | Logic | False | 136 | 7 | 1772 | 30.69 |

5 of 12 final answers passed the checks.

Calls, output tokens, and runtime include all calls within each experiment. Word counts describe the final response. Failed checks can involve wrong answers or missing answer formatting. Timings are single-run observations.

## Attention Models

| Model | Initial loss | Final loss | Training accuracy | Training seconds | Max future weight | Mean future mass |
| --- | --- | --- | --- | --- | --- | --- |
| unmasked | 3.690619 | 0.632527 | 0.777778 | 0.928128 | 1.0 | 0.542781 |
| masked | 3.707081 | 0.462142 | 0.711111 | 0.916433 | 0.0 | 0.0 |

Training accuracy is a fraction between 0 and 1. Training time is in seconds. Both models use the same paragraph; these are training results, not held-out results. The unmasked model can access future target words. The causal model blocks future scores before softmax.
