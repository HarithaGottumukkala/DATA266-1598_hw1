# Homework 4 Metrics

SID4: 1598 | SEED: 1598 | SLICE: 598
HP_ID: 2 | CLS_A: 8 | CLS_B: 5

| Setting | Value |
|---|---|
| Total characters | 1738 |
| Vocabulary size | 49 |
| Training pairs | 1610 |
| Sequence length | 128 |
| Parameters | 425,777 |
| Hidden dimension | 128 |
| Attention heads | 4 |
| Decoder blocks | 2 |
| Optimizer | Adam |
| Learning rate | 0.0003 |
| Batch size | 32 |
| Epochs | 8 |
| Device | cpu |

| Epoch | Average training loss |
|---|---|
| 1 | 3.106401 |
| 2 | 2.479756 |
| 3 | 2.278498 |
| 4 | 2.175339 |
| 5 | 2.104990 |
| 6 | 2.040887 |
| 7 | 1.971539 |
| 8 | 1.896454 |

Prompt: ROMEO:
Generated characters per sample: 300
Decoding: greedy; temperatures 0.5, 1.0, 1.5; top-k 2 and 20.
Each sample uses SEED = 1598.

The loss measures training performance; no held-out evaluation was used.
Generated samples and software versions are saved in results.json.
