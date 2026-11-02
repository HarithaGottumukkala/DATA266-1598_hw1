# Homework 9 Metrics

## Personal Parameters

| Parameter | Value |
|---|---:|
| SID4 | 1598 |
| SEED | 1598 |
| SLICE | 598 |
| HP_ID | 2 |
| CLS_A | 8 |
| CLS_B | 5 |

## Environment

| Item | Value |
|---|---|
| Platform | Google Colab |
| GPU | Tesla T4 |
| PyTorch | 2.11.0+cu130 |
| VAE | stabilityai/sd-vae-ft-mse |
| VAE scaling factor | 0.18215 |

## Task 1 — VAE Compression

| Metric | Value |
|---|---:|
| Input shape | 3 × 64 × 64 |
| Latent shape | 4 × 8 × 8 |
| Input values per image | 12288 |
| Latent values per image | 256 |
| Compression ratio | 48.0:1 |
| Mean reconstruction MSE, [-1, 1] | 0.00954555 |
| Std reconstruction MSE, [-1, 1] | 0.00622775 |
| Mean reconstruction MSE, [0, 1] | 0.00229053 |
| Std reconstruction MSE, [0, 1] | 0.00152609 |

## Task 2 — Forward Diffusion

| Timestep | Alpha bar | Signal coefficient | Noise coefficient |
|---:|---:|---:|---:|
| 0 | 1.00000000 | 1.00000000 | 0.00000000 |
| 200 | 0.65903884 | 0.81181204 | 0.58391881 |
| 500 | 0.07858729 | 0.28033426 | 0.95990247 |
| 800 | 0.00153209 | 0.03914193 | 0.99923366 |
| 1000 | 0.00004036 | 0.00635282 | 0.99997979 |

## Task 4 — Training

| Metric | Value |
|---|---:|
| Epochs | 50 |
| Learning rate | 2e-05 |
| Batch size | 256 |
| Label-drop probability | 0.1 |
| Initial training MSE | 0.57163303 |
| Final training MSE | 0.11632118 |
| Initial validation MSE | 0.32552291 |
| Final validation MSE | 0.11176687 |
| Best validation MSE | 0.11176687 |
| Best validation epoch | 50 |
| Training time | 7.43 minutes |

## Task 5 — Sampling

| Setting | Value |
|---|---:|
| Generated images | 64 |
| Guidance weight | 3.0 |
| Reverse diffusion steps | 1000 |
| Sampling seed | 1598 |
| Final latent shape | 64 × 4 × 8 × 8 |
| Decoded image shape | 64 × 3 × 64 × 64 |

Generation was evaluated qualitatively using the 64-image grid and reverse
diffusion progression. No classifier-based generation accuracy was computed.

## Saved Artifacts

- Final checkpoint: `/content/drive/MyDrive/DATA266/HW9_1598/hw9_sid1598_final.pt`
- Sample grid: `/content/drive/MyDrive/DATA266/HW9_1598/conditional_sample_grid_64_cfg_w3.png`
- Reverse-process figure: `/content/drive/MyDrive/DATA266/HW9_1598/reverse_diffusion_progress_cfg_w3.png`
- Generated latents: `/content/drive/MyDrive/DATA266/HW9_1598/generated_latents_cfg_w3_sid1598.pt`
- Training-loss figure: `/content/drive/MyDrive/DATA266/HW9_1598/training_validation_loss.png`
