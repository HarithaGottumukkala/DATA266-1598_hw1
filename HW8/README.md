# Homework 8: Mini Diffusion Model (DDPM → DDIM)

**Course:** DATA 266 – Generative AI and Large Language Models  
**Student ID (Last 4 Digits):** 1598  
**Framework:** PyTorch  
**Environment:** Google Colab (Tesla T4 GPU)

## 1. Project Overview

In this homework, I implemented a small diffusion model using the MNIST handwritten digit dataset.

The main idea was to understand how diffusion models work by adding Gaussian noise to images and training a CNN to predict that noise. After training, I used DDPM and DDIM sampling to generate handwritten-like images from random noise.

I also compared the results using different numbers of sampling steps to see how sampling speed and image quality changed.

## 2. Assignment Tasks

| Part | Task | Points |
|---|---|---:|
| A | Implement forward diffusion and visualize noisy MNIST images | 5 |
| B | Train a timestep-conditioned CNN to predict Gaussian noise | 7 |
| C | Generate images using DDPM reverse sampling | 4 |
| D | Generate and compare images using DDIM sampling | 4 |
| **Total** | | **20** |

## 3. Dataset and Configuration

I used the MNIST dataset containing 60,000 training images and 10,000 test images. Each image is grayscale with a resolution of 28 × 28 pixels.

| Parameter | Value |
|---|---|
| Dataset | MNIST |
| Diffusion steps | 100 |
| Beta schedule | Linear (0.0001 to 0.02) |
| Batch size | 128 |
| Training epochs | 50 |
| CNN layers | 6 convolutional layers |
| Model parameters | 355,009 |
| Timestep conditioning | Learned embedding |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss function | Mean Squared Error (MSE) |
| Random seed | 42 |

## 4. Implementation

### Part A: Forward Diffusion

I implemented the forward diffusion equation to add Gaussian noise to MNIST images.

I visualized the same handwritten digit at timesteps 0, 10, 30, 60, and 100.

As the timestep increased, the image became noisier and the original digit gradually became harder to recognize.

### Part B: Training the CNN

I built a CNN with six convolutional layers and ReLU activations. I used a learned timestep embedding so the model could take the noise level into account.

The CNN was trained to predict the Gaussian noise added to each image using MSE loss.

### Part C: DDPM Sampling

After training, I implemented DDPM reverse sampling using 100 steps.

The generated images contained handwritten-like strokes, but most were distorted and difficult to recognize as digits.

### Part D: DDIM Sampling

I implemented deterministic DDIM sampling with 50 and 20 steps using the same trained CNN.

I compared the generated images and measured the sampling time over five runs after a GPU warm-up.

## 5. Experimental Results

### Training Performance

| Metric | Result |
|---|---:|
| Initial training MSE | 0.1775 |
| Final training MSE | 0.0602 |
| Lowest recorded MSE | 0.0601 |
| Training loss reduction | 66.07% |
| Training epochs | 50 |

The training loss decreased quickly during the first few epochs and then gradually became more stable.

### Sampling Performance

| Method | Steps | Average Time (seconds) |
|---|---:|---:|
| DDPM | 100 | 0.1836 ± 0.0160 |
| DDIM | 50 | 0.1437 ± 0.0639 |
| DDIM | 20 | 0.0413 ± 0.0001 |

The 20-step DDIM method was approximately **3.48 times faster** than the 50-step method in this benchmark.

### Image Quality

All three methods produced handwritten-like shapes, but the generated digits were mostly unclear.

DDIM with 50 and 20 steps produced similar-looking results. Reducing the number of sampling steps improved the speed without showing a major visible difference in image quality.

## 6. Failure Analysis

One important limitation was the quality of the generated images.

The training loss decreased, but the model still produced distorted digits with overlapping strokes.

A possible reason is the limited noise schedule. With 100 diffusion steps and the selected beta range, the final noisy training images still contain information from the original digits. However, reverse sampling starts from pure Gaussian noise.

This mismatch may have affected generation quality. The small CNN architecture may also have limited the model's ability to produce clear digits.

These are possible explanations rather than experimentally confirmed causes.

## 7. How to Run

1. Open the notebook in Google Colab.
2. Select a GPU runtime, if available.
3. Run the cells in order to install or import dependencies and load MNIST.
4. Run the forward diffusion visualization.
5. Train the CNN for 50 epochs.
6. Generate images using DDPM and DDIM sampling.
7. Run the timing comparison and report-generation cells.

**Main dependencies:** Python, PyTorch, torchvision, NumPy, Matplotlib, and tqdm.

The complete training process is computationally expensive, so a GPU is recommended.

## 8. Project Files

The `HW8/` folder contains the assignment notebook and supporting reports.

- `DATA266_HW8_1598.ipynb` — Complete implementation, visualizations, training, sampling, and report-generation code.
- `METRICS.md` — Model configuration, training results, and sampling benchmarks.
- `RUN_LOG.txt` — Execution details and recorded training loss for all 50 epochs.
- `failure_analysis.md` — Analysis of distorted generated images and possible causes.
- `DATA266_HW8_IEEE_Mini_Report.pdf` — IEEE-style mini report, if included in the repository.

## 9. Conclusion

This homework helped me understand the basic workflow of diffusion models, from adding noise to images to generating new samples using the reverse process.

The CNN learned to predict Gaussian noise, and its training MSE decreased by approximately 66%.

I also observed that DDIM could generate images faster using fewer sampling steps. However, the generated digits were still unclear, showing that a lower training loss does not always guarantee better image quality.

## 10. References

1. J. Ho, A. Jain, and P. Abbeel, "Denoising Diffusion Probabilistic Models," *NeurIPS*, 2020.
2. J. Song, C. Meng, and S. Ermon, "Denoising Diffusion Implicit Models," *ICLR*, 2021.
3. Y. LeCun et al., "Gradient-Based Learning Applied to Document Recognition," *Proceedings of the IEEE*, 1998.

---

*This repository contains my implementation, experiment results, and analysis for DATA 266 Homework 8.*
