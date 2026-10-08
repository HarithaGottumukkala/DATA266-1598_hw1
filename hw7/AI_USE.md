## AI Usage Report

### 1. How did I use AI?

I used ChatGPT to understand GANs, mode collapse, and the
differences between FC-AE and VAE. I also used it for
clarifying PyTorch implementation steps, reviewing code,
organizing explanations, and checking evaluation methods.

I ran the experiments in Google Colab and reviewed the
outputs, training curves, and reconstruction results.

### 2. What issue did I notice?
One issue I checked was how the reconstruction results
of FC-AE and VAE were being compared. Since these models
use different training loss functions, comparing their
total losses directly would be misleading.

### 3. How did I verify it?

I reviewed the loss functions and checked the training
outputs. I confirmed that FC-AE uses MSE, while VAE
uses reconstruction BCE and KL divergence.

### 4. What did I change?

I used the same pixel-level test MSE metric for both
models. I also used the VAE's latent mean during final
reconstruction evaluation to avoid random sampling.

This made the reconstruction comparison more consistent.

### Final Note

AI was useful for learning concepts, clarifying
implementation steps, and reviewing the evaluation.I checked the experimental results using my Colab
notebook and actual model outputs.
