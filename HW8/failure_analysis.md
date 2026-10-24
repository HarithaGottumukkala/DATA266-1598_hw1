# Homework 8 - Failure Analysis

## 1. Experiment Summary

I trained a CNN-based diffusion model on MNIST using
100 diffusion steps and 50 epochs.

The training MSE decreased from 0.1775
to 0.0602, a reduction of
66.07%.

All three sampling configurations produced tensors
with the expected image dimensions.

## 2. Observed Image Quality

From the generated-image plots, I noticed that
most samples were difficult to recognize as digits.

Some samples contained thick, overlapping strokes
and distorted handwritten-like shapes.

The DDIM outputs using 50 and 20 steps looked
similar in the visual comparison.

These observations are qualitative. No numerical
image-quality score was measured.

## 3. Possible Cause: Noise Schedule

The final cumulative alpha was:

alpha_bar_T = 0.363563

The forward diffusion equation at the final step
therefore contains approximately:

- Original-image coefficient: 0.6030
- Gaussian-noise coefficient: 0.7978

Because alpha_bar_T is not close to zero, the final
forward-diffused images still retain information
from the original digits.

However, DDPM and DDIM sampling begin with pure
Gaussian noise.

This difference between the training and sampling
distributions may contribute to poor image quality.

## 4. Possible Cause: Model Capacity

The model uses 6 convolutional layers
and 355,009 parameters.

Although its training loss decreased, a small CNN
may not learn sufficiently detailed noise predictions
for high-quality generation.

This is a possible explanation, not a verified cause.

## 5. Sampling Speed and Quality

Average sampling times:

- DDPM 100 steps: 0.1836 seconds
- DDIM 50 steps: 0.1437 seconds
- DDIM 20 steps: 0.0413 seconds

DDIM with 20 steps was 3.48 times
faster than DDIM with 50 steps in this benchmark.

The visual comparison did not show an obvious
improvement in digit clarity with fewer steps.

## 6. Suggested Improvements

If I extended this experiment, I would:

1. Test a schedule that ends closer to pure noise.
2. Try a small U-Net instead of the current CNN.
3. Compare denoising of real images against
   generation from random noise.
4. Use a quantitative image-quality evaluation.
5. Repeat experiments across multiple random seeds.

These changes were not part of the current run.

## 7. Final Conclusion

The model reduced its training noise-prediction loss,
but the generated digits were still unclear.

The results show that lower MSE does not guarantee
good image generation.

DDIM reduced sampling time, but improving the visual
quality would require additional experiments.
