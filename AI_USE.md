# AI-Use Appendix - Haritha Gottumukkala

## 1. Which parts did you use an assistant for, and which did you write yourself?

I used AI mainly as a guide while working through the assignment. I used it to understand some of the requirements, organize the order of the experiments, and check whether I had missed anything important.

For the neural network part, I used AI to help me understand how to keep the PyTorch and TensorFlow experiments comparable. I also used it to understand how my HP_ID = 2 modification should be applied. My assigned change was increasing the learning rate from 0.001 to 0.003 while keeping the hidden layers [64, 32] and the number of epochs at 30.

I ran the models myself in Colab, checked the outputs, compared the baseline and modified models, and used the actual results from my notebook for the final interpretation. I also checked the three required training seeds, the train/validation/test split, the loss curves, and the mean and standard deviation calculations.

For CUDA, I used AI more for understanding the concepts because CUDA programming was newer to me. I used it to understand how blocks and threads work, how each thread computes one output element in matrix multiplication, why a warm-up kernel is useful, and why kernel time and memory-transfer time should be measured separately.

I wrote and organized the final notebook explanations based on the outputs I obtained from my own runs and checked the reported values against those outputs before finalizing the assignment.


## 2. Give one specific thing it produced that was wrong — a tensor shape error, a deprecated API, a loss function that trained but was wrong, a plausible-looking metric computed incorrectly. Paste the wrong output.

One problem happened after I reran the CUDA benchmark.

AI had previously helped me write an observation using timing values from an earlier run. After I reran the CUDA program, the timing values changed, but the written observation was still using the older numbers.

The outdated AI-produced statement was:

> For the 1024 x 1024 matrix, the CPU took about 204 ms, while the complete GPU execution took about 13 ms in this run.

However, after rerunning that section, the actual output was:

CPU time: 211.871 ms  
GPU end-to-end time: 13.940 ms  
End-to-end speedup: 15.199x  
Maximum error: 0.000107

The explanation looked reasonable because the older values were also from a real run, but it no longer matched the execution that was currently saved in the notebook.


## 3. How did you find out? What did the failure look like?

I found the problem while reviewing the CUDA notebook from top to bottom and comparing the written observations with the console output shown directly above them.

The code itself was still running correctly, so there was no crash or syntax error. The failure was an inconsistency between the explanation and the most recent benchmark results.

After noticing the mismatch in the 1024 x 1024 section, I checked the other CUDA results too. I found that some values in the 4096 x 4096 explanation, the profiler discussion, and the final average table also came from earlier executions.

This showed me that timing experiments need extra checking because the values can change slightly every time the program is run.


## 4. What did you change, and why does your version work?

I updated the CUDA observations so that they matched the latest executed outputs.

For the final CUDA comparison, I used the three repeated timing runs for each matrix size and calculated the averages from those runs. I then made sure that the same final values were used consistently in `cuda.ipynb`, `METRICS.md`, and `RUN_LOG.txt`.

I also checked the `nvprof` output again so that the explanation of kernel time and transfer time matched the profiler results from the saved execution.

This version works because the written analysis, the reported averages, and the raw console outputs now all refer to the same set of runs instead of mixing results from different executions.

I also verified the neural network results separately. I checked that the split used `random_state = 1598`, that the same split was used for both frameworks, that the required training seeds were 1598, 1599, and 1600, and that the final mean and standard deviation values were calculated from those three runs.
