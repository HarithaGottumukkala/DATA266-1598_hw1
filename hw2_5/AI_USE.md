

## 1. What did I use AI for, and what did I do myself?

I used ChatGPT mainly to help me understand some assignment requirements,
organize the experiment workflow, debug errors, and improve the clarity of
my written explanations.

I personally ran all GPU experiments on the assigned RTX 4090 workstation,
collected the measurements, checked the outputs, saved the logs and
screenshots, and verified the final reported results.

## 2. One specific thing AI suggested that was wrong

During the optimized attention experiment, ChatGPT initially suggested forcing
the PyTorch Flash Attention backend:

`SDPBackend.FLASH_ATTENTION`

When I ran this version, every required sequence length failed.

The main error was:

`RuntimeError: No available kernel. Aborting execution.`

The failed experiment is preserved in:

- `raw/attention/attention_fused_coarse_4090.csv`
- `raw/attention/attention_fused_coarse_run_4090.txt`

## 3. How did I detect the problem?

I detected the problem from the actual GPU run because all tested sequence
lengths were reported as failed, including the smallest sequence length.

I then created and ran a separate SDPA backend probe instead of assuming that
all optimized attention implementations were unsupported.

The probe showed:

- `FLASH_ATTENTION` — FAILED
- `EFFICIENT_ATTENTION` — SUCCESS
- `CUDNN_ATTENTION` — SUCCESS
- `MATH` — SUCCESS

The probe output is stored in:

`raw/attention/sdpa_backend_probe_4090.txt`

This showed that the failure was specifically related to the Flash Attention
backend available in this Windows/PyTorch environment.

## 4. What did I change, and why does the corrected version work?

I changed the optimized attention experiment from:

`SDPBackend.FLASH_ATTENTION`

to:

`SDPBackend.EFFICIENT_ATTENTION`

I selected the Efficient Attention backend explicitly instead of allowing
PyTorch to silently fall back to the standard math implementation.

After this change, every required sequence length from `512` through `16384`
completed successfully.

At `L=16384`, the memory-efficient implementation reported approximately
`0.25 GiB` peak allocated memory, compared with approximately `32.26 GiB`
for the naive implementation.

Its median forward latency was approximately `17.26 ms`, compared with
approximately `892.22 ms` for the naive version.

These results came from my own RTX 4090 experiment runs.
