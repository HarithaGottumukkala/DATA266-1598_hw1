# Homework 5 — Fine-Tuning FLAN-T5-Small Using LoRA

This assignment uses LoRA (Low-Rank Adaptation) to fine-tune `google/flan-t5-small` for dialogue summarization on DialogSum.

I generated baseline summaries, trained separate adapters with ranks 4 and 16, and compared their outputs on the same two test dialogues. The notebook includes the code, explanations, training logs, and generated summaries.

## Personal Parameters

| Parameter | Value |
|---|---:|
| SID4 | 1598 |
| SEED | 1598 |
| SLICE | 598 |
| HP_ID | 2 |
| CLS_A | 8 |
| CLS_B | 5 |

HW5 uses the required rank comparison of `r = 4` and `r = 16`. No additional HP_ID-based experiment was specified.

## Files

| File or folder | Purpose |
|---|---|
| `DATA266_1598_hw5.ipynb` | Executed notebook with explanations and outputs |
| `dialogsum_lora_r4_adapter/` | Saved rank 4 adapter and tokenizer |
| `dialogsum_lora_r16_adapter/` | Saved rank 16 adapter and tokenizer |
| `baseline_outputs.json` | Summaries generated before fine-tuning |
| `r4_finetuned_outputs.json` | Summaries generated with rank 4 |
| `r16_finetuned_outputs.json` | Summaries generated with rank 16 |
| `lora_rank_metrics_comparison.csv` | Numerical comparison of both ranks |
| `r4_vs_r16_output_comparison.csv` | Generated-output comparison between ranks |
| `METRICS.md` | Experiment settings and measurement tables |
| `RUN_LOG.txt` | Recorded run information |
| `AI_USE.md` | AI assistance disclosure and verification |
| `README.md` | Overview and instructions |

The notebook also writes `baseline_vs_r4_comparison.csv` when its comparison section is executed.

## Dataset and Preprocessing

Dataset: [neil-code/dialogsum-test](https://huggingface.co/datasets/neil-code/dialogsum-test)

The loaded dataset contained:

| Split | Examples |
|---|---:|
| Training | 1,999 |
| Validation | 499 |
| Test | 499 |

Each input contains the instruction below followed by the original dialogue:

> Summarize the following dialogue:

The target is the corresponding human-written summary.

I used all the available training and validation examples in this dataset version. Inputs were truncated to 512 tokens and targets to 128 tokens. `DataCollatorForSeq2Seq` handled batch padding.

The notebook displays two processed training samples.

## Model and LoRA Setup

Base model: `google/flan-t5-small`

Each rank experiment starts from a fresh copy of the pretrained model. PEFT attaches trainable LoRA adapters while keeping the base weights frozen.

| Setting | Value |
|---|---|
| LoRA ranks | 4 and 16 |
| Alpha | 16 |
| Dropout | 0.05 |
| Target modules | `q`, `v` |
| Task type | `SEQ_2_SEQ_LM` |
| Bias | `none` |

## Training Configuration

| Setting | Value |
|---|---|
| Epochs | 2 |
| Training batch size | 8 |
| Evaluation batch size | 8 |
| Gradient accumulation steps | 2 |
| Learning rate | `2e-4` |
| Weight decay | 0.01 |
| Maximum gradient norm | 1.0 |
| Precision | FP32 |
| Evaluation and checkpoint saving | Every epoch |
| Logging interval | Every 25 steps |
| Best checkpoint criterion | Lowest validation loss |
| Seed | 1598 |

The exact training calls are:

`train_result_r4 = trainer_r4.train()`

`train_result_r16 = trainer_r16.train()`

Run these through the notebook after their respective data preparation, model setup, and trainer creation cells.

## Evaluation

Using seed 1598, I selected test indices **240 and 216**. The baseline and both fine-tuned models use these same examples.

Generation settings:

- Maximum input length: 512 tokens.
- Maximum new tokens: 80.
- Beam count: 4.
- Sampling: disabled.
- Evaluation mode with gradients disabled.

The notebook shows each reference summary alongside the baseline, rank 4, and rank 16 outputs.

## Results

| Metric | Rank 4 | Rank 16 |
|---|---:|---:|
| Total parameters | 77,133,184 | 77,649,280 |
| Trainable parameters | 172,032 | 688,128 |
| Trainable percentage | 0.2230% | 0.8862% |
| Pre-training sanity-check batch loss | 2.2771 | 2.2771 |
| Average training loss | 3.4834 | 3.4987 |
| Validation loss | 1.4146 | 1.4198 |
| Global training steps | 250 | 250 |
| Training time in minutes | 2.70 | 2.79 |

The pre-training loss was measured on only two training examples. It is a sanity check, not a full baseline validation loss.

### What Changed After Fine-Tuning

In the accident example, rank 4 included the injured friend and the emergency response that the baseline missed. However, both fine-tuned models incorrectly described the rental company representative as travelling with the customer. Rank 16 also assigned the emergency call to the wrong person.

In the shopping example, rank 4 produced a shorter summary but still repeated supplies and omitted important actions. Rank 16 repeated the same sentence three times.

Rank 16 used four times as many trainable parameters without improving the inspected outputs. Rank 4 had a slightly lower validation loss and was the more efficient option in this run.

## How to Run

1. Open `DATA266_1598_hw5.ipynb` in Google Colab.
2. Select a GPU runtime. The recorded experiment used a Tesla T4.
3. Run the notebook cells in order, starting with dependency installation.
4. Check the printed library versions, GPU information, and personal parameters.
5. Continue through data preparation, baseline inference, rank 4 training, and rank 16 training.
6. Review the output comparisons and save the executed notebook with its outputs.
7. Download the generated files and both adapter folders before the Colab runtime is cleared.

Internet access is needed to install dependencies and download the dataset, tokenizer, and base model. Rerunning the notebook writes output files and adapter folders in the runtime’s working directory.

### Recorded Environment

| Component | Version or hardware |
|---|---|
| GPU | Tesla T4 |
| PyTorch | 2.11.0+cu128 |
| CUDA | 12.8 |
| Transformers | 5.16.1 |
| Datasets | 4.8.5 |
| PEFT | 0.20.0 |
| Accelerate | 1.14.0 |

The notebook removes `torchao` during setup. Its installation commands do not pin package versions, so a later run may install a different environment.

## Saved Adapters

The two adapters are saved separately:

- `dialogsum_lora_r4_adapter/`
- `dialogsum_lora_r16_adapter/`

These contain adapter artifacts and tokenizer files. They are not standalone copies of the complete base model. Loading either adapter also requires `google/flan-t5-small`.

To load an adapter, first load the base model using `AutoModelForSeq2SeqLM.from_pretrained`, then attach the saved adapter using `PeftModel.from_pretrained`.

## Reproducibility and Limitations

Python, NumPy, and PyTorch seeds are explicitly set to 1598. The trainer also receives the same seed.

The notebook requests deterministic algorithms with `warn_only=True`, but training reports a nondeterministic attention operation. Exact numerical reproduction is therefore not guaranteed, and training time can vary.

Other limitations are:

- Only two generated test examples were manually inspected.
- Each rank was trained once.
- No ROUGE scores were reported.
- Input truncation frequency was not measured.
- Alpha remained fixed at 16, so the standard LoRA scaling factor `alpha / rank` changed between experiments.

The small validation-loss difference and two inspected outputs do not establish that rank 4 is generally better than rank 16.

## Assignment Coverage

- Loaded DialogSum and prepared dialogue–summary pairs.
- Displayed two processed samples.
- Generated and saved baseline summaries for two dialogues.
- Configured LoRA and printed total and trainable parameters.
- Fine-tuned both ranks and recorded training results.
- Saved both adapters and tokenizers.
- Repeated inference on the same two dialogues.
- Compared baseline and fine-tuned outputs.
- Compared ranks 4 and 16, including observed output errors.

## AI Use

See `AI_USE.md` for the description of assistant support, the work completed independently, and the checks or corrections made.
