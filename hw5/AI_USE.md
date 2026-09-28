# AI-Use Appendix - Naga Sai Haritha Gottumukkala

## 1. Which parts did you use an assistant for, and which did you write yourself?

I used an AI assistant mainly to understand a few LoRA concepts, organize the notebook, and help when I ran into errors. I ran all the code myself, checked the outputs, trained both LoRA models, compared the results, and wrote the final observations based on my experiment.

## 2. Give one specific thing it produced that was wrong.

One suggestion was to use FP16 for training the FLAN-T5 LoRA model. That run produced invalid results:

```text
Training Loss: 0.000000
Validation Loss: nan
