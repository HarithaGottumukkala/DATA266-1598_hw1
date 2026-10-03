# Metrics

## Main Results

| Model | Test Loss | Test Accuracy | Neighbor Match Rate |
|---|---:|---:|---:|
| Supervised | 1.6781 | 44.15% | 80.00% |
| Rotation SSL | 1.6903 | 41.49% | 66.67% |
| SimCLR | 1.4308 | 48.48% | 66.67% |

## Training Details

- Supervised images: 500
- Supervised epochs: 15
- Final supervised training accuracy: 75.40%
- Rotation images: 100,000
- Rotation epochs: 15
- Final rotation accuracy: 64.08%
- Rotation linear-evaluation epochs: 20
- SimCLR images: 20,000
- SimCLR epochs: 20
- Temperature: 0.2
- Final SimCLR loss: 1.5637
- SimCLR linear-evaluation epochs: 20
