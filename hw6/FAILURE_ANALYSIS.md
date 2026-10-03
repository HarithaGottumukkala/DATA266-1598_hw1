# Failure Analysis and Limitations

## Overview

All three experiments completed and produced test results. The notebook did show a PyTorch DataLoader cleanup warning during supervised training. It reported an `AssertionError` while shutting down worker processes, but it was marked as an ignored exception; training continued, and the model was evaluated successfully.

## Results

| Model | Test accuracy | Test loss | Nearest-neighbor match rate |
|---|---:|---:|---:|
| Supervised ResNet-18 | 44.15% | 1.6781 | 80.00% |
| Rotation SSL | 41.49% | 1.6903 | 66.67% |
| SimCLR | 48.48% | 1.4308 | 66.67% |

## What did not work as expected

### Supervised model

The supervised model reached 75.40% training accuracy, but its test accuracy was 44.15%. This gap suggests overfitting. The model had about 11.18 million trainable parameters, while training used only 500 labeled images. Data augmentation helped, but it did not fully overcome the small training set.

### Rotation SSL

The rotation model reached 64.08% accuracy on its rotation-prediction task, but its frozen-encoder linear probe reached 41.49% test accuracy. Predicting an image’s rotation teaches useful visual features, but success on that task does not guarantee that the features will separate STL-10 object classes well. In this run, rotation-based features performed below the supervised model on the classification test.

### SimCLR

SimCLR had the highest test accuracy at 48.48%, but its nearest-neighbor match rate was 66.67%, below the supervised model’s 80.00%. These metrics measure different things: the linear probe measures classification accuracy across the test set, while the nearest-neighbor result is based on the selected query images. The two rankings therefore do not have to match.

SimCLR training used 20,000 unlabeled images rather than the full 100,000. This limited the amount of data available for contrastive pretraining and may have limited the learned representation.

## Runtime warning

During supervised training, the notebook displayed an ignored DataLoader worker-cleanup exception with the message `can only test a child process`. The training loop continued through its epochs, and testing completed. The warning did not prevent the reported results, but a future run could use `num_workers=0` to check whether it avoids the cleanup warning in the notebook environment.

## Lessons and possible improvements

- Use more labeled examples or reduce model capacity to help control overfitting.
- Compare more labeled-data fractions to see how performance changes as the training set grows.
- Try the full unlabeled set for SimCLR if runtime and GPU memory allow.
- Test alternative augmentation strengths and SimCLR temperature values.
- Repeat the experiments with additional random seeds to check whether the results are stable.
- Keep the classification results and nearest-neighbor results separate because they evaluate different behavior.

## Conclusion

SimCLR performed best on test classification accuracy in this run, while the supervised model had the strongest nearest-neighbor match rate for the selected queries. Rotation prediction learned its proxy task, but its linear-probe accuracy was the lowest of the three. The experiments show that performance depends on the learning objective and evaluation method, as well as on the amount of labeled and unlabeled data.
