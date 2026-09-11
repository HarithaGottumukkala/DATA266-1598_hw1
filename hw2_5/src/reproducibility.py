import random
import numpy as np
import torch


SID4 = 1598
SEED = 1598
SLICE = 598
HP_ID = 2
CLS_A = 8
CLS_B = 5


def set_reproducibility():
    """
    Set the assignment reproducibility seed for Python,
    NumPy, and PyTorch.
    """
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)