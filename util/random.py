"""Utility functions regarding randomness."""
import numpy as np
import torch


def set_seed(seed=9001):
    """Sets random seed in all frameworks. Also adjusts options to achievie better determinism."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
