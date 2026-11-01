"""Reproducibility helpers."""
import os
import random

import numpy as np

SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Fix every source of randomness so a run can be repeated exactly."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    try:
        from transformers import set_seed as hf_set_seed
        hf_set_seed(seed)
    except ImportError:
        pass