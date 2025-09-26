"""
Training utilities for Prophet models.
"""

import os
import torch
import pytorch_lightning as pl


def set_seeds(seed: int):
    """Set all random seeds for reproducible training."""
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # os.environ["PYTHONHASHSEED"] = str(seed)
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # if hasattr(torch, "use_deterministic_algorithms"):
    #     torch.use_deterministic_algorithms(True, warn_only=True)
