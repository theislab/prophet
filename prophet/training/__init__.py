"""
Training utilities for Prophet models.

This module contains training logic, callbacks, and training orchestration.
"""

from .trainer import train_transformer
from .callbacks import R2ScoreCallback

# train_model.py contains the main training script but is not typically imported
# as it's meant to be run as a script

__all__ = [
    'train_transformer',
    'R2ScoreCallback'
]
