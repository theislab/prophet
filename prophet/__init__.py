"""
Prophet: Transformer-based model for predicting cellular responses to perturbations.

This module provides comprehensive functionality for training, fine-tuning, and
evaluating Prophet models on biological data.
"""

from .core.prophet import Prophet
from .training import ProphetTrainer, set_seeds

__version__ = "1.0.0"
__all__ = [
    "Prophet",
    "ProphetTrainer", 
    "set_seeds",
]
