"""
Utility functions for Prophet.

This module contains validation, experimental space utilities, and other
helper functions.
"""

# Validation utilities
from .validation import (
    ValidationError,
    DataFrameValidator,
    EmbeddingValidator,
    ModelValidator,
    validate_prophet_inputs,
)

# Experimental space utilities
from .experimental_space import (
    create_experimental_space,
    experimental_space_summary,
    save_experimental_space,
    interventions_x_cells,
    cells_x_interventions,
)

# Model hub utilities
from .model_hub import (
    download_model,
    download_embeddings,
    print_available_models,
    get_available_datasets,
    get_available_splits,
    get_available_seeds,
    get_available_folds,
    list_available_checkpoints,
    get_checkpoint_name,
)

from .callbacks import R2ScoreCallback, CosineWarmupScheduler, HitRatioCallback

__all__ = [
    # Validation
    "ValidationError",
    "DataFrameValidator",
    "EmbeddingValidator",
    "ModelValidator",
    "validate_prophet_inputs",
    # Experimental space
    "create_experimental_space",
    "experimental_space_summary",
    "save_experimental_space",
    "interventions_x_cells",
    "cells_x_interventions",
    # Model hub
    "download_model",
    "download_embeddings",
    "print_available_models",
    "get_available_datasets",
    "get_available_splits",
    "get_available_seeds",
    "get_available_folds",
    "list_available_checkpoints",
    "get_checkpoint_name",
    # Callbacks
    "R2ScoreCallback",
    "CosineWarmupScheduler",
    "HitRatioCallback",
]
