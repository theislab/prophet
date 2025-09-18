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
    list_available_models,
    download_model_files,
    download_custom_model,
    get_model_info,
    print_available_models,
    check_model_availability,
    get_available_datasets,
    get_available_splits,
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
    "list_available_models",
    "download_model_files",
    "download_custom_model",
    "get_model_info",
    "print_available_models",
    "check_model_availability",
    "get_available_datasets",
    "get_available_splits",
    "R2ScoreCallback",
    "CosineWarmupScheduler",
    "HitRatioCallback",
]
