"""
Data handling and loading utilities for Prophet.

This module provides comprehensive data processing, loading, and validation
functionality for the Prophet transformer model.
"""

# Core data loading functionality
from .dataloader import (
    dataloader_phenotypes,
    get_split_indices,
    get_data_by_setting,
    process_priors,
    remove_nonexistent_cat,
    universal_processing
)

# Dataset classes
from .dataset import (
    PhenotypeDataset
)

# Data processing and validation utilities
from .processing import (
    DataSplitter,
    DataProcessor, 
    DataValidator,
    create_cross_validation_splits
)

__all__ = [
    # Data loading
    'dataloader_phenotypes',
    'get_split_indices', 
    'get_data_by_setting',
    'process_priors',
    'remove_nonexistent_cat',
    'universal_processing',
    
    # Datasets
    'PhenotypeDataset',
    
    # Processing and validation
    'DataSplitter',
    'DataProcessor',
    'DataValidator', 
    'create_cross_validation_splits'
]