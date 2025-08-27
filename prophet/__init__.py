"""Prophet: Transformer-based model for predicting cellular responses to perturbations.

Prophet is a machine learning framework that predicts cellular responses to biological
perturbations (drugs, genetic modifications, etc.) by decomposing experiments into
cell state, treatment, and functional readout components.

Main Components:
    Prophet: Main model class for training and prediction
    data_processing: Utilities for data splitting and preprocessing  
    validation: Input validation and data quality checks
    config: Configuration management utilities

Quick Start:
    >>> from prophet import Prophet
    >>> model = Prophet(
    ...     iv_emb_path="interventions.csv",
    ...     cl_emb_path="cell_lines.csv", 
    ...     model_pth="trained_model.ckpt"
    ... )
    >>> predictions = model.predict(
    ...     target_ivs=["DRUG1", "GENE1"],
    ...     target_cls=["CELLLINE1", "CELLLINE2"],
    ...     save=False
    ... )
"""

# Core functionality (always available)
from .core import Prophet, set_config

# Data processing utilities  
from .data import DataSplitter, DataProcessor, DataValidator, create_cross_validation_splits

# Validation utilities
from .utils import (
    ValidationError, 
    DataFrameValidator, 
    EmbeddingValidator, 
    ModelValidator,
    validate_prophet_inputs
)

# Experimental space utilities
from .utils import (
    create_experimental_space,
    experimental_space_summary,
    save_experimental_space,
    interventions_x_cells,
    cells_x_interventions
)

# Model hub utilities
from .utils import (
    list_available_models,
    print_available_models
)

# Version info
__version__ = "0.1.0"
__author__ = "Alejandro Tejada-Lapuerta, Yuge Ji"

def print_installation_info():
    """Print information about current Prophet installation."""
    print(f"Prophet v{__version__}")
    print("=" * 40)
    
    print("✅ Complete Prophet installation includes:")
    print("  - Prophet model training and inference")
    print("  - Data processing utilities (DataSplitter, DataProcessor, DataValidator)")
    print("  - Input validation utilities")
    print("  - Visualization tools (Matplotlib, Seaborn, Plotly)")
    print("  - Jupyter notebook support")
    print("  - Experiment tracking (Weights & Biases, TensorBoard)")
    print("\n🎉 Everything is ready to use!")

__all__ = [
    # Core functionality
    'Prophet', 
    'set_config',
    
    # Data processing utilities
    'DataSplitter',
    'DataProcessor', 
    'DataValidator',
    'create_cross_validation_splits',
    
    # Validation utilities
    'ValidationError',
    'DataFrameValidator',
    'EmbeddingValidator', 
    'ModelValidator',
    'validate_prophet_inputs',
    
    # Experimental space utilities
    'create_experimental_space',
    'experimental_space_summary',
    'save_experimental_space',
    'interventions_x_cells',
    'cells_x_interventions',
    
    # Installation utilities
    'print_installation_info',
    
    # Model hub utilities
    'list_available_models',
    'print_available_models',
    
    # Package info
    '__version__',
    '__author__'
]
