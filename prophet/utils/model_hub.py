"""HuggingFace Hub integration for Prophet models.

This module provides utilities for downloading and managing pretrained Prophet models
and embeddings from HuggingFace Hub, making it easy for users to get started.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from huggingface_hub import hf_hub_download, list_repo_files
import tempfile

# Registry of available Prophet models on HuggingFace
MODEL_REGISTRY = {
    "prophet-base": {
        "repo_id": "theislab/prophet-base",
        "description": "Base Prophet model trained on large-scale perturbation data",
        "model_file": "prophet_base.ckpt",
        "gene_embeddings": "gene_embeddings.csv",
        "cell_embeddings": "cell_line_embeddings.csv",
        "phenotype_embeddings": None,
        "paper": "https://doi.org/10.1016/j.cell.2024.01.035"
    },
    "prophet-large": {
        "repo_id": "theislab/prophet-large", 
        "description": "Large Prophet model with enhanced capacity",
        "model_file": "prophet_large.ckpt",
        "gene_embeddings": "gene_embeddings.csv",
        "cell_embeddings": "cell_line_embeddings.csv", 
        "phenotype_embeddings": None,
        "paper": "https://doi.org/10.1016/j.cell.2024.01.035"
    },
    "prophet-finetuned": {
        "repo_id": "theislab/prophet-finetuned",
        "description": "Prophet model fine-tuned on additional datasets",
        "model_file": "prophet_finetuned.ckpt", 
        "gene_embeddings": "gene_embeddings.csv",
        "cell_embeddings": "cell_line_embeddings.csv",
        "phenotype_embeddings": None,
        "paper": "https://doi.org/10.1016/j.cell.2024.01.035"
    }
}

def list_available_models() -> Dict[str, Dict]:
    """List all available pretrained Prophet models.
    
    Returns:
        Dictionary mapping model names to their metadata.
        
    Example:
        >>> models = list_available_models()
        >>> for name, info in models.items():
        ...     print(f"{name}: {info['description']}")
    """
    return MODEL_REGISTRY.copy()

def download_model_files(
    model_name: str,
    cache_dir: Optional[str] = None,
    force_download: bool = False
) -> Tuple[str, str, str, Optional[str]]:
    """Download model and embedding files from HuggingFace Hub.
    
    Args:
        model_name: Name of the model to download (e.g., "prophet-base").
        cache_dir: Directory to cache downloaded files. If None, uses default cache.
        force_download: Whether to force re-download even if files exist.
        
    Returns:
        Tuple of (model_path, gene_emb_path, cell_emb_path, phenotype_emb_path).
        
    Raises:
        ValueError: If model_name is not in the registry.
        ConnectionError: If download fails.
        
    Example:
        >>> model_path, gene_emb, cell_emb, pheno_emb = download_model_files("prophet-base")
        >>> # Use the paths with Prophet
    """
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Model '{model_name}' not found. Available models: {available}")
    
    model_info = MODEL_REGISTRY[model_name]
    repo_id = model_info["repo_id"]
    
    # Prepare cache directory
    if cache_dir is None:
        cache_dir = os.path.join(Path.home(), ".cache", "prophet", model_name)
    os.makedirs(cache_dir, exist_ok=True)
    
    try:
        # Download model checkpoint
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=model_info["model_file"],
            cache_dir=cache_dir,
            force_download=force_download
        )
        
        # Download gene embeddings
        gene_emb_path = hf_hub_download(
            repo_id=repo_id,
            filename=model_info["gene_embeddings"],
            cache_dir=cache_dir,
            force_download=force_download
        )
        
        # Download cell line embeddings
        cell_emb_path = hf_hub_download(
            repo_id=repo_id,
            filename=model_info["cell_embeddings"],
            cache_dir=cache_dir,
            force_download=force_download
        )
        
        # Download phenotype embeddings (if available)
        phenotype_emb_path = None
        if model_info["phenotype_embeddings"]:
            phenotype_emb_path = hf_hub_download(
                repo_id=repo_id,
                filename=model_info["phenotype_embeddings"],
                cache_dir=cache_dir,
                force_download=force_download
            )
        
        print(f"✅ Successfully downloaded {model_name} to {cache_dir}")
        return model_path, gene_emb_path, cell_emb_path, phenotype_emb_path
        
    except Exception as e:
        raise ConnectionError(f"Failed to download {model_name}: {str(e)}")

def get_model_info(model_name: str) -> Dict:
    """Get detailed information about a specific model.
    
    Args:
        model_name: Name of the model.
        
    Returns:
        Dictionary with model metadata.
        
    Raises:
        ValueError: If model_name is not found.
    """
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Model '{model_name}' not found. Available models: {available}")
    
    return MODEL_REGISTRY[model_name].copy()

def print_available_models():
    """Print a formatted list of available models."""
    print("📋 Available Prophet Models:")
    print("=" * 50)
    
    for name, info in MODEL_REGISTRY.items():
        print(f"🔬 {name}")
        print(f"   Description: {info['description']}")
        print(f"   Repository: {info['repo_id']}")
        print(f"   Paper: {info['paper']}")
        print()
    
    print("Usage:")
    print("   model = Prophet.from_pretrained('prophet-base')")
    print("   model = Prophet.from_pretrained('prophet-large')")

def check_model_availability(model_name: str) -> bool:
    """Check if a model is available for download.
    
    Args:
        model_name: Name of the model to check.
        
    Returns:
        True if model is available, False otherwise.
    """
    if model_name not in MODEL_REGISTRY:
        return False
    
    try:
        repo_id = MODEL_REGISTRY[model_name]["repo_id"]
        # Try to list files in the repository
        files = list_repo_files(repo_id)
        required_files = [
            MODEL_REGISTRY[model_name]["model_file"],
            MODEL_REGISTRY[model_name]["gene_embeddings"],
            MODEL_REGISTRY[model_name]["cell_embeddings"]
        ]
        
        # Check if all required files exist
        return all(file in files for file in required_files)
    except:
        return False

