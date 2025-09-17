"""HuggingFace Hub integration for Prophet models.

This module provides utilities for downloading and managing pretrained Prophet models
and embeddings from HuggingFace Hub, making it easy for users to get started.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from huggingface_hub import hf_hub_download, list_repo_files

# Registry of available Prophet models on HuggingFace
# Structure: theislab/Prophet/dataset_name/split/seed/ckpt
MODEL_REGISTRY = {
    # Base pretrained models
    "prophet-base": {
        "repo_id": "theislab/Prophet",
        "description": "Base Prophet model trained on large-scale perturbation data",
        "model_path": "base_pretrained/best_model.ckpt",
        "gene_embeddings": "embeddings/gene_embeddings.csv",
        "cell_embeddings": "embeddings/cell_line_embeddings.csv",
        "phenotype_embeddings": None,
        "paper": "https://doi.org/10.1016/j.cell.2024.01.035",
    },
    "prophet-gdsc": {
        "repo_id": "theislab/Prophet",
        "description": "Prophet model trained on GDSC (Genomics of Drug Sensitivity in Cancer) dataset with unseen cell lines split",
        "model_path": "GDSC/unseen_cell_lines/fold_0/110/unbalanced_False/best_model.ckpt",
        "gene_embeddings": "embeddings/gene_embeddings.csv",
        "cell_embeddings": "embeddings/cell_line_embeddings.csv",
        "phenotype_embeddings": None,
        "dataset": "GDSC",
        "split": "unseen_cell_lines",
        "seed": 110,
        "fold": 0,
    },
    "prophet-ctrp": {
        "repo_id": "theislab/Prophet",
        "description": "Prophet model trained on CTRP (Cancer Therapeutics Response Portal) dataset with unseen cell lines split",
        "model_path": "CTRP/unseen_cell_lines/fold_0/110/unbalanced_False/best_model.ckpt",
        "gene_embeddings": "embeddings/gene_embeddings.csv",
        "cell_embeddings": "embeddings/cell_line_embeddings.csv",
        "phenotype_embeddings": None,
        "dataset": "CTRP",
        "split": "unseen_cell_lines",
        "seed": 110,
        "fold": 0,
    },
    "prophet-gdsc-comb": {
        "repo_id": "theislab/Prophet",
        "description": "Prophet model trained on GDSC combination therapy dataset with unseen cell lines split",
        "model_path": "GDSCcomb/unseen_cell_lines/fold_0/110/unbalanced_False/best_model.ckpt",
        "gene_embeddings": "embeddings/gene_embeddings.csv",
        "cell_embeddings": "embeddings/cell_line_embeddings.csv",
        "phenotype_embeddings": None,
        "dataset": "GDSCcomb",
        "split": "unseen_cell_lines",
        "seed": 110,
        "fold": 0,
    },
    "prophet-jump": {
        "repo_id": "theislab/Prophet",
        "description": "Prophet model trained on JUMP (Joint Undertaking in Morphological Profiling) dataset with unseen cell lines split",
        "model_path": "JUMP/unseen_cell_lines/fold_0/110/unbalanced_False/best_model.ckpt",
        "gene_embeddings": "embeddings/gene_embeddings.csv",
        "cell_embeddings": "embeddings/cell_line_embeddings.csv",
        "phenotype_embeddings": None,
        "dataset": "JUMP",
        "split": "unseen_cell_lines",
        "seed": 110,
        "fold": 0,
    },
    "prophet-lincs": {
        "repo_id": "theislab/Prophet",
        "description": "Prophet model trained on LINCS (Library of Integrated Network-based Cellular Signatures) dataset with unseen cell lines split",
        "model_path": "LINCS/unseen_cell_lines/fold_0/110/unbalanced_False/best_model.ckpt",
        "gene_embeddings": "embeddings/gene_embeddings.csv",
        "cell_embeddings": "embeddings/cell_line_embeddings.csv",
        "phenotype_embeddings": None,
        "dataset": "LINCS",
        "split": "unseen_cell_lines",
        "seed": 110,
        "fold": 0,
    },
    "prophet-score": {
        "repo_id": "theislab/Prophet",
        "description": "Prophet model trained on SCORE dataset with unseen cell lines split",
        "model_path": "SCORE/unseen_cell_lines/fold_0/110/unbalanced_False/best_model.ckpt",
        "gene_embeddings": "embeddings/gene_embeddings.csv",
        "cell_embeddings": "embeddings/cell_line_embeddings.csv",
        "phenotype_embeddings": None,
        "dataset": "SCORE",
        "split": "unseen_cell_lines",
        "seed": 110,
        "fold": 0,
    },
    "prophet-horlbeck": {
        "repo_id": "theislab/Prophet",
        "description": "Prophet model trained on Horlbeck dataset with unseen cell lines split",
        "model_path": "Horlbeck/unseen_cell_lines/fold_0/110/unbalanced_False/best_model.ckpt",
        "gene_embeddings": "embeddings/gene_embeddings.csv",
        "cell_embeddings": "embeddings/cell_line_embeddings.csv",
        "phenotype_embeddings": None,
        "dataset": "Horlbeck",
        "split": "unseen_cell_lines",
        "seed": 110,
        "fold": 0,
    },
}


def construct_model_path(
    dataset: str,
    split: str = "unseen_cell_lines",
    seed: int = 110,
    fold: int = 0,
    unbalanced: bool = False,
    checkpoint_name: str = "best_model.ckpt",
) -> str:
    """Construct model path for HuggingFace Prophet repository structure.

    Args:
        dataset: Dataset name (e.g., 'GDSC', 'CTRP', 'LINCS')
        split: Split method ('unseen_perturbations' or 'unseen_cell_lines')
        seed: Random seed used for training (110, 1995, or 2024)
        fold: Cross-validation fold number (0-4)
        unbalanced: Whether unbalanced sampling was used
        checkpoint_name: Name of checkpoint file

    Returns:
        Path string for the model checkpoint

    Example:
        >>> construct_model_path("GDSC", "unseen_cell_lines", 110, 0, False)
        'GDSC/unseen_cell_lines/fold_0/110/unbalanced_False/best_model.ckpt'
    """
    return f"{dataset}/{split}/fold_{fold}/{seed}/unbalanced_{unbalanced}/{checkpoint_name}"


def get_available_datasets() -> List[str]:
    """Get list of available datasets in the Prophet HuggingFace repository.

    Returns:
        List of dataset names available for download
    """
    return ["CTRP", "GDSC", "GDSCcomb", "Horlbeck", "JUMP", "LINCS", "SCORE"]


def get_available_splits() -> List[str]:
    """Get list of available split methods.

    Returns:
        List of split method names
    """
    return ["unseen_perturbations", "unseen_cell_lines"]


def get_available_seeds() -> List[int]:
    """Get list of available seeds used for training.

    Returns:
        List of seed values
    """
    return [110, 1995, 2024]


def get_available_folds() -> List[int]:
    """Get list of available fold numbers.

    Returns:
        List of fold numbers (0-4)
    """
    return [0, 1, 2, 3, 4]


def create_custom_model_config(
    dataset: str,
    split: str = "unseen_cell_lines",
    seed: int = 110,
    fold: int = 0,
    unbalanced: bool = False,
    model_name: Optional[str] = None,
) -> Dict[str, Union[str, int, bool]]:
    """Create a custom model configuration for the registry.

    Args:
        dataset: Dataset name
        split: Split method
        seed: Random seed
        fold: Fold number
        unbalanced: Whether unbalanced sampling was used
        model_name: Custom name for the model (auto-generated if None)

    Returns:
        Model configuration dictionary
    """
    if model_name is None:
        model_name = f"prophet-{dataset.lower()}-{split}-s{seed}-f{fold}"

    return {
        "repo_id": "theislab/Prophet",
        "description": f"Prophet model trained on {dataset} dataset using {split} split",
        "model_path": construct_model_path(dataset, split, seed, fold, unbalanced),
        "gene_embeddings": "embeddings/gene_embeddings.csv",
        "cell_embeddings": "embeddings/cell_line_embeddings.csv",
        "phenotype_embeddings": None,
        "dataset": dataset,
        "split": split,
        "seed": seed,
        "fold": fold,
        "unbalanced": unbalanced,
        "paper": "https://doi.org/10.1016/j.cell.2024.01.035",
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
    model_name: str, cache_dir: Optional[str] = None, force_download: bool = False
) -> Tuple[str, str, str, Optional[str]]:
    """Download model and embedding files from HuggingFace Hub using registry.

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
    return _download_model_files_impl(
        model_name=model_name,
        cache_dir=cache_dir,
        force_download=force_download,
        model_config=None,
    )


def download_custom_model(
    dataset: str,
    split: str = "leave_cl_out",
    seed: int = 42,
    fold: int = 0,
    unbalanced: bool = False,
    cache_dir: Optional[str] = None,
    force_download: bool = False,
) -> Tuple[str, str, str, Optional[str]]:
    """Download a custom Prophet model with specific parameters.

    Args:
        dataset: Dataset name (e.g., 'GDSC', 'CTRP', 'LINCS')
        split: Split method (e.g., 'leave_cl_out', 'leave_iv_out')
        seed: Random seed used for training
        fold: Cross-validation fold number
        unbalanced: Whether unbalanced sampling was used
        cache_dir: Directory to cache downloaded files
        force_download: Whether to force re-download even if files exist

    Returns:
        Tuple of (model_path, gene_emb_path, cell_emb_path, phenotype_emb_path)

    Example:
        >>> # Download GDSC model with specific parameters
        >>> model_path, gene_emb, cell_emb, _ = download_custom_model(
        ...     dataset="GDSC",
        ...     split="leave_cl_out",
        ...     seed=42,
        ...     fold=0
        ... )
    """
    # Create custom model config
    model_config = create_custom_model_config(
        dataset=dataset, split=split, seed=seed, fold=fold, unbalanced=unbalanced
    )

    model_name = f"custom-{dataset.lower()}-{split}-s{seed}-f{fold}"

    return _download_model_files_impl(
        model_name=model_name,
        cache_dir=cache_dir,
        force_download=force_download,
        model_config=model_config,
    )


def _download_model_files_impl(
    model_name: str,
    cache_dir: Optional[str] = None,
    force_download: bool = False,
    model_config: Optional[Dict] = None,
) -> Tuple[str, str, str, Optional[str]]:
    """Internal implementation for downloading model files.

    Args:
        model_name: Name of the model to download
        cache_dir: Directory to cache downloaded files
        force_download: Whether to force re-download even if files exist
        model_config: Custom model configuration (overrides registry)

    Returns:
        Tuple of (model_path, gene_emb_path, cell_emb_path, phenotype_emb_path)
    """
    # Use custom config if provided, otherwise lookup in registry
    if model_config is not None:
        model_info = model_config
    else:
        if model_name not in MODEL_REGISTRY:
            available = ", ".join(MODEL_REGISTRY.keys())
            raise ValueError(
                f"Model '{model_name}' not found. Available models: {available}"
            )
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
            filename=model_info["model_path"],
            cache_dir=cache_dir,
            force_download=force_download,
        )

        # Download gene embeddings
        gene_emb_path = hf_hub_download(
            repo_id=repo_id,
            filename=model_info["gene_embeddings"],
            cache_dir=cache_dir,
            force_download=force_download,
        )

        # Download cell line embeddings
        cell_emb_path = hf_hub_download(
            repo_id=repo_id,
            filename=model_info["cell_embeddings"],
            cache_dir=cache_dir,
            force_download=force_download,
        )

        # Download phenotype embeddings (if available)
        phenotype_emb_path = None
        if model_info.get("phenotype_embeddings"):
            phenotype_emb_path = hf_hub_download(
                repo_id=repo_id,
                filename=model_info["phenotype_embeddings"],
                cache_dir=cache_dir,
                force_download=force_download,
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
        raise ValueError(
            f"Model '{model_name}' not found. Available models: {available}"
        )

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
            MODEL_REGISTRY[model_name]["model_path"],
            MODEL_REGISTRY[model_name]["gene_embeddings"],
            MODEL_REGISTRY[model_name]["cell_embeddings"],
        ]

        # Check if all required files exist
        return all(file in files for file in required_files)
    except Exception:
        return False
