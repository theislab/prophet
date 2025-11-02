"""HuggingFace Hub integration for Prophet models.

This module provides utilities for downloading and managing pretrained Prophet models
and embeddings from HuggingFace Hub, making it easy for users to get started.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from huggingface_hub import hf_hub_download

CHECKPOINT_MAPPING = {
    ("base_pretrained", "perturbations", 0, 110): "epoch=31-step=44800.ckpt",
    ("base_pretrained", "perturbations", 0, 1995): "epoch=19-step=28040.ckpt",
    ("base_pretrained", "perturbations", 0, 2024): "epoch=39-step=55960.ckpt",
    ("base_pretrained", "perturbations", 1, 110): "epoch=50-step=71706.ckpt",
    ("base_pretrained", "perturbations", 1, 1995): "epoch=55-step=78400.ckpt",
    ("base_pretrained", "perturbations", 1, 2024): "epoch=15-step=22448.ckpt",
    ("base_pretrained", "perturbations", 2, 110): "epoch=33-step=47838.ckpt",
    ("base_pretrained", "perturbations", 2, 1995): "epoch=39-step=56040.ckpt",
    ("base_pretrained", "perturbations", 2, 2024): "epoch=26-step=37800.ckpt",
    ("base_pretrained", "perturbations", 3, 110): "epoch=17-step=25254.ckpt",
    ("base_pretrained", "perturbations", 3, 1995): "epoch=13-step=19544.ckpt",
    ("base_pretrained", "perturbations", 3, 2024): "epoch=26-step=37854.ckpt",
    ("base_pretrained", "perturbations", 4, 2024): "epoch=20-step=29568.ckpt",
    ("Horlbeck", "perturbations", 0, 110): "epoch=19-v1.ckpt",
    ("Horlbeck", "perturbations", 0, 1995): "epoch=19-v2.ckpt",
    ("Horlbeck", "perturbations", 0, 2024): "epoch=19-v4.ckpt",
    ("Horlbeck", "perturbations", 1, 110): "epoch=19-v1.ckpt",
    ("Horlbeck", "perturbations", 1, 1995): "epoch=29.ckpt",
    ("Horlbeck", "perturbations", 1, 2024): "epoch=19-v1.ckpt",
    ("Horlbeck", "perturbations", 2, 110): "epoch=29.ckpt",
    ("Horlbeck", "perturbations", 2, 1995): "epoch=19-v1.ckpt",
    ("Horlbeck", "perturbations", 2, 2024): "epoch=19-v1.ckpt",
    ("Horlbeck", "perturbations", 3, 110): "epoch=19-v1.ckpt",
    ("Horlbeck", "perturbations", 3, 1995): "epoch=19-v1.ckpt",
    ("Horlbeck", "perturbations", 3, 2024): "epoch=29.ckpt",
    ("Horlbeck", "perturbations", 4, 2024): "epoch=19-v1.ckpt",
    ("CTRP", "perturbations", 0, 110): "epoch=9.ckpt",
    ("CTRP", "perturbations", 0, 1995): "epoch=9.ckpt",
    ("CTRP", "perturbations", 0, 2024): "epoch=9.ckpt",
    ("CTRP", "perturbations", 1, 110): "epoch=9.ckpt",
    ("CTRP", "perturbations", 1, 1995): "epoch=9.ckpt",
    ("CTRP", "perturbations", 1, 2024): "epoch=9.ckpt",
    ("CTRP", "perturbations", 2, 110): "epoch=9.ckpt",
    ("CTRP", "perturbations", 2, 1995): "epoch=9.ckpt",
    ("CTRP", "perturbations", 2, 2024): "epoch=9.ckpt",
    ("CTRP", "perturbations", 3, 110): "epoch=9.ckpt",
    ("CTRP", "perturbations", 3, 1995): "epoch=9.ckpt",
    ("CTRP", "perturbations", 3, 2024): "epoch=9.ckpt",
    ("CTRP", "perturbations", 4, 2024): "epoch=9.ckpt",
    ("LINCS", "perturbations", 0, 110): "epoch=3-step=19240.ckpt",
    ("LINCS", "perturbations", 0, 1995): "epoch=1-step=9638.ckpt",
    ("LINCS", "perturbations", 0, 2024): "epoch=0-step=4850-v3.ckpt",
    ("LINCS", "perturbations", 1, 110): "epoch=3-step=19276.ckpt",
    ("LINCS", "perturbations", 1, 1995): "epoch=2-step=14433.ckpt",
    ("LINCS", "perturbations", 1, 2024): "epoch=2-step=14511-v1.ckpt",
    ("LINCS", "perturbations", 2, 110): "epoch=2-step=14709.ckpt",
    ("LINCS", "perturbations", 2, 1995): "epoch=1-step=9808.ckpt",
    ("LINCS", "perturbations", 2, 2024): "epoch=1-step=9870-v1.ckpt",
    ("LINCS", "perturbations", 3, 110): "epoch=0-step=4824.ckpt",
    ("LINCS", "perturbations", 3, 1995): "epoch=2-step=14382.ckpt",
    ("LINCS", "perturbations", 3, 2024): "epoch=0-step=4855.ckpt",
    ("LINCS", "perturbations", 4, 2024): "epoch=2-step=14673.ckpt",
    ("PRISM", "perturbations", 0, 110): "epoch=2-step=18456.ckpt",
    ("PRISM", "perturbations", 0, 1995): "epoch=2-step=18438.ckpt",
    ("PRISM", "perturbations", 0, 2024): "epoch=0-step=6146-v2.ckpt",
    ("PRISM", "perturbations", 1, 110): "epoch=0-step=6150.ckpt",
    ("PRISM", "perturbations", 1, 1995): "epoch=1-step=12282.ckpt",
    ("PRISM", "perturbations", 1, 2024): "epoch=1-step=12296-v1.ckpt",
    ("PRISM", "perturbations", 2, 110): "epoch=0-step=6158.ckpt",
    ("PRISM", "perturbations", 2, 1995): "epoch=1-step=12306.ckpt",
    ("PRISM", "perturbations", 2, 2024): "epoch=0-step=6157-v1.ckpt",
    ("PRISM", "perturbations", 3, 110): "epoch=2-step=18459.ckpt",
    ("PRISM", "perturbations", 3, 1995): "epoch=0-step=6153.ckpt",
    ("PRISM", "perturbations", 3, 2024): "epoch=1-step=12304.ckpt",
    ("PRISM", "perturbations", 4, 2024): "epoch=2-step=18441.ckpt",
    ("GDSC", "perturbations", 0, 110): "epoch=19.ckpt",
    ("GDSC", "perturbations", 0, 1995): "epoch=19.ckpt",
    ("GDSC", "perturbations", 0, 2024): "epoch=19-v3.ckpt",
    ("GDSC", "perturbations", 1, 110): "epoch=25-step=19838.ckpt",
    ("GDSC", "perturbations", 1, 1995): "epoch=19.ckpt",
    ("GDSC", "perturbations", 1, 2024): "epoch=23-step=18552.ckpt",
    ("GDSC", "perturbations", 2, 110): "epoch=19.ckpt",
    ("GDSC", "perturbations", 2, 1995): "epoch=19.ckpt",
    ("GDSC", "perturbations", 2, 2024): "epoch=19.ckpt",
    ("GDSC", "perturbations", 3, 110): "epoch=19.ckpt",
    ("GDSC", "perturbations", 3, 1995): "epoch=19.ckpt",
    ("GDSC", "perturbations", 3, 2024): "epoch=19.ckpt",
    ("GDSC", "perturbations", 4, 2024): "epoch=19.ckpt",
    ("JUMP", "perturbations", 0, 110): "epoch=19.ckpt",
    ("JUMP", "perturbations", 0, 1995): "epoch=19.ckpt",
    ("JUMP", "perturbations", 0, 2024): "epoch=20-step=18585-v4.ckpt",
    ("JUMP", "perturbations", 1, 110): "epoch=21-step=19470.ckpt",
    ("JUMP", "perturbations", 1, 1995): "epoch=19.ckpt",
    ("JUMP", "perturbations", 1, 2024): "epoch=19.ckpt",
    ("JUMP", "perturbations", 2, 110): "epoch=19.ckpt",
    ("JUMP", "perturbations", 2, 1995): "epoch=19.ckpt",
    ("JUMP", "perturbations", 2, 2024): "epoch=19.ckpt",
    ("JUMP", "perturbations", 3, 110): "epoch=19.ckpt",
    ("JUMP", "perturbations", 3, 1995): "epoch=19.ckpt",
    ("JUMP", "perturbations", 3, 2024): "epoch=19.ckpt",
    ("JUMP", "perturbations", 4, 2024): "epoch=19.ckpt",
    ("ShifrutMarson", "perturbations", 0, 110): "epoch=29.ckpt",
    ("ShifrutMarson", "perturbations", 0, 1995): "epoch=39.ckpt",
    ("ShifrutMarson", "perturbations", 0, 2024): "epoch=29.ckpt",
    ("ShifrutMarson", "perturbations", 1, 110): "epoch=49.ckpt",
    ("ShifrutMarson", "perturbations", 1, 1995): "epoch=29.ckpt",
    ("ShifrutMarson", "perturbations", 1, 2024): "epoch=29.ckpt",
    ("ShifrutMarson", "perturbations", 2, 110): "epoch=29.ckpt",
    ("ShifrutMarson", "perturbations", 2, 1995): "epoch=39.ckpt",
    ("ShifrutMarson", "perturbations", 2, 2024): "epoch=29.ckpt",
    ("ShifrutMarson", "perturbations", 3, 110): "epoch=29.ckpt",
    ("ShifrutMarson", "perturbations", 3, 1995): "epoch=29.ckpt",
    ("ShifrutMarson", "perturbations", 3, 2024): "epoch=29.ckpt",
    ("ShifrutMarson", "perturbations", 4, 2024): "epoch=49.ckpt",
    ("GDSCcomb", "perturbations", 0, 110): "epoch=49.ckpt",
    ("GDSCcomb", "perturbations", 0, 1995): "epoch=59.ckpt",
    ("GDSCcomb", "perturbations", 0, 2024): "epoch=19-v4.ckpt",
    ("GDSCcomb", "perturbations", 1, 110): "epoch=49.ckpt",
    ("GDSCcomb", "perturbations", 1, 1995): "epoch=49.ckpt",
    ("GDSCcomb", "perturbations", 1, 2024): "epoch=29-v1.ckpt",
    ("GDSCcomb", "perturbations", 2, 110): "epoch=19.ckpt",
    ("GDSCcomb", "perturbations", 2, 1995): "epoch=39.ckpt",
    ("GDSCcomb", "perturbations", 2, 2024): "epoch=39.ckpt",
    ("GDSCcomb", "perturbations", 3, 110): "epoch=39.ckpt",
    ("GDSCcomb", "perturbations", 3, 1995): "epoch=39.ckpt",
    ("GDSCcomb", "perturbations", 3, 2024): "epoch=19.ckpt",
    ("GDSCcomb", "perturbations", 4, 2024): "epoch=39.ckpt",
    ("SCORE", "perturbations", 0, 110): "epoch=0-step=6994.ckpt",
    ("SCORE", "perturbations", 0, 1995): "epoch=0-step=6998.ckpt",
    ("SCORE", "perturbations", 0, 2024): "epoch=1-step=14002-v3.ckpt",
    ("SCORE", "perturbations", 1, 110): "epoch=1-step=13996.ckpt",
    ("SCORE", "perturbations", 1, 1995): "epoch=0-step=6996.ckpt",
    ("SCORE", "perturbations", 1, 2024): "epoch=0-step=7003-v1.ckpt",
    ("SCORE", "perturbations", 2, 110): "epoch=0-step=6992.ckpt",
    ("SCORE", "perturbations", 2, 1995): "epoch=0-step=6991.ckpt",
    ("SCORE", "perturbations", 2, 2024): "epoch=1-step=13984.ckpt",
    ("SCORE", "perturbations", 3, 110): "epoch=0-step=6999.ckpt",
    ("SCORE", "perturbations", 3, 1995): "epoch=0-step=6994.ckpt",
    ("SCORE", "perturbations", 3, 2024): "epoch=0-step=6996.ckpt",
    ("SCORE", "perturbations", 4, 2024): "epoch=0-step=6993.ckpt",
    ("base_pretrained", "cell_lines", 0, 110): "epoch=29-step=45360.ckpt",
    ("base_pretrained", "cell_lines", 0, 1995): "epoch=6-step=10584.ckpt",
    ("base_pretrained", "cell_lines", 0, 2024): "epoch=7-step=11744.ckpt",
    ("base_pretrained", "cell_lines", 1, 110): "epoch=19-step=29420.ckpt",
    ("base_pretrained", "cell_lines", 1, 1995): "epoch=19-step=29440.ckpt",
    ("base_pretrained", "cell_lines", 1, 2024): "epoch=7-step=11424.ckpt",
    ("base_pretrained", "cell_lines", 2, 110): "epoch=14-step=23190.ckpt",
    ("base_pretrained", "cell_lines", 2, 1995): "epoch=5-step=9114.ckpt",
    ("base_pretrained", "cell_lines", 2, 2024): "epoch=5-step=9270.ckpt",
    ("base_pretrained", "cell_lines", 3, 110): "epoch=5-step=9150.ckpt",
    ("base_pretrained", "cell_lines", 3, 1995): "epoch=10-step=16753.ckpt",
    ("base_pretrained", "cell_lines", 3, 2024): "epoch=15-step=24384.ckpt",
    ("base_pretrained", "cell_lines", 4, 110): "epoch=7-step=12520.ckpt",
    ("base_pretrained", "cell_lines", 4, 1995): "epoch=7-step=12120.ckpt",
    ("base_pretrained", "cell_lines", 4, 2024): "epoch=6-step=10948.ckpt",
    ("CTRP", "cell_lines", 0, 110): "epoch=9-step=19590.ckpt",
    ("CTRP", "cell_lines", 0, 1995): "epoch=9.ckpt",
    ("CTRP", "cell_lines", 0, 2024): "epoch=9.ckpt",
    ("CTRP", "cell_lines", 1, 110): "epoch=9.ckpt",
    ("CTRP", "cell_lines", 1, 1995): "epoch=9.ckpt",
    ("CTRP", "cell_lines", 1, 2024): "epoch=9.ckpt",
    ("CTRP", "cell_lines", 2, 110): "epoch=9-step=19630.ckpt",
    ("CTRP", "cell_lines", 2, 1995): "epoch=9.ckpt",
    ("CTRP", "cell_lines", 2, 2024): "epoch=9.ckpt",
    ("CTRP", "cell_lines", 3, 110): "epoch=9.ckpt",
    ("CTRP", "cell_lines", 3, 1995): "epoch=9.ckpt",
    ("CTRP", "cell_lines", 3, 2024): "epoch=9.ckpt",
    ("CTRP", "cell_lines", 4, 110): "epoch=9.ckpt",
    ("CTRP", "cell_lines", 4, 1995): "epoch=9-step=19610.ckpt",
    ("CTRP", "cell_lines", 4, 2024): "epoch=9.ckpt",
    ("LINCS", "cell_lines", 0, 110): "epoch=2-step=15861.ckpt",
    ("LINCS", "cell_lines", 0, 1995): "epoch=1-step=10576.ckpt",
    ("LINCS", "cell_lines", 0, 2024): "epoch=2-step=13764-v1.ckpt",
    ("LINCS", "cell_lines", 1, 110): "epoch=1-step=9278.ckpt",
    ("LINCS", "cell_lines", 1, 1995): "epoch=3-step=18560.ckpt",
    ("LINCS", "cell_lines", 1, 2024): "epoch=2-step=11970.ckpt",
    ("LINCS", "cell_lines", 2, 110): "epoch=2-step=17481.ckpt",
    ("LINCS", "cell_lines", 2, 1995): "epoch=0-step=5402.ckpt",
    ("LINCS", "cell_lines", 2, 2024): "epoch=0-step=5827.ckpt",
    ("LINCS", "cell_lines", 3, 110): "epoch=2-step=16203.ckpt",
    ("LINCS", "cell_lines", 3, 1995): "epoch=0-step=5410.ckpt",
    ("LINCS", "cell_lines", 3, 2024): "epoch=2-step=16206.ckpt",
    ("LINCS", "cell_lines", 4, 110): "epoch=0-step=6081.ckpt",
    ("LINCS", "cell_lines", 4, 1995): "epoch=2-step=15870.ckpt",
    ("LINCS", "cell_lines", 4, 2024): "epoch=0-step=6065.ckpt",
    ("PRISM", "cell_lines", 0, 110): "epoch=2-step=18504.ckpt",
    ("PRISM", "cell_lines", 0, 1995): "epoch=2-step=18558.ckpt",
    ("PRISM", "cell_lines", 0, 2024): "epoch=2-step=18495-v1.ckpt",
    ("PRISM", "cell_lines", 1, 110): "epoch=2-step=18417.ckpt",
    ("PRISM", "cell_lines", 1, 1995): "epoch=1-step=12332.ckpt",
    ("PRISM", "cell_lines", 1, 2024): "epoch=2-step=18405-v1.ckpt",
    ("PRISM", "cell_lines", 2, 110): "epoch=0-step=6161.ckpt",
    ("PRISM", "cell_lines", 2, 1995): "epoch=1-step=12336.ckpt",
    ("PRISM", "cell_lines", 2, 2024): "epoch=2-step=18450.ckpt",
    ("PRISM", "cell_lines", 3, 110): "epoch=2-step=18522.ckpt",
    ("PRISM", "cell_lines", 3, 1995): "epoch=2-step=18477.ckpt",
    ("PRISM", "cell_lines", 3, 2024): "epoch=0-step=6142.ckpt",
    ("PRISM", "cell_lines", 4, 110): "epoch=0-step=6164.ckpt",
    ("PRISM", "cell_lines", 4, 1995): "epoch=1-step=12320.ckpt",
    ("PRISM", "cell_lines", 4, 2024): "epoch=2-step=18459.ckpt",
    ("GDSC", "cell_lines", 0, 110): "epoch=19.ckpt",
    ("GDSC", "cell_lines", 0, 1995): "epoch=19-v1.ckpt",
    ("GDSC", "cell_lines", 0, 2024): "epoch=19-v2.ckpt",
    ("GDSC", "cell_lines", 1, 110): "epoch=21-step=16852.ckpt",
    ("GDSC", "cell_lines", 1, 1995): "epoch=19.ckpt",
    ("GDSC", "cell_lines", 1, 2024): "epoch=19.ckpt",
    ("GDSC", "cell_lines", 2, 110): "epoch=19.ckpt",
    ("GDSC", "cell_lines", 2, 1995): "epoch=19.ckpt",
    ("GDSC", "cell_lines", 2, 2024): "epoch=19.ckpt",
    ("GDSC", "cell_lines", 3, 110): "epoch=19.ckpt",
    ("GDSC", "cell_lines", 3, 1995): "epoch=22-step=17503.ckpt",
    ("GDSC", "cell_lines", 3, 2024): "epoch=19.ckpt",
    ("GDSC", "cell_lines", 4, 110): "epoch=19.ckpt",
    ("GDSC", "cell_lines", 4, 1995): "epoch=19.ckpt",
    ("GDSC", "cell_lines", 4, 2024): "epoch=19.ckpt",
    ("GDSCcomb", "cell_lines", 0, 110): "epoch=19.ckpt",
    ("GDSCcomb", "cell_lines", 0, 1995): "epoch=29.ckpt",
    ("GDSCcomb", "cell_lines", 0, 2024): "epoch=19-v4.ckpt",
    ("GDSCcomb", "cell_lines", 1, 110): "epoch=29.ckpt",
    ("GDSCcomb", "cell_lines", 1, 1995): "epoch=19.ckpt",
    ("GDSCcomb", "cell_lines", 1, 2024): "epoch=19.ckpt",
    ("GDSCcomb", "cell_lines", 2, 110): "epoch=19.ckpt",
    ("GDSCcomb", "cell_lines", 2, 1995): "epoch=19.ckpt",
    ("GDSCcomb", "cell_lines", 2, 2024): "epoch=19.ckpt",
    ("GDSCcomb", "cell_lines", 3, 110): "epoch=19.ckpt",
    ("GDSCcomb", "cell_lines", 3, 1995): "epoch=19.ckpt",
    ("GDSCcomb", "cell_lines", 3, 2024): "epoch=19.ckpt",
    ("GDSCcomb", "cell_lines", 4, 110): "epoch=19.ckpt",
    ("GDSCcomb", "cell_lines", 4, 1995): "epoch=19.ckpt",
    ("GDSCcomb", "cell_lines", 4, 2024): "epoch=19.ckpt",
    ("SCORE", "cell_lines", 0, 110): "epoch=0-step=7021.ckpt",
    ("SCORE", "cell_lines", 0, 1995): "epoch=0-step=7019.ckpt",
    ("SCORE", "cell_lines", 0, 2024): "epoch=1-step=14042-v1.ckpt",
    ("SCORE", "cell_lines", 1, 110): "epoch=1-step=14048.ckpt",
    ("SCORE", "cell_lines", 1, 1995): "epoch=1-step=14052.ckpt",
    ("SCORE", "cell_lines", 1, 2024): "epoch=1-step=14054.ckpt",
    ("SCORE", "cell_lines", 2, 110): "epoch=1-step=14048.ckpt",
    ("SCORE", "cell_lines", 2, 1995): "epoch=0-step=7030.ckpt",
    ("SCORE", "cell_lines", 2, 2024): "epoch=1-step=14056.ckpt",
    ("SCORE", "cell_lines", 3, 110): "epoch=1-step=14042.ckpt",
    ("SCORE", "cell_lines", 3, 1995): "epoch=1-step=14038.ckpt",
    ("SCORE", "cell_lines", 3, 2024): "epoch=1-step=14046.ckpt",
    ("SCORE", "cell_lines", 4, 110): "epoch=1-step=14130.ckpt",
    ("SCORE", "cell_lines", 4, 1995): "epoch=1-step=14122.ckpt",
    ("SCORE", "cell_lines", 4, 2024): "epoch=1-step=14128.ckpt",
}


def get_checkpoint_name(dataset: str, split: str, fold: int, seed: int) -> str:
    """Get the checkpoint filename for a specific model configuration.

    Args:
        dataset: Dataset name
        split: 'cell_lines' or 'perturbations'
        fold: Fold number (0-4)
        seed: Seed value (110, 1995, 2024)

    Returns:
        Checkpoint filename

    Raises:
        ValueError: If configuration not found
    """
    key = (dataset, split, fold, seed)
    if key not in CHECKPOINT_MAPPING:
        raise ValueError(
            f"No checkpoint found for {dataset} with split={split}, fold={fold}, seed={seed}. "
            f"This combination may not be available. Available seeds for some configurations "
            f"may be limited (e.g., fold 4 only has seed 2024)."
        )
    return CHECKPOINT_MAPPING[key]


def get_available_datasets() -> List[str]:
    """Get list of available datasets in the Prophet HuggingFace repository.

    Returns:
        List of dataset names available for download
    """
    return ["CTRP", "GDSC", "GDSCcomb", "Horlbeck", "JUMP", "LINCS", "SCORE", "PRISM", "ShifrutMarson", "base"]


def get_available_splits() -> List[str]:
    """Get list of available split methods.

    Returns:
        List of split method names
    """
    return ["cell_lines", "perturbations"]


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


def get_available_configurations() -> Dict[str, List]:
    """Get available model configurations.

    Returns:
        Dictionary with available options for each parameter

    Example:
        >>> configs = get_available_configurations()
        >>> print(configs['datasets'])
        ['CTRP', 'GDSC', 'GDSCcomb', 'Horlbeck', 'JUMP', 'LINCS', 'SCORE', 'PRISM', 'ShifrutMarson', 'base']
    """
    return {
        "datasets": get_available_datasets(),
        "splits": get_available_splits(),
        "folds": get_available_folds(),
        "seeds": get_available_seeds()
    }


def download_embeddings(
    cache_dir: Optional[str] = None,
    force_download: bool = False,
) -> Tuple[str, str]:
    """Download Prophet embeddings separately from model checkpoints.

    Downloads the cell line and intervention (gene/perturbation) embeddings
    that are used across all Prophet models. These embeddings are universal
    and don't depend on which specific model you're using.

    Args:
        cache_dir: Directory to cache downloaded files. If None, uses default cache.
        force_download: Whether to force re-download even if files exist.

    Returns:
        Tuple of (intervention_emb_path, cell_line_emb_path)

    Example:
        >>> from prophet.utils.model_hub import download_embeddings
        >>> iv_emb, cl_emb = download_embeddings()
        >>> print(f"Intervention embeddings: {iv_emb}")
        >>> print(f"Cell line embeddings: {cl_emb}")
    """
    repo_id = "theislab/Prophet"

    if cache_dir is None:
        cache_dir = os.path.join(Path.home(), ".cache", "prophet", "embeddings")
    os.makedirs(cache_dir, exist_ok=True)

    print("Downloading Prophet embeddings from HuggingFace Hub...")

    intervention_emb_path = hf_hub_download(
        repo_id=repo_id,
        filename="embeddings/intervention_embeddings/global_iv_scaledv3.csv",
        cache_dir=cache_dir,
        force_download=force_download,
        repo_type="dataset"
    )
    print(f"  Intervention embeddings: {intervention_emb_path}")

    cell_emb_path = hf_hub_download(
        repo_id=repo_id,
        filename="embeddings/cell_line_embeddings/cell_line_embedding_full_ccle_300_scaled.csv",
        cache_dir=cache_dir,
        force_download=force_download,
        repo_type="dataset"
    )
    print(f"  Cell line embeddings: {cell_emb_path}")

    print("Successfully downloaded all embeddings!")

    return intervention_emb_path, cell_emb_path


def download_model(
    dataset: str,
    split: str = "cell_lines",
    fold: int = 0,
    seed: int = 110,
    cache_dir: Optional[str] = None,
    force_download: bool = False,
    download_embeddings_flag: bool = True,
) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
    """Download Prophet model with specific configuration.

    Note:
        For most users, use the high-level API instead:
        >>> from prophet import Prophet
        >>> model = Prophet.from_pretrained("GDSC", split="cell_lines", fold=0, seed=110)

        This function is for advanced users who need direct file path access.

    Args:
        dataset: Dataset name ('GDSC', 'CTRP', 'LINCS', 'JUMP', 'Horlbeck',
                'SCORE', 'GDSCcomb', 'ShifrutMarson', 'PRISM', or 'base')
        split: Split type - 'cell_lines' or 'perturbations' (default: 'cell_lines')
        fold: Cross-validation fold number 0-4 (default: 0)
        seed: Random seed - 110, 1995, or 2024 (default: 110)
        cache_dir: Directory to cache downloaded files
        force_download: Whether to force re-download even if files exist
        download_embeddings_flag: Whether to download embeddings (default: True)

    Returns:
        Tuple of (model_path, intervention_emb_path, cell_emb_path, phenotype_emb_path)
        If download_embeddings_flag=False, embedding paths will be None

    Raises:
        ValueError: If invalid parameters are provided

    Example:
        >>> # Download GDSC model with unseen cell lines, fold 0, seed 110
        >>> model, gene_emb, cell_emb, _ = download_model("GDSC", "cell_lines", 0, 110)
        >>>
        >>> # Download CTRP model with unseen perturbations, fold 2, seed 1995
        >>> model, gene_emb, cell_emb, _ = download_model("CTRP", "perturbations", 2, 1995)
        >>>
        >>> # Download only model checkpoint, skip embeddings
        >>> model, _, _, _ = download_model("GDSC", "cell_lines", 0, 110, download_embeddings_flag=False)
    """
    valid_datasets = get_available_datasets()
    valid_splits = get_available_splits()
    valid_seeds = get_available_seeds()

    if dataset not in valid_datasets:
        raise ValueError(f"Invalid dataset '{dataset}'. Choose from: {', '.join(valid_datasets)}")

    if split not in valid_splits:
        raise ValueError(f"Invalid split '{split}'. Choose from: {', '.join(valid_splits)}")

    if fold not in range(5):
        raise ValueError(f"Invalid fold {fold}. Must be 0-4")

    if seed not in valid_seeds:
        raise ValueError(f"Invalid seed {seed}. Choose from: {valid_seeds}")

    repo_id = "theislab/Prophet"

    checkpoint_name = get_checkpoint_name(dataset, split, fold, seed)

    if dataset == "base":
        model_path = f"base_pretrained/unseen_{split}_fold_{fold}/seed_{seed}/{checkpoint_name}"
    else:
        model_path = f"{dataset}/unseen_{split}_fold_{fold}/seed_{seed}/{checkpoint_name}"

    if cache_dir is None:
        cache_dir = os.path.join(Path.home(), ".cache", "prophet",
                                f"{dataset}_{split}_fold{fold}_seed{seed}")
    os.makedirs(cache_dir, exist_ok=True)

    model_file = hf_hub_download(
        repo_id=repo_id,
        filename=model_path,
        cache_dir=cache_dir,
        force_download=force_download,
        repo_type="dataset"
    )

    print(f"Successfully downloaded {dataset} model: {checkpoint_name}")
    print(f"  Configuration: split=unseen_{split}, fold={fold}, seed={seed}")
    print(f"  Model file: {model_file}")

    if download_embeddings_flag:
        intervention_emb_path = hf_hub_download(
            repo_id=repo_id,
            filename="embeddings/intervention_embeddings/global_iv_scaledv3.csv",
            cache_dir=cache_dir,
            force_download=force_download,
            repo_type="dataset"
        )

        cell_emb_path = hf_hub_download(
            repo_id=repo_id,
            filename="embeddings/cell_line_embeddings/cell_line_embedding_full_ccle_300_scaled.csv",
            cache_dir=cache_dir,
            force_download=force_download,
            repo_type="dataset"
        )
    else:
        intervention_emb_path = None
        cell_emb_path = None

    return model_file, intervention_emb_path, cell_emb_path, None


def list_available_checkpoints(dataset: Optional[str] = None, split: Optional[str] = None) -> List[Tuple[str, str, int, int]]:
    """List all available checkpoint configurations.

    Args:
        dataset: Optional filter by dataset name
        split: Optional filter by split type ('cell_lines' or 'perturbations')

    Returns:
        List of tuples (dataset, split, fold, seed) for available checkpoints

    Example:
        >>> # List all available checkpoints
        >>> checkpoints = list_available_checkpoints()
        >>> # List checkpoints for GDSC only
        >>> gdsc_ckpts = list_available_checkpoints(dataset="GDSC")
        >>> # List checkpoints for perturbations split only
        >>> pert_ckpts = list_available_checkpoints(split="perturbations")
    """
    checkpoints = list(CHECKPOINT_MAPPING.keys())

    if dataset is not None:
        checkpoints = [c for c in checkpoints if c[0] == dataset]

    if split is not None:
        checkpoints = [c for c in checkpoints if c[1] == split]

    return sorted(checkpoints)


def print_available_models(dataset: Optional[str] = None):
    """Print a formatted list of available models.

    Args:
        dataset: Optional filter to show only specific dataset

    Example:
        >>> # Print all available models
        >>> print_available_models()
        >>> # Print only GDSC models
        >>> print_available_models(dataset="GDSC")
    """
    checkpoints = list_available_checkpoints(dataset=dataset)

    if not checkpoints:
        print(f"No checkpoints found for dataset: {dataset}")
        return

    print("Available Prophet Model Checkpoints:")
    print("=" * 70)

    current_dataset = None
    for ds, split, fold, seed in checkpoints:
        if ds != current_dataset:
            if current_dataset is not None:
                print()
            print(f"\nDataset: {ds}")
            print("-" * 70)
            current_dataset = ds

        ckpt_name = CHECKPOINT_MAPPING[(ds, split, fold, seed)]
        print(f"  split={split:<15} fold={fold}  seed={seed:<4}  -> {ckpt_name}")

    print("\n" + "=" * 70)
    print("\nRecommended Usage:")
    print("  from prophet import Prophet")
    print('  model = Prophet.from_pretrained("GDSC", split="perturbations", fold=0, seed=110)')
    print("\nAdvanced (low-level API):")
    print("  from prophet.utils.model_hub import download_model")
    print('  model_path, gene_emb, cell_emb, _ = download_model("GDSC", "perturbations", fold=0, seed=110)')
