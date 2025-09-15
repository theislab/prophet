"""Dataset registry for predefined dataset combinations."""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import os


class DatasetRegistry:
    """Registry for predefined dataset configurations."""

    def __init__(self):
        # Base paths - these should be configurable via environment variables
        base_data_path = os.getenv(
            "PROPHET_DATA_PATH",
            "/lustre/groups/ml01/projects/super_rad_project/HF/data_hf",
        )
        base_emb_path = os.getenv(
            "PROPHET_EMB_PATH",
            "/lustre/groups/ml01/projects/super_rad_project/HF/emb_hf",
        )

        self.registry = {
            "SCORE": {
                "data_paths": [f"{base_data_path}/SCORE_dataset.csv"],
                "iv_embeddings": [
                    f"{base_emb_path}/intervention_embeddings/global_iv_scaledv3.csv"
                ],
                "cl_embeddings": [
                    f"{base_emb_path}/cell_line_embeddings/cell_line_embedding_full_ccle_300_scaled.csv"
                ],
                "ph_embeddings": None,
            },
            "Rad": {
                "data_paths": [f"{base_data_path}/Rad_dataset.csv"],
                "iv_embeddings": [
                    f"{base_emb_path}/intervention_embeddings/global_iv_scaledv3.csv"
                ],
                "cl_embeddings": [
                    f"{base_emb_path}/cell_line_embeddings/cell_line_embedding_full_ccle_300_scaled.csv"
                ],
                "ph_embeddings": None,
            },
            "GDSC": {
                "data_paths": [f"{base_data_path}/GDSC_dataset.csv"],
                "iv_embeddings": [
                    f"{base_emb_path}/intervention_embeddings/global_iv_scaledv3.csv"
                ],
                "cl_embeddings": [
                    f"{base_emb_path}/cell_line_embeddings/cell_line_embedding_full_ccle_300_scaled.csv"
                ],
                "ph_embeddings": None,
            },
            "CTRP": {
                "data_paths": [f"{base_data_path}/CTRP_dataset.csv"],
                "iv_embeddings": [
                    f"{base_emb_path}/intervention_embeddings/global_iv_scaledv3.csv",
                    f"{base_emb_path}/intervention_embeddings/CTRP_with_smiles_simscaled.csv",
                ],
                "cl_embeddings": [
                    f"{base_emb_path}/cell_line_embeddings/cell_line_embedding_full_ccle_300_scaled.csv"
                ],
                "ph_embeddings": None,
            },
            "JUMP": {
                "data_paths": [f"{base_data_path}/JUMP_dataset.csv"],
                "iv_embeddings": [
                    f"{base_emb_path}/intervention_embeddings/global_iv_scaledv3.csv"
                ],
                "cl_embeddings": [
                    f"{base_emb_path}/cell_line_embeddings/cell_line_embedding_full_ccle_300_scaled.csv"
                ],
                "ph_embeddings": None,
            },
            "LINCS": {
                "data_paths": [f"{base_data_path}/LINCS_dataset.csv"],
                "iv_embeddings": [
                    f"{base_emb_path}/intervention_embeddings/global_iv_scaledv3.csv"
                ],
                "cl_embeddings": [
                    f"{base_emb_path}/cell_line_embeddings/cell_line_embedding_full_ccle_300_scaled.csv"
                ],
                "ph_embeddings": None,
            },
            "GDSCcomb": {
                "data_paths": [f"{base_data_path}/GDSCcomb_dataset.csv"],
                "iv_embeddings": [
                    f"{base_emb_path}/intervention_embeddings/global_iv_scaledv3.csv"
                ],
                "cl_embeddings": [
                    f"{base_emb_path}/cell_line_embeddings/cell_line_embedding_full_ccle_300_scaled.csv"
                ],
                "ph_embeddings": None,
            },
            "PRISM": {
                "data_paths": [f"{base_data_path}/PRISM_dataset.csv"],
                "iv_embeddings": [
                    f"{base_emb_path}/intervention_embeddings/global_iv_scaledv3.csv"
                ],
                "cl_embeddings": [
                    f"{base_emb_path}/cell_line_embeddings/cell_line_embedding_full_ccle_300_scaled.csv"
                ],
                "ph_embeddings": None,
            },
            "Shifru": {
                "data_paths": [f"{base_data_path}/Shifru_dataset.csv"],
                "iv_embeddings": [
                    f"{base_emb_path}/intervention_embeddings/global_iv_scaledv3.csv"
                ],
                "cl_embeddings": [
                    f"{base_emb_path}/cell_line_embeddings/cell_line_embedding_full_ccle_300_scaled.csv"
                ],
                "ph_embeddings": None,
            },
            "everything": {
                "data_paths": [
                    # f"{base_data_path}/SCORE_dataset.csv",
                    f"{base_data_path}/GDSC_dataset.csv",
                    f"{base_data_path}/CTRP_dataset.csv",
                    # f"{base_data_path}/JUMP_dataset.csv",
                    # f"{base_data_path}/LINCS_dataset.csv",
                    # f"{base_data_path}/GDSCcomb_dataset.csv",
                    # f"{base_data_path}/PRISM_dataset.csv",
                ],
                "iv_embeddings": [
                    f"{base_emb_path}/intervention_embeddings/global_iv_scaledv3.csv",
                    f"{base_emb_path}/intervention_embeddings/CTRP_with_smiles_simscaled.csv",
                ],
                "cl_embeddings": [
                    f"{base_emb_path}/cell_line_embeddings/cell_line_embedding_full_ccle_300_scaled.csv"
                ],
                "ph_embeddings": None,
            },
        }

    def get_dataset_config(self, setting: str) -> Dict:
        """Get dataset configuration for a given setting."""
        if setting not in self.registry:
            available = list(self.registry.keys())
            raise ValueError(
                f"Unknown setting: '{setting}'. Available settings: {available}"
            )
        return self.registry[setting]

    def list_available_settings(self) -> List[str]:
        """List all available dataset settings."""
        return list(self.registry.keys())

    def validate_paths(self, setting: str) -> Dict[str, List[str]]:
        """Validate that all paths for a setting exist."""
        config = self.get_dataset_config(setting)
        missing_paths = {
            "data_paths": [],
            "iv_embeddings": [],
            "cl_embeddings": [],
            "ph_embeddings": [],
        }

        # Check data paths
        for path in config["data_paths"]:
            if not os.path.exists(path):
                missing_paths["data_paths"].append(path)

        # Check embedding paths
        for path in config["iv_embeddings"]:
            if not os.path.exists(path):
                missing_paths["iv_embeddings"].append(path)

        for path in config["cl_embeddings"]:
            if not os.path.exists(path):
                missing_paths["cl_embeddings"].append(path)

        if config["ph_embeddings"]:
            for path in config["ph_embeddings"]:
                if not os.path.exists(path):
                    missing_paths["ph_embeddings"].append(path)

        return missing_paths


# Global instance
dataset_registry = DatasetRegistry()
