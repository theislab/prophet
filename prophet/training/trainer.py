"""
Prophet Training Module

Core training implementation for Prophet models.
"""

import os
from pathlib import Path
from typing import Union, List, Optional, Tuple, Dict
import pandas as pd
import numpy as np
import yaml
import torch
import pytorch_lightning as pl
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import pickle
import json

from ..core.prophet import Prophet
from ..core.config import set_config
from ..models import load_models_config
from ..data import dataset_registry
from ..data.processing import DataSplitter
from ..utils import validate_prophet_inputs


class ProphetTrainer:
    """Clean Prophet training implementation."""

    def __init__(self, config_path: str, seed: int = 42):
        self.config = self._load_config(config_path)
        self.seed = seed
        self.prophet_model = None
        # Store embedding paths after loading data
        self.iv_emb_paths = None
        self.cl_emb_paths = None
        self.ph_emb_paths = None
        self.split_assignments = None  # Store current split assignments

    def _load_config(self, config_path: str) -> dict:
        """Load and validate configuration."""
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Validate data configuration for custom paths only
        data_config = config["data"]
        if "setting" not in data_config or not data_config["setting"]:
            # Validate custom paths
            required_fields = ["data_path", "iv_embeddings", "cl_embeddings"]
            for field in required_fields:
                if field not in data_config:
                    raise ValueError(f"Missing required field: {field}")

        return config

    def load_data(self, data_config: dict) -> pd.DataFrame:
        """Load training data from setting or custom paths."""
        if "setting" in data_config and data_config["setting"]:
            # Use predefined dataset setting
            dataset_config = dataset_registry.get_dataset_config(data_config["setting"])
            data_paths = dataset_config["data_paths"]
            self.iv_emb_paths = dataset_config["iv_embeddings"]
            self.cl_emb_paths = dataset_config["cl_embeddings"]
            self.ph_emb_paths = dataset_config["ph_embeddings"]
        else:
            # Use custom dataset paths
            data_paths = (
                [data_config["data_path"]]
                if isinstance(data_config["data_path"], str)
                else data_config["data_path"]
            )
            self.iv_emb_paths = data_config["iv_embeddings"]
            self.cl_emb_paths = data_config["cl_embeddings"]
            self.ph_emb_paths = data_config.get("ph_embeddings")

        # Load and combine datasets
        datasets = []
        for path in tqdm(data_paths, desc="Loading datasets"):
            if path.endswith(".parquet"):
                df = pd.read_parquet(path)
            elif path.endswith(".csv"):
                df = pd.read_csv(path)
            else:
                raise ValueError(f"Unsupported format: {path}")

            # Subsample SCORE dataset to 33% if it's being loaded
            if "SCORE_dataset.csv" in path:
                df = df.sample(frac=0.33, random_state=self.seed).reset_index(drop=True)

            datasets.append(df)

        if len(datasets) > 1:
            df = pd.concat(datasets, ignore_index=True)
            print(f"Combined dataset size: {len(df)}")
        else:
            df = datasets[0]

        # Validate data
        validation_results = validate_prophet_inputs(
            df=df,
            iv_emb_path=self.iv_emb_paths,
            cl_emb_path=self.cl_emb_paths,
            ph_emb_path=self.ph_emb_paths,
            iv_col=data_config["iv_cols"],
            cl_col=data_config["cl_col"],
            ph_col=data_config["ph_col"],
            readout_col=data_config["readout_col"],
            mode="train",
        )

        return validation_results["processed_inputs"]["df"]

    def save_split_assignments(self, split_assignments: Dict, output_dir: str):
        """Save split assignments to the model output directory."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        splits_file = output_path / "split_assignments.json"

        # Convert sets to lists for JSON serialization
        serializable_assignments = {}
        for fold_idx, assignments in split_assignments.items():
            serializable_assignments[fold_idx] = {
                "train": list(assignments["train"]),
                "val": list(assignments["val"]),
                "test": list(assignments["test"]),
            }

        with open(splits_file, "w") as f:
            json.dump(serializable_assignments, f, indent=2)

        print(f"💾 Saved split assignments to {splits_file}")

    def load_split_assignments(self, checkpoint_path: str) -> Optional[Dict]:
        """Load split assignments from a pretrained model directory."""
        if not checkpoint_path:
            return None

        checkpoint_dir = (
            Path(checkpoint_path).parent
            if Path(checkpoint_path).is_file()
            else Path(checkpoint_path)
        )
        splits_file = checkpoint_dir / "split_assignments.json"

        if not splits_file.exists():
            print(f"⚠️  No split assignments found at {splits_file}")
            print(f"   Creating new splits for this experiment")
            return None

        print(f"📁 Loading split assignments from {splits_file}")

        with open(splits_file, "r") as f:
            serializable_assignments = json.load(f)

        # Convert lists back to sets and ensure integer keys
        split_assignments = {}
        for fold_idx, assignments in serializable_assignments.items():
            split_assignments[int(fold_idx)] = {
                "train": set(assignments["train"]),
                "val": set(assignments["val"]),
                "test": set(assignments["test"]),
            }

        return split_assignments

    def create_new_split_assignments(
        self,
        df: pd.DataFrame,
        iv_cols: List[str],
        seed: int,
        n_splits: int,
        val_fraction: float,
    ) -> Dict:
        """Create stratified split assignments ensuring equal contribution from each dataset (by phenotype)."""

        # Get phenotype column
        ph_col = self.config["data"]["ph_col"]

        # Get unique phenotypes (datasets)
        unique_phenotypes = df[ph_col].unique()
        print(f"📊 Found {len(unique_phenotypes)} unique phenotypes/datasets")

        # Get interventions per phenotype
        phenotype_interventions = {}
        for phenotype in unique_phenotypes:
            phenotype_df = df[df[ph_col] == phenotype]

            # Get unique interventions for this phenotype
            phenotype_ivs = set()
            for col in iv_cols:
                if col in phenotype_df.columns:
                    phenotype_ivs.update(phenotype_df[col].dropna().unique())

            phenotype_interventions[phenotype] = sorted(list(phenotype_ivs))
            print(f"  📈 {phenotype}: {len(phenotype_ivs)} unique interventions")

        # Create stratified split assignments for each fold
        split_assignments = {}

        for i in range(n_splits):
            fold_seed = seed + i
            np.random.seed(fold_seed)

            # Collect interventions from each phenotype separately
            all_train_interventions = set()
            all_val_interventions = set()
            all_test_interventions = set()

            for phenotype, interventions in phenotype_interventions.items():
                if not interventions:
                    continue

                # Shuffle interventions for this phenotype
                shuffled_ivs = np.random.permutation(interventions)

                # Calculate sizes for this phenotype
                n_test = int(len(interventions) / n_splits)
                n_val = int(len(interventions) * val_fraction)

                # Split this phenotype's interventions
                test_ivs = set(shuffled_ivs[:n_test])
                val_ivs = set(shuffled_ivs[n_test : n_test + n_val])
                train_ivs = set(shuffled_ivs[n_test + n_val :])

                # Add to global sets
                all_test_interventions.update(test_ivs)
                all_val_interventions.update(val_ivs)
                all_train_interventions.update(train_ivs)

                if len(interventions) > 100:  # Only show details for larger phenotypes
                    print(
                        f"    {phenotype} fold {i}: {len(train_ivs)} train, {len(val_ivs)} val, {len(test_ivs)} test"
                    )

            split_assignments[i] = {
                "train": all_train_interventions,
                "val": all_val_interventions,
                "test": all_test_interventions,
            }

            print(
                f"  🎯 Total fold {i}: {len(all_train_interventions)} train, {len(all_val_interventions)} val, {len(all_test_interventions)} test interventions"
            )

        return split_assignments

    def create_splits(
        self, df: pd.DataFrame, splitting_config: dict, data_config: dict
    ) -> List[Tuple]:
        """Create train/validation/test splits."""
        method = splitting_config["method"]
        val_fraction = splitting_config["val_fraction"]
        n_splits = splitting_config["n_splits"]
        seed = self.seed

        splits = []

        print("Creating splits using method: ", method)

        if method == "random":
            # Random cross-validation splits
            for i in range(n_splits):
                fold_seed = seed + i

                # Create random train/val/test split
                shuffled_df = df.sample(frac=1, random_state=fold_seed).reset_index(
                    drop=True
                )
                n_val = int(len(shuffled_df) * val_fraction)
                n_test = int(len(shuffled_df) * val_fraction)  # Same size as validation

                val_df = shuffled_df.iloc[:n_val]
                test_df = shuffled_df.iloc[n_val : n_val + n_test]
                train_df = shuffled_df.iloc[n_val + n_test :]

                descriptor = f"random_fold_{i}"
                splits.append((train_df, val_df, test_df, descriptor))

        elif method == "intervention_holdout" or method == "leave_iv_out":
            # Check if we're finetuning (have checkpoint) and load existing splits
            checkpoint_path = self.config.get("checkpoint_path")

            if checkpoint_path:
                print(f"FINETUNING MODE: Looking for existing split assignments...")
                split_assignments = self.load_split_assignments(checkpoint_path)
            else:
                print(f"TRAINING FROM SCRATCH: Will create new split assignments...")
                split_assignments = None

            # If no existing splits found, create new ones
            if split_assignments is None:
                split_assignments = self.create_new_split_assignments(
                    df, data_config["iv_cols"], seed, n_splits, val_fraction
                )
                self.split_assignments = split_assignments  # Store for later saving
            else:
                print(f"Using existing split assignments from pretrained model")
                self.split_assignments = split_assignments

            # Apply split assignments to current dataset
            for i in range(n_splits):
                assignments = split_assignments[i]
                train_interventions = assignments["train"]
                val_interventions = assignments["val"]
                test_interventions = assignments["test"]

                # Create masks for each split
                test_mask = pd.Series(False, index=df.index)
                val_mask = pd.Series(False, index=df.index)

                for col in data_config["iv_cols"]:
                    test_mask |= df[col].isin(test_interventions)
                    val_mask |= df[col].isin(val_interventions)

                # Split the data
                test_df = df[test_mask].copy()
                val_df = df[val_mask & ~test_mask].copy()
                train_df = df[~test_mask & ~val_mask].copy()

                descriptor = f"iv_fold_{i}"

                # Show what's actually present in this dataset
                current_train_ivs = set()
                current_val_ivs = set()
                current_test_ivs = set()

                for col in data_config["iv_cols"]:
                    current_train_ivs.update(train_df[col].dropna().unique())
                    current_val_ivs.update(val_df[col].dropna().unique())
                    current_test_ivs.update(test_df[col].dropna().unique())

                print(
                    f"Split {i}: {len(train_df)} training samples, {len(val_df)} validation samples, {len(test_df)} test samples"
                )
                print(
                    f"  Assigned interventions - Train: {len(train_interventions)}, Val: {len(val_interventions)}, Test: {len(test_interventions)}"
                )
                print(
                    f"  Present interventions - Train: {len(current_train_ivs)}, Val: {len(current_val_ivs)}, Test: {len(current_test_ivs)}"
                )

                splits.append((train_df, val_df, test_df, descriptor))

        elif method == "cell_line_holdout":
            # Cell line holdout splits
            for i in range(n_splits):
                fold_seed = seed + i

                train_test_df, test_df = DataSplitter.cell_line_holdout_split(
                    df,
                    cl_col=data_config["cl_col"],
                    holdout_fraction=1.0 / n_splits,
                    random_state=fold_seed,
                )

                # Split training data into train/val
                n_val = int(len(train_test_df) * val_fraction)
                train_test_df = train_test_df.sample(
                    frac=1, random_state=fold_seed
                ).reset_index(drop=True)
                val_df = train_test_df.iloc[:n_val]
                train_df = train_test_df.iloc[n_val:]

                descriptor = f"cl_fold_{i}"
                splits.append((train_df, val_df, test_df, descriptor))
        else:
            raise ValueError(f"Unknown splitting method: {method}")

        return splits

    def run_cross_validation(
        self,
        splits: List[Tuple],
        data_config: dict,
        output_dir: str,
        checkpoint_path: Optional[str] = None,
    ) -> List:
        """Run cross-validation training."""

        model_split_path = (
            Path(output_dir)
            / data_config["setting"]
            / self.config["splitting"]["method"]
            / str(self.seed)
        )

        # Save split assignments if we created new ones (i.e., training from scratch)
        if self.split_assignments and not checkpoint_path:
            self.save_split_assignments(self.split_assignments, model_split_path)

        # Create config once for all folds
        config_dict = self.config.copy()
        config_dict.update(
            {
                "setting": data_config["setting"],
                "leaveout_method": self.config["splitting"]["method"],
                "max_steps": self.config["max_steps"],
                "batch_size": self.config["batch_size"],
                "early_stopping": self.config["early_stopping"],
                "patience": self.config["patience"],
                "pert_len": self.config["pert_len"],
                "random_seed": self.seed,
            }
        )
        prophet_config = set_config(config_dict)

        trained_models = []
        for i, (train_df, val_df, test_df, descriptor) in enumerate(splits):
            print(f"Training fold {i + 1}/{len(splits)}: {descriptor}")

            # Construct checkpoint path: {output_dir}/{setting}/{leaveout_method}/{seed}/{fold_number}/unbalanced_{unbalanced}/
            unbalanced = self.config.get("unbalanced", False)
            checkpoint_dir = model_split_path / f"fold_{i}" / f"unbalanced_{unbalanced}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            # Train model on this split (pass the pre-created config)
            model = self.train_single_split(
                train_df,
                val_df,
                test_df,
                prophet_config,
                data_config,
                descriptor,
                checkpoint_path,
                str(checkpoint_dir),
            )

            # Store reference to trained model (checkpoint is saved automatically by ModelCheckpoint)
            trained_models.append(
                {
                    "model": model,
                    "checkpoint_dir": str(checkpoint_dir),
                    "descriptor": descriptor,
                    "fold": i,
                }
            )

        return trained_models

    def train_single_split(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        prophet_config: object,
        data_config: dict,
        split_name: str,
        checkpoint_path: Optional[str] = None,
        checkpoint_dirpath: Optional[str] = None,
    ):
        """Train or fine-tune on a single split."""

        # Normalize the readout column per phenotype to prevent data leakage
        readout_col = data_config["readout_col"]
        phenotype_col = data_config["ph_col"]

        # Create copies to avoid modifying original data
        train_df_norm = train_df.copy()
        val_df_norm = val_df.copy()
        test_df_norm = test_df.copy()

        # Store scalers per phenotype for later inverse transformation if needed
        self.current_scalers = {}

        # Get all phenotypes present in training data
        train_phenotypes = train_df_norm[phenotype_col].unique()

        print(
            f"Applying per-phenotype MinMax scaling for {len(train_phenotypes)} phenotypes"
        )

        # Apply MinMax scaling per phenotype
        for phenotype in train_phenotypes:
            # Get training data for this phenotype to fit scaler
            train_phe_mask = train_df_norm[phenotype_col] == phenotype

            train_phe_values = train_df_norm.loc[
                train_phe_mask, readout_col
            ].values.reshape(-1, 1)

            if len(train_phe_values) == 0:
                continue

            # Fit scaler on this phenotype's training data only
            scaler = MinMaxScaler()
            scaler.fit(train_phe_values)

            # Transform training data for this phenotype
            train_df_norm.loc[train_phe_mask, readout_col] = scaler.transform(
                train_phe_values
            ).flatten()

            # Transform validation data for this phenotype (if present)
            val_phe_mask = val_df_norm[phenotype_col] == phenotype
            if val_phe_mask.any():
                val_phe_values = val_df_norm.loc[
                    val_phe_mask, readout_col
                ].values.reshape(-1, 1)
                val_df_norm.loc[val_phe_mask, readout_col] = scaler.transform(
                    val_phe_values
                ).flatten()

            # Transform test data for this phenotype (if present)
            test_phe_mask = test_df_norm[phenotype_col] == phenotype
            if test_phe_mask.any():
                test_phe_values = test_df_norm.loc[
                    test_phe_mask, readout_col
                ].values.reshape(-1, 1)
                test_df_norm.loc[test_phe_mask, readout_col] = scaler.transform(
                    test_phe_values
                ).flatten()

            # Print scaling info for this phenotype
            original_min = train_phe_values.min()
            original_max = train_phe_values.max()
            print(
                f"  {phenotype}: [{original_min:.3f}, {original_max:.3f}] -> [0.000, 1.000] ({len(train_phe_values)} train samples)"
            )

        # Handle phenotypes that appear in val/test but not in training
        # These will use the scaler from the most similar phenotype or be left unscaled with a warning
        all_phenotypes = (
            set(train_df_norm[phenotype_col].unique())
            | set(val_df_norm[phenotype_col].unique())
            | set(test_df_norm[phenotype_col].unique())
        )

        missing_phenotypes = all_phenotypes - set(train_phenotypes)
        if missing_phenotypes:
            print(
                f"⚠️  Warning: Phenotypes {missing_phenotypes} appear in val/test but not in training."
            )
            print(f"   These will not be scaled. Consider adjusting your data splits.")

        # Create Prophet instance with the pre-created config
        self.prophet_model = Prophet(
            iv_emb_path=self.iv_emb_paths,
            cl_emb_path=self.cl_emb_paths,
            ph_emb_path=self.ph_emb_paths,
            model_pth=checkpoint_path,
            architecture=self.config.get("architecture", "Transformer"),
            config=prophet_config,
        )

        # Train the model with normalized data
        self.prophet_model.train(
            df=train_df_norm,
            val_df=val_df_norm,
            test_df=test_df_norm,
            iv_col=data_config["iv_cols"],
            cl_col=data_config["cl_col"],
            ph_col=data_config["ph_col"],
            readout_col=readout_col,
            model_config=prophet_config,
            wandb_config=self.config["wandb"],
            checkpoint_dirpath=checkpoint_dirpath,
        )

        return self.prophet_model
