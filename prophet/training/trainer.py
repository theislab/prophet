"""
Prophet Training Module

Core training implementation for Prophet models.
"""

import os
from pathlib import Path
from typing import Union, List, Optional, Tuple
import pandas as pd
import numpy as np
import yaml
import torch
import pytorch_lightning as pl
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

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
    
    def _load_config(self, config_path: str) -> dict:
        """Load and validate configuration."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate data configuration for custom paths only
        data_config = config['data']
        if 'setting' not in data_config or not data_config['setting']:
            # Validate custom paths
            required_fields = ['data_path', 'iv_embeddings', 'cl_embeddings']
            for field in required_fields:
                if field not in data_config:
                    raise ValueError(f"Missing required field: {field}")
        
        return config
    
    def load_data(self, data_config: dict) -> pd.DataFrame:
        """Load training data from setting or custom paths."""
        if 'setting' in data_config and data_config['setting']:
            # Use predefined dataset setting
            dataset_config = dataset_registry.get_dataset_config(data_config['setting'])
            data_paths = dataset_config["data_paths"]
            self.iv_emb_paths = dataset_config["iv_embeddings"]
            self.cl_emb_paths = dataset_config["cl_embeddings"]
            self.ph_emb_paths = dataset_config["ph_embeddings"]
        else:
            # Use custom dataset paths
            data_paths = [data_config['data_path']] if isinstance(data_config['data_path'], str) else data_config['data_path']
            self.iv_emb_paths = data_config['iv_embeddings']
            self.cl_emb_paths = data_config['cl_embeddings']
            self.ph_emb_paths = data_config.get('ph_embeddings')
        
        # Load and combine datasets
        datasets = []
        for path in tqdm(data_paths, desc="Loading datasets"):
            if path.endswith('.parquet'):
                df = pd.read_parquet(path)
            elif path.endswith('.csv'):
                df = pd.read_csv(path)
            else:
                raise ValueError(f"Unsupported format: {path}")
            
            # Subsample SCORE dataset to 33% if it's being loaded
            if 'SCORE_dataset.csv' in path:
                print(f"Original SCORE dataset size: {len(df)}")
                df = df.sample(frac=0.33, random_state=self.seed).reset_index(drop=True)
                print(f"Subsampled SCORE dataset size: {len(df)} (33%)")
            
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
            iv_col=data_config['iv_cols'],
            cl_col=data_config['cl_col'],
            ph_col=data_config['ph_col'],
            readout_col=data_config['readout_col'],
            mode="train"
        )

        return validation_results['processed_inputs']['df']
    
    def create_splits(self, df: pd.DataFrame, splitting_config: dict, data_config: dict) -> List[Tuple]:
        """Create train/validation/test splits."""
        method = splitting_config['method']
        val_fraction = splitting_config['val_fraction']
        n_splits = splitting_config['n_splits']
        seed = self.seed
        
        splits = []
        
        print("Creating splits using method: ", method)
        
        if method == "random":
            # Random cross-validation splits
            for i in range(n_splits):
                fold_seed = seed + i
                
                # Create random train/val/test split
                shuffled_df = df.sample(frac=1, random_state=fold_seed).reset_index(drop=True)
                n_val = int(len(shuffled_df) * val_fraction)
                n_test = int(len(shuffled_df) * val_fraction)  # Same size as validation
                
                val_df = shuffled_df.iloc[:n_val]
                test_df = shuffled_df.iloc[n_val:n_val+n_test]
                train_df = shuffled_df.iloc[n_val+n_test:]
                
                descriptor = f"random_fold_{i}"
                splits.append((train_df, val_df, test_df, descriptor))
                
        elif method == "intervention_holdout" or method == "leave_iv_out":
            # Use existing function for each fold
            for i in range(n_splits):
                # Use different seed for each fold
                fold_seed = seed + i
                
                # Get train/test split using existing function
                train_test_df, test_df = DataSplitter.intervention_holdout_split(
                    df, 
                    iv_cols=data_config['iv_cols'],
                    holdout_fraction=1.0/n_splits,  # Each fold holds out 1/n_splits
                    random_state=fold_seed
                )
                
                # Split training data into train/val
                n_val = int(len(train_test_df) * val_fraction)
                train_test_df = train_test_df.sample(frac=1, random_state=fold_seed).reset_index(drop=True)
                val_df = train_test_df.iloc[:n_val]
                train_df = train_test_df.iloc[n_val:]
                
                descriptor = f"iv_fold_{i}"
                
                print(f"Split {i}: {len(train_df)} training samples, {len(val_df)} validation samples, {len(test_df)} test samples")
                splits.append((train_df, val_df, test_df, descriptor))
                
        elif method == "cell_line_holdout":
            # Cell line holdout splits
            for i in range(n_splits):
                fold_seed = seed + i
                
                train_test_df, test_df = DataSplitter.cell_line_holdout_split(
                    df,
                    cl_col=data_config['cl_col'],
                    holdout_fraction=1.0/n_splits,
                    random_state=fold_seed
                )
                
                # Split training data into train/val
                n_val = int(len(train_test_df) * val_fraction)
                train_test_df = train_test_df.sample(frac=1, random_state=fold_seed).reset_index(drop=True)
                val_df = train_test_df.iloc[:n_val]
                train_df = train_test_df.iloc[n_val:]
                
                descriptor = f"cl_fold_{i}"
                splits.append((train_df, val_df, test_df, descriptor))
        else:
            raise ValueError(f"Unknown splitting method: {method}")
            
        return splits
    
    def run_cross_validation(self, splits: List[Tuple], data_config: dict, 
                           output_dir: str, checkpoint_path: Optional[str] = None) -> List:
        """Run cross-validation training."""
        
        # Create config once for all folds
        config_dict = self.config.copy()
        config_dict.update({
            'setting': data_config['setting'],
            'leaveout_method': self.config['splitting']['method'],
            'max_steps': self.config['max_steps'],
            'batch_size': self.config['batch_size'],
            'early_stopping': self.config['early_stopping'],
            'patience': self.config['patience'],
            'pert_len': self.config['pert_len'],
            'random_seed': self.seed
        })
        prophet_config = set_config(config_dict)
        
        trained_models = []
        for i, (train_df, val_df, test_df, descriptor) in enumerate(splits):
            print(f"Training fold {i+1}/{len(splits)}: {descriptor}")
            
            # Construct checkpoint path: {output_dir}/{setting}/{leaveout_method}/{seed}/{fold_number}/unbalanced_{unbalanced}/
            unbalanced = self.config.get('unbalanced', False)
            checkpoint_dir = Path(output_dir) / data_config['setting'] / self.config['splitting']['method'] / str(self.seed) / f"fold_{i}" / f"unbalanced_{unbalanced}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Train model on this split (pass the pre-created config)
            model = self.train_single_split(
                train_df, val_df, test_df, prophet_config, data_config, descriptor, 
                checkpoint_path, str(checkpoint_dir)
            )
            
            # Store reference to trained model (checkpoint is saved automatically by ModelCheckpoint)
            trained_models.append({
                'model': model,
                'checkpoint_dir': str(checkpoint_dir),
                'descriptor': descriptor,
                'fold': i
            })
            
        return trained_models

    def train_single_split(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                          prophet_config: object, data_config: dict, split_name: str, 
                          checkpoint_path: Optional[str] = None, checkpoint_dirpath: Optional[str] = None):
        """Train or fine-tune on a single split."""
        
        # Normalize the readout column to prevent data leakage
        readout_col = data_config['readout_col']
        
        # Fit scaler on training data only
        scaler = MinMaxScaler()
        
        # Create copies to avoid modifying original data
        train_df_norm = train_df.copy()
        val_df_norm = val_df.copy()
        test_df_norm = test_df.copy()
        
        # Fit on train, transform all
        train_df_norm[readout_col] = scaler.fit_transform(train_df[[readout_col]])
        val_df_norm[readout_col] = scaler.transform(val_df[[readout_col]])
        test_df_norm[readout_col] = scaler.transform(test_df[[readout_col]])
        
        # Store scaler for later inverse transformation if needed
        self.current_scaler = scaler
        
        # Create Prophet instance with the pre-created config
        self.prophet_model = Prophet(
            iv_emb_path=self.iv_emb_paths,
            cl_emb_path=self.cl_emb_paths,
            ph_emb_path=self.ph_emb_paths,
            model_pth=checkpoint_path,
            architecture="Transformer",
            config=prophet_config
        )
        
        # Train the model with normalized data
        self.prophet_model.train(
            df=train_df_norm,
            val_df=val_df_norm,
            test_df=test_df_norm,
            iv_col=data_config['iv_cols'],
            cl_col=data_config['cl_col'],  
            ph_col=data_config['ph_col'],
            readout_col=readout_col,
            model_config=prophet_config,
            wandb_config=self.config['wandb'],
            checkpoint_dirpath=checkpoint_dirpath
        )
        
        return self.prophet_model
