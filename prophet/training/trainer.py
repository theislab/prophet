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
from sklearn.model_selection import LeaveOneGroupOut, KFold
from tqdm import tqdm
import pickle
import json

from ..core.prophet import Prophet
from ..core.config import set_config
from ..models import load_models_config
from ..data import dataset_registry
from ..data.processing import DataSplitter
from ..utils import validate_prophet_inputs

SEED = 42  # the true, baseline seed (that sets test splits)

def _choose(a, size, seed):
    """Guaranteed deterministic choosing."""
    np.random.seed(seed)  # reset the generator
    return np.random.choice(a, size=size, replace=False)

def _permute(a, seed):
    """Guaranteed deterministic permutation."""
    np.random.seed(seed)  # reset the generator
    return np.random.permutation(a)

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
                df = df.sample(frac=0.33, replace=False,random_state=42).reset_index(drop=True)

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
        data_label = df.copy()
        leaveout_method = splitting_config["method"]
        seed = self.seed
        print('using seed: ', seed)
        print("Creating splits using method: ", leaveout_method)

        # some initial processing to retain exact reproducibility
        data_label['value'] = data_label['value'].astype('f4')
        data_label = data_label.reset_index(drop=True)

        data_label_flipped = data_label.rename(
            columns={'iv1': 'iv2', 'iv2': 'iv1'})
        data_label = pd.concat([data_label, data_label_flipped], axis=0, ignore_index=True)
        data_label = data_label.reset_index(drop=True)

        np.random.seed(SEED)

        original_values = data_label['value'].values
        indices = []
        valid_indices = None

        ivs = list(data_label.iv1.unique())
        
        try:
            ivs.remove('negative_gene')
        except ValueError: pass
        try:
            ivs.remove('negative_drug')
        except ValueError: pass
        cell_lines = list(data_label[data_config["cl_col"]].unique())

        def random_validation(train_indices, frac=.2, seed=42):
            # Calculate the number of validation indices to select
            num_validation_indices = round(len(train_indices) * frac)

            # Select the validation indices using _choose()
            valid_indices = _choose(train_indices, num_validation_indices, seed=seed)
            valid_indices = valid_indices.tolist()
            train_indices = np.setdiff1d(train_indices, valid_indices)
            return train_indices, valid_indices

        def unseen_cl_validation(indices, seed, n=3, data=data_label):
            cls = data.loc[indices][data_config["cl_col"]].unique().tolist()
            if len(cls) <= n:
                print('Validation is random split because there are not enough cell lines left!')
                return random_validation(indices, frac=.2, seed=seed)
            else:
                cl_exclude = _choose(cls, n, seed=seed)
                valid_indices = data.index[data['cell_line'].isin(cl_exclude)].to_list()
                indices = np.setdiff1d(indices, valid_indices)
                return indices, valid_indices
        
        def unseen_ivs_validation(indices, seed, n=3):
            """Randomly completely leave out a fraction of ivs."""
            ivs = list(sorted(set(data_label.loc[indices].iv1.values) & set(data_label.loc[indices].iv2.values) - set(['negative_gene', 'negative_drug'])))
            ivs_exclude = _choose(ivs, n, seed=seed)
            valid_indices = data_label.index[data_label['iv1'].isin(ivs_exclude) | data_label['iv2'].isin(ivs_exclude)].tolist()
            valid_indices = sorted(set(valid_indices) & set(indices))  # validation must be selected from train
            indices = np.setdiff1d(indices, valid_indices)
            return indices, valid_indices

        def arrange_bigger_smaller(data_label):
            """Helper function for wref splits."""
            if 'dataset' not in data_label.columns:
                data_label['dataset'] = data_label['phenotype']
            bigger = data_label.dataset.value_counts().index[0]
            smaller = data_label.dataset.value_counts().index[1]
            data_label['id'] = list(range(data_label.shape[0]))
            sm_data = data_label[data_label.dataset == smaller].copy()
            bg_data = data_label[data_label.dataset == bigger].copy()
            return bg_data, sm_data, data_label

        def realign_indices(full_data, subset, indices):
            """Given the subset on which the split was done and the original data, returns the corrected indices.
            Both dataframes must have an `id` column which matches. """
            new_indices = []
            for index in indices:
                train, val, test, label = index
                real_train = subset.loc[train].id.values
                real_val = subset.loc[val].id.values
                real_test = subset.loc[test].id.values
                real_train_indices = full_data[full_data.id.isin(real_train)].index.to_list()
                real_val_indices = full_data[full_data.id.isin(real_val)].index.to_list()
                real_test_indices = full_data[full_data.id.isin(real_test)].index.to_list()
                new_indices.append((real_train_indices, real_val_indices, real_test_indices, label))
            return new_indices

        def compute_overlap(a, b):
            """Returns everything overlapping, but overlap is computed without case."""
            # Create lowercase to original case mappings for both lists
            lower_to_orig_a = {item.lower(): item for item in a}
            lower_to_orig_b = {item.lower(): item for item in b}

            # Compute the intersection of lowercase sets to find overlaps without case sensitivity
            overlap = set(lower_to_orig_a.keys()) & set(lower_to_orig_b.keys())

            # Collect overlapping elements, preserving original case from both lists
            result = [lower_to_orig_a[item] for item in overlap] + [lower_to_orig_b[item] for item in overlap]
            return sorted(set(result))

        def combine_for_multitest(dataset_indices, split_label):
            """Takes in a dictionary of dataset:indices and combines them."""
            idxs = []
            for i in range(5):
                vl_idxs = set()
                tsidx_all = set()
                ts_idxs = {}
                for ds, _idxs in dataset_indices.items():
                    vl_idxs.update(_idxs[i][1])
                    tsidx_all.update(_idxs[i][2])
                    ts_idxs[ds] = _idxs[i][2]
                ts_idxs['all'] = sorted(tsidx_all)
                tr_idxs = np.setdiff1d(data_label.index, list(tsidx_all | vl_idxs))
                idxs.append((tr_idxs, sorted(vl_idxs), ts_idxs, f'{split_label}_{i}'))  # sort to prevent python reordering
            return idxs

        def combine_for_multitest_using_label(dataset_indices):
            """Takes in a dictionary of dataset:indices and combines them but only where the labels match.
            Returns a list of indices, retaining labels. If the label is 'include_all_of_this', it includes
            everything in train."""
            label_idx_dict = {}
            for ds, _idxs in dataset_indices.items():
                for i in range(len(_idxs)):
                    train, val, test, label = _idxs[i]
                    if label not in label_idx_dict.keys():
                        label_idx_dict[label] = [(train, val, test)]
                    else:
                        label_idx_dict[label].append((train, val, test))

            datasets_always_in_train = set()
            for label, split_indices in label_idx_dict.items():
                if label != 'include_all_of_this':
                    continue
                print(len(split_indices), 'datasets always in train')
                for split in split_indices:
                    datasets_always_in_train.update(split[0])

            idxs = []
            for label, split_indices in label_idx_dict.items():
                if label == 'include_all_of_this':
                    continue
                tr_idxs = set()
                vl_idxs = set()
                ts_idxs = set()
                for split in split_indices:
                    train, val, test = split
                    tr_idxs.update(train)
                    vl_idxs.update(val)
                    ts_idxs.update(test)
                tr_idxs.update(datasets_always_in_train)
                idxs.append((list(tr_idxs), list(vl_idxs), list(ts_idxs), label))
            return idxs

        def aggregate_multitest(data_label, seed, leaveout_method):
            if 'dataset' not in data_label.columns:
                raise ValueError("The setting must be created such that different datasets are delimited.")

            data_label['id'] = list(range(data_label.shape[0]))
            dataset_indices = {}
            for ds in data_label.dataset.unique():
                print('splitting', ds)
                sub_data = data_label[data_label.dataset == ds]
                try:
                    leave_x_out_indices = self.create_splits(sub_data, splitting_config)
                    leave_x_out_indices = realign_indices(data_label, sub_data, leave_x_out_indices)
                    dataset_indices[ds] = leave_x_out_indices
                except (ValueError, NotImplementedError):
                    print(f'{leaveout_method} does not work for {ds}, including all data')
                    dataset_indices[ds] = [(sub_data.index.to_list(), [], [], 'include_all_of_this')]
                    continue

            return dataset_indices

        if leaveout_method == "leave_one_cl_out":
            
            cell_line_groups = data_label["cell_line"].to_numpy()
            logo = LeaveOneGroupOut()
            for _, (train_indices, test_indices) in enumerate(
                logo.split(X=cell_line_groups, groups=cell_line_groups)
            ):
                cl_holdout = data_label["cell_line"][test_indices].values[0]
                train_indices, valid_indices = unseen_cl_validation(train_indices, seed=seed)
                indices.append((train_indices, valid_indices, test_indices, cl_holdout))

                if _ > 15:  # reduce for runtime in datasets with more than 15 cell lines
                    break

        elif leaveout_method == "leave_cl_out":

            n_splits = 5

            kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
            for i, (_, test_idx) in enumerate(kf.split(cell_lines)):
                cls_exclude = np.array(cell_lines)[test_idx]
                test_indices = data_label.index[data_label['cell_line'].isin(cls_exclude)].tolist()
                train_indices = list(np.setdiff1d(data_label.index, test_indices))
                train_indices, valid_indices = unseen_cl_validation(train_indices, n=int(len(cell_lines)*.1), seed=seed)
                indices.append((train_indices, valid_indices, test_indices, f'cl_{i}_TrainedOn{len(_)}'))

        elif leaveout_method == "leave_tissue_out":
            """Leaves out 20% of the tissue types in CCLE at a time, based on frequency of occurrence,
            unless there are 3 or fewer cell lines. Validation split is still unseen cell line so that
            we don't change the training procedure."""
            ccle = pd.read_csv('../info_files/sample_info.csv', index_col=0)
            ccle.stripped_cell_line_name = ccle.stripped_cell_line_name.str.upper()
            tissue_dict = ccle.groupby('lineage')['stripped_cell_line_name'].apply(list).to_dict()

            n_splits = 5
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
            all_tissues = list(tissue_dict.keys())
            for i, (_, test_idx) in enumerate(kf.split(all_tissues)):
                tissues_exclude = np.array(all_tissues)[test_idx]
                cls_exclude = [cell_line for tissue in tissues_exclude for cell_line in tissue_dict[tissue]]
                if len(set(cls_exclude) & set(cell_lines)) <= 3:
                    continue
                test_indices = data_label.index[data_label['cell_line'].isin(cls_exclude)].tolist()
                train_indices = list(np.setdiff1d(data_label.index, test_indices))
                train_indices, valid_indices = unseen_cl_validation(train_indices, n=int(len(cell_lines)*.1), seed=seed)
                indices.append((train_indices, valid_indices, test_indices, f'tissue_split{i}'))

        elif leaveout_method == 'leave_cl_cluster_out':
            """Leaves out 20% of the cell line clusters in CCLE at a time. Leiden was run with scanpy default parameters
            and are generally more coarse than tissue labels."""
            ccle_obs = pd.read_csv('../info_files/sample_info_leiden.csv', index_col=0)
            leiden_dict = ccle_obs.groupby('leiden')['stripped_cell_line_name'].apply(list).to_dict()
            n_splits = 5
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
            all_clusters = list(leiden_dict.keys())
            for i, (_, test_idx) in enumerate(kf.split(all_clusters)):
                clusters_exclude = np.array(all_clusters)[test_idx]
                cls_exclude = [cell_line for cluster in clusters_exclude for cell_line in leiden_dict[cluster]]
                if len(set(cls_exclude) & set(cell_lines)) <= 3:
                    continue
                test_indices = data_label.index[data_label['cell_line'].isin(cls_exclude)].tolist()
                train_indices = list(np.setdiff1d(data_label.index, test_indices))
                train_indices, valid_indices = unseen_cl_validation(train_indices, n=int(len(cell_lines)*.1), seed=seed)
                clusters_exclude = [str(c) for c in clusters_exclude]
                indices.append((train_indices, valid_indices, test_indices, f'cluster_split{i}'))

        elif leaveout_method == "leave_iv_out":

            n_splits = 5

            kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
            for i, (_, test_idx) in enumerate(kf.split(ivs)):
                ivs_exclude = np.array(ivs)[test_idx]
                test_indices = data_label.index[data_label['iv1'].isin(ivs_exclude) | data_label['iv2'].isin(ivs_exclude)].tolist()
                train_indices = list(np.setdiff1d(data_label.index, test_indices))
                train_indices, valid_indices = unseen_ivs_validation(train_indices, n=int(len(ivs)*.1), seed=seed)
                indices.append((train_indices, valid_indices, test_indices, f'iv_{i}_TrainedOn{len(_)}'))

        elif leaveout_method == "leave_ivs_fish":
            """A split specifically for the zebrafish dataset, where we only have 22 perturbations."""
            fish_ivs = np.setdiff1d(list(set(data_label['iv1'].unique()) | set(data_label['iv2'].unique())), ['negative_gene', 'negative_drug'])
            print(f'number of ivs {fish_ivs.shape}')

            n_splits = len(fish_ivs)

            kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
            for i, (_, test_idx) in enumerate(kf.split(fish_ivs)):
                ivs_exclude = fish_ivs[test_idx]
                test_indices = data_label.index[data_label['iv1'].isin(ivs_exclude) | data_label['iv2'].isin(ivs_exclude)].tolist()
                train_indices = list(np.setdiff1d(data_label.index, test_indices))
                train_indices, valid_indices = unseen_ivs_validation(train_indices, n=int(len(fish_ivs)*.1), seed=seed)
                indices.append((train_indices, valid_indices, test_indices, f'iv_{ivs_exclude[0]}_TrainedOn{len(_)}'))
        
        elif leaveout_method == "leave_scaffold_out":
            """Leaves out (20% of) chemical scaffolds as in https://tdcommons.ai/functions/data_split#scaffold-split"""
            smiles_obs = pd.read_csv('../info_files/iv_info.csv', index_col=0)
            smiles_obs.smiles = smiles_obs.smiles.str.lower()
            scaffold_vc = smiles_obs.scaffold.value_counts()
            scaffold_dict = smiles_obs.groupby('scaffold')['smiles'].apply(list).to_dict()

            n_splits = 5
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
            all_scaffolds = list(scaffold_dict.keys())
            for i, (_, test_idx) in enumerate(kf.split(all_scaffolds)):
                scaffolds_exclude = np.array(all_scaffolds)[test_idx]
                ivs_exclude = [iv for scaffold in scaffolds_exclude for iv in scaffold_dict[scaffold]]
                if len(set(ivs_exclude) & set(ivs)) <= 3:
                    continue    
                test_indices = data_label.index[data_label['iv1'].isin(ivs_exclude) | data_label['iv2'].isin(ivs_exclude)].tolist()
                train_indices = list(np.setdiff1d(data_label.index, test_indices))
                train_indices, valid_indices = unseen_ivs_validation(train_indices, n=int(len(ivs)*.1), seed=seed)
                indices.append((train_indices, valid_indices, test_indices, f'scaffold_split{i}'))
        
        elif leaveout_method == "leave_drug_cluster_out":
            """Leaves out (20% of) drug clusters similar to leave_cl_cluster_out."""
            smiles_obs = pd.read_csv('../info_files/iv_info.csv', index_col=0)
            smiles_obs.smiles = smiles_obs.smiles.str.lower()
            leiden_vc = smiles_obs.leiden.value_counts()
            leiden_dict = smiles_obs.groupby('leiden')['smiles'].apply(list).to_dict()
        
            n_splits = 5
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
            all_clusters = list(leiden_dict.keys())
            for i, (_, test_idx) in enumerate(kf.split(all_clusters)):
                clusters_exclude = np.array(all_clusters)[test_idx]
                ivs_exclude = [iv for cluster in clusters_exclude for iv in leiden_dict[cluster]]
                if len(set(ivs_exclude) & set(ivs)) <= 3:
                    continue    
                test_indices = data_label.index[data_label['iv1'].isin(ivs_exclude) | data_label['iv2'].isin(ivs_exclude)].tolist()
                train_indices = list(np.setdiff1d(data_label.index, test_indices))
                train_indices, valid_indices = unseen_ivs_validation(train_indices, n=int(len(ivs)*.1), seed=seed)
                clusters_exclude = [str(c) for c in clusters_exclude]  # convert to str for join
                indices.append((train_indices, valid_indices, test_indices, f'cluster_split{i}'))

        elif leaveout_method == "leave_iv_cluster_out":
            """Leaves out (20% of) iv clusters similar to leave_cl_cluster_out."""
            iv_obs = pd.read_csv('../info_files/iv_info.csv', index_col=0)
            iv_obs.iv1 = iv_obs.iv1.str.lower()
            leiden_vc = iv_obs.leiden.value_counts()
            leiden_dict = iv_obs.groupby('leiden')['iv1'].apply(list).to_dict()
        
            n_splits = 5
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
            all_clusters = list(leiden_dict.keys())
            for i, (_, test_idx) in enumerate(kf.split(all_clusters)):
                clusters_exclude = np.array(all_clusters)[test_idx]
                ivs_exclude = [iv for cluster in clusters_exclude for iv in leiden_dict[cluster]]
                if len(set(ivs_exclude) & set(ivs)) <= 3:
                    continue    
                test_indices = data_label.index[data_label['iv1'].isin(ivs_exclude) | data_label['iv2'].isin(ivs_exclude)].tolist()
                train_indices = list(np.setdiff1d(data_label.index, test_indices))
                train_indices, valid_indices = unseen_ivs_validation(train_indices, n=int(len(ivs)*.1), seed=seed)
                clusters_exclude = [str(c) for c in clusters_exclude]  # convert to str for join
                indices.append((train_indices, valid_indices, test_indices, f'cluster_split{i}'))

        elif leaveout_method == 'leave_cl_out_wref':
            """Replicates the same val+test split as leave_cl_out but where an additional dataset is included in training.
            Assumes only two phenotypes which encode the datasets. Assumes the larger dataset is always included in training."""
            bg_data, sm_data, data_label = arrange_bigger_smaller(data_label)
            train_indices = bg_data.index.to_list()

            new_splitting_config = splitting_config.copy()
            new_splitting_config['method'] = 'leave_cl_out'
            leave_cl_out_indices = self.create_splits(sm_data, new_splitting_config, data_config)
            leave_cl_out_indices = realign_indices(data_label, sm_data, leave_cl_out_indices)
            for index in leave_cl_out_indices:
                indices.append((list(index[0]) + train_indices, index[1], index[2], f'wref_{index[3]}'))

        elif leaveout_method == 'leave_iv_out_wref':
            """Replicates the same val+test split as leave_iv_out but where an additional dataset is included in training.
            Assumes only two phenotypes which encode the datasets. Assumes the larger dataset is always included in training."""
            bg_data, sm_data, data_label = arrange_bigger_smaller(data_label)
            train_indices = bg_data.index.to_list()

            new_splitting_config = splitting_config.copy()
            new_splitting_config['method'] = 'leave_iv_out'
            leave_iv_out_indices = self.create_splits(sm_data, new_splitting_config, data_config)
            leave_iv_out_indices = realign_indices(data_label, sm_data, leave_iv_out_indices)
            for index in leave_iv_out_indices:
                indices.append((list(index[0]) + train_indices, index[1], index[2], f'wref_{index[3]}'))

        elif leaveout_method == 'leave_more_cl_out':
            """Leave more and more cell lines out while keeping the test set the same. Repeated with three different test sets,
            thereby replicating the first 3 splits of leave_iv_out. Validation is 10% selected from the fraction that is
            neither in train nor test."""

            fracs = [0.5, 0.3, 0.2, 0.1, 0.05]
            if len(cell_lines) < 10:
                raise ValueError("Fewer than 10 cell lines, it is not recommended to run leave_more_cl_out")
            elif len(cell_lines) < 20:
                print("fewer than 20 cell lines, removing .05 frac split")
                fracs = fracs[:-1]
            n_tests = 5

            kf = KFold(n_splits=n_tests, shuffle=True, random_state=SEED)
            for i, (_, test_idx) in enumerate(kf.split(cell_lines)):
                exclude = np.array(cell_lines)[test_idx]
                test_indices = data_label.index[data_label[data_config["cl_col"]].isin(exclude)].tolist()
                for frac in fracs:
                    include = _choose(np.setdiff1d(cell_lines, exclude), int(len(cell_lines)*frac), seed=SEED)
                    train_indices = data_label.index[data_label[data_config["cl_col"]].isin(include)].tolist()
                    remaining_indices = np.setdiff1d(np.setdiff1d(data_label.index.tolist(), train_indices), test_indices)
                    _, valid_indices = unseen_cl_validation(remaining_indices, n=int(len(cell_lines)*.1), seed=seed)
                    indices.append((train_indices, valid_indices, test_indices, f'split_{i}_frac{frac}'))
                if i >= 2:
                    break

        elif leaveout_method == 'leave_more_ivs_out':

            fracs = [0.5, 0.3, 0.2, 0.1, 0.05]
            n_tests = 5

            kf = KFold(n_splits=n_tests, shuffle=True, random_state=SEED)
            for i, (_, test_idx) in enumerate(kf.split(ivs)):
                ivs_exclude = np.array(ivs)[test_idx]
                test_indices = data_label.index[data_label['iv1'].isin(ivs_exclude) | data_label['iv2'].isin(ivs_exclude)].tolist()
                for frac in fracs:
                    ivs_include = _choose(np.setdiff1d(ivs, ivs_exclude), int(len(ivs)*frac), seed=SEED)
                    train_indices = data_label.index[data_label['iv1'].isin(ivs_include) & data_label['iv2'].isin(ivs_include)].tolist()
                    if len(train_indices) <= 100:  # in sparsely complete combinations, this can end up exceedingly small
                        continue
                    remaining_indices = np.setdiff1d(np.setdiff1d(data_label.index.tolist(), train_indices), test_indices)
                    _, valid_indices = unseen_ivs_validation(remaining_indices, n=int(len(ivs)*.1), seed=seed)
                    indices.append((train_indices, valid_indices, test_indices, f'split_{i}_frac{frac}'))
                    
        elif leaveout_method == 'leave_common_cl_out':
            """Leaves out incrementally greater percentages of the cell lines shared between the two datasets. Always takes from the smaller dataset.
            It uses as validation set a 10% of the remaining common cell lines. Test set is 1/3 of the common cell lines.
            Test set is the same regardless of fraction, and repeated 3 times. Combined with training without the reference, there are 30 splits within.
            """
            bg_data, sm_data, data_label = arrange_bigger_smaller(data_label)

            commons = compute_overlap(sm_data['cell_line'].unique(), bg_data['cell_line'].unique())
            print(f'{len(commons)} in common (including dups)')
            sm_train_fixed = sm_data[~sm_data['cell_line'].isin(commons)].index.to_list()
            print(f'small and large always included in train: {len(sm_train_fixed)}, {bg_data.shape[0]}')

            sm_data['id'] = list(range(sm_data.shape[0]))
            subset = sm_data[sm_data['cell_line'].isin(commons)]
            print(f'size of overlapping portion from which to select val/test: {subset.shape[0]}')
            new_splitting_config = splitting_config.copy()
            new_splitting_config['method'] = 'leave_more_cl_out'
            leave_cl_out_indices = self.create_splits(subset, new_splitting_config, data_config)
            leave_cl_out_indices = realign_indices(sm_data, subset, leave_cl_out_indices)
            for index in leave_cl_out_indices:
                train, val, test, label = index
                for expanded_dataset in [True, False]:
                    train_indices = sm_train_fixed + train
                    if expanded_dataset:
                        train_indices = train_indices + bg_data.index.tolist()
                    indices.append((train_indices, val, test, f'{label}_withref_{expanded_dataset}'))

        elif leaveout_method == 'leave_common_ivs_out':
            """Same procedure as in leave_common_cl_out."""
            bg_data, sm_data, data_label = arrange_bigger_smaller(data_label)

            commons = compute_overlap(sm_data['iv1'].unique(), bg_data['iv1'].unique())
            print(f'{len(commons)} in common (including dups)')
            sm_train_fixed = sm_data[~sm_data['iv1'].isin(commons)]
            sm_train_fixed = sm_train_fixed[~sm_train_fixed['iv2'].isin(commons)].index.to_list()
            print(f'small and large always included in train: {len(sm_train_fixed)}, {bg_data.shape[0]}')

            sm_data['id'] = list(range(sm_data.shape[0]))
            subset = sm_data[sm_data['iv1'].isin(commons) | sm_data['iv2'].isin(commons)]
            print(f'size of overlapping portion from which to select val/test: {subset.shape[0]}')
            new_splitting_config = splitting_config.copy()
            new_splitting_config['method'] = 'leave_more_ivs_out'
            leave_iv_out_indices = self.create_splits(subset, new_splitting_config, data_config)
            leave_iv_out_indices = realign_indices(sm_data, subset, leave_iv_out_indices)
            for index in leave_iv_out_indices:
                train, val, test, label = index
                for expanded_dataset in [True, False]:
                    train_indices = sm_train_fixed + train
                    if expanded_dataset:
                        train_indices += bg_data.index.tolist()
                    indices.append((train_indices, val, test, f'{label}_withref_{expanded_dataset}'))

        elif leaveout_method == 'leave_both_out':
            """Combines leave_iv_out and leave_cl_out, but the test set only contains
            the intersection of the two."""
            new_splitting_config = splitting_config.copy()
            new_splitting_config['method'] = 'leave_iv_out'
            leave_iv_out_indices = self.create_splits(data_label, new_splitting_config, data_config)
            new_splitting_config = splitting_config.copy()
            new_splitting_config['method'] = 'leave_cl_out'
            leave_cl_out_indices = self.create_splits(data_label, new_splitting_config, data_config)
            for i in range(5):
                train_iv, val_iv, test_iv, label_iv = leave_iv_out_indices[i]
                label_iv = label_iv.split('TrainedOn')[-1]
                train_cl, val_cl, test_cl, label_cl = leave_cl_out_indices[i]
                label_cl = label_cl.split('TrainedOn')[-1]
                test_indices = list(set(test_iv) & set(test_cl))
                val_indices = list(set(val_iv) & set(val_cl))
                train_indices = list(set(train_iv) & set(train_cl))
                indices.append((train_indices, val_indices, test_indices, f'bothsplit_{i}'))

        elif leaveout_method == 'leave_cl_out_multitest':
            if 'dataset' not in data_label.columns:
                raise ValueError("The setting must be created such that different datasets are delimited.")

            data_label['id'] = list(range(data_label.shape[0]))
            dataset_indices = {}
            for ds in data_label.dataset.unique():
                sub_data = data_label[data_label.dataset == ds]

                try:
                    new_splitting_config = splitting_config.copy()
                    new_splitting_config['method'] = 'leave_cl_out'
                    leave_cl_out_indices = self.create_splits(sub_data, new_splitting_config, data_config)
                    leave_cl_out_indices = realign_indices(data_label, sub_data, leave_cl_out_indices)
                    dataset_indices[ds] = leave_cl_out_indices
                except:
                    pass

            indices = combine_for_multitest(dataset_indices, 'cl')

        elif leaveout_method == 'leave_iv_out_multitest':
            if 'dataset' not in data_label.columns:
                raise ValueError("The setting must be created such that different datasets are delimited.")

            data_label['id'] = list(range(data_label.shape[0]))
            dataset_indices = {}
            for ds in data_label.dataset.unique():
                sub_data = data_label[data_label.dataset == ds]

                new_splitting_config = splitting_config.copy()
                new_splitting_config['method'] = 'leave_iv_out'
                leave_iv_out_indices = self.create_splits(sub_data, new_splitting_config, data_config)
                leave_iv_out_indices = realign_indices(data_label, sub_data, leave_iv_out_indices)
                dataset_indices[ds] = leave_iv_out_indices

            indices = combine_for_multitest(dataset_indices, 'iv')

        elif leaveout_method in [
            'leave_cl_cluster_out_multitest',
            'leave_tissue_out_multitest',
            'leave_drug_cluster_out_multitest',
            'leave_iv_cluster_out_multitest',
            'leave_scaffold_out_multitest',
            'leave_both_out_multitest']:
            dataset_indices = aggregate_multitest(data_label, seed, leaveout_method.split('_multitest')[0])
            indices = combine_for_multitest_using_label(dataset_indices)

        else:
            raise ValueError("Must pass in a valid leave out method.")

        if len(indices) == 0:
            raise ValueError(f"Stratification {leaveout_method} is invalid, likely not enough categories")
        # verify that there's no overlap between train, validation, and test
        for split in indices:
            test = split[2]
            if type(split[2]) == dict:
                test = split[2]['all']
            if len(set(split[0]) & set(split[1])) > 0 or len(set(split[0]) & set(test)) > 0 or len(set(split[1]) & set(test)) > 0:
                raise NotImplementedError(f"leavout method {leaveout_method} has overlapping indices!!")
            if len(split[0]) < 1 or len(split[1]) < 1 or len(test) < 1:
                raise NotImplementedError(f"leavout method {leaveout_method} has no examples in one of train:{len(split[0])}, val:{len(split[1])}, test:{len(test)} with description {split[3]}")

        assert(sum(original_values != data_label['value'].values) == 0)

        final_splits = [[data_label.loc[split[0]], data_label.loc[split[1]], data_label.loc[split[2]], split[3]] for split in indices]

        return final_splits

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
            wandb_config={**self.config["wandb"], 'name':'split_name'},
            checkpoint_dirpath=checkpoint_dirpath,
        )

        return self.prophet_model
