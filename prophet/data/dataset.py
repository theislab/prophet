import numpy as np
import pandas as pd
from torch.utils.data import Dataset, Sampler
import itertools


class StratifiedPhenotypeSampler(Sampler):
    """
    Custom sampler that creates balanced batches across phenotypes without using multinomial sampling.
    This avoids PyTorch's 16M limit while ensuring balanced representation of all phenotypes.
    """

    def __init__(
        self, data: pd.DataFrame, indices: list, batch_size: int, key: str = "phenotype"
    ):
        """
        Args:
            data: DataFrame containing the experimental data
            indices: List of indices to sample from (e.g., training indices)
            batch_size: Size of each batch
            key: Column name to stratify on ('phenotype' or 'dataset')
        """
        self.batch_size = batch_size
        self.key = key

        # Group indices by phenotype/dataset
        self.phenotype_groups = {}
        for idx in indices:
            phenotype = data.loc[idx, key]
            if phenotype not in self.phenotype_groups:
                self.phenotype_groups[phenotype] = []
            self.phenotype_groups[phenotype].append(idx)

        self.phenotypes = list(self.phenotype_groups.keys())
        self.n_phenotypes = len(self.phenotypes)
        self.total_samples = len(indices)

        # Calculate how many samples per phenotype per batch
        self.samples_per_phenotype = max(1, batch_size // self.n_phenotypes)

        print(f"StratifiedPhenotypeSampler initialized:")
        print(f"  • Total samples: {self.total_samples:,}")
        print(f"  • Phenotypes: {self.n_phenotypes:,}")
        print(f"  • Samples per phenotype per batch: {self.samples_per_phenotype}")

    def __iter__(self):
        # Create cyclic iterators for each phenotype
        phenotype_iterators = {}
        for phenotype in self.phenotypes:
            indices = self.phenotype_groups[phenotype].copy()
            np.random.shuffle(indices)
            # Create infinite iterator that cycles through indices
            phenotype_iterators[phenotype] = itertools.cycle(indices)

        # Generate batches
        num_batches = self.total_samples // self.batch_size
        for _ in range(num_batches):
            batch = []

            # Sample from each phenotype
            for phenotype in self.phenotypes:
                for _ in range(self.samples_per_phenotype):
                    if len(batch) < self.batch_size:
                        batch.append(next(phenotype_iterators[phenotype]))

            # Fill remaining slots randomly from all phenotypes
            while len(batch) < self.batch_size:
                phenotype = np.random.choice(self.phenotypes)
                batch.append(next(phenotype_iterators[phenotype]))

            yield batch

    def __len__(self):
        return self.total_samples // self.batch_size


class PhenotypeDataset(Dataset):
    """
    Memory-efficient dataset that gathers multiple phenotypes with optimized data access patterns.
    Avoids pre-computing large arrays to prevent OOM on datasets with 50M+ samples.
    """

    def __init__(
        self,
        experimental_data: pd.DataFrame,
        label_key: str,
        iv_embeddings: pd.DataFrame,
        cell_line_embeddings: pd.DataFrame,
        phenotype_embeddings: pd.DataFrame = None,
        phenotypes: list = None,
        cl_embedding: bool = False,
        pert_len: int = 2,
    ):
        """
        Args:
            experimental_data (pd.DataFrame): experimental data, contains the label and the training data (combinations of gRNA)
            label_key (str): key that identifies the label in the experimental data
            iv_embeddings (pd.DataFrame): pandas dataframe with the embeddings of the perturbations
            cell_line_embeddings (pd.DataFrame): pandas dataframe with the embedding of cell lines
            cl_embedding (bool): if True, use predfined embedding; if False, retrieve just index cause it will be learn
            phenotypes (list): if phenotype embeddings are not provided, then a sorted list of phenotypes must be provided.
            pert_len (int): number of perturbations to provide to the model, context length will be pert_len + 2, which comes from phenotype + cell_type
        """
        self.pert_len = pert_len

        # Store labels as contiguous float32 array for better performance
        self.labels = np.ascontiguousarray(
            experimental_data[label_key].values, dtype=np.float32
        )

        # Pre-compute attention mask efficiently (small memory footprint)
        self.attn_mask = self._precompute_attention_masks(experimental_data, pert_len)

        # Store embeddings in contiguous memory with optimal dtypes
        self.iv = np.ascontiguousarray(
            iv_embeddings.iloc[:, 1:].values, dtype=np.float32
        )
        self.type_to_int = {"gene": 0, "drug": 1}
        iv_embs_types_str = iv_embeddings.iloc[:, 0].values
        self.iv_embs_types = np.array(
            [self.type_to_int.get(s, -1) for s in iv_embs_types_str], dtype=np.int8
        )
        self.cell_line = np.ascontiguousarray(
            cell_line_embeddings.values, dtype=np.float32
        )

        # Create optimized mapping dictionaries using numpy arrays for faster lookup
        iv_names = iv_embeddings.index.values
        cl_names = cell_line_embeddings.index.values

        self.iv_to_index = dict(zip(iv_names, np.arange(len(iv_names), dtype=np.int32)))
        self.cl_to_index = dict(zip(cl_names, np.arange(len(cl_names), dtype=np.int32)))
        self.ph_to_index = dict(
            zip(phenotypes, np.arange(len(phenotypes), dtype=np.int32))
        )

        # Handle phenotype embeddings
        if phenotype_embeddings is not None:
            phenotype_embeddings = phenotype_embeddings.T[phenotypes].T  # reorder
            self.phenotype_embeddings = np.ascontiguousarray(
                phenotype_embeddings.values, dtype=np.float32
            )
        else:
            self.phenotype_embeddings = None

        # Pre-convert experimental data strings to integer indices for faster lookups
        columns = ["cell_line", "phenotype"] + [
            f"iv{i}" for i in range(1, pert_len + 1)
        ]

        # Use a temporary DataFrame for mapping
        exp_data_indices = pd.DataFrame(index=experimental_data.index)
        exp_data_indices["cell_line"] = experimental_data["cell_line"].map(
            self.cl_to_index
        )
        exp_data_indices["phenotype"] = experimental_data["phenotype"].map(
            self.ph_to_index
        )
        for i in range(1, pert_len + 1):
            exp_data_indices[f"iv{i}"] = experimental_data[f"iv{i}"].map(
                self.iv_to_index
            )

        self.experimental_data_indices = np.ascontiguousarray(
            exp_data_indices[columns].values, dtype=np.int32
        )

        # Pre-compute type conversion mapping for faster access
        self.type_to_int = {"gene": 0, "drug": 1}

    def _precompute_attention_masks(
        self, experimental_data: pd.DataFrame, pert_len: int
    ) -> np.ndarray:
        """Pre-compute all attention masks - only ~220MB for 55M samples."""
        masks = []

        # CLS token mask (always False - pay attention)
        masks.append(np.zeros(experimental_data.shape[0], dtype=bool))

        # Intervention masks
        for i in range(1, pert_len + 1):
            col = f"iv{i}"
            mask_values = (
                experimental_data[col].isin(["negative_drug", "negative_gene"]).values
            )
            masks.append(mask_values)

        # Cell line and phenotype masks (always False - pay attention)
        masks.append(np.zeros(experimental_data.shape[0], dtype=bool))
        masks.append(np.zeros(experimental_data.shape[0], dtype=bool))

        return np.array(masks, dtype=bool).T

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Memory-efficient __getitem__ method that computes embeddings on-demand.
        Uses pre-computed integer indices for fast lookups.
        """
        # Get pre-computed indices for this sample
        item_indices = self.experimental_data_indices[idx]
        cell_line_idx = item_indices[0]
        phenotype_idx = item_indices[1]

        # Fast numpy array indexing
        cell_line_emb = self.cell_line[cell_line_idx]

        # Handle phenotype context
        if self.phenotype_embeddings is None:
            context = phenotype_idx + 1  # CLS is 0
        else:
            context = self.phenotype_embeddings[phenotype_idx]

        iv_indices = item_indices[2:]
        perturbations = self.iv[iv_indices]
        iv_types = self.iv_embs_types[iv_indices]

        return {
            "phenotype": context,
            "cell_line": cell_line_emb,
            "label": self.labels[idx],
            "attn_mask": self.attn_mask[idx],
            "idx": idx,
            "pert_type": iv_types,
            "perturbations": perturbations,
        }
