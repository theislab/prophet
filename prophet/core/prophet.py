import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from ..utils.callbacks import HitRatioCallback
import numpy as np
import pandas as pd
from typing import List, Union, Optional, Dict, Tuple
from ..utils import R2ScoreCallback
import functools
from joblib import load
from ..data import (
    dataloader_phenotypes,
    process_priors,
    remove_nonexistent_cat,
)
from ..models import load_models_config, TransformerPredictor
from ..utils import (
    download_model,
    print_available_models,
    get_available_datasets,
    get_available_splits,
    get_available_seeds,
    get_available_folds,
)
from pytorch_lightning.loggers import WandbLogger
import torch.optim as optim
import types
import wandb


def inherit_docs_and_signature(from_method):
    def decorator(to_method):
        @functools.wraps(from_method)
        def wrapper(self, *args, **kwargs):
            return to_method(self, *args, **kwargs)

        wrapper.__doc__ = from_method.__doc__
        wrapper.__signature__ = from_method.__signature__
        return wrapper

    return decorator


class Prophet:
    """Prophet: Transformer-based model for predicting cellular responses to perturbations.

    Prophet decomposes biological experiments into three key components:
    1. Cell state (represented by cell line embeddings)
    2. Treatment/intervention (represented by perturbation embeddings)
    3. Functional readout (the phenotypic measurement being predicted)

    The model can predict outcomes for drug treatments, genetic perturbations, and
    combinatorial interventions across different cell lines without requiring
    actual experiments to be performed.

    Examples:
        Basic usage for inference:
        >>> model = Prophet(
        ...     iv_emb_path="gene_embeddings.csv",
        ...     cl_emb_path="cell_line_embeddings.csv",
        ...     model_pth="trained_model.ckpt"
        ... )
        >>> predictions = model.predict(
        ...     target_ivs=["GENE1", "DRUG1"],
        ...     target_cls=["CELLLINE1", "CELLLINE2"]
        ... )

        Fine-tuning on custom data:
        >>> model.train(
        ...     df=training_data,
        ...     iv_col=["treatment1", "treatment2"],
        ...     cl_col="cell_line",
        ...     ph_col="phenotype",
        ...     readout_col="response"
        ... )
    """

    def __init__(
        self,
        iv_emb_path: Optional[Union[str, List[str]]] = None,
        cl_emb_path: Optional[Union[str, List[str]]] = None,
        ph_emb_path: Optional[Union[str, List[str]]] = None,
        model_pth: Optional[str] = None,
        architecture: str = "Transformer",
        config: Optional[object] = None,  # Add config parameter
    ) -> None:
        """Initialize the Prophet model.

        Args:
            iv_emb_path: Path(s) to intervention embeddings (genes, drugs, etc.).
                Can be a single CSV file path or list of paths that will be concatenated.
                Expected format: CSV with interventions as index and embedding dimensions as columns.
            cl_emb_path: Path(s) to cell line embeddings.
                Can be a single CSV file path or list of paths that will be concatenated.
                Expected format: CSV with cell lines as index and embedding dimensions as columns.
            ph_emb_path: Path(s) to phenotype embeddings. Only required if the model
                was trained with explicit phenotype embeddings. Defaults to None.
            model_pth: Path to pre-trained model checkpoint (.ckpt file). Required for
                inference and fine-tuning. Defaults to None.
            architecture: Model architecture to use. Currently supports "Transformer"
                and "RandomForest". Defaults to "Transformer".

        Raises:
            ValueError: If model was trained with explicit phenotype but ph_emb_path is None.
            ValueError: If architecture is not supported.

        Note:
            For inference or fine-tuning, model_pth must be provided. For training from
            scratch, model_pth can be None (though pre-trained models are recommended).
        """

        self.architecture = architecture
        self.iv_emb_path = iv_emb_path
        self.cl_emb_path = cl_emb_path
        self.ph_emb_path = ph_emb_path
        # set phenotypes (must be in the same order regardless of what is passed in predict)
        self.phenotypes = None
        self.column_map = None
        self.pert_len = None
        self.config = config  # Store the config

        if model_pth and architecture == "RandomForest":
            self.model = load(model_pth)
            # For loaded RandomForest, set attributes appropriately
            self.phenotypes = None  # Will be set during training
            self.iv_embedding, self.cl_embedding, self.ph_embedding = process_priors(
                self.iv_emb_path, self.cl_emb_path, self.ph_emb_path
            )
        else:
            self.model_pth = model_pth
            self.model = self._build_model(architecture)

            if architecture == "RandomForest":
                # RandomForest doesn't have hparams, handle differently
                self.phenotypes = None  # Will be set during training
                self.iv_embedding, self.cl_embedding, self.ph_embedding = (
                    process_priors(self.iv_emb_path, self.cl_emb_path, self.ph_emb_path)
                )
            else:
                # Transformer models have hparams
                self.phenotypes = self.model.hparams["phenotypes"]
                self.iv_embedding, self.cl_embedding, self.ph_embedding = (
                    process_priors(self.iv_emb_path, self.cl_emb_path, self.ph_emb_path)
                )
                if self.model.hparams.explicit_phenotype and self.ph_embedding is None:
                    raise ValueError(
                        "model was run with explicit phenotype! must pass a ph_emb_path"
                    )

    @classmethod
    def download_embeddings(
        cls,
        cache_dir: Optional[str] = None,
        force_download: bool = False,
    ) -> Tuple[str, str]:
        """Download Prophet embeddings separately from model checkpoints.

        This method provides explicit control over downloading embeddings,
        allowing you to download them independently from model weights.
        The embeddings are universal across all Prophet models.

        Args:
            cache_dir: Directory to cache downloaded files. If None, uses default cache.
            force_download: Whether to force re-download even if files exist.

        Returns:
            Tuple of (intervention_emb_path, cell_line_emb_path)

        Example:
            Download embeddings separately:
            >>> iv_emb, cl_emb = Prophet.download_embeddings()
            >>> print(f"Intervention embeddings: {iv_emb}")
            >>> print(f"Cell line embeddings: {cl_emb}")

            Then use them to create a model:
            >>> model = Prophet(
            ...     iv_emb_path=iv_emb,
            ...     cl_emb_path=cl_emb,
            ...     model_pth="my_model.ckpt"
            ... )
        """
        from prophet.utils.model_hub import download_embeddings as _download_embeddings
        return _download_embeddings(cache_dir=cache_dir, force_download=force_download)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "base",
        split: str = "cell_lines",
        fold: int = 0,
        seed: int = 110,
        cache_dir: Optional[str] = None,
        force_download: bool = False,
        download_embeddings: bool = True,
        **kwargs,
    ) -> "Prophet":
        """Load a pretrained Prophet model from HuggingFace Hub.

        This is the easiest way to get started with Prophet! Simply specify a dataset name
        and all required files (model checkpoint, embeddings) will be automatically downloaded.

        Args:
            model_name: Dataset name (e.g., "base", "GDSC", "CTRP", "LINCS").
                For backward compatibility, also accepts old names like "prophet-base".
                Use Prophet.list_models() to see available options.
            split: Split type - 'cell_lines' or 'perturbations' (default: 'cell_lines')
            fold: Cross-validation fold number 0-4 (default: 0)
            seed: Random seed - 110, 1995, or 2024 (default: 110)
            cache_dir: Directory to cache downloaded files. If None, uses default cache.
            force_download: Whether to force re-download even if files exist.
            download_embeddings: Whether to download embeddings with the model (default: True).
                Set to False if you want to manage embeddings separately using Prophet.download_embeddings().
            **kwargs: Additional arguments passed to Prophet constructor.

        Returns:
            Prophet model ready for inference or fine-tuning.

        Raises:
            ValueError: If model_name is not available.
            ConnectionError: If download fails.

        Examples:
            Load base pretrained model:
            >>> model = Prophet.from_pretrained("base")
            >>> predictions = model.predict(data)

            Load GDSC model with specific configuration:
            >>> model = Prophet.from_pretrained("GDSC", split="perturbations", fold=0, seed=110)

            Load CTRP model with different seed:
            >>> model = Prophet.from_pretrained("CTRP", split="cell_lines", fold=2, seed=1995)

            Download embeddings separately for better control:
            >>> iv_emb, cl_emb = Prophet.download_embeddings()
            >>> model = Prophet.from_pretrained("GDSC", split="perturbations", fold=0, seed=110,
            ...                                 download_embeddings=False)

            See available models:
            >>> Prophet.list_models()
        """
        legacy_mapping = {
            "prophet-base": "base",
            "prophet-gdsc": "GDSC",
            "prophet-ctrp": "CTRP",
            "prophet-lincs": "LINCS",
            "prophet-jump": "JUMP",
            "prophet-score": "SCORE",
            "prophet-horlbeck": "Horlbeck",
            "prophet-gdsc-comb": "GDSCcomb",
        }

        dataset = legacy_mapping.get(model_name, model_name)

        print(f"🔄 Downloading {dataset} model from HuggingFace Hub...")
        print(f"   Configuration: split={split}, fold={fold}, seed={seed}")

        model_path, gene_emb_path, cell_emb_path, phenotype_emb_path = (
            download_model(
                dataset=dataset,
                split=split,
                fold=fold,
                seed=seed,
                cache_dir=cache_dir,
                force_download=force_download,
                download_embeddings_flag=download_embeddings,
            )
        )

        return cls(
            iv_emb_path=gene_emb_path,
            cl_emb_path=cell_emb_path,
            ph_emb_path=phenotype_emb_path,
            model_pth=model_path,
            **kwargs,
        )

    @classmethod
    def from_dataset(
        cls,
        dataset: str,
        split: str = "cell_lines",
        fold: int = 0,
        seed: int = 110,
        cache_dir: Optional[str] = None,
        force_download: bool = False,
        **kwargs,
    ) -> "Prophet":
        """Load a Prophet model trained on a specific dataset with custom parameters.

        This method allows you to load models with specific training configurations
        from the structured HuggingFace repository.

        Args:
            dataset: Dataset name (e.g., "GDSC", "CTRP", "LINCS", "JUMP", "SCORE", "Horlbeck", "GDSCcomb").
                Use Prophet.available_datasets() to see all options.
            split: Split type - 'cell_lines' or 'perturbations' (default: 'cell_lines')
            fold: Cross-validation fold number 0-4 (default: 0)
            seed: Random seed - 110, 1995, or 2024 (default: 110)
            cache_dir: Directory to cache downloaded files. If None, uses default cache.
            force_download: Whether to force re-download even if files exist.
            **kwargs: Additional arguments passed to Prophet constructor.

        Returns:
            Prophet model ready for inference or fine-tuning.

        Raises:
            ValueError: If dataset or split is not available.
            ConnectionError: If download fails.

        Examples:
            Load GDSC model with default parameters:
            >>> model = Prophet.from_dataset("GDSC")

            Load CTRP model with specific split and seed:
            >>> model = Prophet.from_dataset(
            ...     dataset="CTRP",
            ...     split="perturbations",
            ...     seed=1995,
            ...     fold=2
            ... )

            Load LINCS model with unseen cell lines:
            >>> model = Prophet.from_dataset(
            ...     dataset="LINCS",
            ...     split="cell_lines",
            ...     fold=0,
            ...     seed=110
            ... )

            See available datasets and splits:
            >>> Prophet.available_datasets()
            >>> Prophet.available_splits()
        """
        available_datasets = get_available_datasets()
        if dataset not in available_datasets:
            raise ValueError(
                f"Dataset '{dataset}' not available. Available datasets: {available_datasets}"
            )

        available_splits = get_available_splits()
        if split not in available_splits:
            raise ValueError(
                f"Split '{split}' not available. Available splits: {available_splits}"
            )

        print(f"🔄 Downloading Prophet model: {dataset}/unseen_{split}_fold_{fold}/seed_{seed}")

        model_path, gene_emb_path, cell_emb_path, phenotype_emb_path = (
            download_model(
                dataset=dataset,
                split=split,
                fold=fold,
                seed=seed,
                cache_dir=cache_dir,
                force_download=force_download,
            )
        )

        return cls(
            iv_emb_path=gene_emb_path,
            cl_emb_path=cell_emb_path,
            ph_emb_path=phenotype_emb_path,
            model_pth=model_path,
            **kwargs,
        )

    @staticmethod
    def list_models() -> None:
        """Print available pretrained models.

        Shows all Prophet models available for download from HuggingFace Hub
        with their descriptions and usage information.

        Example:
            >>> Prophet.list_models()
        """
        print_available_models()

    @staticmethod
    def available_models() -> Dict[str, List]:
        """Get programmatic access to available model configurations.

        Returns:
            Dictionary with lists of available options for each parameter.

        Example:
            >>> configs = Prophet.available_models()
            >>> print(configs['datasets'])
            ['CTRP', 'GDSC', 'GDSCcomb', 'Horlbeck', 'JUMP', 'LINCS', 'SCORE', ...]
            >>> print(configs['splits'])
            ['cell_lines', 'perturbations']
        """
        return {
            "datasets": get_available_datasets(),
            "splits": get_available_splits(),
            "folds": get_available_folds(),
            "seeds": get_available_seeds()
        }

    @staticmethod
    def available_datasets() -> List[str]:
        """Get list of available datasets for training-specific models.

        Returns:
            List of dataset names available in the HuggingFace repository.

        Example:
            >>> datasets = Prophet.available_datasets()
            >>> print(datasets)
            ['CTRP', 'GDSC', 'GDSCcomb', 'Horlbeck', 'JUMP', 'LINCS', 'SCORE']
        """
        return get_available_datasets()

    @staticmethod
    def available_splits() -> List[str]:
        """Get list of available split methods used during training.

        Returns:
            List of split method names.

        Example:
            >>> splits = Prophet.available_splits()
            >>> print(splits)
            ['cell_lines', 'perturbations']
        """
        return get_available_splits()

    @staticmethod
    def available_seeds() -> List[int]:
        """Get list of available random seeds used during training.

        Returns:
            List of seed values.

        Example:
            >>> seeds = Prophet.available_seeds()
            >>> print(seeds)
            [110, 1995, 2024]
        """
        return get_available_seeds()

    @staticmethod
    def available_folds() -> List[int]:
        """Get list of available cross-validation folds.

        Returns:
            List of fold numbers (0-4).

        Example:
            >>> folds = Prophet.available_folds()
            >>> print(folds)
            [0, 1, 2, 3, 4]
        """
        return get_available_folds()

    def _build_model(self, arch, model_config=None):
        if arch == "RandomForest":
            self.torch_dataset = False
            from ..models.random_forest import RandomForestPredictor

            # Extract RandomForest-specific config if provided
            rf_config = {}
            if model_config and hasattr(model_config, "RandomForest"):
                rf_config = model_config.RandomForest
            elif (
                model_config
                and isinstance(model_config, dict)
                and "RandomForest" in model_config
            ):
                rf_config = model_config["RandomForest"]

            return RandomForestPredictor(**rf_config)
        elif arch == "Transformer":
            self.torch_dataset = True

            if self.model_pth is not None:
                # Load from checkpoint (fine-tuning)
                model = TransformerPredictor.load_from_checkpoint(
                    checkpoint_path=self.model_pth, map_location=torch.device("cpu")
                )

                # Change learning rate for active learning
                if hasattr(model, "hparams") and "lr" in model.hparams:
                    model.hparams.lr = 1e-5
                    model.hparams.weight_decay = 1e-6
                    print(f"Learning rate set to {model.hparams.lr}")

                # Override optimizer configuration for fine-tuning (disable scheduler)
                def configure_optimizers_no_scheduler(self):
                    optimizer = optim.AdamW(
                        self.parameters(),
                        lr=self.hparams.lr,
                        weight_decay=self.hparams.weight_decay,
                    )
                    return optimizer

                # Bind the new method to the model instance
                model.configure_optimizers = types.MethodType(
                    configure_optimizers_no_scheduler, model
                )

            else:
                # Create new model from scratch
                from ..models import load_models_config

                # Use seed from config, fallback to 42 if not available
                seed = getattr(self.config, "random_seed", 42)
                model, _ = load_models_config(self.config, seed=seed)

            # working backwards from config
            if model.hparams.simpler:
                self.pert_len = model.hparams.ctx_len - 1
            else:
                self.pert_len = model.hparams.ctx_len - 3

            return model
        else:
            raise ValueError(arch, " is not a valid model architecture.")

    def _remove_nonexistent_cat(
        self,
        data_label: Optional[pd.DataFrame] = None,
        verbose=True,
    ):
        embeddings = [self.iv_embedding, self.cl_embedding, self.ph_embedding]
        cols = [self.iv_cols, self.cl_col, self.ph_col]
        for i, embedding in enumerate(embeddings):
            if embedding is None:  # phenotype embedding can be None
                continue
            data_label = remove_nonexistent_cat(data_label, embedding, cols[i], verbose)
        data_label = data_label.reset_index(drop=True)

        if len(data_label) == 0 and not verbose:
            self._remove_nonexistent_cat(data_label=data_label, verbose=True)
            raise ValueError("labels did not match embeddings passed!")
        return data_label

    def _init_input(
        self,
        iv_col: Union[List[str], str] = ["iv1", "iv2"],
        cl_col: str = "cell_line",
        ph_col: str = "phenotype",
        readout_col: str = "value",
    ):
        """Sets some state variables in the model, but is always overwritten by
        either train or predict.
        """
        if isinstance(iv_col, str):
            iv_col = [iv_col]
        intervention_mapping = {col: f"iv{i + 1}" for i, col in enumerate(iv_col)}
        self.iv_cols = list(intervention_mapping.values())

        # store the columns used for training for reference
        self.cl_col = cl_col
        self.ph_col = ph_col
        self.readout_col = readout_col

        # create the mapping to the internal variables used
        self.column_map = {
            self.cl_col: "cell_line",
            self.ph_col: "phenotype",
            self.readout_col: "value",
            **intervention_mapping,
        }
        if self.pert_len is None:
            self.pert_len = len(self.iv_cols)
        else:
            if self.pert_len != len(self.iv_cols):
                raise ValueError(
                    f"Are you sure you passed the right number of intervention columns? Currently receiving {self.iv_cols}"
                )

    def train(
        self,
        df: pd.DataFrame,
        iv_col: Union[List[str], str] = ["iv1", "iv2"],
        cl_col: str = "cell_line",
        ph_col: str = "phenotype",
        readout_col: str = "value",
        model_config: Optional[dict] = None,
        val_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
        wandb_config: Optional[dict] = None,
        checkpoint_dirpath: Optional[str] = None,
    ) -> None:
        """Train or fine-tune the Prophet model on experimental data.

        This method takes a DataFrame of experimental results and trains the model to predict
        the readout values based on cell line, intervention, and phenotype combinations.
        The model can be fine-tuned from a pre-trained checkpoint or trained from scratch.

        Args:
            df: DataFrame containing experimental data with the following required columns:
                - Cell line identifiers (specified by cl_col)
                - Intervention identifiers (specified by iv_col)
                - Phenotype identifiers (specified by ph_col)
                - Readout values (specified by readout_col)
            iv_col: Column name(s) for interventions. For single interventions, use a string
                or single-item list. For combinatorial interventions, use a list of column names
                (e.g., ['drug1', 'drug2']). Maximum 2 interventions supported.
            cl_col: Column name containing cell line identifiers. Cell lines must exist
                in the provided cell line embeddings.
            ph_col: Column name containing phenotype identifiers. These are used to distinguish
                different experimental readouts (e.g., 'viability', 'IC50').
            readout_col: Column name containing the numerical readout values to predict.
                Values should be normalized (e.g., min-max scaled to 0-1) for best performance.
            model_config: Configuration dictionary or object containing training hyperparameters.
                If None, default configuration will be used. Required for transformer models.
            val_df: Optional validation DataFrame. If provided, this will be used for validation
                instead of automatically splitting the training data. Must have the same column
                structure as df.
            wandb_config: Optional dictionary with WandB configuration.
                         Keys: project, entity, name, tags, notes, save_dir

        Raises:
            ValueError: If specified columns are not found in the DataFrame.
            ValueError: If intervention embeddings don't match the data.
            ValueError: If more than 2 interventions are specified.

        Examples:
            Single intervention training:
            >>> model.train(
            ...     df=data,
            ...     iv_col="drug",
            ...     cl_col="cell_line",
            ...     ph_col="assay_type",
            ...     readout_col="response"
            ... )

            Training with pre-split validation data:
            >>> model.train(
            ...     df=train_data,
            ...     val_df=val_data,
            ...     readout_col="response"
            ... )

            Combinatorial intervention training:
            >>> model.train(
            ...     df=combo_data,
            ...     iv_col=["drug_A", "drug_B"],
            ...     cl_col="cell_line",
            ...     readout_col="synergy_score"
            ... )

        Note:
            - For Transformer models, the method performs fine-tuning on a pre-trained checkpoint
            - If val_df is not provided, data is automatically split 90/10 for training/validation
            - Duplicate entries are automatically removed
            - Missing embeddings are automatically filtered out with warnings
        """
        print(
            f"=== TRAIN BEGINNING - Torch RNG state: {torch.get_rng_state()[:10].tolist()}"
        )
        self._init_input(iv_col, cl_col, ph_col, readout_col)

        # Data should already be clean and validated at this point
        df = df.rename(columns=self.column_map).copy()
        val_df = val_df.rename(columns=self.column_map).copy()

        if test_df is not None:
            test_df = test_df.rename(columns=self.column_map).copy()
            test_df = test_df.reset_index(drop=True)
            test_indices = np.arange(
                len(df) + len(val_df), len(df) + len(val_df) + len(test_df)
            )
        else:
            test_df = pd.DataFrame()
            test_indices = []

        # Combine all data
        combined_data = pd.concat([df, val_df, test_df], ignore_index=True)

        # Create indices for the combined dataset
        train_indices = np.arange(len(df))
        valid_indices = np.arange(len(df), len(df) + len(val_df))

        print("Fitting model.")
        if not self.torch_dataset:
            # For non-PyTorch models (like RandomForest)
            # Rebuild model with config if provided
            if model_config is not None:
                self.model = self._build_model("RandomForest", model_config)

            unbalanced = (
                getattr(model_config, "unbalanced", False) if model_config else False
            )
            split = dataloader_phenotypes(
                gene_embedding=self.iv_embedding,
                cell_lines_embedding=self.cl_embedding,
                phenotype_embedding=self.ph_embedding
                if self.ph_embedding is not None
                else None,
                data_label=combined_data,
                label_name="value",
                index=(
                    train_indices,  # train_indices
                    valid_indices,  # valid_indices
                    test_indices,  # test_indices
                    "",  # cl_holdout
                ),
                torch_dataset=self.torch_dataset,
                pert_len=len(self.iv_cols),
                unbalanced=unbalanced,
            )
            print(f"Using unbalanced sampling: {unbalanced}")
            X_train, y_train = split[0]
            X_test, y_test = split[2] if len(test_indices) > 0 else (None, None)

            # Initialize WandB logging if config provided
            logger = None
            if wandb_config is not None:
                if wandb_config == {}:
                    wandb_config = {}

                # Default WandB settings
                default_wandb = {
                    "project": "prophet",
                    "entity": None,
                    "name": "prophet-randomforest-training",
                    "tags": ["RandomForest"],
                    "notes": "Random Forest training with Prophet",
                    "save_dir": "./wandb",
                }

                # Update with provided config
                default_wandb.update(wandb_config)

                # Create WandB logger
                logger = WandbLogger(
                    project=default_wandb["project"],
                    entity=default_wandb["entity"],
                    name=default_wandb["name"],
                    tags=default_wandb["tags"],
                    notes=default_wandb["notes"],
                    save_dir=default_wandb["save_dir"],
                )

                # Log model config
                logger.experiment.config.update(
                    {
                        "architecture": "RandomForest",
                        "n_estimators": getattr(self.model, "n_estimators", 100),
                        "max_depth": getattr(self.model, "max_depth", None),
                        "min_samples_split": getattr(
                            self.model, "min_samples_split", 2
                        ),
                        "min_samples_leaf": getattr(self.model, "min_samples_leaf", 1),
                        "bootstrap": getattr(self.model, "bootstrap", True),
                        "n_features": X_train.shape[1],
                        "n_train_samples": X_train.shape[0],
                        "n_test_samples": X_test.shape[0] if X_test is not None else 0,
                        "unbalanced_sampling": unbalanced,
                        "pert_len": len(self.iv_cols),
                    }
                )

            # Train the model
            print(
                f"🌲 Training RandomForest with {X_train.shape[0]:,} samples and {X_train.shape[1]:,} features"
            )
            self.model.fit(X_train, y_train)

            # Evaluate on test set if available
            if X_test is not None and len(X_test) > 0:
                print("🔮 Evaluating RandomForest on test set...")
                y_pred = self.model.predict(X_test)

                # Compute basic metrics
                from sklearn.metrics import r2_score
                from scipy.stats import spearmanr
                from ..utils.callbacks import compute_hit_ratio

                r2_test = r2_score(y_test, y_pred)
                spearman_test = spearmanr(y_pred, y_test).statistic

                print(f"✅ Test R² Score: {r2_test:.4f}")
                print(f"✅ Test Spearman Correlation: {spearman_test:.4f}")

                # Prepare data for HitRatio computation
                test_data = combined_data.iloc[test_indices].copy()

                # Extract cell lines, interventions, and phenotypes
                cl_names = test_data["cell_line"].tolist()
                iv_combinations = []
                for idx in test_data.index:
                    iv_combo = tuple(test_data.loc[idx, col] for col in self.iv_cols)
                    iv_combinations.append(iv_combo)

                phenotypes = test_data["phenotype"].values

                # Compute HitRatio metrics
                topk_values = (
                    [5, 10]
                    if model_config is None
                    else getattr(model_config, "topk", [5, 10])
                )
                hitratio_metrics = compute_hit_ratio(
                    predictions=y_pred,
                    targets=y_test,
                    cl=cl_names,
                    phenotypes=phenotypes,
                    iv=iv_combinations,
                    topk=topk_values,
                    suffix="_test",
                )

                # Print HitRatio results
                for metric_name, value in hitratio_metrics.items():
                    if "precision_both" in metric_name:
                        print(f"✅ {metric_name}: {value:.4f}")

                # Log all metrics to WandB if logger available
                if logger is not None:
                    logger.experiment.log(
                        {
                            "R2_test": r2_test,
                            "Spearman_test": spearman_test,
                            **hitratio_metrics,
                            "n_test_samples": len(y_test),
                            "n_features": X_train.shape[1],
                        }
                    )

                    # Finish WandB run
                    logger.experiment.finish()
                    print("📊 Metrics logged to WandB!")
            else:
                print("⚠️  No test set available for evaluation")
                if logger is not None:
                    logger.experiment.finish()
        else:
            # Create dataloader with test setj
            unbalanced = (
                getattr(model_config, "unbalanced", False) if model_config else False
            )
            split = dataloader_phenotypes(
                gene_embedding=self.iv_embedding,
                cell_lines_embedding=self.cl_embedding,
                phenotype_embedding=self.ph_embedding,
                data_label=combined_data,
                label_name="value",
                index=(
                    train_indices,  # train_indices
                    valid_indices,  # valid_indices
                    test_indices,
                    "",  # cl_holdout
                ),
                torch_dataset=self.torch_dataset,
                pert_len=len(self.iv_cols),
                valid_set=True,
                test_set=len(test_indices) > 0,
                batch_size=model_config.batch_size,
                unbalanced=unbalanced,
            )
            print(f"Using unbalanced sampling: {unbalanced}")

            # Use existing model if available, otherwise load from config
            if hasattr(self, "model") and self.model is not None:
                model = self.model
                model = model.float()

                if checkpoint_dirpath is not None:
                    dirpath = checkpoint_dirpath
                elif model_config is None:
                    dirpath = "./checkpoints"
                else:
                    dirpath = model_config.dirpath
            else:
                if model_config is None:
                    raise ValueError(
                        "model_config is required when no pre-trained model is loaded"
                    )
                model, model_config = load_models_config(
                    model_config, seed=42, phenotypes=None
                )
                self.model = model
                model = model.float()
                dirpath = (
                    checkpoint_dirpath
                    if checkpoint_dirpath is not None
                    else model_config.dirpath
                )

            lr_monitor = LearningRateMonitor(logging_interval="step")
            model_checkpointer = ModelCheckpoint(
                dirpath=dirpath,
                save_top_k=1,
                every_n_epochs=1,
                monitor="R2_validation",
                mode="max",
            )
            r2_average = (
                getattr(model_config, "r2_average", False) if model_config else False
            )
            r2_callback = R2ScoreCallback(device=model.device, average=r2_average)
            hitratio_callback = HitRatioCallback(device=model.device, topk=[5, 10])
            print(f"R2 average: {r2_average}")
            early_stopping = EarlyStopping(
                monitor="R2_validation", mode="max", patience=10, min_delta=0.0
            )

            if wandb_config is None:
                wandb_config = {}

            # Default WandB settings
            default_wandb = {
                "project": "prophet",
                "entity": None,  # Use default entity
                "name": "prophet-training",
                "tags": [],
                "notes": None,
                "save_dir": "./wandb",
            }

            # Update with provided config
            default_wandb.update(wandb_config)

            # Create WandB logger
            logger = WandbLogger(
                project=default_wandb["project"],
                entity=default_wandb["entity"],
                name=default_wandb["name"],
                tags=default_wandb["tags"],
                notes=default_wandb["notes"],
                save_dir=default_wandb["save_dir"],
            )

            callbacks = [
                r2_callback,
                hitratio_callback,
                model_checkpointer,
                lr_monitor,
                early_stopping,
            ]

            trainer = pl.Trainer(
                max_steps=model_config.max_steps,
                accelerator="gpu",
                devices=-1,
                check_val_every_n_epoch=1,
                callbacks=callbacks,
                strategy="ddp" if torch.cuda.device_count() > 1 else "auto",
                enable_progress_bar=True,
                log_every_n_steps=1,
                deterministic=True,
                enable_model_summary=True,
                logger=logger,
                num_sanity_val_steps=0,
                gradient_clip_val=1.0,
            )

            print(
                f"Dataset sizes:\n"
                f"  Training:    {len(split[0].dataset.labels):,d} samples\n"
                f"  Validation:  {len(split[1].dataset.labels):,d} samples\n"
                f"  Test:        {len(split[2].dataset.labels):,d} samples"
            )
            train_dataloader = split[0]
            print(
                "Indices in first batch:",
                train_dataloader.batch_sampler.__iter__().__next__(),
            )
            trainer.fit(
                model=model, train_dataloaders=split[0], val_dataloaders=split[1]
            )

            # Load the best checkpoint after training
            best_model_path = model_checkpointer.best_model_path
            if best_model_path:
                model = type(model).load_from_checkpoint(best_model_path)
                model = model.float()
                self.model = model
            else:
                print("No checkpoint was saved during training")

            # 🆕 NEW: Evaluate on test set if provided
            if len(test_indices) > 0:
                print("Evaluating on test set...")
                test_results = trainer.test(
                    model, split[2]
                )  # split[2] is already test dataloader
                print("✅ Test set evaluation completed!")
                print(f"   Test metrics: {test_results}")

            wandb.finish()

    def _generate_predict_df(
        self,
        run_index: int,
        num_iterations: int,
        target_ivs: List[str],
        target_cls: List[str],
        target_phs: List[str] = ["_"],
    ):
        subset_cl = pd.DataFrame(target_cls, columns=["cell_line"])
        subset_iv = pd.DataFrame(target_ivs, columns=["iv"])
        if target_phs is None:
            target_phs = ["_"]
        subset_ph = pd.DataFrame(target_phs, columns=["phenotype"])
        if len(self.iv_cols) > 2:
            raise ValueError(
                "Currently only 1 or 2 interventions are supported when using list mode. "
                f"You specified {len(self.iv_cols)} interventions: {self.iv_cols}. "
                "For more complex interventions, please create a DataFrame with your "
                "specific experimental combinations and use df mode instead."
            )

        batch_size = int(len(subset_iv) // num_iterations)
        start_idx = run_index * batch_size
        end_idx = (
            (start_idx + batch_size)
            if (run_index < num_iterations - 1)
            else len(subset_iv["iv"])
        )

        data_label = pd.merge(
            subset_iv[["iv"]][start_idx:end_idx], subset_cl, how="cross"
        )
        data_label = pd.merge(data_label, subset_ph, how="cross")

        if len(self.iv_cols) == 1:
            data_label.rename(columns={"iv": "iv1"}, inplace=True)
        else:
            data_label = pd.merge(
                subset_iv[["iv"]], data_label, how="cross", suffixes=("1", "2")
            )
            # A+B and B+A should be the same, so we remove all duplicates in favor of A+B (was pretty sure this shouldn't exist in the implementation @John)
            data_label["iv1+iv2"] = [
                "+".join(sorted([row["iv1"], row["iv2"]]))
                for _, row in data_label.iterrows()
            ]
            data_label = data_label.drop_duplicates(
                subset=["iv1+iv2", "cell_line", "phenotype"]
            )

        data_label["value"] = "_"

        return data_label

    def _decide_iteration_num(
        self,
        total_size: int,
        single_run_size: int = None,
        memory_size: int = None,
    ):
        if total_size <= single_run_size:
            num_iterations = 1
        else:
            num_iterations = total_size // single_run_size

        return int(num_iterations)

    def predict(
        self,
        df: pd.DataFrame,
        save: bool = False,
    ) -> pd.DataFrame:
        """Generate predictions using the trained Prophet model.

        Args:
            df: DataFrame containing experimental combinations to predict.
                Must contain columns: iv1, iv2, cell_line, phenotype.
            save: If True, saves predictions to parquet file. If False, returns DataFrame.

        Returns:
            DataFrame with predictions in 'pred' column.

        Raises:
            ValueError: If no valid rows remain after filtering invalid embeddings.
        """
        import warnings

        if self.column_map is not None:
            df = df.rename(columns=self.column_map).copy()
        df = df.reset_index(drop=True)

        # Validate data against available embeddings
        original_len = len(df)
        available_cls = set(self.cl_embedding.index)
        available_ivs = set(self.iv_embedding.index)
        available_phs = set(self.phenotypes) if self.phenotypes else None

        # Check cell lines
        missing_cls = set(df["cell_line"].unique()) - available_cls
        if missing_cls:
            warnings.warn(
                f"Found {len(missing_cls)} cell line(s) not in embeddings: {list(missing_cls)[:5]}"
                f"{'...' if len(missing_cls) > 5 else ''}. These rows will be skipped."
            )
            df = df[df["cell_line"].isin(available_cls)]

        # Check interventions
        for iv_col in self.iv_cols:
            if iv_col in df.columns:
                missing_ivs = set(df[iv_col].unique()) - available_ivs
                if missing_ivs:
                    warnings.warn(
                        f"Found {len(missing_ivs)} intervention(s) in '{iv_col}' not in embeddings: "
                        f"{list(missing_ivs)[:5]}{'...' if len(missing_ivs) > 5 else ''}. These rows will be skipped."
                    )
                    df = df[df[iv_col].isin(available_ivs)]

        # Check phenotypes
        if available_phs is not None:
            missing_phs = set(df["phenotype"].unique()) - available_phs
            if missing_phs:
                warnings.warn(
                    f"Found {len(missing_phs)} phenotype(s) not in model: {list(missing_phs)}. "
                    f"Available phenotypes: {list(available_phs)[:10]}. These rows will be skipped."
                )
                df = df[df["phenotype"].isin(available_phs)]

        df = df.reset_index(drop=True)

        # Check if any valid rows remain
        if len(df) == 0:
            raise ValueError(
                "No valid rows remain after filtering. Check that your cell lines, "
                "interventions, and phenotypes exist in the model's embeddings. "
                f"Available cell lines: {len(available_cls)}, "
                f"Available interventions: {len(available_ivs)}, "
                f"Available phenotypes: {list(available_phs) if available_phs else 'any'}"
            )

        if len(df) < original_len:
            print(f"Predicting on {len(df)}/{original_len} valid rows ({original_len - len(df)} rows filtered out)")

        # Add dummy value column for dataloader
        df["_"] = 0

        # Create dataloader
        split = dataloader_phenotypes(
            gene_embedding=self.iv_embedding,
            cell_lines_embedding=self.cl_embedding,
            phenotype_embedding=self.ph_embedding,
            data_label=df,
            label_name="_",
            index=(
                np.array(df.index).tolist(),
                [],
                np.array(df.index).tolist(),  # test indices
                "",
            ),
            torch_dataset=self.torch_dataset,
            pert_len=self.pert_len,
        )

        # Get test dataloader
        (
            train_dataloader,
            valid_dataloader,
            test_dataloader,
            train_indices,
            test_indices,
            descriptor,
        ) = split

        # Make predictions
        trainer = pl.Trainer(devices=1)
        predictions = trainer.predict(self.model, test_dataloader)
        predictions = [t[0] for t in predictions]
        predictions = torch.cat(predictions, dim=0)

        # Add predictions to dataframe
        df["pred"] = predictions
        df.drop(columns=["_"], inplace=True)

        if save:
            df.to_parquet("prophet_predictions.parquet")
            return None
        else:
            return df
