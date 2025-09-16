"""Input validation utilities for Prophet.

This module provides comprehensive validation functions to ensure data
quality and compatibility before training or prediction.
"""

import pandas as pd
from typing import List, Dict, Optional, Union, Any
import warnings
from pathlib import Path
from prophet.data import remove_nonexistent_cat, process_priors


class ValidationError(Exception):
    """Custom exception for validation errors."""

    pass


class DataFrameValidator:
    """Validator for experimental DataFrames used in Prophet."""

    @staticmethod
    def validate_columns(
        df: pd.DataFrame,
        required_columns: List[str],
        optional_columns: Optional[List[str]] = None,
    ) -> None:
        """Validate that DataFrame contains required columns.

        Args:
            df: DataFrame to validate.
            required_columns: List of column names that must be present.
            optional_columns: List of column names that may be present.

        Raises:
            ValidationError: If required columns are missing.
        """
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValidationError(
                f"Missing required columns: {missing_cols}. "
                f"Available columns: {list(df.columns)}"
            )

        # Warn about unexpected columns
        expected_cols = set(required_columns + (optional_columns or []))
        unexpected_cols = set(df.columns) - expected_cols
        if unexpected_cols:
            warnings.warn(
                f"Unexpected columns found: {list(unexpected_cols)}. "
                "These will be ignored during processing."
            )

    @staticmethod
    def validate_data_types(
        df: pd.DataFrame,
        column_types: Dict[str, type],
        convert_if_possible: bool = True,
    ) -> pd.DataFrame:
        """Validate and optionally convert column data types.

        Args:
            df: DataFrame to validate.
            column_types: Dictionary mapping column names to expected types.
            convert_if_possible: Whether to attempt type conversion.

        Returns:
            DataFrame with validated/converted types.

        Raises:
            ValidationError: If types cannot be validated or converted.
        """
        df_validated = df.copy()

        for col, expected_type in column_types.items():
            if col not in df_validated.columns:
                continue

            current_type = df_validated[col].dtype

            # Check if conversion is needed
            if expected_type is float and not pd.api.types.is_numeric_dtype(
                current_type
            ):
                if convert_if_possible:
                    try:
                        df_validated[col] = pd.to_numeric(
                            df_validated[col], errors="coerce"
                        )
                        if df_validated[col].isnull().any():
                            warnings.warn(
                                f"Some values in column '{col}' could not be converted to numeric"
                            )
                    except Exception as e:
                        raise ValidationError(
                            f"Cannot convert column '{col}' to numeric: {e}"
                        )
                else:
                    raise ValidationError(
                        f"Column '{col}' has type {current_type}, expected numeric"
                    )

            elif expected_type is str and current_type != "object":
                if convert_if_possible:
                    df_validated[col] = df_validated[col].astype(str)
                else:
                    raise ValidationError(
                        f"Column '{col}' has type {current_type}, expected string"
                    )

        return df_validated

    @staticmethod
    def validate_value_ranges(
        df: pd.DataFrame, column_ranges: Dict[str, tuple], clip_outliers: bool = False
    ) -> pd.DataFrame:
        """Validate that numeric columns fall within expected ranges.

        Args:
            df: DataFrame to validate.
            column_ranges: Dictionary mapping column names to (min, max) tuples.
            clip_outliers: Whether to clip values to the valid range.

        Returns:
            DataFrame with validated ranges.

        Raises:
            ValidationError: If values are outside expected ranges and clip_outliers=False.
        """
        df_validated = df.copy()

        for col, (min_val, max_val) in column_ranges.items():
            if col not in df_validated.columns:
                continue

            out_of_range = (df_validated[col] < min_val) | (df_validated[col] > max_val)

            if out_of_range.any():
                n_outliers = out_of_range.sum()
                if clip_outliers:
                    df_validated[col] = df_validated[col].clip(min_val, max_val)
                    warnings.warn(
                        f"Clipped {n_outliers} values in column '{col}' to range [{min_val}, {max_val}]"
                    )
                else:
                    raise ValidationError(
                        f"Column '{col}' contains {n_outliers} values outside "
                        f"valid range [{min_val}, {max_val}]"
                    )

        return df_validated

    @staticmethod
    def validate_no_missing_values(
        df: pd.DataFrame, columns: Optional[List[str]] = None, action: str = "error"
    ) -> pd.DataFrame:
        """Validate that specified columns contain no missing values.

        Args:
            df: DataFrame to validate.
            columns: List of columns to check. If None, checks all columns.
            action: Action to take if missing values found ('error', 'warn', 'drop').

        Returns:
            DataFrame with missing values handled.

        Raises:
            ValidationError: If missing values found and action='error'.
        """
        if columns is None:
            columns = df.columns.tolist()

        df_validated = df.copy()
        missing_info = {}

        for col in columns:
            if col in df_validated.columns:
                missing_count = df_validated[col].isnull().sum()
                if missing_count > 0:
                    missing_info[col] = missing_count

        if missing_info:
            total_missing = sum(missing_info.values())
            message = f"Found {total_missing} missing values in columns: {missing_info}"

            if action == "error":
                raise ValidationError(message)
            elif action == "warn":
                warnings.warn(message)
            elif action == "drop":
                warnings.warn(f"{message}. Dropping rows with missing values.")
                df_validated = df_validated.dropna(subset=columns)
            else:
                raise ValueError(f"Unknown action: {action}")

        return df_validated


class EmbeddingValidator:
    """Validator for embedding files and compatibility."""

    @staticmethod
    def validate_embedding_files(
        file_paths: Union[str, List[str]], required_format: str = "csv"
    ) -> List[str]:
        """Validate that embedding files exist and have correct format.

        Args:
            file_paths: Path or list of paths to embedding files.
            required_format: Expected file format ('csv', 'parquet', etc.).

        Returns:
            List of validated file paths.

        Raises:
            ValidationError: If files don't exist or have wrong format.
        """
        if isinstance(file_paths, str):
            file_paths = [file_paths]

        validated_paths = []

        for path in file_paths:
            path_obj = Path(path)

            if not path_obj.exists():
                raise ValidationError(f"Embedding file not found: {path}")

            if required_format and not path.endswith(f".{required_format}"):
                raise ValidationError(
                    f"Embedding file {path} does not have expected format .{required_format}"
                )

            validated_paths.append(str(path_obj.absolute()))

        return validated_paths

    @staticmethod
    def validate_embedding_compatibility(
        df: pd.DataFrame,
        embeddings: pd.DataFrame,
        data_columns: List[str],
        embedding_name: str = "embedding",
    ) -> Dict[str, List[str]]:
        """Check compatibility between data and embeddings.

        Args:
            df: Experimental data DataFrame.
            embeddings: Embedding DataFrame with entities as index.
            data_columns: Columns in df containing entity identifiers.
            embedding_name: Name of embedding for error messages.

        Returns:
            Dictionary with compatibility information.
        """
        results = {
            "missing_entities": [],
            "available_entities": list(embeddings.index),
            "data_entities": [],
            "coverage": 0.0,
        }

        # Collect all entities from data
        data_entities = set()
        for col in data_columns:
            if col in df.columns:
                data_entities.update(df[col].unique())

        results["data_entities"] = list(data_entities)

        # Find missing entities
        missing = data_entities - set(embeddings.index)
        results["missing_entities"] = list(missing)

        # Calculate coverage
        if data_entities:
            coverage = len(data_entities - missing) / len(data_entities)
            results["coverage"] = coverage

        return results

    @staticmethod
    def validate_embedding_dimensions(
        embeddings: List[pd.DataFrame],
        expected_dims: Optional[List[int]] = None,
        embedding_names: Optional[List[str]] = None,
    ) -> None:
        """Validate that embeddings have expected dimensions.

        Args:
            embeddings: List of embedding DataFrames.
            expected_dims: List of expected dimensions for each embedding.
            embedding_names: Names of embeddings for error messages.

        Raises:
            ValidationError: If dimensions don't match expectations.
        """
        if embedding_names is None:
            embedding_names = [f"embedding_{i}" for i in range(len(embeddings))]

        for i, emb in enumerate(embeddings):
            if emb is None:
                continue

            actual_dim = emb.shape[1]
            name = embedding_names[i]

            if expected_dims and i < len(expected_dims):
                expected_dim = expected_dims[i]
                if actual_dim != expected_dim:
                    raise ValidationError(
                        f"{name} has {actual_dim} dimensions, expected {expected_dim}"
                    )

            # Check for reasonable dimension range
            if actual_dim < 10 or actual_dim > 10000:
                warnings.warn(
                    f"{name} has {actual_dim} dimensions, which seems unusual. "
                    "Typical embedding dimensions are between 50-1000."
                )


class ModelValidator:
    """Validator for model configuration and compatibility."""

    @staticmethod
    def validate_model_config(
        config: Dict[str, Any], required_fields: Optional[List[str]] = None
    ) -> None:
        """Validate model configuration dictionary.

        Args:
            config: Configuration dictionary.
            required_fields: List of required configuration fields.

        Raises:
            ValidationError: If required fields are missing or invalid.
        """
        if required_fields is None:
            required_fields = ["max_steps", "batch_size", "Transformer"]

        missing_fields = [field for field in required_fields if field not in config]
        if missing_fields:
            raise ValidationError(f"Missing required config fields: {missing_fields}")

        # Validate specific field types and ranges
        if "max_steps" in config:
            if not isinstance(config["max_steps"], int) or config["max_steps"] <= 0:
                raise ValidationError("max_steps must be a positive integer")

        if "batch_size" in config:
            if not isinstance(config["batch_size"], int) or config["batch_size"] <= 0:
                raise ValidationError("batch_size must be a positive integer")

    @staticmethod
    def validate_checkpoint_compatibility(
        checkpoint_path: str, embedding_dims: Dict[str, int]
    ) -> None:
        """Validate that model checkpoint is compatible with embeddings.

        Args:
            checkpoint_path: Path to model checkpoint file.
            embedding_dims: Dictionary of embedding dimensions.

        Raises:
            ValidationError: If checkpoint is incompatible.
        """
        if not Path(checkpoint_path).exists():
            raise ValidationError(f"Checkpoint file not found: {checkpoint_path}")

        try:
            # Load checkpoint metadata without loading full model
            import torch

            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            hparams = checkpoint.get("hyper_parameters", {})

            # Check dimension compatibility
            if "dim_iv" in hparams and "intervention" in embedding_dims:
                if hparams["dim_iv"] != embedding_dims["intervention"]:
                    raise ValidationError(
                        f"Checkpoint expects intervention dim {hparams['dim_iv']}, "
                        f"but embeddings have dim {embedding_dims['intervention']}"
                    )

            if "dim_cl" in hparams and "cell_line" in embedding_dims:
                if hparams["dim_cl"] != embedding_dims["cell_line"]:
                    raise ValidationError(
                        f"Checkpoint expects cell line dim {hparams['dim_cl']}, "
                        f"but embeddings have dim {embedding_dims['cell_line']}"
                    )

        except Exception as e:
            raise ValidationError(f"Cannot load checkpoint for validation: {e}")


def _convert_strings_to_lowercase(df: pd.DataFrame, cl_col: str) -> pd.DataFrame:
    """Convert string columns to lowercase efficiently."""
    string_columns = df.select_dtypes(include=["object"]).columns
    string_columns = [col for col in string_columns if col != cl_col]

    if string_columns:
        # Vectorized operation on all columns at once
        df_copy = df.copy()
        df_copy[string_columns] = df_copy[string_columns].apply(
            lambda x: x.str.lower() if x.dtype == "object" else x
        )
        return df_copy

    return df


def _remove_nonexistent_categories(
    df: pd.DataFrame,
    iv_embedding: dict,
    cl_embedding: dict,
    ph_embedding: Optional[dict],
    iv_col: Union[str, List[str]],
    cl_col: str,
    ph_col: str,
) -> pd.DataFrame:
    """Remove rows with nonexistent categories efficiently."""

    # Build comprehensive filter in one pass
    mask = pd.Series(True, index=df.index)

    # Check intervention columns
    if isinstance(iv_col, str):
        iv_cols = [iv_col]
    else:
        iv_cols = iv_col

    valid_ivs = sorted(set(iv_embedding.index))
    valid_ivs_set = set(valid_ivs)  # Convert back to set for fast lookup

    for col in iv_cols:
        if col in df.columns:
            mask &= df[col].isin(valid_ivs_set)

    # Check cell line column
    if cl_col in df.columns:
        valid_cls = sorted(set(cl_embedding.index))
        valid_cls_set = set(valid_cls)  # Convert back to set for fast lookup
        mask &= df[cl_col].isin(valid_cls_set)

    # Check phenotype column
    if ph_embedding is not None and ph_col in df.columns:
        valid_phs = sorted(set(ph_embedding.index))
        valid_phs_set = set(valid_phs)  # Convert back to set for fast lookup
        mask &= df[ph_col].isin(valid_phs_set)

    return df[mask].copy()


def validate_prophet_inputs(
    df: Optional[pd.DataFrame] = None,
    iv_emb_path: Optional[Union[str, List[str]]] = None,
    cl_emb_path: Optional[Union[str, List[str]]] = None,
    ph_emb_path: Optional[Union[str, List[str]]] = None,
    iv_col: Optional[Union[str, List[str]]] = None,
    cl_col: Optional[str] = None,
    ph_col: Optional[str] = None,
    readout_col: Optional[str] = None,
    mode: str = "train",
) -> Dict:
    """
    Validate Prophet inputs and return processed data.

    Args:
        df: Input DataFrame.
        iv_emb_path: Path(s) to intervention embeddings.
        cl_emb_path: Path(s) to cell line embeddings.
        ph_emb_path: Path(s) to phenotype embeddings.
        model_pth: Path to model checkpoint.
        iv_col: Intervention column name(s).
        cl_col: Cell line column name.
        ph_col: Phenotype column name.
        readout_col: Readout column name.
        mode: Validation mode ('train', 'predict').

    Returns:
        Dictionary with validation results and processed inputs.

    Raises:
        ValidationError: If validation fails.
    """
    print("Starting input validation...")
    results = {"status": "success", "warnings": [], "processed_inputs": {}}

    # Validate embedding files
    if iv_emb_path or cl_emb_path or ph_emb_path:
        print("Validating embedding files...")

    if iv_emb_path:
        iv_paths = EmbeddingValidator.validate_embedding_files(iv_emb_path)
        results["processed_inputs"]["iv_emb_path"] = iv_paths

    if cl_emb_path:
        cl_paths = EmbeddingValidator.validate_embedding_files(cl_emb_path)
        results["processed_inputs"]["cl_emb_path"] = cl_paths

    if ph_emb_path:
        ph_paths = EmbeddingValidator.validate_embedding_files(ph_emb_path)
        results["processed_inputs"]["ph_emb_path"] = ph_paths

    # Validate DataFrame if provided
    if df is not None:
        print(
            f"Processing DataFrame with {df.shape[0]} rows and {df.shape[1]} columns..."
        )

        required_cols = []
        if iv_col:
            if isinstance(iv_col, str):
                required_cols.append(iv_col)
            else:
                required_cols.extend(iv_col)
        if cl_col:
            required_cols.append(cl_col)
        if ph_col:
            required_cols.append(ph_col)
        if readout_col and mode == "train":
            required_cols.append(readout_col)

        print("Validating DataFrame structure and columns...")
        DataFrameValidator.validate_columns(df, required_cols)

        # Validate data types
        column_types = {}
        if readout_col and mode == "train":
            column_types[readout_col] = float

        df_validated = DataFrameValidator.validate_data_types(df, column_types)
        results["processed_inputs"]["df"] = df_validated

        # Check for missing values in critical columns
        print("Checking for missing values...")
        critical_cols = [col for col in required_cols if col != readout_col]
        df_validated = DataFrameValidator.validate_no_missing_values(
            df_validated, critical_cols, action="warn"
        )

        if mode == "train" and readout_col:
            # Validate readout values
            df_validated = DataFrameValidator.validate_no_missing_values(
                df_validated, [readout_col], action="warn"
            )

        # Convert strings to lowercase efficiently
        print("Converting text to lowercase...")
        df_validated = _convert_strings_to_lowercase(df_validated, cl_col)

        # Process priors
        print("Loading embedding files...")
        iv_embedding, cl_embedding, ph_embedding = process_priors(
            iv_emb_path, cl_emb_path, ph_emb_path
        )

        # Remove nonexistent categories in one efficient pass
        print("Filtering data to match available embeddings...")
        initial_rows = df_validated.shape[0]
        df_validated = _remove_nonexistent_categories(
            df_validated,
            iv_embedding,
            cl_embedding,
            ph_embedding,
            iv_col,
            cl_col,
            ph_col,
        )
        final_rows = df_validated.shape[0]
        if final_rows < initial_rows:
            print(
                f"Filtered {initial_rows - final_rows} rows with missing embeddings ({final_rows} rows remaining)"
            )

        # Single reset_index at the end
        df_validated = df_validated.reset_index(drop=True)
        results["processed_inputs"]["df"] = df_validated

    print("Input validation completed successfully")
    return results
