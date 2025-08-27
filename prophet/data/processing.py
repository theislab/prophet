"""Data processing and splitting utilities for Prophet.

This module provides utilities for preparing data for training and evaluation,
including various data splitting strategies, preprocessing functions, and
validation utilities.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional, Union, Literal
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class DataSplitter:
    """Utilities for splitting experimental data into train/validation/test sets.

    Supports various splitting strategies appropriate for biological data,
    including random splits, cell line holdouts, intervention holdouts,
    and temporal splits.
    """

    @staticmethod
    def random_split(
        df: pd.DataFrame,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
        random_state: int = 42,
        stratify_col: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Perform random split of data into train/validation/test sets.

        Args:
            df: Input DataFrame to split.
            train_size: Fraction of data for training (0-1).
            val_size: Fraction of data for validation (0-1).
            test_size: Fraction of data for testing (0-1).
                Must satisfy train_size + val_size + test_size = 1.
            random_state: Random seed for reproducibility.
            stratify_col: Column name to stratify split on (e.g., 'phenotype').

        Returns:
            Tuple of (train_df, val_df, test_df).

        Raises:
            ValueError: If size fractions don't sum to 1.
        """
        if not np.isclose(train_size + val_size + test_size, 1.0):
            raise ValueError("train_size + val_size + test_size must equal 1.0")

        stratify = df[stratify_col] if stratify_col else None

        # First split: train vs (val + test)
        train_df, temp_df = train_test_split(
            df, train_size=train_size, random_state=random_state, stratify=stratify
        )
        val_df = temp_df

        return train_df, val_df

    @staticmethod
    def cell_line_holdout_split(
        df: pd.DataFrame,
        cl_col: str = "cell_line",
        holdout_fraction: float = 0.2,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data by holding out specific cell lines for testing.

        This creates a more realistic evaluation where the model must predict
        on cell lines it has never seen during training.

        Args:
            df: Input DataFrame containing experimental data.
            cl_col: Column name containing cell line identifiers.
            holdout_fraction: Fraction of cell lines to hold out for testing.
            random_state: Random seed for reproducibility.

        Returns:
            Tuple of (train_df, test_df).
        """
        unique_cell_lines = df[cl_col].unique()
        n_holdout = int(len(unique_cell_lines) * holdout_fraction)

        np.random.seed(random_state)
        holdout_cls = np.random.choice(unique_cell_lines, size=n_holdout, replace=False)

        test_df = df[df[cl_col].isin(holdout_cls)].copy()
        train_df = df[~df[cl_col].isin(holdout_cls)].copy()

        return train_df, test_df

    @staticmethod
    def intervention_holdout_split(
        df: pd.DataFrame,
        iv_cols: Union[str, List[str]] = "iv1",
        holdout_fraction: float = 0.2,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data by holding out specific interventions for testing.

        This evaluates how well the model can predict effects of novel
        interventions not seen during training.

        Args:
            df: Input DataFrame containing experimental data.
            iv_cols: Column name(s) containing intervention identifiers.
            holdout_fraction: Fraction of interventions to hold out for testing.
            random_state: Random seed for reproducibility.

        Returns:
            Tuple of (train_df, test_df).
        """
        if isinstance(iv_cols, str):
            iv_cols = [iv_cols]

        # Get unique intervention combinations
        intervention_combinations = df[iv_cols].drop_duplicates()
        n_holdout = int(len(intervention_combinations) * holdout_fraction)

        np.random.seed(random_state)
        holdout_indices = np.random.choice(
            len(intervention_combinations), size=n_holdout, replace=False
        )
        holdout_interventions = intervention_combinations.iloc[holdout_indices]

        # Create test set with holdout interventions
        test_mask = pd.Series(False, index=df.index)
        for _, row in holdout_interventions.iterrows():
            mask = pd.Series(True, index=df.index)
            for col in iv_cols:
                mask &= df[col] == row[col]
            test_mask |= mask

        test_df = df[test_mask].copy()
        train_df = df[~test_mask].copy()

        return train_df, test_df

    @staticmethod
    def temporal_split(
        df: pd.DataFrame,
        time_col: str,
        split_date: Optional[str] = None,
        train_fraction: float = 0.8,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data based on temporal ordering.

        Useful when experimental data has temporal structure and you want
        to evaluate on future time points.

        Args:
            df: Input DataFrame with temporal information.
            time_col: Column name containing dates/timestamps.
            split_date: Specific date to split on (YYYY-MM-DD format).
                If None, uses train_fraction to determine split point.
            train_fraction: Fraction of earliest data for training
                (ignored if split_date is provided).

        Returns:
            Tuple of (train_df, test_df).
        """
        df_sorted = df.sort_values(time_col).copy()

        if split_date is not None:
            split_point = pd.to_datetime(split_date)
            train_df = df_sorted[df_sorted[time_col] < split_point].copy()
            test_df = df_sorted[df_sorted[time_col] >= split_point].copy()
        else:
            split_idx = int(len(df_sorted) * train_fraction)
            train_df = df_sorted.iloc[:split_idx].copy()
            test_df = df_sorted.iloc[split_idx:].copy()

        return train_df, test_df


class DataProcessor:
    """Utilities for preprocessing experimental data for Prophet training."""

    @staticmethod
    def normalize_readouts(
        df: pd.DataFrame,
        readout_col: str = "value",
        method: Literal["minmax", "standard", "robust"] = "minmax",
        clip_outliers: bool = True,
        outlier_percentiles: Tuple[float, float] = (1, 99),
    ) -> Tuple[pd.DataFrame, object]:
        """Normalize readout values for better model performance.

        Args:
            df: Input DataFrame with readout values.
            readout_col: Column name containing readout values.
            method: Normalization method ('minmax', 'standard', 'robust').
            clip_outliers: Whether to clip extreme outliers before normalization.
            outlier_percentiles: Percentile range for outlier clipping.

        Returns:
            Tuple of (normalized_df, scaler_object).
        """
        df_norm = df.copy()
        values = df_norm[readout_col].values.reshape(-1, 1)

        # Clip outliers if requested
        if clip_outliers:
            lower_bound = np.percentile(values, outlier_percentiles[0])
            upper_bound = np.percentile(values, outlier_percentiles[1])
            values = np.clip(values, lower_bound, upper_bound)

        # Apply normalization
        if method == "minmax":
            scaler = MinMaxScaler()
        elif method == "standard":
            scaler = StandardScaler()
        elif method == "robust":
            from sklearn.preprocessing import RobustScaler

            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        normalized_values = scaler.fit_transform(values)
        df_norm[readout_col] = normalized_values.flatten()

        return df_norm, scaler

    @staticmethod
    def handle_missing_values(
        df: pd.DataFrame,
        strategy: Literal["drop", "median", "mean"] = "drop",
        missing_threshold: float = 0.5,
    ) -> pd.DataFrame:
        """Handle missing values in experimental data.

        Args:
            df: Input DataFrame potentially containing missing values.
            strategy: How to handle missing values ('drop', 'median', 'mean').
            missing_threshold: If strategy is 'drop', drop rows with this fraction
                of missing values or more.

        Returns:
            DataFrame with missing values handled.
        """
        df_clean = df.copy()

        if strategy == "drop":
            # Drop rows with too many missing values
            missing_fraction = df_clean.isnull().sum(axis=1) / len(df_clean.columns)
            df_clean = df_clean[missing_fraction < missing_threshold]
            # Drop any remaining rows with missing values
            df_clean = df_clean.dropna()
        elif strategy == "median":
            df_clean = df_clean.fillna(df_clean.median())
        elif strategy == "mean":
            df_clean = df_clean.fillna(df_clean.mean())
        else:
            raise ValueError(f"Unknown missing value strategy: {strategy}")

        return df_clean

    @staticmethod
    def balance_dataset(
        df: pd.DataFrame,
        group_col: str,
        max_samples_per_group: Optional[int] = None,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """Balance dataset by sampling equal numbers from each group.

        Useful when certain experimental conditions are over-represented.

        Args:
            df: Input DataFrame to balance.
            group_col: Column to group by (e.g., 'phenotype', 'cell_line').
            max_samples_per_group: Maximum samples per group. If None,
                uses the size of the smallest group.
            random_state: Random seed for reproducibility.

        Returns:
            Balanced DataFrame.
        """
        if max_samples_per_group is None:
            max_samples_per_group = df[group_col].value_counts().min()

        balanced_dfs = []
        for group in df[group_col].unique():
            group_df = df[df[group_col] == group]
            if len(group_df) > max_samples_per_group:
                group_df = group_df.sample(
                    n=max_samples_per_group, random_state=random_state
                )
            balanced_dfs.append(group_df)

        return pd.concat(balanced_dfs, ignore_index=True)


class DataValidator:
    """Utilities for validating experimental data before training."""

    @staticmethod
    def validate_dataframe(
        df: pd.DataFrame,
        required_cols: List[str],
        iv_embeddings: Optional[pd.DataFrame] = None,
        cl_embeddings: Optional[pd.DataFrame] = None,
        iv_cols: Optional[List[str]] = None,
        cl_col: Optional[str] = None,
    ) -> Dict[str, List[str]]:
        """Validate experimental DataFrame for Prophet training.

        Args:
            df: DataFrame to validate.
            required_cols: List of required column names.
            iv_embeddings: Intervention embeddings for validation.
            cl_embeddings: Cell line embeddings for validation.
            iv_cols: Intervention column names.
            cl_col: Cell line column name.

        Returns:
            Dictionary with validation results and any issues found.
        """
        issues = {
            "missing_columns": [],
            "missing_interventions": [],
            "missing_cell_lines": [],
            "data_type_issues": [],
            "other_issues": [],
        }

        # Check required columns
        missing_cols = [col for col in required_cols if col not in df.columns]
        issues["missing_columns"] = missing_cols

        # Check for missing interventions
        if iv_embeddings is not None and iv_cols is not None:
            for iv_col in iv_cols:
                if iv_col in df.columns:
                    missing_ivs = set(df[iv_col].unique()) - set(iv_embeddings.index)
                    if missing_ivs:
                        issues["missing_interventions"].extend(list(missing_ivs))

        # Check for missing cell lines
        if cl_embeddings is not None and cl_col is not None:
            if cl_col in df.columns:
                missing_cls = set(df[cl_col].unique()) - set(cl_embeddings.index)
                if missing_cls:
                    issues["missing_cell_lines"].extend(list(missing_cls))

        # Check data types
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if "value" in df.columns and "value" not in numeric_cols:
            issues["data_type_issues"].append("'value' column should be numeric")

        # Check for duplicates
        if df.duplicated().any():
            issues["other_issues"].append(
                f"Found {df.duplicated().sum()} duplicate rows"
            )

        # Check for missing values
        missing_counts = df.isnull().sum()
        if missing_counts.any():
            missing_info = missing_counts[missing_counts > 0].to_dict()
            issues["other_issues"].append(f"Missing values found: {missing_info}")

        return issues

    @staticmethod
    def print_validation_report(issues: Dict[str, List[str]]) -> None:
        """Print a formatted validation report.

        Args:
            issues: Dictionary of validation issues from validate_dataframe.
        """
        print("=== Data Validation Report ===")

        total_issues = sum(len(issue_list) for issue_list in issues.values())
        if total_issues == 0:
            print("✅ No issues found - data looks good!")
            return

        print(f"⚠️  Found {total_issues} potential issues:")

        for category, issue_list in issues.items():
            if issue_list:
                print(f"\n{category.replace('_', ' ').title()}:")
                for issue in issue_list:
                    print(f"  - {issue}")

        print("\nRecommendations:")
        if issues["missing_columns"]:
            print("  - Add missing columns or check column naming")
        if issues["missing_interventions"] or issues["missing_cell_lines"]:
            print("  - Update embeddings or filter data to remove missing entities")
        if issues["data_type_issues"]:
            print("  - Convert columns to appropriate data types")
        if any("duplicate" in str(issue) for issue in issues["other_issues"]):
            print("  - Remove duplicate rows: df.drop_duplicates()")
        if any("Missing values" in str(issue) for issue in issues["other_issues"]):
            print(
                "  - Handle missing values using DataProcessor.handle_missing_values()"
            )


def create_cross_validation_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    split_type: Literal["random", "cell_line", "intervention"] = "random",
    group_col: Optional[str] = None,
    random_state: int = 42,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Create cross-validation splits for robust model evaluation.

    Args:
        df: Input DataFrame to split.
        n_splits: Number of CV folds.
        split_type: Type of split strategy.
        group_col: Column to group by for stratified splits.
        random_state: Random seed for reproducibility.

    Returns:
        List of (train_df, val_df) tuples for each fold.
    """
    if split_type == "random":
        if group_col is not None:
            kfold = StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=random_state
            )
            splits = kfold.split(df, df[group_col])
        else:
            kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            splits = kfold.split(df)

        cv_splits = []
        for train_idx, val_idx in splits:
            train_df = df.iloc[train_idx].copy()
            val_df = df.iloc[val_idx].copy()
            cv_splits.append((train_df, val_df))

    elif split_type == "cell_line":
        # Split by cell lines
        unique_cls = df[group_col or "cell_line"].unique()
        np.random.seed(random_state)
        np.random.shuffle(unique_cls)

        fold_size = len(unique_cls) // n_splits
        cv_splits = []

        for i in range(n_splits):
            start_idx = i * fold_size
            end_idx = (i + 1) * fold_size if i < n_splits - 1 else len(unique_cls)
            val_cls = unique_cls[start_idx:end_idx]

            val_df = df[df[group_col or "cell_line"].isin(val_cls)].copy()
            train_df = df[~df[group_col or "cell_line"].isin(val_cls)].copy()
            cv_splits.append((train_df, val_df))

    else:
        raise ValueError(f"Unknown split_type: {split_type}")

    return cv_splits
