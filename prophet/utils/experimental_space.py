"""
Experimental space utilities for Prophet predictions.

This module provides functions to convert Prophet prediction results into
matrix formats suitable for downstream analysis and visualization.
"""

import pandas as pd
import numpy as np
from typing import Optional, Union, Literal, Tuple
import warnings

try:
    import anndata as ad

    ANNDATA_AVAILABLE = True
except ImportError:
    ANNDATA_AVAILABLE = False
    ad = None


def create_experimental_space(
    predictions_df: pd.DataFrame,
    intervention_col: str = "iv1",
    cell_line_col: str = "cell_line",
    prediction_col: str = "pred",
    phenotype_col: Optional[str] = "phenotype",
    orientation: Literal[
        "interventions_x_cells", "cells_x_interventions"
    ] = "interventions_x_cells",
    output_format: Literal["anndata", "dataframe"] = "anndata",
    fill_missing: float = np.nan,
    aggregate_func: str = "mean",
) -> Union[pd.DataFrame, "anndata.AnnData"]:
    """
    Convert Prophet predictions into experimental space matrix format.

    Takes Prophet prediction results (long format) and converts them into a matrix
    where either interventions or cell lines are observations, suitable for
    downstream analysis and visualization.

    Args:
        predictions_df: DataFrame with Prophet predictions containing at least
            intervention, cell line, and prediction columns.
        intervention_col: Column name containing intervention identifiers.
        cell_line_col: Column name containing cell line identifiers.
        prediction_col: Column name containing prediction values.
        phenotype_col: Column name containing phenotype identifiers. If provided,
            creates separate matrices for each phenotype.
        orientation: Matrix orientation:
            - "interventions_x_cells": interventions as rows, cell lines as columns (m × n)
            - "cells_x_interventions": cell lines as rows, interventions as columns (n × m)
        output_format: Output format:
            - "anndata": Returns AnnData object (requires scanpy/anndata)
            - "dataframe": Returns pandas DataFrame
        fill_missing: Value to use for missing intervention-cell line combinations.
        aggregate_func: Function to use when multiple predictions exist for the same
            intervention-cell line combination ("mean", "median", "min", "max").

    Returns:
        Either AnnData object or DataFrame with interventions/cell lines as observations
        and cell lines/interventions as features.

    Examples:
        Basic usage with DataFrame output:
        >>> predictions = model.predict(target_ivs=["DRUG1", "DRUG2"],
        ...                            target_cls=["CELL1", "CELL2"], save=False)
        >>> matrix_df = create_experimental_space(predictions, output_format="dataframe")
        >>> # Result: 2×2 DataFrame with drugs as rows, cell lines as columns

        Create AnnData for downstream analysis:
        >>> adata = create_experimental_space(predictions, orientation="cells_x_interventions")
        >>> # Result: AnnData with cell lines as observations, interventions as variables

        Handle multiple phenotypes:
        >>> # If predictions contain multiple phenotypes, creates separate matrices
        >>> adata = create_experimental_space(predictions, phenotype_col="phenotype")

    Raises:
        ImportError: If output_format="anndata" but anndata is not installed.
        ValueError: If required columns are missing from predictions_df.
        ValueError: If orientation or output_format values are invalid.
    """

    # Validate inputs
    required_cols = [intervention_col, cell_line_col, prediction_col]
    missing_cols = [col for col in required_cols if col not in predictions_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if output_format == "anndata" and not ANNDATA_AVAILABLE:
        raise ImportError(
            "AnnData output requires anndata package. Install with: "
            "pip install anndata, or use output_format='dataframe'"
        )

    # Work with a copy to avoid modifying original
    df = predictions_df.copy()

    # Handle multiple phenotypes
    if phenotype_col and phenotype_col in df.columns:
        phenotypes = df[phenotype_col].unique()
        if len(phenotypes) > 1:
            print(f"Found {len(phenotypes)} phenotypes: {phenotypes}")
            print("Creating separate matrices for each phenotype...")

            results = {}
            for phenotype in phenotypes:
                pheno_df = df[df[phenotype_col] == phenotype].copy()
                result = create_experimental_space(
                    pheno_df,
                    intervention_col=intervention_col,
                    cell_line_col=cell_line_col,
                    prediction_col=prediction_col,
                    phenotype_col=None,  # Don't recurse
                    orientation=orientation,
                    output_format=output_format,
                    fill_missing=fill_missing,
                    aggregate_func=aggregate_func,
                )
                results[phenotype] = result
            return results

    # Aggregate duplicates if present
    group_cols = [intervention_col, cell_line_col]
    if len(df) > len(df[group_cols].drop_duplicates()):
        warnings.warn(
            f"Found duplicate intervention-cell line combinations. "
            f"Aggregating using {aggregate_func}."
        )
        agg_funcs = {"mean": "mean", "median": "median", "min": "min", "max": "max"}
        if aggregate_func not in agg_funcs:
            raise ValueError(f"aggregate_func must be one of {list(agg_funcs.keys())}")

        df = (
            df.groupby(group_cols)[prediction_col]
            .agg(agg_funcs[aggregate_func])
            .reset_index()
        )

    # Create pivot table
    pivot_df = df.pivot(
        index=intervention_col, columns=cell_line_col, values=prediction_col
    )

    # Fill missing values
    if not pd.isna(fill_missing):
        pivot_df = pivot_df.fillna(fill_missing)

    # Handle orientation
    if orientation == "cells_x_interventions":
        pivot_df = pivot_df.T
    elif orientation != "interventions_x_cells":
        raise ValueError(
            f"orientation must be 'interventions_x_cells' or 'cells_x_interventions', "
            f"got '{orientation}'"
        )

    # Return appropriate format
    if output_format == "dataframe":
        return pivot_df
    elif output_format == "anndata":
        # Create AnnData object
        adata = ad.AnnData(X=pivot_df.values)
        adata.obs_names = pivot_df.index.astype(str)
        adata.var_names = pivot_df.columns.astype(str)

        # Add metadata
        if orientation == "interventions_x_cells":
            adata.obs["intervention"] = adata.obs_names
            adata.var["cell_line"] = adata.var_names
        else:
            adata.obs["cell_line"] = adata.obs_names
            adata.var["intervention"] = adata.var_names

        # Add information to uns
        adata.uns["prophet_prediction"] = {
            "orientation": orientation,
            "intervention_col": intervention_col,
            "cell_line_col": cell_line_col,
            "prediction_col": prediction_col,
            "n_interventions": len(pivot_df.index)
            if orientation == "interventions_x_cells"
            else len(pivot_df.columns),
            "n_cell_lines": len(pivot_df.columns)
            if orientation == "interventions_x_cells"
            else len(pivot_df.index),
            "fill_missing": fill_missing,
            "aggregate_func": aggregate_func,
        }

        return adata
    else:
        raise ValueError(
            f"output_format must be 'dataframe' or 'anndata', got '{output_format}'"
        )


def experimental_space_summary(
    experimental_space: Union[pd.DataFrame, "anndata.AnnData"],
) -> None:
    """
    Print a summary of the experimental space matrix.

    Args:
        experimental_space: Matrix created by create_experimental_space.
    """
    if hasattr(experimental_space, "X"):  # AnnData object
        matrix = experimental_space.X
        obs_names = experimental_space.obs_names
        var_names = experimental_space.var_names
        metadata = experimental_space.uns.get("prophet_prediction", {})

        print("🧬 Prophet Experimental Space (AnnData)")
        print("=" * 50)
        print(
            f"Observations: {len(obs_names)} ({metadata.get('orientation', 'unknown').split('_x_')[0]})"
        )
        print(
            f"Variables: {len(var_names)} ({metadata.get('orientation', 'unknown').split('_x_')[1]})"
        )
        print(f"Matrix shape: {matrix.shape}")
        print(f"Data type: {matrix.dtype}")

        if not np.isnan(matrix).all():
            print(f"Value range: {np.nanmin(matrix):.3f} - {np.nanmax(matrix):.3f}")
            print(
                f"Missing values: {np.isnan(matrix).sum()} ({np.isnan(matrix).mean() * 100:.1f}%)"
            )
        else:
            print("All values are NaN")

    else:  # DataFrame
        print("📊 Prophet Experimental Space (DataFrame)")
        print("=" * 50)
        print(f"Shape: {experimental_space.shape}")
        print(f"Rows (observations): {len(experimental_space.index)}")
        print(f"Columns (features): {len(experimental_space.columns)}")
        print(f"Data type: {experimental_space.dtypes.iloc[0]}")

        values = experimental_space.values
        if not np.isnan(values).all():
            print(f"Value range: {np.nanmin(values):.3f} - {np.nanmax(values):.3f}")
            print(
                f"Missing values: {np.isnan(values).sum()} ({np.isnan(values).mean() * 100:.1f}%)"
            )
        else:
            print("All values are NaN")

    print("\n💡 Use this matrix for:")
    print("  - Heatmap visualization")
    print("  - Clustering analysis")
    print("  - Dimensionality reduction")
    print("  - Statistical analysis")
    print("  - Integration with other omics data")


def save_experimental_space(
    experimental_space: Union[pd.DataFrame, "anndata.AnnData"],
    filename: str,
    format: Optional[str] = None,
) -> None:
    """
    Save experimental space matrix to file.

    Args:
        experimental_space: Matrix to save.
        filename: Output filename.
        format: File format ('csv', 'h5ad', 'excel'). If None, inferred from filename.
    """
    if format is None:
        if filename.endswith(".h5ad"):
            format = "h5ad"
        elif filename.endswith(".csv"):
            format = "csv"
        elif filename.endswith((".xlsx", ".xls")):
            format = "excel"
        else:
            format = "csv"
            filename += ".csv"

    if hasattr(experimental_space, "X"):  # AnnData
        if format == "h5ad":
            experimental_space.write(filename)
        elif format == "csv":
            pd.DataFrame(
                experimental_space.X,
                index=experimental_space.obs_names,
                columns=experimental_space.var_names,
            ).to_csv(filename)
        elif format == "excel":
            pd.DataFrame(
                experimental_space.X,
                index=experimental_space.obs_names,
                columns=experimental_space.var_names,
            ).to_excel(filename)
    else:  # DataFrame
        if format == "csv":
            experimental_space.to_csv(filename)
        elif format == "excel":
            experimental_space.to_excel(filename)
        elif format == "h5ad":
            raise ValueError("Cannot save DataFrame as h5ad. Convert to AnnData first.")

    print(f"✅ Saved experimental space to {filename}")


# Convenience functions for common use cases
def interventions_x_cells(
    predictions_df: pd.DataFrame, **kwargs
) -> Union[pd.DataFrame, "anndata.AnnData"]:
    """Create interventions × cell lines matrix (m × n)."""
    return create_experimental_space(
        predictions_df, orientation="interventions_x_cells", **kwargs
    )


def cells_x_interventions(
    predictions_df: pd.DataFrame, **kwargs
) -> Union[pd.DataFrame, "anndata.AnnData"]:
    """Create cell lines × interventions matrix (n × m)."""
    return create_experimental_space(
        predictions_df, orientation="cells_x_interventions", **kwargs
    )
