import pytest
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from prophet.dataloader import (
    dataloader_phenotypes,
    process_priors,
    remove_nonexistent_cat,
    universal_processing,
)


# Mock data
@pytest.fixture
def mock_data():
    iv_embedding = pd.DataFrame(
        {
            "type": ["gene", "gene", "drug", "gene", "drug"],
            "feature1": [0.1, 0.2, 0.3, 0.0, 0.0],
            "feature2": [0.4, 0.5, 0.6, 0.0, 0.0],
        },
        index=["gene1", "gene2", "drug1", "negative_gene", "negative_drug"],
    )  # assume embedding for drug2 is not exist

    cell_lines_embedding = pd.DataFrame(
        {"feature1": [0.7, 0.8], "feature2": [0.9, 1.0]},
        index=["cell_line_1", "cell_line_2"],
    )

    phenotype_embedding = None  # usually None

    data_label = pd.DataFrame(
        {
            "phenotype": ["phenotype1", "phenotype2", "phenotype1", "phenotype2"],
            "cell_line": ["cell_line_1", "cell_line_2", "cell_line_1", "cell_line_2"],
            "iv1": ["gene1", "gene2", "drug1", "drug2"],
            "iv2": ["negative_gene", "negative_gene", "negative_drug", "negative_drug"],
            "value": [0.5, 0.6, 0.9, 0.3],
        }
    )

    index = (
        np.array([0, 2]),  # train_indices
        np.array([1]),  # validation_indices
        [],  # test_indices
        None,  # cl_holdout
    )

    return iv_embedding, cell_lines_embedding, phenotype_embedding, data_label, index


def test_dataloader_phenotypes(mock_data):
    iv_embedding, cell_lines_embedding, phenotype_embedding, data_label, index = (
        mock_data
    )

    # Just filter out rows where iv1 or iv2 values that dont have embeddings
    valid_iv1 = data_label["iv1"].isin(iv_embedding.index)
    valid_iv2 = data_label["iv2"].isin(iv_embedding.index)
    data_label = data_label[valid_iv1 & valid_iv2]

    (
        train_dataloader,
        valid_dataloader,
        test_dataloader,
        train_indices,
        test_indices,
        cl_holdout,
    ) = dataloader_phenotypes(
        gene_embedding=iv_embedding,
        cell_lines_embedding=cell_lines_embedding,
        phenotype_embedding=phenotype_embedding,
        data_label=data_label,
        index=index,
        batch_size=2,
        label_name="value",
        unbalanced=False,
        torch_dataset=True,
        valid_set=True,
    )

    assert isinstance(train_dataloader, DataLoader)
    assert len(train_dataloader.dataset) == 2  # indices: 0,2

    assert isinstance(valid_dataloader, DataLoader)
    assert len(valid_dataloader.dataset) == 1  # indices: 1

    assert test_dataloader is None  # no test samples given


def test_process_priors(mock_data):
    iv_embedding, cell_lines_embedding, phenotype_embedding, _, _ = mock_data

    # setup temp paths
    iv_path = "./iv_embedding.csv"
    cl_path = "./cell_lines_embedding.csv"
    ph_path = "./phenotype_embedding.csv"

    iv_embedding.to_csv(iv_path)
    cell_lines_embedding.to_csv(cl_path)
    if (
        phenotype_embedding is not None
    ):  # create dummy files for testing of process_priors
        pd.DataFrame().to_csv(ph_path)
    else:
        ph_path = None

    iv_prior, cl_prior, phe_prior = process_priors(
        genes_prior=[str(iv_path)],
        cell_lines_prior=[str(cl_path)],
        phenotype_prior=[str(ph_path)] if ph_path else None,
    )

    assert "negative_gene" in iv_prior.index
    assert "negative_drug" in iv_prior.index
    assert iv_prior.loc["negative_gene", "type"] == "gene"

    assert cl_prior.shape == cell_lines_embedding.shape

    assert phe_prior == None


def test_remove_nonexistent_cat(mock_data):
    iv_embedding, cell_lines_embedding, phenotype_embedding, data_label, _ = mock_data
    embeddings = [iv_embedding, cell_lines_embedding, phenotype_embedding]

    iv_cols = ["iv1", "iv2"]
    cl_col = "cell_line"
    ph_col = "phenotype"
    cols = [iv_cols, cl_col, ph_col]

    prior = pd.DataFrame(index=["gene1", "gene2", "drug1"])
    print("embeddings:", embeddings)

    for i, embedding in enumerate(embeddings):
        if embedding is None:  # phenotype embedding can be None
            continue
        data_label = remove_nonexistent_cat(
            data_label, embedding, cols[i], verbose=True
        )

    assert (
        len(data_label) == 3
    )  # 4-1= 3 According to mock embeddings and  datalabel, one row contains
    # "drug2" should be removed since its embedding does not exist!
    assert "gene1" in data_label["iv1"].values
    assert "gene2" in data_label["iv1"].values
    assert "drug1" in data_label["iv1"].values
    assert "drug2" not in data_label["iv1"].values  # removed


def test_universal_processing(mock_data):
    _, _, _, data_label, _ = mock_data

    processed_data = universal_processing(data_label)

    assert "phenotype" in processed_data.columns
    assert "value" in processed_data.columns
    assert "iv1" in processed_data.columns
    assert "iv2" in processed_data.columns
    assert len(processed_data) == 8  # original (4) + flipped rows (4)
