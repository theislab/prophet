from sklearn.model_selection import LeaveOneGroupOut, KFold
from torch.utils.data import DataLoader
from dataset import PhenotypeDataset
from typing import Literal, List, Tuple, Optional
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import warnings
import pandas as pd
from functools import reduce
import math

SEED = 42  # the true, baseline seed (that sets test splits)

def _choose(a, size, seed):
    """Guaranteed deterministic choosing."""
    np.random.seed(seed)  # reset the generator
    return np.random.choice(a, size=size, replace=False)



def dataloader_phenotypes(
    gene_embedding: List[pd.DataFrame],
    cell_lines_embedding: List[pd.DataFrame],
    phenotype_embedding: List[pd.DataFrame],
    data_label,
    index,
    batch_size = 2048,
    label_name=None,
    unbalanced: bool = False,
    torch_dataset: bool = True,
    pert_len: int = 2,
    valid_set: bool = True, # no idea Yuge, check this
    test_set: bool = True,
    phenotypes: list = None,
) -> List[Tuple[DataLoader, DataLoader, DataLoader, np.array, np.array]] :
    """Dataloader for multiple sources of information

    Args:
        gene_embedding (pd.DataFrame): index needs to be gene or gRNA
        cell_lines_embedding (pd.DataFrame): index needs to be cancer cell line name
        phenotype_embedding (pd.DataFrame): index needs to be phenotype name
        data_label (_type_): experimental data with columns that match indices in gene_embedding dataframes and cell line embedings dataframes
        label_name (_type_): label to predict
        indices (_type_, optional): _description_. Defaults to None.
        batch_size (_type_, optional): -1 to indicate work with sklearn
        embedding (Optional[List[bool]], optional): whether gene and cell_line should send index for embedding of vectors. [F, F] send vectors, [F, T] sends vector for genes and index for cell_line
        pert_len (int): number of perturbations to give to the model
        phenotypes (list): Pass a list of phenotypes to force indexing to occur correctly at prediction time for a pytorch model. If None, automatically determined from data_label.
    Returns:
        _type_: _description_
    """
    train_indices, valid_indices, test_indices, cl_holdout = index
    if len(valid_indices) == 0:
        valid_set = False
    if len(test_indices) == 0:
        test_set = False

    # accounting for multiple datasets
    test_dict = None
    if type(test_indices) == dict:
        test_dict = test_indices.copy()
        test_indices = test_indices['all']

    # value checks
    if 'type' not in gene_embedding.columns:
        raise ValueError("No column 'type' in gene_embedding")

    if not torch_dataset:
        # Note: this doesn't evaluate on multiple test sets because probably it will never be used at scale
        # create input dataframes
        # ge = pd.concat(gene_embeddings, axis=1).dropna()
        ge = gene_embedding.dropna()
        # ce = pd.concat(cell_lines_embeddings, axis=1).dropna()
        ce = cell_lines_embedding.dropna()
        ge = ge.drop(columns=['type'])

        if phenotype_embedding is not None:
            pe = phenotype_embedding.dropna()
            X_phenotype = pe.loc[data_label.phenotype].to_numpy()
        else:
            X_phenotype = pd.get_dummies(data_label['phenotype']).astype(float).values  # 1he
        X_iv = np.concatenate(
            [ge.loc[data_label[f'iv{i}']].to_numpy() for i in range(1, pert_len + 1)]
            , axis=1)
        X_cellline = ce.loc[data_label.cell_line].to_numpy()
        if label_name is None:  
            return [
                (np.concatenate([X_phenotype[idxs], X_cellline[idxs], X_iv[idxs]], axis=1),
                None) \
                for idxs in [train_indices, valid_indices, test_indices]]
        else:
            y_label = data_label[label_name].to_numpy()

            return [
                (np.concatenate([X_phenotype[idxs], X_cellline[idxs], X_iv[idxs]], axis=1),
                y_label[idxs]) \
                for idxs in [train_indices, valid_indices, test_indices]]

    data = data_label.copy()
    if phenotypes is None:
        phenotypes = sorted(list(data_label.phenotype.unique()))
        
        # for i in range(len(phenotypes)):
        #     print(f"Phenotype {i} is {phenotypes[i]}")

    train_set = PhenotypeDataset(
        experimental_data = data.loc[train_indices], 
        label_key = label_name,
        iv_embeddings = gene_embedding,
        cell_line_embeddings = cell_lines_embedding,
        phenotype_embeddings = phenotype_embedding if phenotype_embedding is not None else None, # if it's None, None[0] will return error
        phenotypes = phenotypes,
        pert_len=pert_len
        ) 
    if valid_set:
        valid_set = PhenotypeDataset( 
            experimental_data = data.loc[valid_indices], 
            label_key = label_name,
            iv_embeddings = gene_embedding,
            cell_line_embeddings = cell_lines_embedding,
            phenotype_embeddings = phenotype_embedding if phenotype_embedding is not None else None,
            phenotypes = phenotypes,
            pert_len=pert_len
            )            
        valid_dataloader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=4)
    else:
        valid_dataloader = None
     
    if test_set:
        test_set = PhenotypeDataset( 
                experimental_data = data.loc[test_indices], 
                label_key = label_name,
                iv_embeddings = gene_embedding,
                cell_line_embeddings = cell_lines_embedding,
                phenotype_embeddings = phenotype_embedding if phenotype_embedding is not None else None,
                phenotypes = phenotypes,
                pert_len=pert_len
                )
        test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4)
        # convert to dict if there are multiple test sets
        if test_dict is not None:
            test_dl_dict = {'all': test_dataloader}
            for k, test_indices in test_dict.items():
                if k == 'all':  # already loaded in test_dataloader
                    pass
                test_set = PhenotypeDataset(
                    experimental_data = data.loc[test_indices],
                    label_key = label_name,
                    iv_embeddings = gene_embedding,
                    cell_line_embeddings = cell_lines_embedding,
                    phenotype_embeddings = phenotype_embedding if phenotype_embedding is not None else None,
                    phenotypes = phenotypes,
                    pert_len=pert_len
                )
                test_dl_dict[k] = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4)

    else:
        test_dataloader = None

    if unbalanced: # unbalanced means that one phenotype is way more measured that the other (not sure whether this should be implemented tbh)

        # # Count the occurrences of each class
        key = 'phenotype'
        if 'dataset' in data.columns:
            key = 'dataset'
        class_counts = data.loc[train_indices][key].value_counts()
        num_samples = len(data)
        class_weights = [num_samples / class_counts.values[i] for i in range(len(class_counts))]
        scaling_factor = 1 / min(class_weights)
        class_weights = [x * scaling_factor for x in class_weights]
        class_weight_dict = dict(zip(class_counts.index, class_weights))

        # Create a custom WeightedRandomSampler to oversample the minority class
        weights = [class_weight_dict[c] for c in data.loc[train_indices][key].values]
        train_sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        train_dataloader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, num_workers=4)
    else:
        train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)

    if test_dict:
        return (train_dataloader, valid_dataloader, test_dl_dict, train_indices, test_indices, cl_holdout)
    else:
        return (train_dataloader, valid_dataloader, test_dataloader, train_indices, test_indices, cl_holdout)

def read_in_priors(prior_files):
    """Convert list of filenames to dfs. When presented with multiple files, assumes they contain different
    indices and adds additional rows. Assumes columns which contain the same feature are named accordingly."""
    prior = []
    if isinstance(prior_files, str):
        prior_files = [prior_files]
    for file in range(len(prior_files)):
        emb = pd.read_csv(prior_files[file], index_col=0)
        emb.index = emb.index.astype(str)
        prior.append(emb)

    return pd.concat(prior).fillna(0) if len(prior) > 0 else None

def process_priors(genes_prior, cell_lines_prior, phenotype_prior):
    gene_prior = read_in_priors(genes_prior)
    try:
        gene_prior = gene_prior.set_index('smiles')
    except KeyError:  # still works even if there's no smiles column, like for genetic interventions
        pass
    gene_prior.index = [str(x).lower() for x in gene_prior.index]  # allow translatability across organisms and drugs
    cl_prior = read_in_priors(cell_lines_prior)
    phe_prior = None
    if phenotype_prior is not None:
        phe_prior = read_in_priors(phenotype_prior)

    if gene_prior is not None and "type" not in gene_prior.columns:
        raise ValueError("type not in iv_embedding columns")

    # add 0 for control
    gene_prior.loc['negative_gene'] = 0
    gene_prior.loc['negative_drug'] = 0
    gene_prior.loc['negative_gene', 'type'] = 'gene'
    gene_prior.loc['negative_drug', 'type'] = 'drug'

    # I wanted to remove duplicate smiles for efficient runtime but idk what this actually does
    # gene_prior = gene_prior[~gene_prior.index.duplicated(keep='first')]

    return gene_prior, cl_prior, phe_prior


def remove_nonexistent_cat(data_label, prior, columns, verbose=True):
    """Takes in a cell line or gene prior embedding dataframe and removes rows from
    data_label where the embedding doesn't exist.

    Parameters
    ----------
    data_label : pandas.DataFrame
        A dataframe with phenotype, cell context, and iv columns. This dataframe is modified by the function.
    prior : pandas.DataFrame
        A dataframe representing prior embeddings. The index of this dataframe should contain the categories to be filtered on.
    columns : list[str]
        A list of column names in `data_label` where the values are checked against the categories in `prior`.

    Returns
    -------
    pandas.DataFrame
    """
    if isinstance(columns, str):
        columns = [columns]
    data_label_cats = reduce(lambda x, y: np.union1d(x, y), [data_label[col].astype(str).values for col in columns])
    emb_cats = prior.index.to_list()
    strings_to_remove = list(np.setdiff1d(data_label_cats, emb_cats))
    for col in columns:
        data_label = data_label[~data_label[col].isin(strings_to_remove)]
    if verbose:
        print(f"Removing {len(strings_to_remove)} such as {strings_to_remove[:5]} from {columns}. {data_label.shape[0]} rows remaining.", flush=True)
    return data_label

def check_iv_emb(emb):
    if "type" not in emb.columns:
        raise KeyError("Intervention embedding has no `type` in columns.")

def check_data(data_label):
    cols = set(data_label.columns)
    needed = set(["phenotype", "cell_line", "iv1"])
    if not needed.issubset(cols):
        raise KeyError(f"Cols is missing {cols-needed}")
