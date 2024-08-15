import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class PhenotypeDataset(Dataset):
    """
    Dataset that gathers multiple phenotypes. The splits are done before. Experimental_data is the dataframe with the train, test or validation data.
    Each train, test or validation data is a different PhenotypeDataset dataset with different 'experimental_data' according to the splits
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
        pert_len: int = 2
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
        # precompute the attention mask
        self.attn_mask = [[False]*experimental_data.shape[0]] # always pay attention to CLS, which is first token
        for i in range(1, pert_len + 1):
            col = f'iv{i}'
            mask_values = experimental_data[col].isin(['negative_drug', 'negative_gene']).values  # mask if negative
            self.attn_mask.append(mask_values)
        self.attn_mask.append([False]*experimental_data.shape[0])  # once for cell_line
        self.attn_mask.append([False]*experimental_data.shape[0])  # once for phenotype
        self.attn_mask = np.array(self.attn_mask).T

        columns = ['cell_line', 'phenotype'] + [f'iv{i}' for i in range(1, pert_len + 1)]
        self.experimental_data = experimental_data[columns].values  # ordered
        self.labels = experimental_data[label_key].values
        self.iv = iv_embeddings.values
        self.cell_line = cell_line_embeddings.values
        self.iv_to_index = dict(zip(iv_embeddings.index, range(iv_embeddings.shape[0])))
        self.cl_to_index = dict(zip(cell_line_embeddings.index, range(cell_line_embeddings.shape[0])))

        # special handling for phenotypes
        self.ph_to_index = dict(zip(phenotypes, range(len(phenotypes))))
        if phenotype_embeddings is not None:
            phenotype_embeddings = phenotype_embeddings.T[phenotypes].T  # reorder the embedding so that ph_to_index matches
            self.phenotype_embeddings = phenotype_embeddings.values
        else:
            self.phenotype_embeddings = None
        
        self.pert_len = pert_len
        
        # print("Interventions: ", self.iv.shape)
        # print("Cell line: ", self.cell_line.shape)
        # print("Order of phenotypes: ", phenotypes)
        # print("Don't using explicit phenotype embeddings") if self.phenotype_embeddings is None else print(f"Explicit phenotype {self.phenotype_embeddings.shape} was passed")

    def __len__(self):
        return len(self.experimental_data)

    def __getitem__(self, idx):
        """
        Returns a dictionary with:
            phenotype: index to query
            cell_line: embedding
            label: scalar to predict
            names: names of the perturbations
            idx: index of the observation
            pert_type: list that says whether perturbations are genes or drugs
            **iv_values_dict: dictionary with keys iv1, iv2 etc. and respective embedding values. Same size as pert_len.
        """

        item = self.experimental_data[idx]
        cell_line = item[0]
        phenotype = item[1]

        iv_values_dict = {}
        iv_type = []
        for i in range(2, self.pert_len + 2):
            name = item[i]  # perturbation name
            emb_entry = self.iv[self.iv_to_index[name]]
            iv_type.append(emb_entry[0]) # first item of the embedding is 'gene' or 'drug' - this could probably have been precomputed
            iv_values_dict[f'iv{i-1}'] = emb_entry[1:].astype('float64') # use all dimensions but the first one
                
        # if there isn't phenotype embedding 
        if self.phenotype_embeddings is None:
            context = self.ph_to_index[phenotype] # retrieve index
            context = context + 1 # CLS is 0 
        else: # if there's embedding
            context = self.phenotype_embeddings[self.ph_to_index[phenotype]] # retrieve embedding

        # Gene = 0, Drug = 1
        iv_types = [0 if item == 'gene' else (1 if item == 'drug' else item) for item in iv_type]
        iv_types = np.array(iv_types)

        return {'phenotype': context, # sometimes an int, sometimes an embedidng
                'cell_line': self.cell_line[self.cl_to_index[cell_line]],
                'label': self.labels[idx],
                # 'names': names,
                'attn_mask': self.attn_mask[idx],
                'idx': idx,
                'pert_type': iv_types, 
                **iv_values_dict
        }