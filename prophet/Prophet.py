import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm
from typing import List, Union, Dict, Optional
from itertools import permutations
import functools
from pathlib import Path
from joblib import load
from sklearn.ensemble import RandomForestRegressor
from .dataloader import (
    dataloader_phenotypes,
    process_priors,
    remove_nonexistent_cat,
)
from .model import load_models_config, TransformerPredictor

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
    def __init__(
        self,
        iv_emb_path: Union[str, List[str]] = None,  # TODO default values
        cl_emb_path: Union[str, List[str]] = None,  # TODO default values
        ph_emb_path: Union[str, List[str]] = None,  # TODO default values
        model_pth=None,
        architecture="Transformer",
    ):
        """Initialize the Prophet model.

        Args:
            iv_emb_path (Union[str, List[str]], optional): The path to the gene embeddings. Defaults to None.
            cl_emb_path (Union[str, List[str]], optional): The path to the cell line embeddings. Defaults to None.
            ph_emb_path (Union[str, List[str]], optional): The path to the phenotype embeddings. Defaults to None.
            model_pth ([type], optional): The path to the trained model. Defaults to None.
            architecture (str, optional): The architecture of the model. Defaults to "Transformer".
        """

        self.architecture = architecture
        self.iv_emb_path = iv_emb_path
        self.cl_emb_path = cl_emb_path
        self.ph_emb_path = ph_emb_path
        # set phenotypes (must be in the same order regardless of what is passed in predict)
        self.phenotypes = list(pd.read_csv('./embeddings/phenotypes.csv', index_col=0).values.flatten())
        self.column_map = None
        self.pert_len = None

        if model_pth and architecture == "RandomForest":
            self.model = load(model_pth)
        else:
            self.model_pth = model_pth
            self.model = self._build_model(architecture)
            self.iv_embedding, self.cl_embedding, self.ph_embedding = process_priors(self.iv_emb_path, self.cl_emb_path, self.ph_emb_path)
            if self.model.hparams.explicit_phenotype and self.ph_embedding is None:
                raise ValueError('model was run with explicit phenotype! must pass a ph_emb_path')

    def _build_model(self, arch):
        if arch == "RandomForest":
            self.torch_dataset = False
            return RandomForestRegressor()
        elif arch == "Transformer":
            self.torch_dataset = True
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print('returning trained model!')
            model = TransformerPredictor.load_from_checkpoint(checkpoint_path=self.model_pth, map_location=device)
            model.eval()
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
            raise ValueError('labels did not match embeddings passed!')
        return data_label

    def _init_input(
        self,
        iv_col: Union[List[str], str] = ['iv1', 'iv2'],
        cl_col: str = "cell_line",
        ph_col: str = "phenotype",
        readout_col: str = "value",
    ):
        """Sets some state variables in the model, but is always overwritten by
        either train or predict.
        """
        if isinstance(iv_col, str):
            iv_col = [iv_col]
        intervention_mapping = {col: f"iv{i+1}" for i, col in enumerate(iv_col)}
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
            **intervention_mapping
        }
        if self.pert_len is None:
            self.pert_len = len(self.iv_cols)
        else:
            if self.pert_len != len(self.iv_cols):
                raise ValueError(f"Are you sure you passed the right number of intervention columns? Currently receiving {self.iv_cols}")

    def train(
        self,
        df: pd.DataFrame,
        iv_col: Union[List[str], str] = ['iv1', 'iv2'],
        cl_col: str = "cell_line",
        ph_col: str = "phenotype",
        readout_col: str = "value",
    ):
        """Train the Prophet model on the provided DataFrame.

        This function reformats the DataFrame according to the specified settings and intervention columns,
        then trains the model using the reformatted data.

        Warning: this has not yet been compared to train_model.py, use at your own risk.

        Args:
            df (pd.DataFrame): The DataFrame containing the experimental data.
            iv_col (Union[List[str], str]): The names of the intervention columns in the DataFrame. Can be a single column name or a list of names.
            cl_col (str, optional): The name of the column in df that contains the setting labels. Defaults to "cell_line".
            ph_col (str, optional): The name of the column in df that contains the phenotype labels. Defaults to "phenotype".
            readout_col (str, optional): The name of the column in df that contains the readout data. Defaults to "value".
            flip_iv_col (bool, optional): Whether to flip the intervention columns for data augmentation. Defaults to False. If using the Transformer architecture, should be turned off to save memory.
        """
        self._init_input(iv_col, cl_col, ph_col, readout_col)
        # user-friendly check that the columns were passed in correctly
        for _, (old_name, new_name) in enumerate(self.column_map.items()):
            if old_name not in df.columns:
                raise ValueError(f"{old_name} not in df columns.")

        df = df.rename(columns=self.column_map).copy()

        # Formatting
        df = df.drop_duplicates()
        df = df.reset_index(drop=True)

        ## generate training dataloader
        df = self._remove_nonexistent_cat(data_label=df, verbose=False)
        split = dataloader_phenotypes(
            gene_embedding=self.iv_embedding,
            cell_lines_embedding=self.cl_embedding,
            phenotype_embedding=self.ph_embedding if self.ph_embedding is not None else None,
            data_label=df,
            label_name="value",
            index=(
                np.array(df.index),
                [],
                [],
                "",
            ),  # (train, test, val, descr)
            torch_dataset=self.torch_dataset,
            pert_len=len(self.iv_cols)
        )

        print("Fitting model.")
        if not self.torch_dataset:
            X_train, y_train = split[2]
            self.model.fit(X_train, y_train)
        else:
            print('pytorch model, already fit')  # train does not currently support finetuning
            pass
            #self.model.fit(split[2])  # TODO: convert train_transformer to this or something

    def _generate_predict_df(self,
                             run_index: int,
                             num_iterations: int,
                             target_ivs: List[str],
                             target_cls: List[str],
                             target_phs: List[str] = ['_'],
                             ):
        subset_cl = pd.DataFrame(target_cls, columns=["cell_line"])
        subset_iv = pd.DataFrame(target_ivs, columns=["iv"])
        if target_phs is None:
            target_phs = ['_']
        subset_ph = pd.DataFrame(target_phs, columns=["phenotype"])
        if len(self.iv_cols) > 2:
            raise NotImplementedError("Only support 1 or 2 interventions if you input a list of interventions. Please create the data label dataframe yourself and input to predict()!")
        
        batch_size = int(len(subset_iv) // num_iterations)
        start_idx = run_index * batch_size
        end_idx = (
            (start_idx + batch_size)
            if (run_index < num_iterations - 1)
            else len(subset_iv["iv"])
        )

        data_label = pd.merge(subset_iv[["iv"]][start_idx:end_idx], subset_cl, how="cross")
        data_label = pd.merge(data_label, subset_ph, how="cross")
    
        if len(self.iv_cols) == 1:
            data_label.rename(columns={"iv": "iv1"}, inplace=True)
        else:
            data_label = pd.merge(subset_iv[["iv"]], data_label, how="cross", suffixes=("1", "2"))
            # A+B and B+A should be the same, so we remove all duplicates in favor of A+B (was pretty sure this shouldn't exist in the implementation @John)
            data_label['iv1+iv2'] = ['+'.join(sorted([row['iv1'], row['iv2']])) for _, row in data_label.iterrows()] # TODO: make sure that these column names at least always exist, probably in some variable somewhere
            data_label = data_label.drop_duplicates(subset=['iv1+iv2', 'cell_line', 'phenotype'])
        
        data_label['value'] = '_'
        
        return data_label
    
    def _decide_iteration_num(
        self,
        total_size: int,
        single_run_size: int = None,
        memory_size: int = None,
    ):

        # TODO decide single_run_size based on memory_size

        if total_size <= single_run_size:
            num_iterations = 1
        else:
            num_iterations = total_size // single_run_size

        return int(num_iterations)

    def predict(
        self,
        df: pd.DataFrame = None,
        target_ivs: Union[str, List[str]] = None,
        target_cls: Union[str, List[str]] = None,
        target_phs: Union[str, List[str]] = None,
        iv_col: Union[List[str], str] = ['iv1', 'iv2'],
        cl_col: str = "cell_line",
        ph_col: str = "phenotype",
        num_iterations: int = None,
        save: bool = True,
        filename: str = "Prophet_prediction",
    ):
        """Predict outcomes using the trained Prophet model.

        This function can take either a DataFrame or a combination of gene and cell line lists to make predictions.
        If a dataframe is passed, which columns correspond to which inputs must also be passed. If lists are passed,
        all combinations within are taken (not including combinations).

        Args:
            df (pd.DataFrame, optional): The DataFrame containing the data for prediction. If None, predictions will be made for all combinations of provided genes and cell lines.
            target_ivs (Union[str, List[str]], optional): The intervention or list of interventions for prediction. If df and target_ivs are both None, predictions will be made for all available treatments.
            target_cls (Union[str, List[str]], optional): The cell line or list of cell lines for prediction. If df and target_cls are both None, predictions will be made for all available cell lines.
            target_phs (Union[str, List[str]], optional): The phenotype for prediction. If None, it will be set to "_". 
            num_iterations (int, optional): The number of iterations to run the prediction. If None, it will be calculated automatically.
            save (bool, optional): Whether to save the prediction results to a file. If False, return the prediction result as dataframe. Defaults to True.
            filename (str, optional): The filename to save the prediction results. If None, a default name will be used.
        """
        if save:
            print(f"Saving results to {filename}.parquet")

        # If only pass target_cls and target_ivs
        if df is None:

            if isinstance(target_cls, str):
                target_cls = [target_cls]
            if isinstance(target_phs, str):
                target_phs = [target_phs]
            if isinstance(target_ivs, str):
                target_ivs = [target_ivs]
            
            if target_cls is None:
                target_cls = list(self.cl_embedding.index)
                warnings.warn(f"Trying to predict for all cell lines that are from {self.cl_embedding.index}!")
                print(f"There are {len(target_cls)} cell lines in total!")
            if target_ivs is None:
                target_ivs = list(self.iv_embedding.drop_duplicates().index)  # because there compounds with the same embedding but different
                warnings.warn(f"Trying to predict for all gene combinations that are from {self.iv_embedding.index}!")
                print(f"There are {len(target_ivs)} genes in total!")

            # in case the user wants to specify more than one intervention
            if iv_col is None:
                iv_col = ['iv1']

            total_size = len(target_ivs) * len(target_cls) * len(target_phs)
            self._init_input(iv_col, 'cell_line', 'phenotype', 'value')  # since we're constructing it ourselves
        # or pass a dataframe has similiar format with in train
        else:
            self._init_input(iv_col, cl_col, ph_col, 'value')
            # user-friendly column name check
            for _, (old_name, new_name) in enumerate(self.column_map.items()):
                if new_name == 'value':  # prediction does not need a readout col
                    continue
                if old_name not in df.columns:
                    raise ValueError(f"{old_name} not in df columns.")

            df = df.rename(columns=self.column_map).copy()

            # same formatting as in train
            df = df.drop_duplicates()
            df = df.reset_index(drop=True)
            df = self._remove_nonexistent_cat(data_label=df, verbose=False)
            total_size = len(df)
        
        # Divide work into partitions. This allows users to run a large amount of inference
        # in one shot without memory problems.
        if num_iterations:
            if num_iterations < 1:
                raise KeyError("num_iterations must be larger than 0.")
            else:
                if num_iterations - self._decide_iteration_num(total_size=total_size, single_run_size=1e08) > 5:
                    warnings.warn("The num_iterations you passed might be too small.")
        else:
            num_iterations = self._decide_iteration_num(total_size=total_size, single_run_size=1e08)
        
        if num_iterations % 2 == 0:
            num_iterations = num_iterations + 1
        
        print(f"There are {num_iterations} iterations")
        data_label_list = []
        for run_index in tqdm(range(num_iterations)):

            # If the user input is df
            if isinstance(df, pd.DataFrame):
                batch_size = int(total_size // num_iterations)
                start_idx = run_index * batch_size
                end_idx = (
                    start_idx + batch_size
                    if (run_index < num_iterations - 1)
                    else len(df)
                )
                data_label = df.iloc[start_idx:end_idx]

            # If the user input is gene list and cl list
            else:
                data_label = self._generate_predict_df(run_index=run_index,num_iterations=num_iterations,target_ivs=target_ivs,target_cls=target_cls, target_phs=target_phs)

            data_label = data_label.drop_duplicates()  # TODO should be able to remove

            data_label = self._remove_nonexistent_cat(data_label=data_label, verbose=not isinstance(df, pd.DataFrame))
            
            if self.torch_dataset:  # format for pytorch dataloading
                # must have a value column
                data_label['_'] = 0
                # manually add one row per phenotype, ensuring model has all the phenotypes for indexing to be correct
                duplicated_rows = data_label.tail(len(self.phenotypes)).copy()
                duplicated_rows['phenotype'] = self.phenotypes
                data_label = pd.concat([data_label, duplicated_rows], ignore_index=True)

            split = dataloader_phenotypes(
                    gene_embedding=self.iv_embedding,
                    cell_lines_embedding=self.cl_embedding,
                    phenotype_embedding=self.ph_embedding,
                    data_label=data_label,
                    label_name='_',
                    index=(
                        np.array(data_label.index),
                        [],
                        np.array(data_label.index).tolist(),  # test is not shuffled
                        "",
                    ),  # (train, test, val, descr)
                    torch_dataset=self.torch_dataset,
                    pert_len=self.pert_len
                )

            if self.torch_dataset:
                X = split[2] # take test is not shuffled
            else:
                X, _ = split[2]

            train_dataloader, valid_dataloader, test_dataloader, train_indices, test_indices, descriptor = split
            trainer = pl.Trainer(devices=1)
            predictions = trainer.predict(self.model, test_dataloader)
            predictions = [t[0] for t in predictions]
            predictions = torch.cat(predictions, dim=0)
            data_label["pred"] = predictions
            data_label.drop(columns=['_'], inplace=True)

            if save:
                data_label.to_parquet(
                    f"{filename}_{run_index}.parquet", 
                    # engine="fastparquet"
                )
            else:
                data_label_list.append(data_label)
        if not save:
            concatenated_df = pd.concat(data_label_list)
            return concatenated_df
