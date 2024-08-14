import torch
import pandas as pd
import argparse
import pprint

from dataloader import dataloader_phenotypes, get_split_indices, get_data_by_setting
from train import train_transformer
from model import load_models_config
import pytorch_lightning as pl
from prophet.config import set_config
import os
import yaml

# Add arguments
parser = argparse.ArgumentParser()
parser.add_argument("--setting", type=str,
                    default="Rad", required=False) # Rad, Rad_Horlbeck or Horlbeck
parser.add_argument("--leaveout_method", type=str,
                    default="leave_one_cl_out", required=False)
parser.add_argument("--config_file", type=str, required=True) # config file with info regarding the architecture
parser.add_argument("--fine_tune", action='store_true', default=False, required=False)

args = parser.parse_args()
config_file = args.config_file.split('-')[1]
seed = int(args.config_file.split('-')[0])

def get_global_rank():
    return int(os.getenv('SLURM_PROCID', '0'))

if __name__ == "__main__":
    
    # Check the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print("Number of gpus: ", num_gpus)
    
    global_rank = get_global_rank()
    print(f"Global Rank: {global_rank}")

    with open(config_file, 'r') as f:
        models_config = yaml.safe_load(f)

    models_config = set_config(models_config)

    # override config to the default fine tuning model parameters during fine-tuning
    if args.fine_tune:
        with open('config_files/config_file_finetuning.yaml', 'r') as f:
            ft_config = set_config(yaml.safe_load(f))
        ft_config.setting = models_config.setting
        ft_config.leaveout_method = models_config.leaveout_method
        ft_config.dirpath = models_config.dirpath
        models_config = ft_config

    pl.seed_everything(seed, workers=True)
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['TORCH_USE_CUDA_DSA'] = '1'

    data_label, gene_prior, cl_prior, phe_prior, path = get_data_by_setting(models_config.setting, models_config.genes_prior, models_config.cell_lines_prior, models_config.phenotype_prior)

    models_config.path = path # to print the path
    
    pprint.pprint(models_config)

    # Get the indices of the DataFrame
    indices = data_label.index
    indices = get_split_indices(data_label, models_config.leaveout_method, seed)

    for index in indices:

        data = dataloader_phenotypes(
            gene_embedding = gene_prior,
            cell_lines_embedding = cl_prior,
            phenotype_embedding = phe_prior,
            data_label = data_label,
            label_name = "value",
            index = index,
            batch_size = models_config.batch_size,
            unbalanced = models_config.unbalanced,
            pert_len = models_config.pert_len,
        )

        models_config.ohe_dim = 0  # relic of ohe

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model, models_config = load_models_config(models_config, seed)

        train_transformer(data=data, model=model, config=models_config, name=index[-1], seed=seed)
        del data
        del model
