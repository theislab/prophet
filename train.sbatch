#!/bin/bash
#SBATCH -o err/train_%A.out
#SBATCH -e err/train_%A.err
#SBATCH -J train
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
##SBATCH --constraint=a100_80gb
#SBATCH --mem=400G
#SBATCH -t 24:00:00
#SBATCH --nice=1
#SBATCH --exclude=gpusrv35

source /home/icb/alejandro.tejada/.bashrc
micromamba activate prophet

python scripts/train.py --config configs/pretraining.yaml
