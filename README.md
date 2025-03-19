[![License: MIT][mit-shield]][mit]
# Prophet

Prophet is a transformer-based regression model that predicts cellular responses by decomposing experiments into cell state, treatment, and functional readout, leveraging extensive screening datasets and scalability to significantly reduce the number of required experiments and identify effective treatments.

## Model Overview

Prophet decomposes biological experiments into three key components:
1. **Cell state** - represented by cell line embeddings derived from gene expression profiles
2. **Treatment** - represented by intervention embeddings (e.g., small molecules, genetic perturbations)
3. **Functional readout** - the phenotypic measurement being predicted (e.g., viability, IC50)

The model uses a transformer architecture to learn complex interactions between these components and predict experimental outcomes without requiring the experiments to be performed.

### Embeddings

Prophet uses three types of embeddings:
- **Cell line embeddings**: 300-dimensional vectors derived from CCLE gene expression data
- **Intervention embeddings**: 500-dimensional vectors representing small molecules or genetic perturbations
- **Phenotype embeddings**: Representations of different readout types (optional)

These embeddings capture the biological properties of each component and allow the model to generalize across different experimental conditions.

## Training

Prophet was trained on a large dataset of cellular response measurements, including:
- Drug sensitivity screens (GDSC, PRISM, CTRP)
- Genetic perturbation screens (DepMap, Achilles)
- Combinatorial perturbation experiments

The model was trained using a masked attention mechanism to handle variable numbers of perturbations and a cosine learning rate schedule with warmup. Training was performed on NVIDIA A100 GPUs with early stopping based on validation loss.

## Installation
```
mamba create -n prophet_env python=3.10
mamba activate prophet_env

git clone https://github.com/theislab/prophet.git
cd prophet
pip install -e .
```

## Usage

### Downloading Resources

Model checkpoints and input embeddings can be downloaded [here](https://huggingface.co/datasets/aletlvl/Prophet_v1/tree/main) and [here](https://data.mendeley.com/datasets/g7z3pw3bfw). 

If you have used our work in your research, please cite our [preprint](https://www.biorxiv.org/content/10.1101/2024.08.12.607533v2).

[mit]: https://opensource.org/licenses/MIT
[mit-image]: https://img.shields.io/badge/License-MIT-yellow.svg
[mit-shield]: https://img.shields.io/badge/License-MIT-yellow.svg
