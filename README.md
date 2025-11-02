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

## Quick Start

```python
from prophet import Prophet

# Load a pretrained model (automatically downloads everything)
model = Prophet.from_pretrained("base")

# Ready to predict!
predictions = model.predict(your_data)
```

### Available Models

See all available models and configurations:
```python
Prophet.list_models()
```

Prophet provides pretrained models for various datasets including:
- **base**: General purpose pretrained model (recommended for most users)
- **GDSC, CTRP, PRISM**: Drug sensitivity datasets
- **LINCS, JUMP**: Gene expression perturbation datasets
- **Horlbeck**: CRISPR screening data
- And more...

Each model can be loaded with different configurations (split type, fold, seed):
```python
# Load with specific configuration
model = Prophet.from_pretrained(
    model_name="GDSC",
    split="perturbations",  # or "cell_lines"
    fold=0,  # 0-4
    seed=110  # 110, 1995, or 2024
)
```

### Tutorials and Examples

For detailed examples and workflows, check out:
- [Getting Started Tutorial](tutorials/getting_started.ipynb) - Complete walkthrough
- [Fine-tuning Guide](tutorials/finetuning.ipynb) - Adapt models to your data

### Advanced: Manual Download

For advanced users who need direct file access, model checkpoints and embeddings are available at:
- [HuggingFace Hub](https://huggingface.co/datasets/theislab/Prophet)
- [Mendeley Data](https://data.mendeley.com/datasets/g7z3pw3bfw)

## Citation

If you have used our work in your research, please cite our [preprint](https://www.biorxiv.org/content/10.1101/2024.08.12.607533v2).

[mit]: https://opensource.org/licenses/MIT
[mit-image]: https://img.shields.io/badge/License-MIT-yellow.svg
[mit-shield]: https://img.shields.io/badge/License-MIT-yellow.svg
