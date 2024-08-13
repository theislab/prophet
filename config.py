from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class TransformerConfig:
    dim_cl: int = 300
    dim_iv: int = 800
    dim_phe: int = 300 # since now is concatenation
    model_dim: int = 128
    num_heads: int = 1
    num_layers: int = 2
    iv_dropout: float = 0.2
    cl_dropout: float = 0.2
    ph_dropout: float = 0.2
    regressor_dropout: float = 0.2
    lr: float = 0.0001
    weight_decay: float = 0.01
    warmup: int = 10000
    max_iters: int = 70000
    dropout: float = 0.2
    exclude_cl_embedding: bool = False
    pool: str = "cls"
    simpler: bool = True
    mask: bool = True
    sum: bool = False
    explicit_phenotype: bool = False
    linear_predictor: bool = False
    tokenizer_layers: int = 2

    def update_from_dict(self, updates: Dict):
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)

@dataclass
class Config:
    setting: str
    leaveout_method: str
    dirpath: str
    ckpt_path: str = None
    project_name: str = "Prophet_hparams"
    cell_lines_prior: List[str] = field(default_factory=lambda: ["./embeddings/cell_line_embedding_full_ccle_300_scaled.csv"])
    genes_prior: List[str] = field(default_factory=lambda: ["./embeddings/ccle_T_pca_300_enformer_full_gene_mean_PCA_500_scaled.csv"] )
    phenotype_prior: List[str] = None
    unbalanced: bool = False
    pert_len: int = 2
    max_steps: int = 140000
    batch_size: int = 2048
    early_stopping: bool = True
    patience: int = 20
    ckpt_path = None
    fine_tune = False
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    
    def update_from_dict(self, updates: Dict):
        for key, value in updates.items():
            if key == 'Transformer' and isinstance(value, dict):
                self.transformer.update_from_dict(value)
            elif hasattr(self, key):
                setattr(self, key, value)

def set_config(models_config):
    config = Config(
        setting=models_config['setting'],
        leaveout_method=models_config['leaveout_method'],
        dirpath=models_config['dirpath'],
    )
    config.update_from_dict(models_config)

    if config.transformer.simpler:
       config.ctx_len = config.pert_len + 1 # add CLS
    else:
        config.ctx_len = config.pert_len + 3 # add CLS, phenotype and cell line
    return config
