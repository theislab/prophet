import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch import optim
import torch.nn.init as init
from prophet.callbacks import CosineWarmupScheduler

from tqdm import tqdm

class TransformerPredictor(pl.LightningModule):

    #def __init__(self, config):
    def __init__(self, 
                 dim_cl: int,
                 dim_iv: int,
                 dim_phe: int,
                 model_dim: int, 
                 num_heads: int, 
                 num_layers: int, 
                 iv_dropout: float,
                 cl_dropout: float,
                 ph_dropout: float,
                 regressor_dropout: float,
                 lr: float, 
                 warmup: int, 
                 weight_decay: float,
                 max_iters: int, 
                 batch_size: int = 0,
                 dropout = 0.0, 
                 pool: str = 'cls', 
                 simpler: bool = False,
                 ctx_len: int = 4,
                 mask: bool = True,
                 sum: bool = False,
                 explicit_phenotype: bool = False,
                 linear_predictor: bool = False,
                 tokenizer_layers: int = 2,
                 seed=42):
        """
        Inputs:
            dim_cl - Number of dimensions to take from the cell lines
            dim_iv - Number of dimensions of the interventional embeddings
            model_dim - Hidden dimensionality to use inside the Transformer
            num_heads - Number of heads to use in the Multi-Head Attention blocks
            num_layers - Number of encoder blocks to use.
            iv_dropout - dropout in iv layers
            cl_dropout - dropout in cl layers
            regressor_dropout - dropout in regressor
            lr - Learning rate in the optimizer
            warmup - Number of warmup steps. Usually between 50 and 300
            max_iters - Number of maximum iterations the model is trained for. This is needed for the CosineWarmup scheduler
            dropout - Dropout to apply inside the model
            pool - 'mean', 'cls', or 'pool'. Mean takes the mean, CLS predicts just with the CLS, pool takes the max value
            simpler - uses the transformer just for the set of perturbations, then it concatenates the result with the other representations
            ctx_len - context length
            mask - if True, mask attention, otherwise don't do it
            sum - if True, don't use Transformer, just sum embeddings
            explicit_phenotype - if True, the user passes an embedding as phenotype directly
            linear_predictor 
            tokenizer_layers
        """
        super().__init__()

        self.save_hyperparameters()
        self._create_model()
        
        # Initialize the weights
        self.initialize_weights()

    def _create_model(self):
        self.learnable_embedding = torch.nn.Embedding(num_embeddings=1000, 
                                                      embedding_dim=self.hparams.model_dim,
                                                      max_norm=0.5, 
        )
        self.embedding_dropout = nn.Dropout(self.hparams.ph_dropout)
        
        # Tokenizer layer strong enough to non-linearly transform the data
        self.gene_net = nn.Sequential(
        nn.Linear(self.hparams.dim_iv, self.hparams.model_dim),
        nn.GELU(), 
        nn.Dropout(self.hparams.iv_dropout),
        nn.Linear(self.hparams.model_dim, self.hparams.model_dim)
        )
        
        self.drug_net = nn.Sequential(
        nn.Linear(self.hparams.dim_iv, self.hparams.model_dim),
        nn.GELU(),
        nn.Dropout(self.hparams.iv_dropout),
        nn.Linear(self.hparams.model_dim, self.hparams.model_dim)
        )
    
        self.cl_net = nn.Sequential(
        nn.Linear(self.hparams.dim_cl, self.hparams.model_dim),
        nn.GELU(), 
        nn.Dropout(self.hparams.cl_dropout),
        nn.Linear(self.hparams.model_dim, self.hparams.model_dim)
        )
        
        if self.hparams.tokenizer_layers == 1:
            self.gene_net = nn.Sequential(
                nn.Linear(self.hparams.dim_iv, self.hparams.model_dim),
                nn.Dropout(self.hparams.iv_dropout),
                )
            self.drug_net = nn.Sequential(
                nn.Linear(self.hparams.dim_iv, self.hparams.model_dim),
                nn.Dropout(self.hparams.iv_dropout),
                )
            self.cl_net = nn.Sequential(
                nn.Linear(self.hparams.dim_cl, self.hparams.model_dim),
                nn.Dropout(self.hparams.cl_dropout),
                )
        
            
        if self.hparams.explicit_phenotype:
            self.phenotype_net = nn.Sequential(
            nn.Linear(self.hparams.dim_phe, self.hparams.dim_phe),
            nn.GELU(), 
            nn.Dropout(self.hparams.ph_dropout),
            nn.Linear(self.hparams.dim_phe, self.hparams.model_dim)
            )
         
        # Transformer
        layer = nn.TransformerEncoderLayer(d_model=self.hparams.model_dim,
                                           nhead=self.hparams.num_heads,
                                           dim_feedforward=2*self.hparams.model_dim,
                                           dropout=self.hparams.dropout,
                                           batch_first=True,
                                           activation="gelu")
        
        self.transformer = nn.TransformerEncoder(encoder_layer=layer,
                                                 num_layers=self.hparams.num_layers)
        
        # 2 layers regressor
        dim_regressor_input = 2*self.hparams.model_dim + self.learnable_embedding.embedding_dim # CLS (model_dim) | CellLine (model_dim) | phenotype (varies)
        if not self.hparams.simpler:
            dim_regressor_input = self.hparams.model_dim
        if self.hparams.sum:
            dim_regressor_input = dim_regressor_input+self.hparams.model_dim+self.hparams.model_dim
        
        self.output_net = nn.Sequential(
            nn.Linear(dim_regressor_input, self.hparams.model_dim),
            nn.GELU(), 
            nn.Dropout(self.hparams.regressor_dropout),
            nn.Linear(self.hparams.model_dim, self.hparams.model_dim),
            nn.GELU(), 
            nn.Linear(self.hparams.model_dim, 1),
        )
        if self.hparams.linear_predictor:
            self.output_net = nn.Sequential(
            nn.Linear(dim_regressor_input, 1)
            )
        
        print(f'Gene net: ', self.gene_net, flush=True)
        print(f'Cell line net: ', self.cl_net, flush=True)
        print(f'Regressor: ', self.output_net, flush=True)
        if self.hparams.explicit_phenotype:
            print("Using explicit phenotype")
        if self.hparams.linear_predictor:
            print("Using linear predictor")

    def forward(self, phenotype, cl, perturbations, perturbations_type, attn_mask):
        """
        Inputs:
            x - Input features of shape [Batch, SeqLen, 1] 
        """
        cl = cl[:,:self.hparams.dim_cl]
        perturbations = [pert[:, :self.hparams.dim_iv] for pert in perturbations]
        attn_mask = attn_mask[:, :self.hparams.ctx_len] 
        
        if self.hparams.explicit_phenotype:
            phenotype_emb = self.phenotype_net(phenotype[:,:self.hparams.dim_phe])
        else:
            phenotype_emb = self.learnable_embedding(phenotype) # Phenotype 
        
        phenotype_emb = self.embedding_dropout(phenotype_emb) # dropout
        # shape is (batch_size x n_dim)
            
                
        # Drugs to drug network and genes to gene network
        # We mask the attention to the negative perturbations, so it's like not using the networks
        drug_perturbations = [self.drug_net(tensor).unsqueeze(1) for tensor in perturbations] # all perts to drug
        gene_perturbations = [self.gene_net(tensor).unsqueeze(1) for tensor in perturbations] # all perts to gene

        drug_perturbations = torch.cat(drug_perturbations, dim=1) # bs x n x dim
        gene_perturbations = torch.cat(gene_perturbations, dim=1) # bs x n x dim

        perturbations = torch.where(perturbations_type.unsqueeze(2) == 0, gene_perturbations, drug_perturbations)

        cl_embedding = self.cl_net(cl).unsqueeze(1) # unsqueeze just useful if not simpler
        phenotype_emb = phenotype_emb.unsqueeze(1)
        # bs x n x dim
                
        # Regression token (CLS) stored in index 0 
        cls = torch.zeros(size=(phenotype_emb.shape[0], 1), device=phenotype_emb.device, dtype=torch.int32)
        cls = self.learnable_embedding(cls) # we get the embedding, shape (bs x 1 x dim)
        
        if self.hparams.pool == 'cls':
            if self.hparams.simpler: # if simpler, just perturbations and CLS to transformer
                embeddings = torch.cat((cls, perturbations), dim=1)
            else: # otherwise everuthing to transformer
                embeddings = torch.cat((cls, perturbations, cl_embedding, phenotype_emb), dim=1)
        else: # if not CLS, we don't need it
            if self.hparams.simpler:
                embeddings = perturbations
            else:
                embeddings = torch.cat((perturbations, cl_embedding, phenotype_emb), dim=1)
                            
        # Run Transformer Layer
        if not self.hparams.sum:
            if self.hparams.mask:
                x = self.transformer(embeddings, mask=None, src_key_padding_mask=attn_mask)
            else:
                x = self.transformer(embeddings, mask=None)
            
            if self.hparams.pool == 'cls':
                x = x[:, 0, :] # use just the regressor token for regression
            elif self.hparams.pool == 'mean':
                x = torch.mean(x, dim=1) # mean-pool
            else:
                x = torch.max(x, dim=1) # max-pool
            
        # If sum, forget about everything else
        if self.hparams.sum:            
            #x = torch.sum(embeddings, dim=1)
            x = torch.reshape(embeddings, (embeddings.shape[0], -1))

        if self.hparams.simpler:
            x = torch.cat((x, cl_embedding.squeeze(1), phenotype_emb.squeeze(1)), dim=-1) 
            
            
        x = self.output_net(x)
        
        return x
    
    def embedding(self, phenotype, cl, perturbations, perturbations_type, attn_mask):
        """
        Inputs:
            x - Input features of shape [Batch, SeqLen, 1] 
        """
        
        # Cut CL and perturbations to number of selected dimensions
        cl = cl[:,:self.hparams.dim_cl]
        perturbations = [pert[:, :self.hparams.dim_iv] for pert in perturbations]
        attn_mask = attn_mask[:, :self.hparams.ctx_len] 
        
        if self.hparams.explicit_phenotype:
            # if explicit_phenotype, we take the phenotype that the user input. The user must make it dim_model-dimensional
            phenotype_emb = self.phenotype_net(phenotype[:,:self.hparams.model_dim])
        else:
            phenotype_emb = self.learnable_embedding(phenotype) # Phenotype 
        
        phenotype_emb = self.embedding_dropout(phenotype_emb) # dropout
        
        # Drugs to drug network and genes to gene network
        # We mask the attention to the negative perturbations, so it's like not using the networks
        drug_perturbations = [self.drug_net(tensor).unsqueeze(1) for tensor in perturbations] # all perts to drug
        gene_perturbations = [self.gene_net(tensor).unsqueeze(1) for tensor in perturbations] # all perts to gene

        drug_perturbations = torch.cat(drug_perturbations, dim=1) # bs x n x dim
        gene_perturbations = torch.cat(gene_perturbations, dim=1) # bs x n x dim
        
        perturbations = torch.where(perturbations_type.unsqueeze(2) == 0, gene_perturbations, drug_perturbations)
        
        cl_embedding = self.cl_net(cl).unsqueeze(1) # unsqueeze just useful if not simpler
        phenotype_emb = phenotype_emb.unsqueeze(1)
        # bs x n x dim
                
        # Regression token (CLS) stored in index 0
        cls = torch.zeros(size=(phenotype_emb.shape[0], 1), device=phenotype_emb.device, dtype=torch.int32)
        cls = self.learnable_embedding(cls) # we get the embedding, shape (bs x 1 x dim)
        
        if self.hparams.pool == 'cls':
            if self.hparams.simpler: # if simpler, just perturbations and CLS to transformer
                embeddings = torch.cat((cls, perturbations), dim=1)
            else: # otherwise everuthing to transformer
                embeddings = torch.cat((cls, perturbations, cl_embedding, phenotype_emb), dim=1)
        else: # if not CLS, we don't need it
            if self.hparams.simpler:
                embeddings = perturbations
            else:
                embeddings = torch.cat((perturbations, cl_embedding, phenotype_emb), dim=1)
                            
        # Run Transformer Layer
        if not self.hparams.sum:
            if self.hparams.mask:
                x = self.transformer(embeddings, mask=None, src_key_padding_mask=attn_mask)
            else:
                x = self.transformer(embeddings, mask=None)
            
            transformer_output = x
            
            if self.hparams.pool == 'cls':
                pert_emb = x[:, 0, :] # use just the regressor token for regression
            elif self.hparams.pool == 'mean':
                pert_emb = torch.mean(x, dim=1) # mean-pool
            else:
                pert_emb = torch.max(x, dim=1) # max-pool
            
        # If sum, forget about everything else
        if self.hparams.sum:            
            #x = torch.sum(embeddings, dim=1)
            x = torch.reshape(embeddings, (embeddings.shape[0], -1))

        if self.hparams.simpler:
            x = torch.cat((pert_emb, cl_embedding.squeeze(1), phenotype_emb.squeeze(1)), dim=-1) 
            
        x = self.output_net(x)
                
        return {'pert_emb': pert_emb, # output of transformer: CLS or mean
                'output': x, 
                'perturbations_after_transformer': transformer_output[:, 1:, :],
                'perturbations': perturbations, # tokens
                'cl_embedding': cl_embedding, # tokens
                'phenotype': phenotype,
                'pert_type': perturbations_type,
                }

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

        # Apply lr scheduler per step
        lr_scheduler = CosineWarmupScheduler(optimizer,
                                             warmup=self.hparams.warmup,
                                             max_iters=self.hparams.max_iters)
        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]

    def training_step(self, batch, batch_idx):

        # complete_masking(batch, self.hparams.ctx_len)
        attn_mask = batch['attn_mask']
        attn_mask = attn_mask[:, :self.hparams.ctx_len]

        phenotype = batch['phenotype']
        cl = batch['cell_line']
        y = batch['label']
        perturbations_type = batch['pert_type']

        perturbations = []
        for pert in range(1, self.hparams.ctx_len - (2 if not self.hparams.simpler else 0)):
            perturbations.append(batch[f'iv{pert}'].to(torch.float32))
                        
        if self.hparams.explicit_phenotype:
            phenotype, cl = phenotype.to(torch.float32), cl.to(torch.float32)
        else:
            phenotype, cl = phenotype.to(torch.int32), cl.to(torch.float32)

        #y_hat = self.forward(phenotype, cl, perturbations, perturbations_type, attn_mask)
        y_hat = self(phenotype, cl, perturbations, perturbations_type, attn_mask)

        y = y.unsqueeze(1)

        loss = torch.nn.functional.mse_loss(y, y_hat)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True, batch_size=phenotype.shape[0])
        
        return {'loss': loss, 'y_pred': y_hat, 'y_true': y}

    def validation_step(self, batch, batch_idx):
                    
        attn_mask = batch['attn_mask']
        attn_mask = attn_mask[:, :self.hparams.ctx_len]
                
        phenotype = batch['phenotype']
        cl = batch['cell_line']
        y = batch['label']
        perturbations_type = batch['pert_type']
                        
        perturbations = []
        for pert in range(1, self.hparams.ctx_len - (2 if not self.hparams.simpler else 0)):
            perturbations.append(batch[f'iv{pert}'].to(torch.float32))

        if self.hparams.explicit_phenotype:
            phenotype, cl = phenotype.to(torch.float32), cl.to(torch.float32)
        else:
            phenotype, cl = phenotype.to(torch.int32), cl.to(torch.float32)

        # y_hat = self.forward(phenotype, cl, perturbations, perturbations_type, attn_mask)
        y_hat = self(phenotype, cl, perturbations, perturbations_type, attn_mask)
        
        y = y.unsqueeze(1)
        loss = torch.nn.functional.mse_loss(y, y_hat)
                
        self.log("validation_loss", loss, sync_dist=True, batch_size=phenotype.shape[0])
        
        return {'y_pred': y_hat, 'y_true': y, 'phenotype': phenotype}

    def test_step(self, batch, batch_idx):

        attn_mask = batch['attn_mask']
        attn_mask = attn_mask[:, :self.hparams.ctx_len]

        phenotype = batch['phenotype']
        cl = batch['cell_line']
        y = batch['label']
        perturbations_type = batch['pert_type']

        perturbations = []
        for pert in range(1, self.hparams.ctx_len - (2 if not self.hparams.simpler else 0)):
            perturbations.append(batch[f'iv{pert}'].to(torch.float32))
                        
        if self.hparams.explicit_phenotype:
            phenotype, cl = phenotype.to(torch.float32), cl.to(torch.float32)
        else:
            phenotype, cl = phenotype.to(torch.int32), cl.to(torch.float32)

        # y_hat = self.forward(phenotype, cl, perturbations, perturbations_type, attn_mask)
        y_hat = self(phenotype, cl, perturbations, perturbations_type, attn_mask)
        
        y = y.unsqueeze(1)
        loss = torch.nn.functional.mse_loss(y, y_hat)
        
        self.log("test_loss", loss, sync_dist=True, batch_size=phenotype.shape[0])
        
        return {'y_pred': y_hat, 'y_true': y}
    
    def get_embeddings(self, batch):        
        
        attn_mask = batch['attn_mask']
        attn_mask = attn_mask[:, :self.hparams.ctx_len]
                
        x = batch['phenotype']
        cl = batch['cell_line']
        names = batch['names']
        perturbations_type = batch['pert_type']
                
        perturbations = []
        for pert in range(1, self.hparams.ctx_len - (2 if not self.hparams.simpler else 0)):
            perturbations.append(batch[f'iv{pert}'].to(torch.float32))

        x, cl = x.to(torch.int32), cl.to(torch.float32)

        emb_dict = self.embedding(x, cl, perturbations, perturbations_type, attn_mask)
                
        return {'pert_emb': emb_dict['pert_emb'], 
                'output': emb_dict['output'], 
                'perturbations_after_transformer': emb_dict['perturbations_after_transformer'], 
                'perturbations': emb_dict['perturbations'], 
                'cell_line': emb_dict['cl_embedding'], 
                'phenotype': emb_dict['phenotype'], 
                'pert_type': emb_dict['pert_type'],
                'names': names}

    def predict_step(self, batch, batch_idx):

        attn_mask = batch['attn_mask']
        attn_mask = attn_mask[:, :self.hparams.ctx_len]

        phenotype = batch['phenotype']
        cl = batch['cell_line']
        y = batch['label']
        perturbations_type = batch['pert_type']

        perturbations = []
        for pert in range(1, self.hparams.ctx_len - (2 if not self.hparams.simpler else 0)):
            perturbations.append(batch[f'iv{pert}'].to(torch.float32))
                        
        if self.hparams.explicit_phenotype:
            phenotype, cl = phenotype.to(torch.float32), cl.to(torch.float32)
        else:
            phenotype, cl = phenotype.to(torch.int32), cl.to(torch.float32)

        # y_hat = self.forward(phenotype, cl, perturbations, perturbations_type, attn_mask)
        y_hat = self(phenotype, cl, perturbations, perturbations_type, attn_mask)
        
        return y_hat, y

    def on_before_optimizer_step(self, optimizer) -> None:
        
        total_norm = 0
        for p in self.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        self.log('grad_norm', total_norm, sync_dist=True)
        
        return super().on_before_optimizer_step(optimizer)
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # You can choose a different initialization method
                init.xavier_normal_(m.weight)
                init.zeros_(m.bias)


def load_models_config(models_config, seed, hparams=False, trial=None):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if hparams:
        if trial is None:
            raise ValueError("Must pass trial when hparams is True.")
    
    transformer = TransformerPredictor(dim_cl=models_config.transformer.dim_cl, dim_iv=models_config.transformer.dim_iv, dim_phe=models_config.transformer.dim_phe, 
                                       model_dim=models_config.transformer.model_dim, num_heads=models_config.transformer.num_heads, 
                                       num_layers=models_config.transformer.num_layers, iv_dropout=models_config.transformer.iv_dropout,
                                       cl_dropout=models_config.transformer.cl_dropout, ph_dropout=models_config.transformer.ph_dropout, regressor_dropout=models_config.transformer.regressor_dropout,
                                       lr=models_config.transformer.lr, weight_decay=models_config.transformer.weight_decay, warmup=models_config.transformer.warmup, batch_size=models_config.batch_size, 
                                       max_iters=models_config.transformer.max_iters, dropout=models_config.transformer.dropout, 
                                       pool=models_config.transformer.pool, simpler=models_config.transformer.simpler,
                                       ctx_len=models_config.ctx_len, mask=models_config.transformer.mask, sum=models_config.transformer.sum, 
                                       explicit_phenotype=models_config.transformer.explicit_phenotype, linear_predictor=models_config.transformer.linear_predictor, tokenizer_layers=models_config.transformer.tokenizer_layers,
                                       seed=seed)

    return transformer, models_config
