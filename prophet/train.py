from model import TransformerPredictor
from prophet.callbacks import R2ScoreCallback
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torchmetrics.regression import R2Score
import torch
import wandb
import os


def train_transformer(data, model, config, name, seed):
    train_dataloader, valid_dataloader, test_dataloader, train_indices, test_indices, descriptor = data

    wandb_config = {
        'model': 'Transformer',
        'descr':descriptor,
        'seed': seed,
        'leaveout_method':config.leaveout_method,
        'setting':config.setting,
        'path': config.path,
        'pooling': config.transformer.pool,
        'unbalanced':config.unbalanced,
        'n_heads':config.transformer.num_heads,
        'n_layers':config.transformer.num_layers,
        'gene_prior': os.path.basename(config.genes_prior[0]),
        'cell_lines_prior': os.path.basename(config.cell_lines_prior[0]),
        'batch_size': config.batch_size,
        'early_stopping': config.early_stopping,
        'patience': config.patience,
        'ckpt_path': config.ckpt_path,
        'fine_tune': config.fine_tune,
        'max_steps': config.max_steps,
    }
    sub_descr = {f'descr{i}':x for i, x in enumerate(descriptor.split('_'))}

    wandb_logger = WandbLogger(
        project=config.project_name,
        name=f'{descriptor}_{config.setting}_nheads_{config.transformer.num_heads}_nlayers_{config.transformer.num_layers}_{config.transformer.simpler}simpler_{config.transformer.mask}mask_{config.transformer.lr}lr_{config.transformer.warmup}warmup_{config.transformer.max_iters}max_iters',
        config={**wandb_config, **sub_descr},
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')
    dirpath = f"./pretrained_prophet/{config.setting}/{descriptor}_{config.leaveout_method[6:]}_{config.transformer.dim_cl}cl_{config.transformer.dim_iv}iv_{config.transformer.model_dim}model_{config.transformer.num_layers}layers_{config.transformer.simpler}simpler_{config.transformer.mask}mask_{config.transformer.lr}lr_{config.transformer.explicit_phenotype}explicitphenotype_{config.transformer.warmup}warmup_{config.transformer.max_iters}max_iters_{config.unbalanced}unbalanced_{config.transformer.weight_decay}wd_{config.batch_size}bs_{config.fine_tune}ft/{name}_seed_{seed}"
    model_checkpointer = ModelCheckpoint(dirpath=dirpath, save_top_k=1, every_n_epochs=1, monitor='R2', mode='max')
    r2_callback = R2ScoreCallback(device=model.device, average=True if config.setting == 'everything' else False)
    early_stopping = EarlyStopping(monitor="R2", mode="max", patience=config.patience, min_delta=0.0)
    
    callbacks = [r2_callback, model_checkpointer, lr_monitor, early_stopping]
    if not config.early_stopping:
        callbacks = [r2_callback, model_checkpointer, lr_monitor]
    
    test_mode = False  # manual toggle for debugging
        
    trainer = pl.Trainer(
        min_epochs=1,
        max_steps=config.max_steps,
        accelerator='gpu',
        devices=-1,
        check_val_every_n_epoch=1,
        callbacks=callbacks,
        logger=wandb_logger,
        strategy="ddp",
        gradient_clip_val=1,
        deterministic=True)
        
    if config.ckpt_path is not None:
        print("name: ", name)
        axis_name = 'gene' if 'gene' in config.leaveout_method else 'cl'
        if axis_name == 'gene':
            axis = 'iv'
        else:
            axis = 'cl'
        split_num = descriptor.split('_')[1]
        ckpt_path = f"{config.ckpt_path}/{axis_name}_{split_num}_seed_{seed}/"
        if config.fine_tune:
            ckpt_path = f"{config.ckpt_path.replace(f'iv_0_iv', f'{axis}_{split_num}_{axis}')}/{axis_name}_{split_num}_seed_{seed}/"
        files = os.listdir(ckpt_path) # ckpt_path is the path to a folder
        full_paths = [os.path.join(ckpt_path, file) for file in files]
        ckpt_file = max(full_paths, key=os.path.getmtime) # ckpt_file is the .ckpt file
        print(f"Resume training from {ckpt_file}")
        if config.fine_tune:
            print(f"Fine tuning")
            model = TransformerPredictor.load_from_checkpoint(ckpt_file, warmup=config.transformer.warmup)
            trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader) # train from scratch
        else:
            trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader, ckpt_path=config.ckpt_path) 
    else:
        trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader, ckpt_path=config.ckpt_path) 
    
    if not test_mode:
        
        # Get most recent checkpoint
        files = os.listdir(dirpath)
        full_paths = [os.path.join(dirpath, file) for file in files]
        ckpt_file = max(full_paths, key=os.path.getmtime)
        
        print(f"Testing model from {ckpt_file}")
        # Load best model in terms of R2
        model = TransformerPredictor.load_from_checkpoint(checkpoint_path=ckpt_file)
        print(type(test_dataloader))
        if not isinstance(test_dataloader, dict):
            trainer.test(model, test_dataloader)
        else:
            for id_dataset, dataset in test_dataloader.items():
                if id_dataset == 'all':  # logs as R2_test as normal
                    trainer.test(model, dataset)
                    continue
                
                if id_dataset == 'all':
                    trainer.test(model, dataset)
                    continue
                
                predictions_and_targets = trainer.predict(model, dataset)
            
                predictions = [t[0] for t in predictions_and_targets]
                targets = [t[1] for t in predictions_and_targets]
            
                predictions = torch.cat(predictions, dim=0)
                targets = torch.cat(targets, dim=0)

                r2score = R2Score().to(model.device)
                wandb.log({f"R2_test_{id_dataset}": r2score(predictions, targets.unsqueeze(-1))})
        

    wandb.finish()
