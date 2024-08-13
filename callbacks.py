import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch import optim
import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import spearmanr


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


class R2ScoreCallback(pl.Callback):
    def __init__(self, device: torch.device = 'cpu', average = False):
        super().__init__()
        self.predictions = []
        self.targets = []
        
        
        self.prediction_train = []
        self.prediction_test = []
        
        self.targets_train = []
        self.targets_test = []
        
        self.phenotype_validation = []
        
        self.device = device
        self.average = average
        
        self.table_val = None
        self.table_train = None
        
        print("R2 average: ", self.average)
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        y_pred, y_true = outputs['y_pred'].detach(), outputs['y_true'].detach()
        self.prediction_train.append(y_pred)
        self.targets_train.append(y_true)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        y_pred, y_true, phenotype = outputs['y_pred'], outputs['y_true'], outputs['phenotype']
        self.predictions.append(y_pred)
        self.targets.append(y_true)
        self.phenotype_validation.append(phenotype)
        
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        y_pred, y_true = outputs['y_pred'], outputs['y_true']
        self.prediction_test.append(y_pred)
        self.targets_test.append(y_true)
        
    def on_validation_epoch_end(self, trainer, pl_module):
        predictions = torch.cat(self.predictions, dim=0)#.cpu().numpy()
        targets = torch.cat(self.targets, dim=0)#.cpu().numpy()
        phenotypes = torch.cat(self.phenotype_validation, dim=0)
        
                
        predictions = predictions.cpu().numpy()
        targets = targets.cpu().numpy()
        phenotypes = phenotypes.cpu().numpy()
        
        if self.average:
            r2_scores = 0
            spearman_scores = 0
            unique_phe = np.unique(phenotypes)
            for phe in unique_phe:
                indices = np.nonzero(phenotypes == phe)
                phe_predictions = predictions[indices]
                phe_targets = targets[indices]
                
                r2 = r2_score(phe_targets, phe_predictions)
                r2_scores += r2
                
                spearman = spearmanr(phe_predictions, phe_targets).statistic
                spearman_scores += spearman
                
            r2_total = r2_scores / len(unique_phe)
            spearman_total = spearman_scores / len(unique_phe)
            
            self.log("R2", r2_total, sync_dist=True, batch_size=predictions.shape[0])
            self.log("Spearman", spearman_total, sync_dist=True, batch_size=predictions.shape[0])
            
        else:
            
            r2 = r2_score(targets, predictions)
            self.log("R2", r2, sync_dist=True, batch_size=predictions.shape[0])
            
            spearman = spearmanr(predictions, targets).statistic
            self.log("Spearman", spearman, sync_dist=True, batch_size=predictions.shape[0])
        
        self.predictions = []
        self.targets = []
        self.phenotype_validation = []

    def on_train_epoch_end(self, trainer, pl_module):
        predictions = torch.cat(self.prediction_train, dim=0)#.cpu().numpy()
        targets = torch.cat(self.targets_train, dim=0)#.cpu().numpy()
                
        targets_mean = targets.mean(0)
        targets_mean = targets_mean.repeat(targets.shape[0])
            
        targets_mean = targets_mean.cpu().numpy()
        predictions = predictions.cpu().numpy()
        targets = targets.cpu().numpy()
    
        r2 = r2_score(targets, predictions)
        self.log("R2_train", r2, sync_dist=True, batch_size=predictions.shape[0])
        
        spearman = spearmanr(predictions, targets).statistic
        self.log("Spearman_train", spearman, sync_dist=True, batch_size=predictions.shape[0])
        
        self.prediction_train = []
        self.targets_train = []
        
    def on_test_epoch_end(self, trainer, pl_module):
        predictions = torch.cat(self.prediction_test, dim=0)#.cpu().numpy()
        targets = torch.cat(self.targets_test, dim=0)#.cpu().numpy()
                
        targets_mean = targets.mean(0)
        targets_mean = targets_mean.repeat(targets.shape[0])
            
        targets_mean = targets_mean.cpu().numpy()
        predictions = predictions.cpu().numpy()
        targets = targets.cpu().numpy()
    
        r2 = r2_score(targets, predictions)
        self.log("R2_test", r2, sync_dist=True, batch_size=predictions.shape[0])
        
        spearman = spearmanr(predictions, targets).statistic
        self.log("Spearman_test", spearman, sync_dist=True, batch_size=predictions.shape[0])
        
        self.prediction_test = []
        self.targets_test = []