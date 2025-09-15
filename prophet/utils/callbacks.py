import torch
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
    def __init__(self, device: torch.device = "cpu", average=False):
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
        y_pred, y_true = outputs["y_pred"].detach(), outputs["y_true"].detach()
        self.prediction_train.append(y_pred)
        self.targets_train.append(y_true)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        y_pred, y_true, phenotype = (
            outputs["y_pred"],
            outputs["y_true"],
            outputs["phenotype"],
        )
        self.predictions.append(y_pred)
        self.targets.append(y_true)
        self.phenotype_validation.append(phenotype)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        y_pred, y_true = outputs["y_pred"], outputs["y_true"]
        self.prediction_test.append(y_pred)
        self.targets_test.append(y_true)

    def on_validation_epoch_end(self, trainer, pl_module):
        predictions = torch.cat(self.predictions, dim=0)
        targets = torch.cat(self.targets, dim=0)
        phenotypes = torch.cat(self.phenotype_validation, dim=0)

        predictions = predictions.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
        phenotypes = phenotypes.detach().cpu().numpy()

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

            self.log(
                "R2_validation",
                r2_total,
                sync_dist=True,
                batch_size=predictions.shape[0],
            )
            self.log(
                "Spearman_validation",
                spearman_total,
                sync_dist=True,
                batch_size=predictions.shape[0],
            )

        else:
            r2 = r2_score(targets, predictions)
            self.log(
                "R2_validation", r2, sync_dist=True, batch_size=predictions.shape[0]
            )

            spearman = spearmanr(predictions, targets).statistic
            self.log(
                "Spearman_validation",
                spearman,
                sync_dist=True,
                batch_size=predictions.shape[0],
            )

        self.predictions = []
        self.targets = []
        self.phenotype_validation = []

    def on_train_epoch_end(self, trainer, pl_module):
        predictions = torch.cat(self.prediction_train, dim=0)
        targets = torch.cat(self.targets_train, dim=0)

        predictions = predictions.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()

        r2 = r2_score(targets, predictions)
        self.log("R2_train", r2, sync_dist=True, batch_size=predictions.shape[0])

        spearman = spearmanr(predictions, targets).statistic
        self.log(
            "Spearman_train", spearman, sync_dist=True, batch_size=predictions.shape[0]
        )

        self.prediction_train = []
        self.targets_train = []

    def on_test_epoch_end(self, trainer, pl_module):
        predictions = torch.cat(self.prediction_test, dim=0)
        targets = torch.cat(self.targets_test, dim=0)

        predictions = predictions.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()

        r2 = r2_score(targets, predictions)
        self.log("R2_test", r2, sync_dist=True, batch_size=predictions.shape[0])

        spearman = spearmanr(predictions, targets).statistic
        self.log(
            "Spearman_test", spearman, sync_dist=True, batch_size=predictions.shape[0]
        )

        self.prediction_test = []
        self.targets_test = []


def prrc(targets, predictions, i):
    """
    Parameters
    ----------
    targets : np.array
    predictions : np.array
    i : int
        Percentile threshold.

    Returns
    -------
    dict
        Dictionary with hit ratio top and hit ratio bottom.
    """        
    # For top i%
    true_top_threshold = np.percentile(targets, 100-i)
    pred_top_threshold = np.percentile(predictions, 100-i)
    actual_top = targets >= true_top_threshold
    predicted_top = predictions >= pred_top_threshold

    # For bottom i%
    true_bottom_threshold = np.percentile(targets, i)
    pred_bottom_threshold = np.percentile(predictions, i) 
    actual_bottom = targets <= true_bottom_threshold
    predicted_bottom = predictions <= pred_bottom_threshold

    hitratio_top = np.sum(actual_top & predicted_top) / np.sum(predicted_top) if np.sum(predicted_top) > 0 else 0
    hitratio_bottom = np.sum(actual_bottom & predicted_bottom) / np.sum(predicted_bottom) if np.sum(predicted_bottom) > 0 else 0
    
    return {
        'precision_top': hitratio_top,
        'precision_bottom': hitratio_bottom,
    }

def compute_hit_ratio(predictions, targets, cl, phenotypes, iv, topk, suffix=''):
    """Compute hit ratio for top and bottom k% of predictions, given predictions, targets, and all labels.

    Parameters
    ----------
    predictions : np.array
    targets : np.array
    cl : list[str]
        Cell line names
    phenotypes : np.array
        Phenotype indices or names
    iv : list[tuple[str, ...]]
        List of intervention tuples, e.g., [('gene1', 'drug1'), ('gene2', 'drug2'), ...]
    topk : list[int]
        List of percentiles to compute hit ratios for
    suffix : str
        Suffix to add to the logging labels.

    Returns
    -------
    dict
        Dictionary with logging labels as keys.
    """
    unique_cl = set(cl)
    unique_phe = np.unique(phenotypes)
    unique_iv = set(iv)
    logging_dict = {}
    
    for i in topk:
        # Overall metrics for cell lines and interventions separately
        all_cl_precisions_top = []
        all_cl_precisions_bottom = []

        all_iv_precisions_top = []
        all_iv_precisions_bottom = []
        
        # Per phenotype metrics
        for phe in unique_phe:
            phe_cl_precisions_top = []
            phe_cl_precisions_bottom = []
                        
            phe_iv_precisions_top = []
            phe_iv_precisions_bottom = []
            
            # Get indices for this phenotype
            phe_indices = np.where(phenotypes == phe)[0]
            phe_cl = [cl[idx] for idx in phe_indices]
            phe_unique_cl = set(phe_cl)
            
            # Get interventions for this phenotype
            phe_iv = [iv[idx] for idx in phe_indices]
            phe_unique_iv = set(phe_iv)
            
            # Compute metrics per cell line within this phenotype
            for cell_line in phe_unique_cl:
                # Get indices for this cell line within phenotype
                indices = [idx for idx in phe_indices if cl[idx] == cell_line]
                if len(indices) < 2:  # Need at least 2 samples for ranking
                    continue
                    
                cl_predictions = predictions[indices]
                cl_targets = targets[indices]

                metrics = prrc(cl_targets, cl_predictions, i)

                # Top k metrics
                phe_cl_precisions_top.append(metrics['precision_top'])
                all_cl_precisions_top.append(metrics['precision_top'])
                
                # Bottom k metrics
                phe_cl_precisions_bottom.append(metrics['precision_bottom'])
                all_cl_precisions_bottom.append(metrics['precision_bottom'])
            
            # Compute metrics per intervention within this phenotype
            for intervention in phe_unique_iv:
                # Get indices for this intervention within phenotype
                indices = [idx for idx in phe_indices if iv[idx] == intervention]
                if len(indices) < 2:  # Need at least 2 samples for ranking
                    continue
                    
                iv_predictions = predictions[indices]
                iv_targets = targets[indices]

                metrics = prrc(iv_targets, iv_predictions, i)

                # Top k metrics
                phe_iv_precisions_top.append(metrics['precision_top'])
                all_iv_precisions_top.append(metrics['precision_top'])
                
                # Bottom k metrics
                phe_iv_precisions_bottom.append(metrics['precision_bottom'])
                all_iv_precisions_bottom.append(metrics['precision_bottom'])
            
        # Calculate and log overall average hit ratio - cell lines
        if all_cl_precisions_top:
            avg_cl_precision_top = np.mean(all_cl_precisions_top)
            logging_dict[f"cl_average_precision_top_{i}{suffix}"] = avg_cl_precision_top
            
            avg_cl_precision_bottom = np.mean(all_cl_precisions_bottom)
            logging_dict[f"cl_average_precision_bottom_{i}{suffix}"] = avg_cl_precision_bottom

            # Calculate and log overall averages for cell lines - both
            avg_cl_precision_both = (avg_cl_precision_top + avg_cl_precision_bottom) / 2
            logging_dict[f"cl_avg_precision_both_{i}{suffix}"] = avg_cl_precision_both
        
        # Calculate and log overall average hit ratio - interventions
        if all_iv_precisions_top:
            avg_iv_precision_top = np.mean(all_iv_precisions_top)
            logging_dict[f"iv_average_precision_top_{i}{suffix}"] = avg_iv_precision_top
            
            avg_iv_precision_bottom = np.mean(all_iv_precisions_bottom)
            logging_dict[f"iv_average_precision_bottom_{i}{suffix}"] = avg_iv_precision_bottom

            # Calculate and log overall averages for interventions - both
            avg_iv_precision_both = (avg_iv_precision_top + avg_iv_precision_bottom) / 2
            logging_dict[f"iv_avg_precision_both_{i}{suffix}"] = avg_iv_precision_both
    
    return logging_dict


class HitRatioCallback(pl.Callback):
    """
    Callback that computes HitRatio metrics across interventions and cell lines.
    
    HitRatio measures how well the model ranks the top k% of samples:
    - For interventions: Groups by (iv1, iv2, ...) combination and computes ranking performance
    - For cell lines: Groups by cell_line and computes ranking performance
    - Computed per phenotype and overall
    
    The metric answers: "Of the true top k% samples in this group, how many did we 
    correctly predict in our top k%?"
    """
    
    def __init__(self, topk: list = [10, 20], device: torch.device = "cpu"):
        super().__init__()
        self.topk = topk
        self.device = device
        
        # Storage for validation data
        self.val_predictions = []
        self.val_targets = []
        self.val_phenotypes = []
        self.val_dataset_indices = []
        
        # Storage for test data  
        self.test_predictions = []
        self.test_targets = []
        self.test_phenotypes = []
        self.test_dataset_indices = []
        
        # Mappings - will be populated from the dataset
        self.index_to_cl_name = None
        self.index_to_iv_names = None
        self.index_to_ph_name = None
        self.pert_len = None
        self.experimental_data_indices = None

    def _initialize_mappings(self, trainer, dataset_type="val"):
        """Initialize reverse mappings from dataset"""
        if dataset_type == "val" and hasattr(trainer, 'val_dataloaders') and trainer.val_dataloaders:
            dataset = trainer.val_dataloaders.dataset
        elif dataset_type == "test" and hasattr(trainer, 'test_dataloaders') and trainer.test_dataloaders:
            dataset = trainer.test_dataloaders[0].dataset if isinstance(trainer.test_dataloaders, list) else trainer.test_dataloaders.dataset
        else:
            return False
        
        # Create reverse mappings
        self.index_to_cl_name = {v: k for k, v in dataset.cl_to_index.items()}
        self.index_to_iv_names = {v: k for k, v in dataset.iv_to_index.items()}
        self.index_to_ph_name = {v: k for k, v in dataset.ph_to_index.items()}
        self.pert_len = dataset.pert_len
        self.experimental_data_indices = dataset.experimental_data_indices
        
        return True

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.index_to_cl_name is None:
            if not self._initialize_mappings(trainer, "val"):
                return
            
        y_pred, y_true, phenotype = (
            outputs["y_pred"],
            outputs["y_true"], 
            outputs["phenotype"],
        )
        
        dataset_indices = batch["idx"]
        
        self.val_predictions.append(y_pred.detach().cpu())
        self.val_targets.append(y_true.detach().cpu())
        self.val_phenotypes.append(phenotype.detach().cpu())
        self.val_dataset_indices.append(dataset_indices.detach().cpu())

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.index_to_cl_name is None:
            if not self._initialize_mappings(trainer, "test"):
                return
            
        y_pred, y_true = outputs["y_pred"], outputs["y_true"]
        
        # For test, we need to extract phenotype from batch since it's not in outputs
        phenotype = batch["phenotype"]
        dataset_indices = batch["idx"]
        
        self.test_predictions.append(y_pred.detach().cpu())
        self.test_targets.append(y_true.detach().cpu())
        self.test_phenotypes.append(phenotype.detach().cpu())
        self.test_dataset_indices.append(dataset_indices.detach().cpu())

    def _compute_hit_ratios(self, predictions, targets, phenotypes, dataset_indices, suffix=""):
        """Compute hit ratios for interventions and cell lines"""
        predictions = predictions.numpy().flatten()
        targets = targets.numpy().flatten()
        phenotypes = phenotypes.numpy().flatten()
        dataset_indices = dataset_indices.numpy().flatten()
        
        # Map dataset indices back to original data
        cl_names = []
        iv_combinations = []
        ph_names = []
        
        for idx in dataset_indices:
            # Get the experimental data row for this index
            exp_row = self.experimental_data_indices[idx]
            cl_idx = exp_row[0]  # cell_line index
            ph_idx = exp_row[1]  # phenotype index
            iv_indices = exp_row[2:]  # intervention indices
            
            # Map back to original names
            cl_name = self.index_to_cl_name[cl_idx]
            iv_names = tuple(self.index_to_iv_names[iv_idx] for iv_idx in iv_indices)
            
            # For phenotypes, use the name if we have explicit phenotype embeddings,
            # otherwise use the phenotype index from the model output
            if self.index_to_ph_name and ph_idx in self.index_to_ph_name:
                ph_name = self.index_to_ph_name[ph_idx]
            else:
                ph_name = phenotypes[len(cl_names)]  # Use the phenotype from model output
            
            cl_names.append(cl_name)
            iv_combinations.append(iv_names)
            ph_names.append(ph_name)
        
        # Use phenotypes from mapping if available, otherwise from model
        final_phenotypes = np.array(ph_names) if self.index_to_ph_name else phenotypes
        
        # Compute hit ratios using the existing function
        logging_dict = compute_hit_ratio(
            predictions=predictions,
            targets=targets, 
            cl=cl_names,
            phenotypes=final_phenotypes,
            iv=iv_combinations,
            topk=self.topk,
            suffix=suffix
        )
        
        # Add summary statistics
        n_samples = len(predictions)
        n_cell_lines = len(set(cl_names))
        n_interventions = len(set(iv_combinations))
        n_phenotypes = len(set(final_phenotypes))
        
        logging_dict[f"hitratio_n_samples{suffix}"] = n_samples
        logging_dict[f"hitratio_n_cell_lines{suffix}"] = n_cell_lines
        logging_dict[f"hitratio_n_interventions{suffix}"] = n_interventions
        logging_dict[f"hitratio_n_phenotypes{suffix}"] = n_phenotypes
        
        return logging_dict

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self.val_predictions or self.index_to_cl_name is None:
            return
            
        predictions = torch.cat(self.val_predictions, dim=0)
        targets = torch.cat(self.val_targets, dim=0)
        phenotypes = torch.cat(self.val_phenotypes, dim=0)
        dataset_indices = torch.cat(self.val_dataset_indices, dim=0)
        
        # Compute hit ratios
        logging_dict = self._compute_hit_ratios(
            predictions, targets, phenotypes, dataset_indices, suffix="_validation"
        )
        
        # Log all metrics
        for key, value in logging_dict.items():
            pl_module.log(key, value, sync_dist=True, batch_size=predictions.shape[0])
        
        # Clear storage
        self.val_predictions = []
        self.val_targets = []
        self.val_phenotypes = []
        self.val_dataset_indices = []

    def on_test_epoch_end(self, trainer, pl_module):
        if not self.test_predictions or self.index_to_cl_name is None:
            return
            
        predictions = torch.cat(self.test_predictions, dim=0)
        targets = torch.cat(self.test_targets, dim=0)
        phenotypes = torch.cat(self.test_phenotypes, dim=0)
        dataset_indices = torch.cat(self.test_dataset_indices, dim=0)
        
        # Compute hit ratios
        logging_dict = self._compute_hit_ratios(
            predictions, targets, phenotypes, dataset_indices, suffix="_test"
        )
        
        # Log all metrics
        for key, value in logging_dict.items():
            pl_module.log(key, value, sync_dist=True, batch_size=predictions.shape[0])
        
        # Clear storage
        self.test_predictions = []
        self.test_targets = []
        self.test_phenotypes = []
        self.test_dataset_indices = []