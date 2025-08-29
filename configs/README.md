# Prophet Training Configurations

This directory contains YAML configuration files for different Prophet training scenarios. The new YAML-based system provides a clean, comprehensive way to specify all training parameters.

## 🆕 **NEW: YAML-Based Training System**

All training is now configured through YAML files! No more long command lines - just specify everything in a config file.

## 📋 Available Configurations

### 1. `full_training_config.yaml` ⭐ **Complete Template**
- **Use case**: Comprehensive template with all options
- **Contains**: Data paths, splitting strategy, model architecture, training parameters
- **Best for**: Understanding all available options, customizing your own config

### 2. `cv_intervention_holdout.yaml` 🔬 **Cross-Validation**
- **Use case**: Leave-interventions-out cross-validation
- **Splitting**: KFold validation with intervention holdouts
- **Best for**: Evaluating model generalization to unseen interventions

### 3. `finetune_existing.yaml` 🔧 **Fine-tuning**
- **Use case**: Fine-tuning pre-trained models
- **Settings**: Lower learning rate, reduced training steps
- **Best for**: Adapting existing models to new domains

### 4. Legacy configs still available for reference
- `train_from_scratch.yaml` - Basic transformer config
- `quick_experiment.yaml` - Fast prototyping
- `large_model.yaml` - High-performance training
- `single_intervention.yaml` - Single intervention experiments

## 🚀 **New Usage Examples**

### **Simple Training**
```bash
# Just specify the config file!
python train_prophet.py --config configs/full_training_config.yaml
```

### **Cross-Validation Training**
```bash
# 5-fold cross-validation with intervention holdouts
python train_prophet.py --config configs/cv_intervention_holdout.yaml
```

### **Fine-tuning**
```bash
# Fine-tune existing model
python train_prophet.py --config configs/finetune_existing.yaml
```

### **Override Parameters**
```bash
# Override specific parameters from CLI
python train_prophet.py --config configs/full_training_config.yaml \
    --output-dir ./my_custom_output \
    --random-seed 123 \
    --run-cv
```

## 📝 **YAML Configuration Structure**

```yaml
# === DATA CONFIGURATION ===
data:
  data_path: "./data/my_dataset.csv"           # Training data
  iv_embeddings: ["./embeddings/drugs.csv"]   # Intervention embeddings  
  cl_embeddings: ["./embeddings/cells.csv"]   # Cell line embeddings
  ph_embeddings: null                          # Optional phenotype embeddings
  
  iv_cols: ["iv1", "iv2"]                      # Column names
  cl_col: "cell_line"
  ph_col: "phenotype" 
  readout_col: "value"

# === TRAINING MODE ===
mode: "scratch"                                # "scratch" or "finetune"
checkpoint_path: null                          # For fine-tuning

# === SPLITTING STRATEGY ===  
splitting:
  method: "leave_iv_out"                       # Splitting method
  val_fraction: 0.15                           # Validation fraction
  n_splits: 5                                  # KFold splits
  run_cv: true                                 # Cross-validation

# === RANDOMIZATION ===
random_seed: 42                                # Deterministic training

# === OUTPUT ===
output_dir: "./trained_models"                 # Output directory

# === MODEL CONFIGURATION ===
# (Standard Prophet configuration section)
```

## 🎯 **Splitting Methods Available**

- **`random`**: Standard random train/val/test split
- **`leave_iv_out`**: Leave interventions out (KFold over interventions)
- **`leave_cl_out`**: Leave cell lines out (KFold over cell lines)  
- **`leave_one_iv_out`**: Leave single intervention out (each intervention as test)
- **`leave_one_cl_out`**: Leave single cell line out (each cell line as test)

## ✨ **Key Benefits**

- **🧹 Clean**: No more long command lines
- **📖 Readable**: Easy to understand and modify configurations
- **🔄 Reproducible**: Save exact training configurations
- **🎲 Deterministic**: Full seed control for reproducible results
- **🔍 Validated**: Automatic config validation with helpful error messages
- **🚀 Flexible**: Override any parameter from command line if needed

## ⚙️ Configuration Parameters

### Key Parameters to Modify

**Training Scale:**
- `max_steps`: Total training iterations
- `batch_size`: Batch size (adjust based on GPU memory)
- `patience`: Early stopping patience

**Model Architecture:**
- `model_dim`: Hidden dimension size
- `num_layers`: Number of transformer layers
- `num_heads`: Number of attention heads

**Learning:**
- `lr`: Learning rate
- `warmup`: Warmup steps
- `dropout`: Dropout rates

**Data Specific:**
- `pert_len`: Number of interventions per experiment (1 or 2)
- `cell_lines_prior`: Path(s) to cell line embeddings
- `genes_prior`: Path(s) to intervention embeddings

### Embedding Dimensions
Make sure your configuration matches your embedding dimensions:
- `dim_cl`: Should match your cell line embedding size
- `dim_iv`: Should match your intervention embedding size
- `dim_phe`: Phenotype embedding size (if using explicit phenotypes)

## 📁 Directory Structure Expected

```
your_project/
├── data/
│   ├── training_data.csv          # Your experimental data
│   └── embeddings/
│       ├── cell_embeddings.csv    # Cell line embeddings
│       ├── drug_embeddings.csv    # Drug/intervention embeddings
│       └── gene_embeddings.csv    # Gene embeddings (if separate)
├── configs/                       # Configuration files
└── trained_models/               # Output directory
```

## 🎛️ Customizing Configurations

You can:
1. **Copy and modify** existing configurations for your needs
2. **Override parameters** via command line:
   ```bash
   python train_prophet.py --config configs/train_from_scratch.yaml \
       --batch-size 1024 --max-steps 75000
   ```
3. **Create new configurations** by copying and editing existing files

## 💡 Tips

- **Start with `quick_experiment.yaml`** to test your data pipeline
- **Use `train_from_scratch.yaml`** for most production training
- **Scale up to `large_model.yaml`** only if you have sufficient compute and data
- **Always validate your data format** before starting long training runs
- **Monitor WandB logs** to track training progress and tune hyperparameters
