#!/usr/bin/env python3
"""
Prophet Training Script

Clean, research-focused training script using YAML configuration.
"""

import argparse
from pathlib import Path
import yaml
import torch

from prophet.training import ProphetTrainer, set_seeds


def load_config(config_path: str) -> dict:
    """Load training configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Prophet models")
    parser.add_argument("--config", type=str, required=True, help="YAML config file")
    parser.add_argument("--output-dir", type=str, help="Override output directory")
    parser.add_argument("--random-seed", type=int, help="Override random seed")

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Apply CLI overrides
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.random_seed is not None:
        config["random_seed"] = args.random_seed

    # Set seeds
    set_seeds(config["random_seed"])

    # Create output directory
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize trainer
    trainer = ProphetTrainer(args.config, config["random_seed"])

    # Load data
    df = trainer.load_data(config["data"])

    print("\nDataset Summary:")
    print(f"  • Shape: {df.shape[0]:,} rows × {df.shape[1]:,} columns")
    print(f"  • Unique phenotypes: {df['phenotype'].nunique():,}")
    print(f"  • Unique cell lines: {df['cell_line'].nunique():,}")
    print(
        f"  • Unique interventions: {len(set(df['iv1'].unique()) | set(df['iv2'].unique())):,}\n"
    )

    # Create splits
    splits = trainer.create_splits(df, config["splitting"], config["data"])

    # Run cross-validation
    if config["splitting"]["run_cv"]:
        trained_models = trainer.run_cross_validation(
            splits=splits,
            data_config=config["data"],
            output_dir=str(output_dir),
            checkpoint_path=config.get("checkpoint_path"),
        )

        mode = "fine-tuning" if config.get("checkpoint_path") else "training"
        print(f"Cross-validation {mode} completed: {len(trained_models)} models")
    else:
        # Single split training
        train_df, val_df, test_df, descriptor = splits[0]
        model = trainer.train_single_split(
            train_df,
            val_df,
            test_df,
            config["data"],
            descriptor,
            config.get("checkpoint_path"),
        )
        print(f"Single split training completed")


if __name__ == "__main__":
    print("\nGPU Information:")
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"  • Number of GPUs: {gpu_count}")
        for i in range(gpu_count):
            print(f"  • GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("  • No GPUs available - using CPU")
    print()
    main()
