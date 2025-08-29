#!/usr/bin/env python3
"""
Prophet Fine-tuning Script

Fine-tune pre-trained Prophet models on new data.
"""

import argparse
from pathlib import Path
import yaml

from prophet.training import ProphetTrainer, set_seeds


def main():
    """Main fine-tuning function."""
    parser = argparse.ArgumentParser(description="Fine-tune Prophet models")
    parser.add_argument("--config", type=str, required=True, help="YAML config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to pre-trained checkpoint")
    parser.add_argument("--output-dir", type=str, help="Override output directory")
    parser.add_argument("--random-seed", type=int, help="Override random seed")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override checkpoint path
    config['checkpoint_path'] = args.checkpoint
    
    # Apply CLI overrides
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.random_seed is not None:
        config['random_seed'] = args.random_seed
    
    # Set seeds
    set_seeds(config['random_seed'])
    
    # Create output directory
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize trainer
    trainer = ProphetTrainer(args.config, config['random_seed'])
    
    # Load data
    df = trainer.load_data(config['data'])
    
    # Create splits
    splits = trainer.create_splits(df, config['splitting'], config['data'])
    
    # Run fine-tuning
    if config['splitting']['run_cv']:
        trained_models = trainer.run_cross_validation(
            splits=splits,
            data_config=config['data'],
            output_dir=str(output_dir),
            checkpoint_path=config['checkpoint_path']
        )
        print(f"Cross-validation fine-tuning completed: {len(trained_models)} models")
    else:
        # Single split fine-tuning
        train_df, val_df, test_df, descriptor = splits[0]
        model = trainer.train_single_split(
            train_df, val_df, test_df, config['data'], descriptor, 
            config['checkpoint_path']
        )
        print(f"Single split fine-tuning completed")


if __name__ == "__main__":
    main()
