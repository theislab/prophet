#!/usr/bin/env python3
"""
Prophet Evaluation Script

Evaluate trained Prophet models on test data.
"""

import argparse
from pathlib import Path
import yaml
import pandas as pd

from prophet import Prophet


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate Prophet models")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to trained checkpoint"
    )
    parser.add_argument("--data", type=str, required=True, help="Path to test data")
    parser.add_argument(
        "--embeddings", type=str, nargs="+", required=True, help="Embedding paths"
    )
    parser.add_argument(
        "--output", type=str, default="./evaluation_results.csv", help="Output file"
    )

    args = parser.parse_args()

    # Load test data
    if args.data.endswith(".csv"):
        test_df = pd.read_csv(args.data)
    elif args.data.endswith(".parquet"):
        test_df = pd.read_parquet(args.data)
    else:
        raise ValueError("Unsupported data format")

    # Initialize Prophet model
    model = Prophet(
        model_pth=args.checkpoint,
        iv_emb_path=args.embeddings[0] if len(args.embeddings) >= 1 else None,
        cl_emb_path=args.embeddings[1] if len(args.embeddings) >= 2 else None,
        ph_emb_path=args.embeddings[2] if len(args.embeddings) >= 3 else None,
    )

    # Make predictions
    predictions = model.predict(test_df)

    # Save results
    predictions.to_csv(args.output, index=False)
    print(f"Evaluation completed. Results saved to {args.output}")


if __name__ == "__main__":
    main()
