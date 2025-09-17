#!/usr/bin/env python3
"""
Prophet MCP Server

A Model Context Protocol (MCP) server that provides LLMs with access to Prophet,
a transformer-based model for predicting cellular responses to perturbations.

This server enables LLMs to:
- Load and manage Prophet models
- Predict drug efficacy and genetic perturbation effects
- Design biological experiments
- Analyze cellular responses across different conditions

Usage:
    python prophet_mcp_server.py

Requirements:
    pip install mcp pandas numpy torch pytorch-lightning scikit-learn
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Sequence, Union
from pathlib import Path
import traceback

import pandas as pd
import numpy as np

# MCP imports
from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp.types import Resource, Tool, TextContent, ImageContent, EmbeddedResource
import mcp.types as types

# Prophet imports
try:
    from prophet.core.prophet import Prophet
    from prophet.utils import validate_prophet_inputs

    PROPHET_AVAILABLE = True
except ImportError as e:
    PROPHET_AVAILABLE = False
    PROPHET_IMPORT_ERROR = str(e)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProphetMCPServer:
    """MCP Server for Prophet biological prediction models."""

    def __init__(self):
        self.server = Server("prophet-predictor")
        self.loaded_models: Dict[str, Prophet] = {}
        self.model_metadata: Dict[str, Dict] = {}

        # Register tools
        self._register_tools()

    def _register_tools(self):
        """Register all MCP tools."""

        @self.server.list_tools()
        async def handle_list_tools() -> list[types.Tool]:
            """List available Prophet tools."""
            return [
                types.Tool(
                    name="list_available_models",
                    description="List all available pretrained Prophet models from HuggingFace Hub",
                    inputSchema={"type": "object", "properties": {}, "required": []},
                ),
                types.Tool(
                    name="load_prophet_model",
                    description="Load a Prophet model for biological predictions",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "model_name": {
                                "type": "string",
                                "description": "Name of the Prophet model to load (e.g., 'prophet-base', 'prophet-gdsc')",
                                "default": "prophet-base",
                            },
                            "cache_dir": {
                                "type": "string",
                                "description": "Directory to cache downloaded model files",
                            },
                            "force_download": {
                                "type": "boolean",
                                "description": "Whether to force re-download even if files exist",
                                "default": False,
                            },
                        },
                        "required": ["model_name"],
                    },
                ),
                types.Tool(
                    name="load_prophet_dataset_model",
                    description="Load a Prophet model trained on a specific dataset with custom parameters",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "dataset": {
                                "type": "string",
                                "description": "Dataset name (e.g., 'GDSC', 'CTRP', 'LINCS', 'JUMP', 'SCORE')",
                                "enum": [
                                    "CTRP",
                                    "GDSC",
                                    "GDSCcomb",
                                    "Horlbeck",
                                    "JUMP",
                                    "LINCS",
                                    "SCORE",
                                ],
                            },
                            "split": {
                                "type": "string",
                                "description": "Split method used during training",
                                "enum": ["unseen_perturbations", "unseen_cell_lines"],
                                "default": "unseen_cell_lines",
                            },
                            "seed": {
                                "type": "integer",
                                "description": "Random seed used during training",
                                "enum": [110, 1995, 2024],
                                "default": 110,
                            },
                            "fold": {
                                "type": "integer",
                                "description": "Cross-validation fold number (0-4)",
                                "enum": [0, 1, 2, 3, 4],
                                "default": 0,
                            },
                            "unbalanced": {
                                "type": "boolean",
                                "description": "Whether the model used unbalanced sampling during training",
                                "default": False,
                            },
                            "cache_dir": {
                                "type": "string",
                                "description": "Directory to cache downloaded model files",
                            },
                            "force_download": {
                                "type": "boolean",
                                "description": "Whether to force re-download even if files exist",
                                "default": False,
                            },
                        },
                        "required": ["dataset"],
                    },
                ),
                types.Tool(
                    name="get_model_info",
                    description="Get information about a loaded Prophet model",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "model_name": {
                                "type": "string",
                                "description": "Name of the loaded model to inspect",
                            }
                        },
                        "required": ["model_name"],
                    },
                ),
                types.Tool(
                    name="predict_cellular_response",
                    description="Predict cellular responses to interventions using Prophet",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "model_name": {
                                "type": "string",
                                "description": "Name of the loaded Prophet model to use",
                                "default": "prophet-base",
                            },
                            "interventions": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of interventions (genes, drugs, etc.) to test",
                            },
                            "cell_lines": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of cell lines to test interventions on",
                            },
                            "phenotypes": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of phenotypes/readouts to predict",
                                "default": ["viability"],
                            },
                            "combination_mode": {
                                "type": "string",
                                "enum": ["single", "pairwise", "custom"],
                                "description": "How to handle multiple interventions: single (one at a time), pairwise (all pairs), or custom",
                                "default": "single",
                            },
                            "custom_combinations": {
                                "type": "array",
                                "items": {"type": "array", "items": {"type": "string"}},
                                "description": "Custom intervention combinations if combination_mode is 'custom'",
                            },
                        },
                        "required": ["interventions", "cell_lines"],
                    },
                ),
                types.Tool(
                    name="batch_predict_from_csv",
                    description="Run batch predictions from a CSV file containing experimental combinations",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "model_name": {
                                "type": "string",
                                "description": "Name of the loaded Prophet model to use",
                            },
                            "csv_path": {
                                "type": "string",
                                "description": "Path to CSV file with columns: cell_line, iv1, iv2, phenotype",
                            },
                            "output_path": {
                                "type": "string",
                                "description": "Path to save predictions CSV file",
                            },
                        },
                        "required": ["model_name", "csv_path"],
                    },
                ),
                types.Tool(
                    name="find_top_predictions",
                    description="Find top predicted responses for given criteria",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "model_name": {
                                "type": "string",
                                "description": "Name of the loaded Prophet model to use",
                            },
                            "interventions": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of interventions to screen",
                            },
                            "cell_lines": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of cell lines to screen",
                            },
                            "phenotype": {
                                "type": "string",
                                "description": "Phenotype to optimize for",
                                "default": "viability",
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Number of top predictions to return",
                                "default": 10,
                            },
                            "minimize": {
                                "type": "boolean",
                                "description": "Whether to find minimum (True) or maximum (False) predictions",
                                "default": True,
                            },
                        },
                        "required": ["interventions", "cell_lines"],
                    },
                ),
                types.Tool(
                    name="compare_interventions",
                    description="Compare predicted efficacy of different interventions across cell lines",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "model_name": {
                                "type": "string",
                                "description": "Name of the loaded Prophet model to use",
                            },
                            "interventions": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of interventions to compare",
                            },
                            "cell_lines": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of cell lines for comparison",
                            },
                            "phenotype": {
                                "type": "string",
                                "description": "Phenotype to compare",
                                "default": "viability",
                            },
                        },
                        "required": ["interventions", "cell_lines"],
                    },
                ),
            ]

        @self.server.call_tool()
        async def handle_call_tool(
            name: str, arguments: dict
        ) -> list[types.TextContent]:
            """Handle tool calls."""
            try:
                if not PROPHET_AVAILABLE:
                    return [
                        types.TextContent(
                            type="text",
                            text=f"❌ Prophet not available: {PROPHET_IMPORT_ERROR}\n\nPlease install Prophet and its dependencies.",
                        )
                    ]

                if name == "list_available_models":
                    return await self._list_available_models()
                elif name == "load_prophet_model":
                    return await self._load_prophet_model(arguments)
                elif name == "load_prophet_dataset_model":
                    return await self._load_prophet_dataset_model(arguments)
                elif name == "get_model_info":
                    return await self._get_model_info(arguments)
                elif name == "predict_cellular_response":
                    return await self._predict_cellular_response(arguments)
                elif name == "batch_predict_from_csv":
                    return await self._batch_predict_from_csv(arguments)
                elif name == "find_top_predictions":
                    return await self._find_top_predictions(arguments)
                elif name == "compare_interventions":
                    return await self._compare_interventions(arguments)
                else:
                    return [
                        types.TextContent(type="text", text=f"❌ Unknown tool: {name}")
                    ]

            except Exception as e:
                logger.error(f"Error in tool {name}: {str(e)}")
                logger.error(traceback.format_exc())
                return [
                    types.TextContent(
                        type="text",
                        text=f"❌ Error in {name}: {str(e)}\n\nPlease check your inputs and try again.",
                    )
                ]

    async def _list_available_models(self) -> list[types.TextContent]:
        """List available Prophet models."""
        try:
            models_info = Prophet.available_models()

            if not models_info:
                return [
                    types.TextContent(
                        type="text",
                        text="📋 No pretrained models found. Please check your internet connection.",
                    )
                ]

            result = "📋 **Available Prophet Models:**\n\n"
            for model_name, info in models_info.items():
                result += f"**{model_name}**\n"
                result += (
                    f"  - Description: {info.get('description', 'No description')}\n"
                )
                result += f"  - Size: {info.get('size', 'Unknown')}\n"
                result += (
                    f"  - Architecture: {info.get('architecture', 'Transformer')}\n\n"
                )

            result += "\n💡 Use `load_prophet_model` with any of these model names to get started!"

            return [types.TextContent(type="text", text=result)]

        except Exception as e:
            Prophet.list_models()  # Fallback to print version
            return [
                types.TextContent(
                    type="text",
                    text="📋 Available models listed above. Use Prophet.available_models() for programmatic access.",
                )
            ]

    async def _load_prophet_model(self, arguments: dict) -> list[types.TextContent]:
        """Load a Prophet model."""
        model_name = arguments.get("model_name", "prophet-base")
        cache_dir = arguments.get("cache_dir")
        force_download = arguments.get("force_download", False)

        if model_name in self.loaded_models:
            return [
                types.TextContent(
                    type="text",
                    text=f"✅ Model '{model_name}' is already loaded and ready for predictions.",
                )
            ]

        try:
            logger.info(f"Loading Prophet model: {model_name}")

            model = Prophet.from_pretrained(
                model_name=model_name,
                cache_dir=cache_dir,
                force_download=force_download,
            )

            self.loaded_models[model_name] = model

            # Store metadata
            self.model_metadata[model_name] = {
                "architecture": getattr(model, "architecture", "Transformer"),
                "loaded_at": pd.Timestamp.now().isoformat(),
                "cache_dir": cache_dir,
            }

            return [
                types.TextContent(
                    type="text",
                    text=f"✅ Successfully loaded Prophet model '{model_name}'!\n\n"
                    f"🔬 **Model Info:**\n"
                    f"- Architecture: {self.model_metadata[model_name]['architecture']}\n"
                    f"- Loaded at: {self.model_metadata[model_name]['loaded_at']}\n\n"
                    f"🚀 Ready for biological predictions! Use `predict_cellular_response` to get started.",
                )
            ]

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            return [
                types.TextContent(
                    type="text",
                    text=f"❌ Failed to load model '{model_name}': {str(e)}\n\n"
                    f"💡 **Troubleshooting:**\n"
                    f"- Check your internet connection\n"
                    f"- Verify the model name with `list_available_models`\n"
                    f"- Try setting force_download=true if you have cached files",
                )
            ]

    async def _load_prophet_dataset_model(
        self, arguments: dict
    ) -> list[types.TextContent]:
        """Load a Prophet model from a specific dataset with custom parameters."""
        dataset = arguments.get("dataset")
        split = arguments.get("split", "leave_cl_out")
        seed = arguments.get("seed", 42)
        fold = arguments.get("fold", 0)
        unbalanced = arguments.get("unbalanced", False)
        cache_dir = arguments.get("cache_dir")
        force_download = arguments.get("force_download", False)

        # Create a unique model name for this configuration
        model_name = f"{dataset.lower()}-{split}-s{seed}-f{fold}-u{unbalanced}"

        if model_name in self.loaded_models:
            return [
                types.TextContent(
                    type="text",
                    text=f"✅ Model '{model_name}' is already loaded and ready for predictions.",
                )
            ]

        try:
            logger.info(f"Loading Prophet model from dataset: {dataset}")

            model = Prophet.from_dataset(
                dataset=dataset,
                split=split,
                seed=seed,
                fold=fold,
                unbalanced=unbalanced,
                cache_dir=cache_dir,
                force_download=force_download,
            )

            self.loaded_models[model_name] = model

            # Store metadata
            self.model_metadata[model_name] = {
                "architecture": getattr(model, "architecture", "Transformer"),
                "loaded_at": pd.Timestamp.now().isoformat(),
                "cache_dir": cache_dir,
                "dataset": dataset,
                "split": split,
                "seed": seed,
                "fold": fold,
                "unbalanced": unbalanced,
            }

            return [
                types.TextContent(
                    type="text",
                    text=f"✅ Successfully loaded Prophet model from {dataset}!\n\n"
                    f"🔬 **Model Configuration:**\n"
                    f"- Dataset: {dataset}\n"
                    f"- Split: {split}\n"
                    f"- Seed: {seed}\n"
                    f"- Fold: {fold}\n"
                    f"- Unbalanced: {unbalanced}\n"
                    f"- Architecture: {self.model_metadata[model_name]['architecture']}\n"
                    f"- Loaded at: {self.model_metadata[model_name]['loaded_at']}\n\n"
                    f"🚀 Ready for biological predictions! Use `predict_cellular_response` to get started.",
                )
            ]

        except Exception as e:
            logger.error(f"Failed to load model from {dataset}: {str(e)}")
            return [
                types.TextContent(
                    type="text",
                    text=f"❌ Failed to load model from dataset '{dataset}': {str(e)}\n\n"
                    f"💡 **Troubleshooting:**\n"
                    f"- Check your internet connection\n"
                    f"- Verify the dataset name is correct\n"
                    f"- Available datasets: {Prophet.available_datasets()}\n"
                    f"- Available splits: {Prophet.available_splits()}\n"
                    f"- Try setting force_download=true if you have cached files",
                )
            ]

    async def _get_model_info(self, arguments: dict) -> list[types.TextContent]:
        """Get information about a loaded model."""
        model_name = arguments.get("model_name")

        if not model_name:
            loaded_models = list(self.loaded_models.keys())
            if not loaded_models:
                return [
                    types.TextContent(
                        type="text",
                        text="❌ No models loaded. Use `load_prophet_model` first.",
                    )
                ]
            return [
                types.TextContent(
                    type="text",
                    text=f"📋 **Loaded Models:** {', '.join(loaded_models)}\n\n"
                    f"💡 Specify a model_name to get detailed information.",
                )
            ]

        if model_name not in self.loaded_models:
            return [
                types.TextContent(
                    type="text",
                    text=f"❌ Model '{model_name}' not loaded. Available models: {list(self.loaded_models.keys())}",
                )
            ]

        model = self.loaded_models[model_name]
        metadata = self.model_metadata[model_name]

        info = f"🔬 **Prophet Model: {model_name}**\n\n"
        info += f"**Architecture:** {metadata['architecture']}\n"
        info += f"**Loaded:** {metadata['loaded_at']}\n\n"

        # Add model-specific information
        if hasattr(model, "iv_embedding") and model.iv_embedding is not None:
            info += f"**Intervention Embeddings:** {model.iv_embedding.shape[0]} interventions, {model.iv_embedding.shape[1]} dimensions\n"
        if hasattr(model, "cl_embedding") and model.cl_embedding is not None:
            info += f"**Cell Line Embeddings:** {model.cl_embedding.shape[0]} cell lines, {model.cl_embedding.shape[1]} dimensions\n"
        if hasattr(model, "phenotypes") and model.phenotypes is not None:
            info += f"**Phenotypes:** {len(model.phenotypes)} types\n"

        info += (
            f"\n🚀 **Ready for predictions!** Use `predict_cellular_response` to start."
        )

        return [types.TextContent(type="text", text=info)]

    def _validate_entities(
        self, model, interventions: List[str], cell_lines: List[str]
    ) -> tuple[List[str], List[str], str]:
        """Validate interventions and cell lines against model embeddings.

        Returns:
            tuple: (valid_interventions, valid_cell_lines, warning_message)
        """
        warnings = []

        # Check interventions against embeddings
        valid_interventions = []
        invalid_interventions = []

        if hasattr(model, "iv_embedding") and model.iv_embedding is not None:
            available_interventions = set(model.iv_embedding.index.str.lower())

            for intervention in interventions:
                if intervention.lower() in available_interventions:
                    valid_interventions.append(intervention)
                else:
                    invalid_interventions.append(intervention)
        else:
            # If no embedding info available, assume all are valid
            valid_interventions = interventions

        # Check cell lines against embeddings
        valid_cell_lines = []
        invalid_cell_lines = []

        if hasattr(model, "cl_embedding") and model.cl_embedding is not None:
            available_cell_lines = set(model.cl_embedding.index.str.lower())

            for cell_line in cell_lines:
                if cell_line.lower() in available_cell_lines:
                    valid_cell_lines.append(cell_line)
                else:
                    invalid_cell_lines.append(cell_line)
        else:
            # If no embedding info available, assume all are valid
            valid_cell_lines = cell_lines

        # Create warning message
        warning_parts = []
        if invalid_interventions:
            warning_parts.append(
                f"⚠️ **Unknown interventions (will be skipped):** {', '.join(invalid_interventions)}"
            )
        if invalid_cell_lines:
            warning_parts.append(
                f"⚠️ **Unknown cell lines (will be skipped):** {', '.join(invalid_cell_lines)}"
            )

        warning_message = "\n".join(warning_parts)

        return valid_interventions, valid_cell_lines, warning_message

    async def _predict_cellular_response(
        self, arguments: dict
    ) -> list[types.TextContent]:
        """Predict cellular responses to interventions."""
        model_name = arguments.get("model_name", "prophet-base")
        interventions = arguments.get("interventions", [])
        cell_lines = arguments.get("cell_lines", [])
        phenotypes = arguments.get("phenotypes", ["viability"])
        combination_mode = arguments.get("combination_mode", "single")
        custom_combinations = arguments.get("custom_combinations", [])

        if model_name not in self.loaded_models:
            return [
                types.TextContent(
                    type="text",
                    text=f"❌ Model '{model_name}' not loaded. Use `load_prophet_model` first.",
                )
            ]

        if not interventions or not cell_lines:
            return [
                types.TextContent(
                    type="text",
                    text="❌ Please provide both interventions and cell_lines for prediction.",
                )
            ]

        model = self.loaded_models[model_name]

        # Validate interventions and cell lines against embeddings
        valid_interventions, valid_cell_lines, validation_warnings = (
            self._validate_entities(model, interventions, cell_lines)
        )

        if not valid_interventions:
            return [
                types.TextContent(
                    type="text",
                    text="❌ No valid interventions found in model embeddings.\n\n"
                    f"**Provided:** {', '.join(interventions)}\n"
                    f"💡 **Tip:** Use `get_model_info` to see available entities or check spelling.",
                )
            ]

        if not valid_cell_lines:
            return [
                types.TextContent(
                    type="text",
                    text="❌ No valid cell lines found in model embeddings.\n\n"
                    f"**Provided:** {', '.join(cell_lines)}\n"
                    f"💡 **Tip:** Use `get_model_info` to see available entities or check spelling.",
                )
            ]

        try:
            # Create prediction DataFrame based on combination mode using validated entities
            prediction_data = []

            if combination_mode == "single":
                # Single interventions
                for intervention in valid_interventions:
                    for cell_line in valid_cell_lines:
                        for phenotype in phenotypes:
                            prediction_data.append(
                                {
                                    "iv1": intervention,
                                    "iv2": "negative_gene",  # Default negative control
                                    "cell_line": cell_line,
                                    "phenotype": phenotype,
                                }
                            )

            elif combination_mode == "pairwise":
                # All pairwise combinations
                for i, iv1 in enumerate(valid_interventions):
                    for iv2 in valid_interventions[i + 1 :]:  # Avoid duplicates
                        for cell_line in valid_cell_lines:
                            for phenotype in phenotypes:
                                prediction_data.append(
                                    {
                                        "iv1": iv1,
                                        "iv2": iv2,
                                        "cell_line": cell_line,
                                        "phenotype": phenotype,
                                    }
                                )

            elif combination_mode == "custom":
                # Custom combinations
                if not custom_combinations:
                    return [
                        types.TextContent(
                            type="text",
                            text="❌ custom_combinations required when combination_mode is 'custom'",
                        )
                    ]

                for combination in custom_combinations:
                    if len(combination) < 1 or len(combination) > 2:
                        continue  # Skip invalid combinations

                    iv1 = combination[0]
                    iv2 = combination[1] if len(combination) > 1 else "negative_gene"

                    for cell_line in valid_cell_lines:
                        for phenotype in phenotypes:
                            prediction_data.append(
                                {
                                    "iv1": iv1,
                                    "iv2": iv2,
                                    "cell_line": cell_line,
                                    "phenotype": phenotype,
                                }
                            )

            if not prediction_data:
                return [
                    types.TextContent(
                        type="text",
                        text="❌ No valid predictions to make. Check your inputs.",
                    )
                ]

            df = pd.DataFrame(prediction_data)
            logger.info(f"Making {len(df)} predictions...")

            # Make predictions
            predictions = model.predict(df)

            # Format results
            results = []
            for _, row in predictions.iterrows():
                intervention_str = row["iv1"]
                if row["iv2"] != "negative_gene":
                    intervention_str += f" + {row['iv2']}"

                results.append(
                    {
                        "intervention": intervention_str,
                        "cell_line": row["cell_line"],
                        "phenotype": row["phenotype"],
                        "predicted_response": float(row["pred"]),
                    }
                )

            # Sort by predicted response for easier interpretation
            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values("predicted_response")

            # Create summary
            summary = f"🔬 **Prophet Predictions Complete!**\n\n"

            # Add validation warnings if any
            if validation_warnings:
                summary += f"{validation_warnings}\n\n"

            summary += f"**Predictions made:** {len(results)}\n"
            summary += f"**Valid interventions used:** {len(valid_interventions)}\n"
            summary += f"**Valid cell lines used:** {len(valid_cell_lines)}\n"
            summary += f"**Phenotypes:** {len(phenotypes)}\n\n"

            # Show top and bottom predictions
            summary += "**🎯 Most Effective (Lowest Response):**\n"
            for _, row in results_df.head(5).iterrows():
                summary += f"- {row['intervention']} in {row['cell_line']}: {row['predicted_response']:.3f}\n"

            summary += f"\n**⚠️ Least Effective (Highest Response):**\n"
            for _, row in results_df.tail(5).iterrows():
                summary += f"- {row['intervention']} in {row['cell_line']}: {row['predicted_response']:.3f}\n"

            summary += f"\n**📊 Full Results:**\n"
            summary += results_df.to_string(index=False, float_format="%.3f")

            return [types.TextContent(type="text", text=summary)]

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return [
                types.TextContent(
                    type="text",
                    text=f"❌ Prediction failed: {str(e)}\n\n"
                    f"💡 **Possible issues:**\n"
                    f"- Intervention or cell line names not in model embeddings\n"
                    f"- Invalid phenotype names\n"
                    f"- Check spelling and try again",
                )
            ]

    async def _batch_predict_from_csv(self, arguments: dict) -> list[types.TextContent]:
        """Run batch predictions from CSV file."""
        model_name = arguments.get("model_name")
        csv_path = arguments.get("csv_path")
        output_path = arguments.get("output_path")

        if model_name not in self.loaded_models:
            return [
                types.TextContent(
                    type="text",
                    text=f"❌ Model '{model_name}' not loaded. Use `load_prophet_model` first.",
                )
            ]

        if not Path(csv_path).exists():
            return [
                types.TextContent(
                    type="text", text=f"❌ CSV file not found: {csv_path}"
                )
            ]

        try:
            # Load CSV
            df = pd.read_csv(csv_path)
            required_cols = ["cell_line", "iv1", "iv2", "phenotype"]
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                return [
                    types.TextContent(
                        type="text",
                        text=f"❌ Missing required columns: {missing_cols}\n"
                        f"Required: {required_cols}\n"
                        f"Found: {list(df.columns)}",
                    )
                ]

            model = self.loaded_models[model_name]

            logger.info(f"Running batch predictions on {len(df)} rows...")
            predictions = model.predict(df)

            # Save results if output path provided
            if output_path:
                predictions.to_csv(output_path, index=False)
                save_msg = f"💾 Results saved to: {output_path}\n\n"
            else:
                save_msg = ""

            # Create summary
            summary = f"📊 **Batch Predictions Complete!**\n\n"
            summary += save_msg
            summary += f"**Total predictions:** {len(predictions)}\n"
            summary += f"**Unique interventions:** {predictions['iv1'].nunique()}\n"
            summary += f"**Unique cell lines:** {predictions['cell_line'].nunique()}\n"
            summary += (
                f"**Unique phenotypes:** {predictions['phenotype'].nunique()}\n\n"
            )

            # Show statistics
            pred_stats = predictions["pred"].describe()
            summary += f"**Prediction Statistics:**\n"
            summary += f"- Mean: {pred_stats['mean']:.3f}\n"
            summary += f"- Std: {pred_stats['std']:.3f}\n"
            summary += f"- Min: {pred_stats['min']:.3f}\n"
            summary += f"- Max: {pred_stats['max']:.3f}\n\n"

            # Show sample results
            summary += f"**Sample Results:**\n"
            summary += predictions.head(10).to_string(index=False, float_format="%.3f")

            return [types.TextContent(type="text", text=summary)]

        except Exception as e:
            return [
                types.TextContent(
                    type="text", text=f"❌ Batch prediction failed: {str(e)}"
                )
            ]

    async def _find_top_predictions(self, arguments: dict) -> list[types.TextContent]:
        """Find top predicted responses."""
        model_name = arguments.get("model_name", "prophet-base")
        interventions = arguments.get("interventions", [])
        cell_lines = arguments.get("cell_lines", [])
        phenotype = arguments.get("phenotype", "viability")
        top_k = arguments.get("top_k", 10)
        minimize = arguments.get("minimize", True)

        if model_name not in self.loaded_models:
            return [
                types.TextContent(
                    type="text",
                    text=f"❌ Model '{model_name}' not loaded. Use `load_prophet_model` first.",
                )
            ]

        # Use predict_cellular_response to get all predictions
        pred_args = {
            "model_name": model_name,
            "interventions": interventions,
            "cell_lines": cell_lines,
            "phenotypes": [phenotype],
            "combination_mode": "single",
        }

        # Get predictions (reuse existing method)
        model = self.loaded_models[model_name]

        try:
            # Create prediction data
            prediction_data = []
            for intervention in interventions:
                for cell_line in cell_lines:
                    prediction_data.append(
                        {
                            "iv1": intervention,
                            "iv2": "negative_gene",
                            "cell_line": cell_line,
                            "phenotype": phenotype,
                        }
                    )

            df = pd.DataFrame(prediction_data)
            predictions = model.predict(df)

            # Sort and get top K
            predictions = predictions.sort_values("pred", ascending=minimize)
            top_predictions = predictions.head(top_k)

            # Format results
            direction = "Most Effective" if minimize else "Highest Response"
            summary = f"🎯 **{direction} Predictions (Top {top_k}):**\n\n"

            for i, (_, row) in enumerate(top_predictions.iterrows(), 1):
                summary += f"{i:2d}. **{row['iv1']}** in **{row['cell_line']}**: {row['pred']:.3f}\n"

            # Add statistics
            all_preds = predictions["pred"]
            summary += f"\n📊 **Overall Statistics:**\n"
            summary += f"- Total predictions: {len(predictions)}\n"
            summary += f"- Mean response: {all_preds.mean():.3f}\n"
            summary += f"- Best response: {all_preds.min():.3f}\n"
            summary += f"- Worst response: {all_preds.max():.3f}\n"

            return [types.TextContent(type="text", text=summary)]

        except Exception as e:
            return [
                types.TextContent(
                    type="text", text=f"❌ Failed to find top predictions: {str(e)}"
                )
            ]

    async def _compare_interventions(self, arguments: dict) -> list[types.TextContent]:
        """Compare interventions across cell lines."""
        model_name = arguments.get("model_name", "prophet-base")
        interventions = arguments.get("interventions", [])
        cell_lines = arguments.get("cell_lines", [])
        phenotype = arguments.get("phenotype", "viability")

        if model_name not in self.loaded_models:
            return [
                types.TextContent(
                    type="text",
                    text=f"❌ Model '{model_name}' not loaded. Use `load_prophet_model` first.",
                )
            ]

        model = self.loaded_models[model_name]

        try:
            # Create prediction data
            prediction_data = []
            for intervention in interventions:
                for cell_line in cell_lines:
                    prediction_data.append(
                        {
                            "iv1": intervention,
                            "iv2": "negative_gene",
                            "cell_line": cell_line,
                            "phenotype": phenotype,
                        }
                    )

            df = pd.DataFrame(prediction_data)
            predictions = model.predict(df)

            # Create comparison matrix
            comparison_matrix = predictions.pivot(
                index="iv1", columns="cell_line", values="pred"
            )

            summary = f"🔬 **Intervention Comparison Matrix**\n\n"
            summary += f"**Phenotype:** {phenotype}\n"
            summary += f"**Interventions:** {len(interventions)}\n"
            summary += f"**Cell Lines:** {len(cell_lines)}\n\n"

            # Show matrix
            summary += "**Predicted Responses:**\n"
            summary += comparison_matrix.to_string(float_format="%.3f")

            # Add intervention rankings
            summary += f"\n\n📊 **Intervention Rankings (by mean efficacy):**\n"
            intervention_means = comparison_matrix.mean(axis=1).sort_values()

            for i, (intervention, mean_response) in enumerate(
                intervention_means.items(), 1
            ):
                summary += f"{i:2d}. **{intervention}**: {mean_response:.3f} (mean across cell lines)\n"

            # Add cell line sensitivity
            summary += f"\n🧬 **Cell Line Sensitivity (by mean response):**\n"
            cell_line_means = comparison_matrix.mean(axis=0).sort_values()

            for i, (cell_line, mean_response) in enumerate(cell_line_means.items(), 1):
                summary += f"{i:2d}. **{cell_line}**: {mean_response:.3f} (mean across interventions)\n"

            return [types.TextContent(type="text", text=summary)]

        except Exception as e:
            return [
                types.TextContent(type="text", text=f"❌ Comparison failed: {str(e)}")
            ]

    async def run(self):
        """Run the MCP server."""
        from mcp.server.stdio import stdio_server

        logger.info("🚀 Starting Prophet MCP Server...")
        logger.info("Ready to help with biological predictions!")

        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="prophet-predictor",
                    server_version="1.0.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )


def main():
    """Main entry point."""
    if not PROPHET_AVAILABLE:
        print(f"❌ Prophet not available: {PROPHET_IMPORT_ERROR}")
        print("\nPlease install Prophet and its dependencies:")
        print("pip install torch pytorch-lightning scikit-learn pandas numpy")
        print("# Then install Prophet from source or PyPI")
        return

    server = ProphetMCPServer()
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
