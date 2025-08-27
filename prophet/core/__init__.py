"""
Core Prophet functionality.

This module contains the main Prophet class and configuration management.
"""

from .prophet import Prophet
from .config import set_config

__all__ = ["Prophet", "set_config"]
