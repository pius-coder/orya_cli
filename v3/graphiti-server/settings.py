"""Graphiti-server settings — reuses agent config to avoid duplication.

Fixes v2: eliminates the copy-paste Settings class.
"""
from ..agent.core.config import Settings, get_settings

__all__ = ["Settings", "get_settings"]
