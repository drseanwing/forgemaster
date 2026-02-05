"""Agent type definitions for Forgemaster.

This module contains agent definitions including system prompts,
tool permissions, model routing, and specialisation configurations.
"""

from __future__ import annotations

__all__ = ["ArchitectConfig", "get_architect_config"]

from forgemaster.agents.definitions.architect import (
    ArchitectConfig,
    get_architect_config,
)
