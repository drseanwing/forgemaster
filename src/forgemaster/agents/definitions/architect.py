"""Architect agent definition for FORGEMASTER.

This module defines the architect agent configuration, responsible for system
architecture design, technical decision-making, and architectural documentation.
The architect agent uses high-capacity reasoning to design component structures,
evaluate technology choices, and produce comprehensive architecture documents.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ArchitectConfig(BaseModel):
    """Configuration for architect agent.

    The architect agent specializes in system design, architectural patterns,
    component decomposition, and technology evaluation. It produces structured
    architecture documents with component diagrams, interface definitions,
    and justified technical decisions.

    Attributes:
        agent_type: Agent type identifier
        model: Claude model to use (Opus 4.5 for complex reasoning)
        tools: Available tools for architecture work
        max_tokens: Maximum tokens for response generation
        temperature: Sampling temperature (0.3 for focused technical decisions)
        purpose: High-level description of agent role
        output_format: Expected output structure (architecture_document)
    """

    agent_type: str = Field(default="architect")
    model: str = Field(default="claude-opus-4-5-20251101")
    tools: list[str] = Field(
        default=["Read", "Write", "Bash", "Grep", "Glob"]
    )
    max_tokens: int = Field(default=16384)
    temperature: float = Field(default=0.3)
    purpose: str = Field(default="System architecture design and technical decision-making")
    output_format: str = Field(default="architecture_document")


def get_architect_config() -> ArchitectConfig:
    """Create architect agent configuration.

    Returns:
        ArchitectConfig instance with default settings for architecture work

    Example:
        >>> config = get_architect_config()
        >>> print(config.agent_type)
        'architect'
        >>> print(config.model)
        'claude-opus-4-5-20251101'
    """
    return ArchitectConfig()
