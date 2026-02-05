"""Documentation reviewer agent definition for FORGEMASTER.

This module defines the documentation reviewer agent configuration, responsible for
reviewing docstring completeness, README accuracy, API documentation, inline comments
quality, and example code correctness. The docs reviewer uses Haiku 4.5 for efficient
documentation review with higher creativity.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class DocsReviewerConfig(BaseModel):
    """Configuration for documentation reviewer agent.

    The docs reviewer specializes in docstring quality, API reference accuracy,
    README completeness, code comment clarity, example code correctness, and
    documentation structure.

    Attributes:
        agent_type: Agent type identifier
        model: Claude model to use (Haiku 4.5 for efficient docs review)
        tools: Available tools for review work (read-only)
        max_tokens: Maximum tokens for response generation
        temperature: Sampling temperature (0.4 for slightly creative suggestions)
        purpose: High-level description of agent role
        output_format: Expected output structure (review_findings)
    """

    agent_type: str = Field(default="reviewer_docs")
    model: str = Field(default="claude-haiku-4-5-20251001")
    tools: list[str] = Field(default=["Read", "Grep", "Glob"])
    max_tokens: int = Field(default=8192)
    temperature: float = Field(default=0.4)
    purpose: str = Field(
        default="Documentation review: docstring completeness, README accuracy, API documentation, inline comments quality, example code correctness"
    )
    output_format: str = Field(default="review_findings")


def get_docs_reviewer_config() -> DocsReviewerConfig:
    """Create documentation reviewer agent configuration.

    Returns:
        DocsReviewerConfig instance with default settings for docs review

    Example:
        >>> config = get_docs_reviewer_config()
        >>> print(config.agent_type)
        'reviewer_docs'
        >>> print(config.model)
        'claude-haiku-4-5-20251001'
    """
    return DocsReviewerConfig()
