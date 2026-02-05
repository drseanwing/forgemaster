"""Spec compliance reviewer agent definition for FORGEMASTER.

This module defines the spec compliance reviewer agent configuration, responsible for
reviewing interface contracts, required fields, naming conventions, expected behavior,
and completeness against specifications. The spec reviewer uses Sonnet 4.5 for
focused technical review of specification compliance.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SpecReviewerConfig(BaseModel):
    """Configuration for spec compliance reviewer agent.

    The spec reviewer specializes in verifying implementation adherence to
    specifications, checking interface contracts, validating required fields,
    ensuring naming consistency, and confirming behavioral completeness.

    Attributes:
        agent_type: Agent type identifier
        model: Claude model to use (Sonnet 4.5 for spec review)
        tools: Available tools for review work (read-only)
        max_tokens: Maximum tokens for response generation
        temperature: Sampling temperature (0.3 for focused review)
        purpose: High-level description of agent role
        output_format: Expected output structure (review_findings)
    """

    agent_type: str = Field(default="reviewer_spec")
    model: str = Field(default="claude-sonnet-4-5-20250929")
    tools: list[str] = Field(default=["Read", "Grep", "Glob"])
    max_tokens: int = Field(default=8192)
    temperature: float = Field(default=0.3)
    purpose: str = Field(
        default="Spec compliance review: interface contracts, required fields, naming conventions, expected behavior, completeness"
    )
    output_format: str = Field(default="review_findings")


def get_spec_reviewer_config() -> SpecReviewerConfig:
    """Create spec compliance reviewer agent configuration.

    Returns:
        SpecReviewerConfig instance with default settings for spec review

    Example:
        >>> config = get_spec_reviewer_config()
        >>> print(config.agent_type)
        'reviewer_spec'
        >>> print(config.model)
        'claude-sonnet-4-5-20250929'
    """
    return SpecReviewerConfig()
