"""Integration reviewer agent definition for FORGEMASTER.

This module defines the integration reviewer agent configuration, responsible for
reviewing API contract adherence, cross-module compatibility, data flow consistency,
and error propagation. The integration reviewer uses Sonnet 4.5 for focused
integration analysis.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class IntegrationReviewerConfig(BaseModel):
    """Configuration for integration reviewer agent.

    The integration reviewer specializes in API contract compliance, module boundary
    interactions, data flow across components, error propagation through layers,
    and integration point consistency.

    Attributes:
        agent_type: Agent type identifier
        model: Claude model to use (Sonnet 4.5 for integration review)
        tools: Available tools for review work (read-only)
        max_tokens: Maximum tokens for response generation
        temperature: Sampling temperature (0.3 for focused review)
        purpose: High-level description of agent role
        output_format: Expected output structure (review_findings)
    """

    agent_type: str = Field(default="reviewer_integration")
    model: str = Field(default="claude-sonnet-4-5-20250929")
    tools: list[str] = Field(default=["Read", "Grep", "Glob"])
    max_tokens: int = Field(default=8192)
    temperature: float = Field(default=0.3)
    purpose: str = Field(
        default="Integration review: API contract adherence, cross-module compatibility, data flow consistency, error propagation"
    )
    output_format: str = Field(default="review_findings")


def get_integration_reviewer_config() -> IntegrationReviewerConfig:
    """Create integration reviewer agent configuration.

    Returns:
        IntegrationReviewerConfig instance with default settings for integration review

    Example:
        >>> config = get_integration_reviewer_config()
        >>> print(config.agent_type)
        'reviewer_integration'
        >>> print(config.model)
        'claude-sonnet-4-5-20250929'
    """
    return IntegrationReviewerConfig()
