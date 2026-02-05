"""Error handling reviewer agent definition for FORGEMASTER.

This module defines the error handling reviewer agent configuration, responsible for
reviewing bare except clauses, swallowed exceptions, missing error propagation,
inconsistent error types, and missing validation. The error reviewer uses Sonnet 4.5
for focused error handling analysis.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ErrorsReviewerConfig(BaseModel):
    """Configuration for error handling reviewer agent.

    The error reviewer specializes in exception handling patterns, error propagation,
    logging on error paths, retry mechanisms, timeout handling, and validation coverage.

    Attributes:
        agent_type: Agent type identifier
        model: Claude model to use (Sonnet 4.5 for error review)
        tools: Available tools for review work (read-only)
        max_tokens: Maximum tokens for response generation
        temperature: Sampling temperature (0.3 for focused review)
        purpose: High-level description of agent role
        output_format: Expected output structure (review_findings)
    """

    agent_type: str = Field(default="reviewer_errors")
    model: str = Field(default="claude-sonnet-4-5-20250929")
    tools: list[str] = Field(default=["Read", "Grep", "Glob"])
    max_tokens: int = Field(default=8192)
    temperature: float = Field(default=0.3)
    purpose: str = Field(
        default="Error handling review: bare except clauses, swallowed exceptions, missing error propagation, inconsistent error types, missing validation"
    )
    output_format: str = Field(default="review_findings")


def get_errors_reviewer_config() -> ErrorsReviewerConfig:
    """Create error handling reviewer agent configuration.

    Returns:
        ErrorsReviewerConfig instance with default settings for error review

    Example:
        >>> config = get_errors_reviewer_config()
        >>> print(config.agent_type)
        'reviewer_errors'
        >>> print(config.model)
        'claude-sonnet-4-5-20250929'
    """
    return ErrorsReviewerConfig()
