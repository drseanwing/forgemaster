"""Backend reviewer agent definition for FORGEMASTER.

This module defines the backend reviewer agent configuration, responsible for
reviewing error handling, input validation, resource management, API consistency,
and logging in backend code. The backend reviewer uses Sonnet 4.5 for focused
technical review of server-side implementation.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class BackendReviewerConfig(BaseModel):
    """Configuration for backend reviewer agent.

    The backend reviewer specializes in reviewing API endpoints, service layer logic,
    error handling patterns, input validation, resource lifecycle management,
    logging practices, and server-side performance.

    Attributes:
        agent_type: Agent type identifier
        model: Claude model to use (Sonnet 4.5 for backend review)
        tools: Available tools for review work (read-only)
        max_tokens: Maximum tokens for response generation
        temperature: Sampling temperature (0.3 for focused review)
        purpose: High-level description of agent role
        output_format: Expected output structure (review_findings)
    """

    agent_type: str = Field(default="reviewer_backend")
    model: str = Field(default="claude-sonnet-4-5-20250929")
    tools: list[str] = Field(default=["Read", "Grep", "Glob"])
    max_tokens: int = Field(default=8192)
    temperature: float = Field(default=0.3)
    purpose: str = Field(
        default="Backend review: error handling, input validation, resource management, API consistency, logging"
    )
    output_format: str = Field(default="review_findings")


def get_backend_reviewer_config() -> BackendReviewerConfig:
    """Create backend reviewer agent configuration.

    Returns:
        BackendReviewerConfig instance with default settings for backend review

    Example:
        >>> config = get_backend_reviewer_config()
        >>> print(config.agent_type)
        'reviewer_backend'
        >>> print(config.model)
        'claude-sonnet-4-5-20250929'
    """
    return BackendReviewerConfig()
