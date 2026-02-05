"""Frontend reviewer agent definition for FORGEMASTER.

This module defines the frontend reviewer agent configuration, responsible for
reviewing component structure, accessibility, responsive design, state management,
and performance in frontend code. The frontend reviewer uses Sonnet 4.5 for
focused technical review of UI/UX implementation.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class FrontendReviewerConfig(BaseModel):
    """Configuration for frontend reviewer agent.

    The frontend reviewer specializes in reviewing React/Vue/Angular components,
    CSS/styling patterns, accessibility compliance, responsive design implementation,
    state management patterns, and client-side performance optimization.

    Attributes:
        agent_type: Agent type identifier
        model: Claude model to use (Sonnet 4.5 for frontend review)
        tools: Available tools for review work (read-only)
        max_tokens: Maximum tokens for response generation
        temperature: Sampling temperature (0.3 for focused review)
        purpose: High-level description of agent role
        output_format: Expected output structure (review_findings)
    """

    agent_type: str = Field(default="reviewer_frontend")
    model: str = Field(default="claude-sonnet-4-5-20250929")
    tools: list[str] = Field(default=["Read", "Grep", "Glob"])
    max_tokens: int = Field(default=8192)
    temperature: float = Field(default=0.3)
    purpose: str = Field(
        default="Frontend review: component structure, accessibility, responsive design, state management, performance"
    )
    output_format: str = Field(default="review_findings")


def get_frontend_reviewer_config() -> FrontendReviewerConfig:
    """Create frontend reviewer agent configuration.

    Returns:
        FrontendReviewerConfig instance with default settings for frontend review

    Example:
        >>> config = get_frontend_reviewer_config()
        >>> print(config.agent_type)
        'reviewer_frontend'
        >>> print(config.model)
        'claude-sonnet-4-5-20250929'
    """
    return FrontendReviewerConfig()
