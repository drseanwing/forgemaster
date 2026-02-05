"""SCM/CI reviewer agent definition for FORGEMASTER.

This module defines the SCM/CI reviewer agent configuration, responsible for
reviewing branch strategy, commit message quality, CI pipeline configuration,
test coverage gates, and deployment safety. The SCM reviewer uses Sonnet 4.5
for focused SCM/CI analysis.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ScmReviewerConfig(BaseModel):
    """Configuration for SCM/CI reviewer agent.

    The SCM reviewer specializes in Git workflow patterns, commit message conventions,
    CI/CD pipeline configuration, test coverage requirements, deployment strategies,
    and version control best practices.

    Attributes:
        agent_type: Agent type identifier
        model: Claude model to use (Sonnet 4.5 for SCM review)
        tools: Available tools for review work (read-only)
        max_tokens: Maximum tokens for response generation
        temperature: Sampling temperature (0.3 for focused review)
        purpose: High-level description of agent role
        output_format: Expected output structure (review_findings)
    """

    agent_type: str = Field(default="reviewer_scm")
    model: str = Field(default="claude-sonnet-4-5-20250929")
    tools: list[str] = Field(default=["Read", "Grep", "Glob"])
    max_tokens: int = Field(default=8192)
    temperature: float = Field(default=0.3)
    purpose: str = Field(
        default="SCM/CI review: branch strategy, commit message quality, CI pipeline configuration, test coverage gates, deployment safety"
    )
    output_format: str = Field(default="review_findings")


def get_scm_reviewer_config() -> ScmReviewerConfig:
    """Create SCM/CI reviewer agent configuration.

    Returns:
        ScmReviewerConfig instance with default settings for SCM review

    Example:
        >>> config = get_scm_reviewer_config()
        >>> print(config.agent_type)
        'reviewer_scm'
        >>> print(config.model)
        'claude-sonnet-4-5-20250929'
    """
    return ScmReviewerConfig()
