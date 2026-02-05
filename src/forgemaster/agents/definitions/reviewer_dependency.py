"""Dependency reviewer agent definition for FORGEMASTER.

This module defines the dependency reviewer agent configuration, responsible for
reviewing version compatibility, license compliance, security advisories, update
availability, and bundled size impact. The dependency reviewer uses Sonnet 4.5
for focused dependency analysis.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class DependencyReviewerConfig(BaseModel):
    """Configuration for dependency reviewer agent.

    The dependency reviewer specializes in dependency version constraints, license
    compatibility, known security vulnerabilities, available updates, transitive
    dependencies, and bundle size implications.

    Attributes:
        agent_type: Agent type identifier
        model: Claude model to use (Sonnet 4.5 for dependency review)
        tools: Available tools for review work (read-only)
        max_tokens: Maximum tokens for response generation
        temperature: Sampling temperature (0.3 for focused review)
        purpose: High-level description of agent role
        output_format: Expected output structure (review_findings)
    """

    agent_type: str = Field(default="reviewer_dependency")
    model: str = Field(default="claude-sonnet-4-5-20250929")
    tools: list[str] = Field(default=["Read", "Grep", "Glob"])
    max_tokens: int = Field(default=8192)
    temperature: float = Field(default=0.3)
    purpose: str = Field(
        default="Dependency review: version compatibility, license compliance, security advisories, update availability, bundled size impact"
    )
    output_format: str = Field(default="review_findings")


def get_dependency_reviewer_config() -> DependencyReviewerConfig:
    """Create dependency reviewer agent configuration.

    Returns:
        DependencyReviewerConfig instance with default settings for dependency review

    Example:
        >>> config = get_dependency_reviewer_config()
        >>> print(config.agent_type)
        'reviewer_dependency'
        >>> print(config.model)
        'claude-sonnet-4-5-20250929'
    """
    return DependencyReviewerConfig()
