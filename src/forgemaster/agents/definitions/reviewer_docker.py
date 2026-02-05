"""Docker/infrastructure reviewer agent definition for FORGEMASTER.

This module defines the Docker/infrastructure reviewer agent configuration, responsible
for reviewing Dockerfile best practices, layer optimization, security scanning, compose
configuration, and resource limits. The Docker reviewer uses Sonnet 4.5 for focused
infrastructure review.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class DockerReviewerConfig(BaseModel):
    """Configuration for Docker/infrastructure reviewer agent.

    The Docker reviewer specializes in Dockerfile optimization, multi-stage builds,
    image security, docker-compose configuration, resource limits, health checks,
    and container orchestration best practices.

    Attributes:
        agent_type: Agent type identifier
        model: Claude model to use (Sonnet 4.5 for Docker review)
        tools: Available tools for review work (read-only)
        max_tokens: Maximum tokens for response generation
        temperature: Sampling temperature (0.3 for focused review)
        purpose: High-level description of agent role
        output_format: Expected output structure (review_findings)
    """

    agent_type: str = Field(default="reviewer_docker")
    model: str = Field(default="claude-sonnet-4-5-20250929")
    tools: list[str] = Field(default=["Read", "Grep", "Glob"])
    max_tokens: int = Field(default=8192)
    temperature: float = Field(default=0.3)
    purpose: str = Field(
        default="Docker/infra review: Dockerfile best practices, layer optimization, security scanning, compose configuration, resource limits"
    )
    output_format: str = Field(default="review_findings")


def get_docker_reviewer_config() -> DockerReviewerConfig:
    """Create Docker/infrastructure reviewer agent configuration.

    Returns:
        DockerReviewerConfig instance with default settings for Docker review

    Example:
        >>> config = get_docker_reviewer_config()
        >>> print(config.agent_type)
        'reviewer_docker'
        >>> print(config.model)
        'claude-sonnet-4-5-20250929'
    """
    return DockerReviewerConfig()
