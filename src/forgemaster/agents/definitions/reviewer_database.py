"""Database reviewer agent definition for FORGEMASTER.

This module defines the database reviewer agent configuration, responsible for
reviewing schema design, query optimization, migration safety, index usage, and
connection management. The database reviewer uses Sonnet 4.5 for focused
technical review of database implementation.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class DatabaseReviewerConfig(BaseModel):
    """Configuration for database reviewer agent.

    The database reviewer specializes in reviewing database schemas, migration scripts,
    query patterns, index strategies, transaction boundaries, connection pooling,
    and database-specific optimizations.

    Attributes:
        agent_type: Agent type identifier
        model: Claude model to use (Sonnet 4.5 for database review)
        tools: Available tools for review work (read-only)
        max_tokens: Maximum tokens for response generation
        temperature: Sampling temperature (0.3 for focused review)
        purpose: High-level description of agent role
        output_format: Expected output structure (review_findings)
    """

    agent_type: str = Field(default="reviewer_database")
    model: str = Field(default="claude-sonnet-4-5-20250929")
    tools: list[str] = Field(default=["Read", "Grep", "Glob"])
    max_tokens: int = Field(default=8192)
    temperature: float = Field(default=0.3)
    purpose: str = Field(
        default="Database review: schema design, query optimization, migration safety, index usage, connection management"
    )
    output_format: str = Field(default="review_findings")


def get_database_reviewer_config() -> DatabaseReviewerConfig:
    """Create database reviewer agent configuration.

    Returns:
        DatabaseReviewerConfig instance with default settings for database review

    Example:
        >>> config = get_database_reviewer_config()
        >>> print(config.agent_type)
        'reviewer_database'
        >>> print(config.model)
        'claude-sonnet-4-5-20250929'
    """
    return DatabaseReviewerConfig()
