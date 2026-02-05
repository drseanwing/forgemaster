"""Security reviewer agent definition for FORGEMASTER.

This module defines the security reviewer agent configuration, responsible for
reviewing injection vulnerabilities, authentication issues, secrets exposure,
insecure defaults, SSRF, and dependency security. The security reviewer uses
Opus 4.5 for deep security analysis with high token budget.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SecurityReviewerConfig(BaseModel):
    """Configuration for security reviewer agent.

    The security reviewer specializes in identifying SQL injection, XSS, CSRF,
    authentication/authorization flaws, secrets in code, insecure cryptography,
    SSRF vulnerabilities, and dependency security issues.

    Attributes:
        agent_type: Agent type identifier
        model: Claude model to use (Opus 4.5 for deep security analysis)
        tools: Available tools for review work (read-only)
        max_tokens: Maximum tokens for response generation (16384 for thorough analysis)
        temperature: Sampling temperature (0.2 for precise security findings)
        purpose: High-level description of agent role
        output_format: Expected output structure (review_findings)
    """

    agent_type: str = Field(default="reviewer_security")
    model: str = Field(default="claude-opus-4-5-20251101")
    tools: list[str] = Field(default=["Read", "Grep", "Glob"])
    max_tokens: int = Field(default=16384)
    temperature: float = Field(default=0.2)
    purpose: str = Field(
        default="Security review: injection vulnerabilities, auth issues, secrets exposure, insecure defaults, SSRF, dependency security"
    )
    output_format: str = Field(default="review_findings")


def get_security_reviewer_config() -> SecurityReviewerConfig:
    """Create security reviewer agent configuration.

    Returns:
        SecurityReviewerConfig instance with default settings for security review

    Example:
        >>> config = get_security_reviewer_config()
        >>> print(config.agent_type)
        'reviewer_security'
        >>> print(config.model)
        'claude-opus-4-5-20251101'
    """
    return SecurityReviewerConfig()
