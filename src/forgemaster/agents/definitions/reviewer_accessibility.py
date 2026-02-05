"""Accessibility reviewer agent definition for FORGEMASTER.

This module defines the accessibility reviewer agent configuration, responsible for
reviewing WCAG compliance, ARIA attributes, keyboard navigation, screen reader
compatibility, and color contrast. The accessibility reviewer uses Sonnet 4.5
for focused accessibility audit.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class AccessibilityReviewerConfig(BaseModel):
    """Configuration for accessibility reviewer agent.

    The accessibility reviewer specializes in WCAG 2.1/2.2 compliance, proper
    ARIA labeling, keyboard navigation patterns, screen reader compatibility,
    color contrast ratios, and inclusive design practices.

    Attributes:
        agent_type: Agent type identifier
        model: Claude model to use (Sonnet 4.5 for accessibility review)
        tools: Available tools for review work (read-only)
        max_tokens: Maximum tokens for response generation
        temperature: Sampling temperature (0.3 for focused review)
        purpose: High-level description of agent role
        output_format: Expected output structure (review_findings)
    """

    agent_type: str = Field(default="reviewer_accessibility")
    model: str = Field(default="claude-sonnet-4-5-20250929")
    tools: list[str] = Field(default=["Read", "Grep", "Glob"])
    max_tokens: int = Field(default=8192)
    temperature: float = Field(default=0.3)
    purpose: str = Field(
        default="Accessibility review: WCAG compliance, ARIA attributes, keyboard navigation, screen reader compatibility, color contrast"
    )
    output_format: str = Field(default="review_findings")


def get_accessibility_reviewer_config() -> AccessibilityReviewerConfig:
    """Create accessibility reviewer agent configuration.

    Returns:
        AccessibilityReviewerConfig instance with default settings for accessibility review

    Example:
        >>> config = get_accessibility_reviewer_config()
        >>> print(config.agent_type)
        'reviewer_accessibility'
        >>> print(config.model)
        'claude-sonnet-4-5-20250929'
    """
    return AccessibilityReviewerConfig()
