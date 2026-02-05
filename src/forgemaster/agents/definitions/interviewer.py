"""Interviewer agent definition for FORGEMASTER.

This module defines the configuration for the interviewer agent, which is responsible
for specification clarification and requirements gathering through structured Q&A.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class InterviewerConfig(BaseModel):
    """Configuration for the interviewer agent.

    The interviewer agent conducts structured interviews to clarify specifications,
    identify gaps, and gather missing requirements through targeted questioning.

    Attributes:
        agent_type: Fixed agent type identifier
        model: Claude model to use (Opus for best reasoning)
        tools: Available tools (Read for examining specs)
        max_tokens: Maximum tokens per response
        temperature: Sampling temperature for question generation
        purpose: Agent purpose description
        max_questions: Maximum questions to ask per round
        max_rounds: Maximum interview rounds before finalization
    """

    agent_type: str = Field(default="interviewer")
    model: str = Field(default="claude-opus-4-5-20251101")
    tools: list[str] = Field(default=["Read"])
    max_tokens: int = Field(default=8192)
    temperature: float = Field(default=0.7)
    purpose: str = Field(default="Specification clarification and requirements gathering")
    max_questions: int = Field(default=10)
    max_rounds: int = Field(default=3)


def get_interviewer_config() -> InterviewerConfig:
    """Factory function to create interviewer agent configuration.

    Returns:
        InterviewerConfig instance with default settings
    """
    return InterviewerConfig()
