"""Planner agent definition for Forgemaster.

The planner agent specializes in task decomposition, dependency graph generation,
and parallel execution analysis. It breaks down high-level project architecture
into atomic, executable tasks with proper sequencing and parallelization strategies.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PlannerConfig(BaseModel):
    """Configuration for the planner agent.

    Attributes:
        agent_type: Agent type identifier.
        model: Claude model to use for planning.
        tools: List of tool names available to the agent.
        max_tokens: Maximum tokens for agent responses.
        temperature: Sampling temperature (lower = more deterministic).
        purpose: High-level description of the agent's role.
        max_tasks_per_phase: Maximum tasks allowed in a single phase.
        max_dependency_depth: Maximum depth of dependency chains.
    """

    agent_type: str = "planner"
    model: str = "claude-opus-4-5-20251101"
    tools: list[str] = Field(default_factory=lambda: ["Read", "Grep", "Glob"])
    max_tokens: int = 16384
    temperature: float = 0.3
    purpose: str = "Task decomposition and dependency graph generation"
    max_tasks_per_phase: int = 50
    max_dependency_depth: int = 10


def get_planner_config() -> PlannerConfig:
    """Factory function to create a planner configuration.

    Returns:
        PlannerConfig with default settings for the planner agent.

    Example:
        ```python
        from forgemaster.agents.definitions.planner import get_planner_config

        config = get_planner_config()
        print(f"Planner uses {config.model}")
        ```
    """
    return PlannerConfig()
