"""Agent management for Forgemaster.

This module handles agent session creation, lifecycle management, and
communication with the Claude Agent SDK.
"""

from forgemaster.agents.result_schema import (
    AgentResult,
    IssueDiscovered,
    LessonLearned,
    TestResult,
    create_minimal_result,
    parse_agent_result,
    parse_agent_result_safe,
)
from forgemaster.agents.sdk_wrapper import AgentClient, AgentSession, ClaudeSDK
from forgemaster.agents.session import (
    AgentSessionManager,
    HealthStatus,
    SessionInfo,
    SessionMetrics,
    SessionState,
)

__all__ = [
    # SDK Wrapper
    "AgentClient",
    "AgentSession",
    "ClaudeSDK",
    # Session Management
    "AgentSessionManager",
    "SessionState",
    "HealthStatus",
    "SessionInfo",
    "SessionMetrics",
    # Result Schema
    "AgentResult",
    "TestResult",
    "IssueDiscovered",
    "LessonLearned",
    "parse_agent_result",
    "parse_agent_result_safe",
    "create_minimal_result",
]
