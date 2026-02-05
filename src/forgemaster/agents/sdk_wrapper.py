"""Claude Agent SDK wrapper for Forgemaster.

This module provides a Protocol-based abstraction over the Claude Agent SDK,
allowing for type-safe integration without requiring the actual SDK to be
installed during development.

The abstractions defined here will be implemented by the actual SDK when
integrated, providing a clean interface for session management, message
sending, and SDK configuration.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class AgentSession(Protocol):
    """Protocol defining the interface for an active agent session.

    This protocol abstracts the Claude Agent SDK's session object, allowing
    for testing and development without the actual SDK installed.
    """

    @property
    def session_id(self) -> str:
        """Unique identifier for this session.

        Returns:
            Session ID string
        """
        ...

    async def send_message(self, message: str) -> str:
        """Send a message to the agent and receive response.

        Args:
            message: Text message to send to the agent

        Returns:
            Agent's text response

        Raises:
            RuntimeError: If session is closed or invalid
            TimeoutError: If request exceeds configured timeout
        """
        ...

    async def close(self) -> None:
        """Close the agent session and release resources.

        This should be called when the session is no longer needed.
        Multiple calls to close() should be idempotent.
        """
        ...


@runtime_checkable
class ClaudeSDK(Protocol):
    """Protocol defining the interface for the Claude Agent SDK client.

    This protocol abstracts the main SDK client, which is responsible for
    creating and managing agent sessions with specific configurations.
    """

    async def create_session(
        self,
        model: str,
        system_prompt: str,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
    ) -> AgentSession:
        """Create a new agent session with specified configuration.

        Args:
            model: Claude model identifier (e.g., "claude-3-5-sonnet-20241022")
            system_prompt: System prompt to configure agent behavior
            tools: Optional list of tool definitions for the agent
            max_tokens: Maximum tokens for agent responses
            temperature: Sampling temperature (0.0-1.0)

        Returns:
            Active agent session instance

        Raises:
            ValueError: If configuration parameters are invalid
            RuntimeError: If SDK initialization failed
        """
        ...

    async def close(self) -> None:
        """Close the SDK client and clean up resources.

        This should be called during application shutdown to ensure
        all active sessions are properly closed.
        """
        ...


class AgentClient:
    """Wrapper around the Claude Agent SDK for session management.

    This class provides a high-level interface for initializing the SDK
    and creating agent sessions. It manages SDK lifecycle and provides
    type-safe access to SDK functionality.

    Attributes:
        sdk: The underlying Claude SDK client instance
    """

    def __init__(self, api_key: str, base_url: str | None = None):
        """Initialize the agent client.

        Args:
            api_key: Anthropic API key for authentication
            base_url: Optional custom base URL for API endpoint

        Note:
            This is a placeholder implementation. The actual SDK initialization
            will be implemented once claude-agent-sdk is integrated.
        """
        self._api_key = api_key
        self._base_url = base_url
        self._sdk: ClaudeSDK | None = None

    async def initialize(self) -> None:
        """Initialize the Claude Agent SDK.

        This method should be called before creating any sessions.
        It sets up the SDK client with the configured API key and base URL.

        Raises:
            RuntimeError: If SDK initialization fails
        """
        # TODO: Actual SDK initialization when claude-agent-sdk is available
        # from claude_agent_sdk import ClaudeAgent
        # self._sdk = ClaudeAgent(api_key=self._api_key, base_url=self._base_url)
        pass

    async def create_session(
        self,
        model: str,
        system_prompt: str,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
    ) -> AgentSession:
        """Create a new agent session with specified configuration.

        Args:
            model: Claude model identifier (e.g., "claude-3-5-sonnet-20241022")
            system_prompt: System prompt to configure agent behavior
            tools: Optional list of tool definitions for the agent
            max_tokens: Maximum tokens for agent responses
            temperature: Sampling temperature (0.0-1.0)

        Returns:
            Active agent session instance

        Raises:
            RuntimeError: If SDK is not initialized
            ValueError: If configuration parameters are invalid
        """
        if self._sdk is None:
            raise RuntimeError("SDK not initialized. Call initialize() first.")

        return await self._sdk.create_session(
            model=model,
            system_prompt=system_prompt,
            tools=tools,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    async def close(self) -> None:
        """Close the SDK client and clean up resources.

        This should be called during application shutdown to ensure
        all active sessions are properly closed.
        """
        if self._sdk is not None:
            await self._sdk.close()
            self._sdk = None

    @property
    def is_initialized(self) -> bool:
        """Check if the SDK client is initialized.

        Returns:
            True if SDK is ready for use, False otherwise
        """
        return self._sdk is not None
