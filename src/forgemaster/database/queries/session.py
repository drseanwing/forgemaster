"""Agent session CRUD query functions for Forgemaster.

Provides async functions for creating, reading, updating, and managing
AgentSession records, including activity tracking and idle session detection.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

import structlog
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from forgemaster.database.models.session import AgentSession, SessionStatus

logger = structlog.get_logger(__name__)


async def create_session(
    session: AsyncSession,
    task_id: UUID,
    model: str,
    worktree_path: str | None = None,
) -> AgentSession:
    """Create a new agent session.

    Args:
        session: Active async database session.
        task_id: UUID of the associated task.
        model: Claude model identifier for this session.
        worktree_path: Optional git worktree path assigned to this session.

    Returns:
        The newly created AgentSession instance.
    """
    agent_session = AgentSession(
        task_id=task_id,
        model=model,
        worktree_path=worktree_path,
        status=SessionStatus.initialising,
        token_count=0,
    )

    async with session.begin():
        session.add(agent_session)
        await session.flush()
        await session.refresh(agent_session)

    logger.info(
        "agent_session_created",
        session_id=str(agent_session.id),
        task_id=str(task_id),
        model=model,
        status=agent_session.status.value,
    )

    return agent_session


async def get_session(
    session: AsyncSession,
    session_id: UUID,
) -> AgentSession | None:
    """Retrieve an agent session by ID.

    Args:
        session: Active async database session.
        session_id: UUID of the session to retrieve.

    Returns:
        The AgentSession instance if found, None otherwise.
    """
    stmt = select(AgentSession).where(AgentSession.id == session_id)
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def list_sessions(
    session: AsyncSession,
    task_id: UUID | None = None,
    status_filter: SessionStatus | None = None,
) -> list[AgentSession]:
    """List agent sessions with optional filters.

    Args:
        session: Active async database session.
        task_id: Optional task UUID to filter by.
        status_filter: Optional status to filter by.

    Returns:
        List of matching AgentSession instances, ordered by start time.
    """
    stmt = select(AgentSession)

    if task_id is not None:
        stmt = stmt.where(AgentSession.task_id == task_id)

    if status_filter is not None:
        stmt = stmt.where(AgentSession.status == status_filter)

    stmt = stmt.order_by(AgentSession.started_at.desc())
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def update_session_activity(
    session: AsyncSession,
    session_id: UUID,
) -> AgentSession:
    """Update the last activity timestamp for a session.

    Args:
        session: Active async database session.
        session_id: UUID of the session to update.

    Returns:
        The updated AgentSession instance.

    Raises:
        ValueError: If session not found.
    """
    agent_session = await get_session(session, session_id)
    if agent_session is None:
        raise ValueError(f"Session {session_id} not found")

    async with session.begin():
        stmt = (
            update(AgentSession)
            .where(AgentSession.id == session_id)
            .values(last_activity_at=func.now())
        )
        await session.execute(stmt)
        await session.refresh(agent_session)

    logger.debug(
        "session_activity_updated",
        session_id=str(session_id),
    )

    return agent_session


async def end_session(
    session: AsyncSession,
    session_id: UUID,
    status: SessionStatus,
    result: dict[str, Any] | None = None,
    error_message: str | None = None,
) -> AgentSession:
    """End an agent session with a final status and result.

    Args:
        session: Active async database session.
        session_id: UUID of the session to end.
        status: Final session status (should be a terminal status).
        result: Optional result data as dict.
        error_message: Optional error description if session failed.

    Returns:
        The updated AgentSession instance.

    Raises:
        ValueError: If session not found.
    """
    agent_session = await get_session(session, session_id)
    if agent_session is None:
        raise ValueError(f"Session {session_id} not found")

    updates: dict[str, Any] = {
        "status": status,
        "ended_at": datetime.now(timezone.utc),
    }

    if result is not None:
        updates["result"] = result

    if error_message is not None:
        updates["error_message"] = error_message

    async with session.begin():
        stmt = (
            update(AgentSession)
            .where(AgentSession.id == session_id)
            .values(**updates)
        )
        await session.execute(stmt)
        await session.refresh(agent_session)

    logger.info(
        "session_ended",
        session_id=str(session_id),
        status=status.value,
        has_error=error_message is not None,
    )

    return agent_session


async def get_active_sessions(
    session: AsyncSession,
) -> list[AgentSession]:
    """Get all currently active agent sessions.

    Active sessions are those in 'active' or 'idle' status.

    Args:
        session: Active async database session.

    Returns:
        List of active AgentSession instances.
    """
    stmt = (
        select(AgentSession)
        .where(
            AgentSession.status.in_([SessionStatus.active, SessionStatus.idle])
        )
        .order_by(AgentSession.last_activity_at.desc())
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def get_idle_sessions(
    session: AsyncSession,
    idle_threshold_seconds: int = 300,
) -> list[AgentSession]:
    """Get sessions that have been idle for longer than the threshold.

    Args:
        session: Active async database session.
        idle_threshold_seconds: Seconds of inactivity to consider idle (default 300 = 5 minutes).

    Returns:
        List of idle AgentSession instances.
    """
    threshold_time = datetime.now(timezone.utc) - timedelta(seconds=idle_threshold_seconds)

    stmt = (
        select(AgentSession)
        .where(AgentSession.status == SessionStatus.active)
        .where(AgentSession.last_activity_at < threshold_time)
        .order_by(AgentSession.last_activity_at.asc())
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())
