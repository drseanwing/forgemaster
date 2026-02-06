"""Agent session query endpoints for Forgemaster.

This module provides REST API endpoints for querying agent sessions:
- List sessions with optional task_id and status filters
- Retrieve session by ID
- Get all active sessions
- Get idle sessions with configurable threshold

Example:
    >>> from fastapi import FastAPI
    >>> from forgemaster.web.routes.sessions import create_sessions_router
    >>>
    >>> app = FastAPI()
    >>> app.include_router(create_sessions_router())
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from forgemaster.database.models.session import SessionStatus
from forgemaster.database.queries.session import (
    get_active_sessions,
    get_idle_sessions,
    get_session,
    list_sessions,
)
from forgemaster.logging import get_logger

if TYPE_CHECKING:
    from datetime import datetime

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

logger = get_logger(__name__)


class SessionResponse(BaseModel):
    """Agent session response model.

    Attributes:
        id: Session UUID.
        task_id: Associated task UUID.
        model: Claude model identifier.
        status: Current session status.
        worktree_path: Git worktree path assigned to this session.
        token_count: Total tokens consumed during the session.
        result: Session output/result data.
        error_message: Error description if session failed.
        started_at: Session start timestamp.
        last_activity_at: Timestamp of most recent activity.
        ended_at: Session termination timestamp.
    """

    id: UUID
    task_id: UUID
    model: str
    status: str
    worktree_path: str | None
    token_count: int
    result: dict[str, Any] | None
    error_message: str | None
    started_at: datetime
    last_activity_at: datetime | None
    ended_at: datetime | None

    model_config = {"from_attributes": True}


def get_session_factory(request: Request) -> async_sessionmaker[AsyncSession]:
    """Dependency that retrieves session factory from app state.

    Args:
        request: FastAPI request object.

    Returns:
        Session factory from app.state.
    """
    return request.app.state.session_factory  # type: ignore[return-value]


def create_sessions_router() -> APIRouter:
    """Create agent session routes.

    Returns:
        Configured APIRouter with session endpoints.

    Routes:
        GET /sessions/ - List sessions with optional filters
        GET /sessions/{session_id} - Get session by ID
        GET /sessions/active - Get all active sessions
        GET /sessions/idle - Get idle sessions with optional threshold
    """
    router = APIRouter(prefix="/sessions", tags=["sessions"])

    @router.get("/", response_model=list[SessionResponse])
    async def list_sessions_endpoint(
        task_id: UUID | None = Query(None, description="Filter by task UUID"),
        status: SessionStatus | None = Query(None, description="Filter by session status"),
        session_factory: async_sessionmaker[AsyncSession] = Depends(  # noqa: B008
            get_session_factory
        ),
    ) -> list[SessionResponse]:
        """List agent sessions with optional filters.

        Query Parameters:
            task_id: Optional task UUID to filter sessions.
            status: Optional session status to filter (initialising, active, idle, etc.).

        Args:
            task_id: Filter sessions by associated task UUID.
            status: Filter sessions by status.
            session_factory: Injected session factory from app state.

        Returns:
            List of matching agent sessions ordered by start time descending.
        """
        try:
            async with session_factory() as session:
                sessions = await list_sessions(
                    session, task_id=task_id, status_filter=status
                )

            logger.debug(
                "sessions_listed",
                count=len(sessions),
                task_id=str(task_id) if task_id else None,
                status=status.value if status else None,
            )

            return [SessionResponse.model_validate(s) for s in sessions]

        except Exception as exc:
            logger.error(
                "list_sessions_failed",
                error=str(exc),
                task_id=str(task_id) if task_id else None,
                status=status.value if status else None,
            )
            raise HTTPException(
                status_code=500, detail="Failed to list sessions"
            ) from exc

    @router.get("/{session_id}", response_model=SessionResponse)
    async def get_session_endpoint(
        session_id: UUID,
        session_factory: async_sessionmaker[AsyncSession] = Depends(  # noqa: B008
            get_session_factory
        ),
    ) -> SessionResponse:
        """Retrieve an agent session by ID.

        Path Parameters:
            session_id: UUID of the session to retrieve.

        Args:
            session_id: Session UUID.
            session_factory: Injected session factory from app state.

        Returns:
            Agent session details.

        Raises:
            HTTPException: 404 if session not found.
        """
        try:
            async with session_factory() as session:
                agent_session = await get_session(session, session_id)

            if agent_session is None:
                logger.warning("session_not_found", session_id=str(session_id))
                raise HTTPException(
                    status_code=404, detail=f"Session {session_id} not found"
                )

            logger.debug("session_retrieved", session_id=str(session_id))
            return SessionResponse.model_validate(agent_session)

        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "get_session_failed", error=str(exc), session_id=str(session_id)
            )
            raise HTTPException(
                status_code=500, detail="Failed to retrieve session"
            ) from exc

    @router.get("/active", response_model=list[SessionResponse])
    async def get_active_sessions_endpoint(
        session_factory: async_sessionmaker[AsyncSession] = Depends(  # noqa: B008
            get_session_factory
        ),
    ) -> list[SessionResponse]:
        """Get all currently active agent sessions.

        Active sessions are those in 'active' or 'idle' status.

        Args:
            session_factory: Injected session factory from app state.

        Returns:
            List of active agent sessions ordered by last activity descending.
        """
        try:
            async with session_factory() as session:
                sessions = await get_active_sessions(session)

            logger.debug("active_sessions_retrieved", count=len(sessions))
            return [SessionResponse.model_validate(s) for s in sessions]

        except Exception as exc:
            logger.error("get_active_sessions_failed", error=str(exc))
            raise HTTPException(
                status_code=500, detail="Failed to retrieve active sessions"
            ) from exc

    @router.get("/idle", response_model=list[SessionResponse])
    async def get_idle_sessions_endpoint(
        threshold_seconds: int = Query(
            300,
            ge=1,
            description="Seconds of inactivity to consider idle (default 300)",
        ),
        session_factory: async_sessionmaker[AsyncSession] = Depends(  # noqa: B008
            get_session_factory
        ),
    ) -> list[SessionResponse]:
        """Get sessions that have been idle for longer than the threshold.

        Query Parameters:
            threshold_seconds: Seconds of inactivity to consider idle (default 300 = 5 minutes).

        Args:
            threshold_seconds: Idle threshold in seconds.
            session_factory: Injected session factory from app state.

        Returns:
            List of idle agent sessions ordered by last activity ascending (oldest first).
        """
        try:
            async with session_factory() as session:
                sessions = await get_idle_sessions(session, threshold_seconds)

            logger.debug(
                "idle_sessions_retrieved",
                count=len(sessions),
                threshold_seconds=threshold_seconds,
            )
            return [SessionResponse.model_validate(s) for s in sessions]

        except Exception as exc:
            logger.error(
                "get_idle_sessions_failed",
                error=str(exc),
                threshold_seconds=threshold_seconds,
            )
            raise HTTPException(
                status_code=500, detail="Failed to retrieve idle sessions"
            ) from exc

    return router
