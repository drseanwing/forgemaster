"""Lesson learned query endpoints for Forgemaster.

This module provides REST API endpoints for querying LessonLearned records:
- List lessons with optional filters
- Get lesson by ID
- Full-text search across lesson content
- Search lessons by file overlap

Example:
    >>> from fastapi import FastAPI
    >>> from forgemaster.web.routes.lessons import create_lessons_router
    >>>
    >>> app = FastAPI()
    >>> app.include_router(create_lessons_router())
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from forgemaster.database.queries.lesson import (
    get_lesson,
    list_lessons,
    search_lessons_by_files,
    search_lessons_by_text,
)
from forgemaster.logging import get_logger

if TYPE_CHECKING:
    from datetime import datetime

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

logger = get_logger(__name__)


class LessonResponse(BaseModel):
    """Response model for a single lesson learned.

    Attributes:
        id: Unique identifier for the lesson
        project_id: UUID of the associated project
        task_id: UUID of the originating task
        symptom: Description of the observed problem
        root_cause: Identified root cause of the problem
        fix_applied: Description of the fix that resolved it
        files_affected: List of file paths involved in the lesson
        pattern_tags: Classification tags for the pattern
        verification_status: Current verification status
        confidence_score: Confidence in the lesson (0.0 to 1.0)
        created_at: Timestamp when lesson was created
        updated_at: Timestamp of last update
    """

    id: UUID
    project_id: UUID
    task_id: UUID
    symptom: str
    root_cause: str
    fix_applied: str
    files_affected: list[str] | None
    pattern_tags: list[str] | None
    verification_status: str
    confidence_score: float
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class LessonSearchText(BaseModel):
    """Request model for text-based lesson search.

    Attributes:
        project_id: UUID of the project to search within
        query: Text query string (minimum 1 character)
    """

    project_id: UUID
    query: str = Field(..., min_length=1)


class LessonSearchFiles(BaseModel):
    """Request model for file-based lesson search.

    Attributes:
        project_id: UUID of the project to search within
        files: List of file paths to match against (minimum 1 file)
    """

    project_id: UUID
    files: list[str] = Field(..., min_items=1)


def get_session_factory(request: Request) -> async_sessionmaker[AsyncSession]:
    """Dependency that retrieves session factory from app state.

    Args:
        request: FastAPI request object

    Returns:
        Session factory from app.state
    """
    return request.app.state.session_factory  # type: ignore[return-value]


def create_lessons_router() -> APIRouter:
    """Create lessons query router with endpoints.

    Returns:
        Configured APIRouter with lesson query endpoints.

    Routes:
        GET /lessons/ - List lessons with optional filters
        GET /lessons/{lesson_id} - Get lesson by ID
        POST /lessons/search/text - Full-text search lessons
        POST /lessons/search/files - Search lessons by file overlap
    """
    router = APIRouter(prefix="/lessons", tags=["lessons"])

    @router.get("/", response_model=list[LessonResponse])
    async def list_lessons_endpoint(
        project_id: UUID | None = Query(None, description="Filter by project UUID"),
        verification_status: str | None = Query(
            None,
            description="Filter by verification status (unverified, verified, deprecated)",
        ),
        pattern_tags: list[str] | None = Query(
            None, description="Filter by pattern tags (matches ANY tag)"
        ),
        session_factory: async_sessionmaker[AsyncSession] = Depends(  # noqa: B008
            get_session_factory
        ),
    ) -> list[dict[str, Any]]:
        """List lessons with optional filters.

        Retrieves lessons matching the provided filters, ordered by creation time
        (newest first).

        Args:
            project_id: Optional project UUID to filter by
            verification_status: Optional verification status to filter by
            pattern_tags: Optional pattern tags to filter by (matches ANY tag)
            session_factory: Injected session factory from app state

        Returns:
            List of matching lesson records
        """
        try:
            async with session_factory() as session:
                lessons = await list_lessons(
                    session=session,
                    project_id=project_id,
                    verification_status=verification_status,
                    pattern_tags=pattern_tags,
                )

            logger.info(
                "lessons_listed",
                count=len(lessons),
                project_id=str(project_id) if project_id else None,
                verification_status=verification_status,
                pattern_tags=pattern_tags,
            )

            return [LessonResponse.model_validate(lesson).model_dump() for lesson in lessons]

        except Exception as exc:
            logger.error(
                "list_lessons_failed",
                error=str(exc),
                project_id=str(project_id) if project_id else None,
            )
            raise HTTPException(status_code=500, detail="Failed to list lessons") from exc

    @router.get("/{lesson_id}", response_model=LessonResponse)
    async def get_lesson_endpoint(
        lesson_id: UUID,
        session_factory: async_sessionmaker[AsyncSession] = Depends(  # noqa: B008
            get_session_factory
        ),
    ) -> dict[str, Any]:
        """Get a lesson by ID.

        Args:
            lesson_id: UUID of the lesson to retrieve
            session_factory: Injected session factory from app state

        Returns:
            Lesson record matching the ID

        Raises:
            HTTPException: 404 if lesson not found
        """
        try:
            async with session_factory() as session:
                lesson = await get_lesson(session=session, lesson_id=lesson_id)

            if lesson is None:
                logger.warning("lesson_not_found", lesson_id=str(lesson_id))
                raise HTTPException(status_code=404, detail=f"Lesson {lesson_id} not found")

            logger.info("lesson_retrieved", lesson_id=str(lesson_id))
            return LessonResponse.model_validate(lesson).model_dump()

        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "get_lesson_failed",
                error=str(exc),
                lesson_id=str(lesson_id),
            )
            raise HTTPException(status_code=500, detail="Failed to retrieve lesson") from exc

    @router.post("/search/text", response_model=list[LessonResponse])
    async def search_lessons_text_endpoint(
        request_body: LessonSearchText,
        session_factory: async_sessionmaker[AsyncSession] = Depends(  # noqa: B008
            get_session_factory
        ),
    ) -> list[dict[str, Any]]:
        """Search lessons using full-text search.

        Uses PostgreSQL full-text search across symptom, root_cause, and fix_applied
        fields. Results are ranked by relevance.

        Args:
            request_body: Search request containing project_id and query
            session_factory: Injected session factory from app state

        Returns:
            List of matching lesson records, ranked by relevance
        """
        try:
            async with session_factory() as session:
                lessons = await search_lessons_by_text(
                    session=session,
                    project_id=request_body.project_id,
                    query=request_body.query,
                )

            logger.info(
                "lessons_text_search_completed",
                count=len(lessons),
                project_id=str(request_body.project_id),
                query=request_body.query,
            )

            return [LessonResponse.model_validate(lesson).model_dump() for lesson in lessons]

        except Exception as exc:
            logger.error(
                "search_lessons_text_failed",
                error=str(exc),
                project_id=str(request_body.project_id),
            )
            raise HTTPException(status_code=500, detail="Failed to search lessons") from exc

    @router.post("/search/files", response_model=list[LessonResponse])
    async def search_lessons_files_endpoint(
        request_body: LessonSearchFiles,
        session_factory: async_sessionmaker[AsyncSession] = Depends(  # noqa: B008
            get_session_factory
        ),
    ) -> list[dict[str, Any]]:
        """Search lessons by file overlap.

        Finds lessons where files_affected overlaps with the provided file list.
        Results are ordered by creation time (newest first).

        Args:
            request_body: Search request containing project_id and files list
            session_factory: Injected session factory from app state

        Returns:
            List of matching lesson records, ordered by creation time
        """
        try:
            async with session_factory() as session:
                lessons = await search_lessons_by_files(
                    session=session,
                    project_id=request_body.project_id,
                    files=request_body.files,
                )

            logger.info(
                "lessons_file_search_completed",
                count=len(lessons),
                project_id=str(request_body.project_id),
                files_count=len(request_body.files),
            )

            return [LessonResponse.model_validate(lesson).model_dump() for lesson in lessons]

        except Exception as exc:
            logger.error(
                "search_lessons_files_failed",
                error=str(exc),
                project_id=str(request_body.project_id),
            )
            raise HTTPException(status_code=500, detail="Failed to search lessons") from exc

    return router
