"""Lesson learned CRUD and search query functions for Forgemaster.

Provides async functions for creating, reading, updating, and searching
LessonLearned records using full-text search, vector similarity search,
and file overlap matching.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

import structlog
from pgvector.sqlalchemy import Vector
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from forgemaster.database.models.lesson import LessonLearned

logger = structlog.get_logger(__name__)


async def create_lesson(
    session: AsyncSession,
    project_id: UUID,
    task_id: UUID,
    symptom: str,
    root_cause: str,
    fix_applied: str,
    files_affected: list[str] | None = None,
    pattern_tags: list[str] | None = None,
    confidence_score: float = 0.5,
) -> LessonLearned:
    """Create a new lesson learned.

    Args:
        session: Active async database session.
        project_id: UUID of the associated project.
        task_id: UUID of the originating task.
        symptom: Description of the observed problem.
        root_cause: Identified root cause of the problem.
        fix_applied: Description of the fix that resolved it.
        files_affected: Optional list of file paths involved.
        pattern_tags: Optional classification tags for the pattern.
        confidence_score: Confidence in the lesson (0.0 to 1.0).

    Returns:
        The newly created LessonLearned instance.
    """
    lesson = LessonLearned(
        project_id=project_id,
        task_id=task_id,
        symptom=symptom,
        root_cause=root_cause,
        fix_applied=fix_applied,
        files_affected=files_affected,
        pattern_tags=pattern_tags,
        verification_status="unverified",
        confidence_score=confidence_score,
    )

    async with session.begin():
        session.add(lesson)
        await session.flush()
        await session.refresh(lesson)

    logger.info(
        "lesson_created",
        lesson_id=str(lesson.id),
        project_id=str(project_id),
        task_id=str(task_id),
        pattern_tags=pattern_tags,
    )

    return lesson


async def get_lesson(
    session: AsyncSession,
    lesson_id: UUID,
) -> LessonLearned | None:
    """Retrieve a lesson by ID.

    Args:
        session: Active async database session.
        lesson_id: UUID of the lesson to retrieve.

    Returns:
        The LessonLearned instance if found, None otherwise.
    """
    stmt = select(LessonLearned).where(LessonLearned.id == lesson_id)
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def list_lessons(
    session: AsyncSession,
    project_id: UUID | None = None,
    verification_status: str | None = None,
    pattern_tags: list[str] | None = None,
) -> list[LessonLearned]:
    """List lessons with optional filters.

    Args:
        session: Active async database session.
        project_id: Optional project UUID to filter by.
        verification_status: Optional verification status to filter by.
        pattern_tags: Optional pattern tags to filter by (matches ANY tag).

    Returns:
        List of matching LessonLearned instances, ordered by creation time.
    """
    stmt = select(LessonLearned)

    if project_id is not None:
        stmt = stmt.where(LessonLearned.project_id == project_id)

    if verification_status is not None:
        stmt = stmt.where(LessonLearned.verification_status == verification_status)

    if pattern_tags is not None:
        # Match if ANY of the provided tags is present in the lesson's pattern_tags
        stmt = stmt.where(LessonLearned.pattern_tags.overlap(pattern_tags))

    stmt = stmt.order_by(LessonLearned.created_at.desc())
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def update_lesson_verification(
    session: AsyncSession,
    lesson_id: UUID,
    status: str,
    confidence_score: float | None = None,
) -> LessonLearned:
    """Update a lesson's verification status and confidence score.

    Args:
        session: Active async database session.
        lesson_id: UUID of the lesson to update.
        status: New verification status.
        confidence_score: Optional new confidence score (0.0 to 1.0).

    Returns:
        The updated LessonLearned instance.

    Raises:
        ValueError: If lesson not found.
    """
    lesson = await get_lesson(session, lesson_id)
    if lesson is None:
        raise ValueError(f"Lesson {lesson_id} not found")

    updates: dict[str, Any] = {"verification_status": status}
    if confidence_score is not None:
        updates["confidence_score"] = confidence_score

    async with session.begin():
        stmt = (
            update(LessonLearned)
            .where(LessonLearned.id == lesson_id)
            .values(**updates)
        )
        await session.execute(stmt)
        await session.refresh(lesson)

    logger.info(
        "lesson_verification_updated",
        lesson_id=str(lesson_id),
        status=status,
        confidence_score=confidence_score,
    )

    return lesson


async def search_lessons_by_text(
    session: AsyncSession,
    project_id: UUID,
    query: str,
) -> list[LessonLearned]:
    """Search lessons using PostgreSQL full-text search.

    Uses the generated content_tsv column for fast full-text search
    across symptom, root_cause, and fix_applied fields.

    Args:
        session: Active async database session.
        project_id: UUID of the project to search within.
        query: Text query string.

    Returns:
        List of matching LessonLearned instances, ranked by relevance.
    """
    # Convert query to tsquery
    tsquery = func.plainto_tsquery("english", query)

    stmt = (
        select(LessonLearned)
        .where(LessonLearned.project_id == project_id)
        .where(LessonLearned.content_tsv.op("@@")(tsquery))
        .order_by(
            func.ts_rank(LessonLearned.content_tsv, tsquery).desc()
        )
    )

    result = await session.execute(stmt)
    return list(result.scalars().all())


async def search_lessons_by_embedding(
    session: AsyncSession,
    project_id: UUID,
    embedding: list[float],
    limit: int = 10,
) -> list[LessonLearned]:
    """Search lessons using vector similarity search.

    Uses the content_embedding column for semantic similarity search.

    Args:
        session: Active async database session.
        project_id: UUID of the project to search within.
        embedding: Vector embedding to search with (768 dimensions).
        limit: Maximum number of results to return.

    Returns:
        List of matching LessonLearned instances, ordered by similarity.
    """
    # Calculate cosine distance for similarity search
    stmt = (
        select(LessonLearned)
        .where(LessonLearned.project_id == project_id)
        .where(LessonLearned.content_embedding.isnot(None))
        .order_by(
            LessonLearned.content_embedding.cosine_distance(embedding)
        )
        .limit(limit)
    )

    result = await session.execute(stmt)
    return list(result.scalars().all())


async def search_lessons_by_files(
    session: AsyncSession,
    project_id: UUID,
    files: list[str],
) -> list[LessonLearned]:
    """Search lessons by file overlap.

    Finds lessons where files_affected overlaps with the provided file list.

    Args:
        session: Active async database session.
        project_id: UUID of the project to search within.
        files: List of file paths to match against.

    Returns:
        List of matching LessonLearned instances, ordered by creation time.
    """
    stmt = (
        select(LessonLearned)
        .where(LessonLearned.project_id == project_id)
        .where(LessonLearned.files_affected.overlap(files))
        .order_by(LessonLearned.created_at.desc())
    )

    result = await session.execute(stmt)
    return list(result.scalars().all())
