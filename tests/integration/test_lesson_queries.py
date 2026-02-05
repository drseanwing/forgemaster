"""Integration tests for lesson learned query functions.

Tests all lesson CRUD operations including creation, retrieval, listing
with filters, verification updates, full-text search, and file overlap search.

Note: Vector similarity search tests are skipped in SQLite environment as they
require PostgreSQL with pgvector extension. These tests are marked with
@pytest.mark.postgres for future PostgreSQL test environments.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from forgemaster.database.queries.lesson import (
    create_lesson,
    get_lesson,
    list_lessons,
    search_lessons_by_files,
    search_lessons_by_text,
    update_lesson_verification,
)
from forgemaster.database.queries.project import create_project
from forgemaster.database.queries.task import create_task


@pytest.mark.asyncio
async def test_create_lesson_with_all_fields(db_session: AsyncSession) -> None:
    """Test creating a lesson with all optional fields populated."""
    project = await create_project(db_session, name="Lesson Test Project", config={})
    task = await create_task(
        db_session, project_id=project.id, title="Test Task", agent_type="executor"
    )

    lesson = await create_lesson(
        db_session,
        project_id=project.id,
        task_id=task.id,
        symptom="Import statement failed with ModuleNotFoundError",
        root_cause="Package not listed in pyproject.toml dependencies",
        fix_applied="Added 'requests>=2.31.0' to dependencies and ran uv sync",
        files_affected=["pyproject.toml", "src/api/client.py"],
        pattern_tags=["dependency-management", "python", "import-error"],
        confidence_score=0.95,
    )

    assert lesson.id is not None
    assert isinstance(lesson.id, UUID)
    assert lesson.project_id == project.id
    assert lesson.task_id == task.id
    assert lesson.symptom == "Import statement failed with ModuleNotFoundError"
    assert lesson.root_cause == "Package not listed in pyproject.toml dependencies"
    assert lesson.fix_applied == "Added 'requests>=2.31.0' to dependencies and ran uv sync"
    assert lesson.files_affected == ["pyproject.toml", "src/api/client.py"]
    assert lesson.pattern_tags == ["dependency-management", "python", "import-error"]
    assert lesson.verification_status == "unverified"
    assert lesson.confidence_score == 0.95
    assert lesson.created_at is not None


@pytest.mark.asyncio
async def test_create_lesson_minimal(db_session: AsyncSession) -> None:
    """Test creating a lesson with only required fields."""
    project = await create_project(db_session, name="Minimal Lesson Test", config={})
    task = await create_task(
        db_session, project_id=project.id, title="Task", agent_type="executor"
    )

    lesson = await create_lesson(
        db_session,
        project_id=project.id,
        task_id=task.id,
        symptom="Type error in function call",
        root_cause="Wrong argument type passed",
        fix_applied="Added type conversion",
    )

    assert lesson.symptom == "Type error in function call"
    assert lesson.files_affected is None
    assert lesson.pattern_tags is None
    assert lesson.confidence_score == 0.5  # Default value


@pytest.mark.asyncio
async def test_get_lesson_found(db_session: AsyncSession) -> None:
    """Test retrieving an existing lesson by ID."""
    project = await create_project(db_session, name="Retrieval Test", config={})
    task = await create_task(
        db_session, project_id=project.id, title="Task", agent_type="executor"
    )
    created = await create_lesson(
        db_session,
        project_id=project.id,
        task_id=task.id,
        symptom="Database connection timeout",
        root_cause="Connection pool exhausted",
        fix_applied="Increased pool size to 20",
    )

    retrieved = await get_lesson(db_session, created.id)

    assert retrieved is not None
    assert retrieved.id == created.id
    assert retrieved.symptom == "Database connection timeout"


@pytest.mark.asyncio
async def test_get_lesson_not_found(db_session: AsyncSession) -> None:
    """Test retrieving a non-existent lesson returns None."""
    non_existent_id = uuid4()
    result = await get_lesson(db_session, non_existent_id)

    assert result is None


@pytest.mark.asyncio
async def test_list_lessons_no_filters(db_session: AsyncSession) -> None:
    """Test listing all lessons without filters."""
    project = await create_project(db_session, name="List Test", config={})
    task = await create_task(
        db_session, project_id=project.id, title="Task", agent_type="executor"
    )

    await create_lesson(
        db_session,
        project_id=project.id,
        task_id=task.id,
        symptom="Error 1",
        root_cause="Cause 1",
        fix_applied="Fix 1",
    )
    await create_lesson(
        db_session,
        project_id=project.id,
        task_id=task.id,
        symptom="Error 2",
        root_cause="Cause 2",
        fix_applied="Fix 2",
    )

    lessons = await list_lessons(db_session)

    assert len(lessons) >= 2


@pytest.mark.asyncio
async def test_list_lessons_filter_by_project(db_session: AsyncSession) -> None:
    """Test listing lessons filtered by project ID."""
    project_a = await create_project(db_session, name="Project A", config={})
    project_b = await create_project(db_session, name="Project B", config={})
    task_a = await create_task(
        db_session, project_id=project_a.id, title="Task A", agent_type="executor"
    )
    task_b = await create_task(
        db_session, project_id=project_b.id, title="Task B", agent_type="executor"
    )

    lesson_a1 = await create_lesson(
        db_session,
        project_id=project_a.id,
        task_id=task_a.id,
        symptom="A1",
        root_cause="Cause A1",
        fix_applied="Fix A1",
    )
    lesson_a2 = await create_lesson(
        db_session,
        project_id=project_a.id,
        task_id=task_a.id,
        symptom="A2",
        root_cause="Cause A2",
        fix_applied="Fix A2",
    )
    await create_lesson(
        db_session,
        project_id=project_b.id,
        task_id=task_b.id,
        symptom="B1",
        root_cause="Cause B1",
        fix_applied="Fix B1",
    )

    project_a_lessons = await list_lessons(db_session, project_id=project_a.id)

    assert len(project_a_lessons) == 2
    assert {l.id for l in project_a_lessons} == {lesson_a1.id, lesson_a2.id}


@pytest.mark.asyncio
async def test_list_lessons_filter_by_verification_status(db_session: AsyncSession) -> None:
    """Test listing lessons filtered by verification status."""
    project = await create_project(db_session, name="Status Filter Test", config={})
    task = await create_task(
        db_session, project_id=project.id, title="Task", agent_type="executor"
    )

    unverified = await create_lesson(
        db_session,
        project_id=project.id,
        task_id=task.id,
        symptom="Unverified lesson",
        root_cause="Cause",
        fix_applied="Fix",
    )
    verified = await create_lesson(
        db_session,
        project_id=project.id,
        task_id=task.id,
        symptom="Verified lesson",
        root_cause="Cause",
        fix_applied="Fix",
    )

    # Update verification status
    await update_lesson_verification(db_session, verified.id, "verified", confidence_score=0.9)

    verified_lessons = await list_lessons(
        db_session, project_id=project.id, verification_status="verified"
    )
    unverified_lessons = await list_lessons(
        db_session, project_id=project.id, verification_status="unverified"
    )

    assert len(verified_lessons) == 1
    assert verified_lessons[0].id == verified.id
    assert len(unverified_lessons) == 1
    assert unverified_lessons[0].id == unverified.id


@pytest.mark.asyncio
async def test_list_lessons_filter_by_pattern_tags(db_session: AsyncSession) -> None:
    """Test listing lessons filtered by pattern tags (ANY match)."""
    project = await create_project(db_session, name="Tag Filter Test", config={})
    task = await create_task(
        db_session, project_id=project.id, title="Task", agent_type="executor"
    )

    python_lesson = await create_lesson(
        db_session,
        project_id=project.id,
        task_id=task.id,
        symptom="Python error",
        root_cause="Cause",
        fix_applied="Fix",
        pattern_tags=["python", "type-error"],
    )
    js_lesson = await create_lesson(
        db_session,
        project_id=project.id,
        task_id=task.id,
        symptom="JS error",
        root_cause="Cause",
        fix_applied="Fix",
        pattern_tags=["javascript", "async-error"],
    )
    multi_lesson = await create_lesson(
        db_session,
        project_id=project.id,
        task_id=task.id,
        symptom="Multi error",
        root_cause="Cause",
        fix_applied="Fix",
        pattern_tags=["python", "async-error"],
    )

    # Filter by python tag
    python_tagged = await list_lessons(
        db_session, project_id=project.id, pattern_tags=["python"]
    )

    assert len(python_tagged) == 2
    assert {l.id for l in python_tagged} == {python_lesson.id, multi_lesson.id}

    # Filter by async-error tag
    async_tagged = await list_lessons(
        db_session, project_id=project.id, pattern_tags=["async-error"]
    )

    assert len(async_tagged) == 2
    assert {l.id for l in async_tagged} == {js_lesson.id, multi_lesson.id}


@pytest.mark.asyncio
async def test_list_lessons_ordering(db_session: AsyncSession) -> None:
    """Test that lessons are ordered by created_at descending."""
    project = await create_project(db_session, name="Order Test", config={})
    task = await create_task(
        db_session, project_id=project.id, title="Task", agent_type="executor"
    )

    first = await create_lesson(
        db_session,
        project_id=project.id,
        task_id=task.id,
        symptom="First",
        root_cause="Cause",
        fix_applied="Fix",
    )
    second = await create_lesson(
        db_session,
        project_id=project.id,
        task_id=task.id,
        symptom="Second",
        root_cause="Cause",
        fix_applied="Fix",
    )

    lessons = await list_lessons(db_session, project_id=project.id)

    # Most recent first
    lesson_ids = [l.id for l in lessons]
    assert lesson_ids.index(second.id) < lesson_ids.index(first.id)


@pytest.mark.asyncio
async def test_update_lesson_verification_status_only(db_session: AsyncSession) -> None:
    """Test updating only the verification status."""
    project = await create_project(db_session, name="Verification Test", config={})
    task = await create_task(
        db_session, project_id=project.id, title="Task", agent_type="executor"
    )
    lesson = await create_lesson(
        db_session,
        project_id=project.id,
        task_id=task.id,
        symptom="Test symptom",
        root_cause="Test cause",
        fix_applied="Test fix",
        confidence_score=0.5,
    )

    updated = await update_lesson_verification(
        db_session,
        lesson.id,
        "verified",
    )

    assert updated.verification_status == "verified"
    assert updated.confidence_score == 0.5  # Unchanged


@pytest.mark.asyncio
async def test_update_lesson_verification_with_confidence(db_session: AsyncSession) -> None:
    """Test updating verification status and confidence score together."""
    project = await create_project(db_session, name="Confidence Test", config={})
    task = await create_task(
        db_session, project_id=project.id, title="Task", agent_type="executor"
    )
    lesson = await create_lesson(
        db_session,
        project_id=project.id,
        task_id=task.id,
        symptom="Test symptom",
        root_cause="Test cause",
        fix_applied="Test fix",
    )

    updated = await update_lesson_verification(
        db_session,
        lesson.id,
        "human-verified",
        confidence_score=0.99,
    )

    assert updated.verification_status == "human-verified"
    assert updated.confidence_score == 0.99


@pytest.mark.asyncio
async def test_update_lesson_verification_not_found(db_session: AsyncSession) -> None:
    """Test updating verification on non-existent lesson raises ValueError."""
    non_existent_id = uuid4()

    with pytest.raises(ValueError, match=f"Lesson {non_existent_id} not found"):
        await update_lesson_verification(db_session, non_existent_id, "verified")


@pytest.mark.asyncio
@pytest.mark.skipif(True, reason="Requires PostgreSQL with tsvector support")
async def test_search_lessons_by_text(db_session: AsyncSession) -> None:
    """Test full-text search across symptom, root_cause, and fix_applied.

    This test is skipped in SQLite as it requires PostgreSQL's tsvector
    and full-text search features. To run this test, use a PostgreSQL
    test database with the @pytest.mark.postgres marker.
    """
    project = await create_project(db_session, name="Search Test", config={})
    task = await create_task(
        db_session, project_id=project.id, title="Task", agent_type="executor"
    )

    await create_lesson(
        db_session,
        project_id=project.id,
        task_id=task.id,
        symptom="Authentication token expired",
        root_cause="JWT token TTL too short",
        fix_applied="Increased token TTL to 1 hour",
    )
    await create_lesson(
        db_session,
        project_id=project.id,
        task_id=task.id,
        symptom="Database connection timeout",
        root_cause="Pool size too small",
        fix_applied="Increased pool to 20 connections",
    )

    # Search for authentication-related lessons
    results = await search_lessons_by_text(db_session, project.id, "authentication token")

    assert len(results) >= 1
    assert any("authentication" in l.symptom.lower() for l in results)


@pytest.mark.asyncio
async def test_search_lessons_by_files(db_session: AsyncSession) -> None:
    """Test searching lessons by file overlap."""
    project = await create_project(db_session, name="File Search Test", config={})
    task = await create_task(
        db_session, project_id=project.id, title="Task", agent_type="executor"
    )

    auth_lesson = await create_lesson(
        db_session,
        project_id=project.id,
        task_id=task.id,
        symptom="Auth error",
        root_cause="Cause",
        fix_applied="Fix",
        files_affected=["src/auth/jwt.py", "src/auth/middleware.py"],
    )
    db_lesson = await create_lesson(
        db_session,
        project_id=project.id,
        task_id=task.id,
        symptom="DB error",
        root_cause="Cause",
        fix_applied="Fix",
        files_affected=["src/db/connection.py", "src/db/queries.py"],
    )
    mixed_lesson = await create_lesson(
        db_session,
        project_id=project.id,
        task_id=task.id,
        symptom="Mixed error",
        root_cause="Cause",
        fix_applied="Fix",
        files_affected=["src/auth/jwt.py", "src/db/connection.py"],
    )

    # Search for lessons affecting auth files
    auth_results = await search_lessons_by_files(
        db_session, project.id, ["src/auth/jwt.py", "src/auth/utils.py"]
    )

    # Should match lessons with any overlapping files
    auth_result_ids = {l.id for l in auth_results}
    assert auth_lesson.id in auth_result_ids
    assert mixed_lesson.id in auth_result_ids
    assert db_lesson.id not in auth_result_ids


@pytest.mark.asyncio
async def test_search_lessons_by_files_no_matches(db_session: AsyncSession) -> None:
    """Test file search with no matching lessons."""
    project = await create_project(db_session, name="No Match Test", config={})
    task = await create_task(
        db_session, project_id=project.id, title="Task", agent_type="executor"
    )

    await create_lesson(
        db_session,
        project_id=project.id,
        task_id=task.id,
        symptom="Error",
        root_cause="Cause",
        fix_applied="Fix",
        files_affected=["src/other/file.py"],
    )

    results = await search_lessons_by_files(
        db_session, project.id, ["src/unrelated/file.py"]
    )

    assert len(results) == 0


@pytest.mark.asyncio
async def test_search_lessons_by_files_empty_files_affected(db_session: AsyncSession) -> None:
    """Test that lessons with no files_affected are not matched."""
    project = await create_project(db_session, name="Empty Files Test", config={})
    task = await create_task(
        db_session, project_id=project.id, title="Task", agent_type="executor"
    )

    lesson_with_files = await create_lesson(
        db_session,
        project_id=project.id,
        task_id=task.id,
        symptom="Has files",
        root_cause="Cause",
        fix_applied="Fix",
        files_affected=["src/test.py"],
    )
    lesson_without_files = await create_lesson(
        db_session,
        project_id=project.id,
        task_id=task.id,
        symptom="No files",
        root_cause="Cause",
        fix_applied="Fix",
        files_affected=None,
    )

    results = await search_lessons_by_files(db_session, project.id, ["src/test.py"])

    assert len(results) == 1
    assert results[0].id == lesson_with_files.id


@pytest.mark.asyncio
@pytest.mark.skipif(True, reason="Requires PostgreSQL with pgvector extension")
async def test_search_lessons_by_embedding(db_session: AsyncSession) -> None:
    """Test vector similarity search.

    This test is skipped in SQLite as it requires PostgreSQL with the
    pgvector extension installed. To run this test, use a PostgreSQL
    test database with pgvector enabled and mark it with @pytest.mark.postgres.

    In a real test with pgvector:
    1. Create lessons with content_embedding populated
    2. Call search_lessons_by_embedding with a query embedding
    3. Verify results are ordered by similarity
    """
    # This test requires actual embeddings and pgvector
    # Would need to:
    # - Generate embeddings for lesson content
    # - Store in content_embedding column
    # - Search with query embedding
    pass


@pytest.mark.asyncio
async def test_lesson_with_multiple_pattern_tags(db_session: AsyncSession) -> None:
    """Test creating and filtering lessons with multiple pattern tags."""
    project = await create_project(db_session, name="Multi-Tag Test", config={})
    task = await create_task(
        db_session, project_id=project.id, title="Task", agent_type="executor"
    )

    lesson = await create_lesson(
        db_session,
        project_id=project.id,
        task_id=task.id,
        symptom="Complex error",
        root_cause="Multiple causes",
        fix_applied="Multi-step fix",
        pattern_tags=["python", "async", "database", "timeout"],
    )

    # Should match with any of these tags
    python_results = await list_lessons(
        db_session, project_id=project.id, pattern_tags=["python"]
    )
    async_results = await list_lessons(
        db_session, project_id=project.id, pattern_tags=["async"]
    )
    multi_tag_results = await list_lessons(
        db_session, project_id=project.id, pattern_tags=["python", "async"]
    )

    assert lesson.id in {l.id for l in python_results}
    assert lesson.id in {l.id for l in async_results}
    assert lesson.id in {l.id for l in multi_tag_results}
