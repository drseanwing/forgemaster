"""Integration tests for project query functions.

Tests all CRUD operations for Project records including creation, retrieval,
listing with filters, updates, and deletion. Uses in-memory SQLite database
for fast, isolated testing.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from forgemaster.database.models.project import ProjectStatus
from forgemaster.database.queries.project import (
    create_project,
    delete_project,
    get_project,
    list_projects,
    update_project,
)


@pytest.mark.asyncio
async def test_create_project_with_all_fields(db_session: AsyncSession) -> None:
    """Test creating a project with all optional fields populated."""
    spec_doc = {
        "requirements": ["Feature A", "Feature B"],
        "constraints": ["Must use Python 3.12"],
    }
    arch_doc = {
        "components": ["API", "Database", "Worker"],
        "patterns": ["Event sourcing", "CQRS"],
    }
    config = {
        "max_workers": 5,
        "timeout_seconds": 300,
        "retry_policy": "exponential",
    }

    project = await create_project(
        db_session,
        name="E-Commerce Platform",
        config=config,
        spec_document=spec_doc,
        architecture_document=arch_doc,
    )

    assert project.id is not None
    assert isinstance(project.id, UUID)
    assert project.name == "E-Commerce Platform"
    assert project.status == ProjectStatus.draft
    assert project.spec_document == spec_doc
    assert project.architecture_document == arch_doc
    assert project.config == config
    assert project.created_at is not None
    assert project.updated_at is not None


@pytest.mark.asyncio
async def test_create_project_minimal(db_session: AsyncSession) -> None:
    """Test creating a project with only required fields."""
    project = await create_project(
        db_session,
        name="Minimal Project",
        config={"environment": "development"},
    )

    assert project.id is not None
    assert project.name == "Minimal Project"
    assert project.status == ProjectStatus.draft
    assert project.spec_document is None
    assert project.architecture_document is None
    assert project.config == {"environment": "development"}


@pytest.mark.asyncio
async def test_get_project_found(db_session: AsyncSession) -> None:
    """Test retrieving an existing project by ID."""
    created = await create_project(
        db_session,
        name="Retrieval Test Project",
        config={"key": "value"},
    )

    retrieved = await get_project(db_session, created.id)

    assert retrieved is not None
    assert retrieved.id == created.id
    assert retrieved.name == "Retrieval Test Project"
    assert retrieved.config == {"key": "value"}


@pytest.mark.asyncio
async def test_get_project_not_found(db_session: AsyncSession) -> None:
    """Test retrieving a non-existent project returns None."""
    non_existent_id = uuid4()
    result = await get_project(db_session, non_existent_id)

    assert result is None


@pytest.mark.asyncio
async def test_list_projects_no_filter(db_session: AsyncSession) -> None:
    """Test listing all projects without status filtering."""
    # Create projects with different statuses
    await create_project(db_session, name="Draft Project", config={"status": "draft"})
    await create_project(db_session, name="Active Project", config={"status": "active"})
    active_project = await create_project(
        db_session, name="Second Active", config={"status": "active"}
    )

    # Update second project to active status
    await update_project(db_session, active_project.id, status=ProjectStatus.active)

    projects = await list_projects(db_session)

    assert len(projects) >= 2
    # Projects are ordered by created_at descending
    assert all(hasattr(p, "name") for p in projects)


@pytest.mark.asyncio
async def test_list_projects_with_status_filter(db_session: AsyncSession) -> None:
    """Test listing projects filtered by status."""
    # Create projects with different statuses
    draft = await create_project(db_session, name="Draft Project", config={})
    active = await create_project(db_session, name="Active Project", config={})
    completed = await create_project(db_session, name="Completed Project", config={})

    # Update statuses
    await update_project(db_session, draft.id, status=ProjectStatus.draft)
    await update_project(db_session, active.id, status=ProjectStatus.active)
    await update_project(db_session, completed.id, status=ProjectStatus.completed)

    # Filter by active status
    active_projects = await list_projects(db_session, status_filter=ProjectStatus.active)

    assert len(active_projects) == 1
    assert active_projects[0].name == "Active Project"
    assert active_projects[0].status == ProjectStatus.active


@pytest.mark.asyncio
async def test_list_projects_ordering(db_session: AsyncSession) -> None:
    """Test that projects are ordered by created_at descending."""
    # Create projects in sequence
    first = await create_project(db_session, name="First Project", config={})
    second = await create_project(db_session, name="Second Project", config={})
    third = await create_project(db_session, name="Third Project", config={})

    projects = await list_projects(db_session)

    # Most recent should be first
    project_ids = [p.id for p in projects]
    assert project_ids.index(third.id) < project_ids.index(second.id)
    assert project_ids.index(second.id) < project_ids.index(first.id)


@pytest.mark.asyncio
async def test_update_project_status(db_session: AsyncSession) -> None:
    """Test updating a project's status field."""
    project = await create_project(
        db_session,
        name="Status Update Test",
        config={},
    )

    updated = await update_project(
        db_session,
        project.id,
        status=ProjectStatus.active,
    )

    assert updated.status == ProjectStatus.active
    assert updated.name == "Status Update Test"  # Other fields unchanged


@pytest.mark.asyncio
async def test_update_project_multiple_fields(db_session: AsyncSession) -> None:
    """Test updating multiple project fields at once."""
    project = await create_project(
        db_session,
        name="Original Name",
        config={"original": "config"},
        spec_document={"original": "spec"},
    )

    updated = await update_project(
        db_session,
        project.id,
        name="Updated Name",
        status=ProjectStatus.paused,
        config={"updated": "config"},
        architecture_document={"new": "architecture"},
    )

    assert updated.name == "Updated Name"
    assert updated.status == ProjectStatus.paused
    assert updated.config == {"updated": "config"}
    assert updated.architecture_document == {"new": "architecture"}
    # spec_document should remain unchanged
    assert updated.spec_document == {"original": "spec"}


@pytest.mark.asyncio
async def test_update_project_not_found(db_session: AsyncSession) -> None:
    """Test updating a non-existent project raises ValueError."""
    non_existent_id = uuid4()

    with pytest.raises(ValueError, match=f"Project {non_existent_id} not found"):
        await update_project(
            db_session,
            non_existent_id,
            name="Should Fail",
        )


@pytest.mark.asyncio
async def test_delete_project_success(db_session: AsyncSession) -> None:
    """Test successfully deleting an existing project."""
    project = await create_project(
        db_session,
        name="Project to Delete",
        config={},
    )

    deleted = await delete_project(db_session, project.id)

    assert deleted is True

    # Verify project is actually gone
    retrieved = await get_project(db_session, project.id)
    assert retrieved is None


@pytest.mark.asyncio
async def test_delete_project_not_found(db_session: AsyncSession) -> None:
    """Test deleting a non-existent project returns False."""
    non_existent_id = uuid4()

    deleted = await delete_project(db_session, non_existent_id)

    assert deleted is False


@pytest.mark.asyncio
async def test_delete_project_idempotent(db_session: AsyncSession) -> None:
    """Test that deleting the same project twice returns False on second attempt."""
    project = await create_project(
        db_session,
        name="Idempotent Delete Test",
        config={},
    )

    # First deletion succeeds
    first_delete = await delete_project(db_session, project.id)
    assert first_delete is True

    # Second deletion returns False
    second_delete = await delete_project(db_session, project.id)
    assert second_delete is False
