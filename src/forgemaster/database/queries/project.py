"""Project CRUD query functions for Forgemaster.

Provides async functions for creating, reading, updating, and deleting
Project records using SQLAlchemy 2.0 select() API.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

import structlog
from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from forgemaster.database.models.project import Project, ProjectStatus

logger = structlog.get_logger(__name__)


async def create_project(
    session: AsyncSession,
    name: str,
    config: dict[str, Any],
    spec_document: dict[str, Any] | None = None,
    architecture_document: dict[str, Any] | None = None,
) -> Project:
    """Create a new project.

    Args:
        session: Active async database session.
        name: Human-readable project name.
        config: Project-specific configuration as dict.
        spec_document: Optional project specification as dict.
        architecture_document: Optional architecture description as dict.

    Returns:
        The newly created Project instance.
    """
    project = Project(
        name=name,
        config=config,
        spec_document=spec_document,
        architecture_document=architecture_document,
        status=ProjectStatus.draft,
    )

    async with session.begin():
        session.add(project)
        await session.flush()
        await session.refresh(project)

    logger.info(
        "project_created",
        project_id=str(project.id),
        name=name,
        status=project.status.value,
    )

    return project


async def get_project(
    session: AsyncSession,
    project_id: UUID,
) -> Project | None:
    """Retrieve a project by ID.

    Args:
        session: Active async database session.
        project_id: UUID of the project to retrieve.

    Returns:
        The Project instance if found, None otherwise.
    """
    stmt = select(Project).where(Project.id == project_id)
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def list_projects(
    session: AsyncSession,
    status_filter: ProjectStatus | None = None,
) -> list[Project]:
    """List projects, optionally filtered by status.

    Args:
        session: Active async database session.
        status_filter: Optional status to filter by.

    Returns:
        List of matching Project instances.
    """
    stmt = select(Project)

    if status_filter is not None:
        stmt = stmt.where(Project.status == status_filter)

    stmt = stmt.order_by(Project.created_at.desc())
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def update_project(
    session: AsyncSession,
    project_id: UUID,
    **updates: Any,
) -> Project:
    """Update a project's fields.

    Args:
        session: Active async database session.
        project_id: UUID of the project to update.
        **updates: Field names and values to update.

    Returns:
        The updated Project instance.

    Raises:
        ValueError: If project not found.
    """
    # First fetch the project
    project = await get_project(session, project_id)
    if project is None:
        raise ValueError(f"Project {project_id} not found")

    # Apply updates
    async with session.begin():
        stmt = (
            update(Project)
            .where(Project.id == project_id)
            .values(**updates)
        )
        await session.execute(stmt)
        await session.refresh(project)

    logger.info(
        "project_updated",
        project_id=str(project_id),
        fields_updated=list(updates.keys()),
    )

    return project


async def delete_project(
    session: AsyncSession,
    project_id: UUID,
) -> bool:
    """Delete a project.

    Args:
        session: Active async database session.
        project_id: UUID of the project to delete.

    Returns:
        True if the project was deleted, False if not found.
    """
    async with session.begin():
        stmt = delete(Project).where(Project.id == project_id)
        result = await session.execute(stmt)

    deleted = result.rowcount > 0

    if deleted:
        logger.info("project_deleted", project_id=str(project_id))
    else:
        logger.warning("project_not_found", project_id=str(project_id))

    return deleted
