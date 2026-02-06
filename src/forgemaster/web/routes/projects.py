"""Project CRUD endpoints for Forgemaster.

This module provides REST API endpoints for managing Project resources:
- List projects with optional status filtering
- Get individual project by ID
- Create new projects
- Update existing projects
- Delete projects

All endpoints use dependency injection for database session factories
and return appropriate HTTP status codes for error conditions.

Example:
    >>> from fastapi import FastAPI
    >>> from forgemaster.web.routes.projects import create_projects_router
    >>>
    >>> app = FastAPI()
    >>> app.include_router(create_projects_router())
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi import status as http_status
from pydantic import BaseModel, Field

from forgemaster.database.models.project import ProjectStatus
from forgemaster.database.queries import project as project_queries
from forgemaster.logging import get_logger

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

logger = get_logger(__name__)


class ProjectCreate(BaseModel):
    """Request schema for creating a new project.

    Attributes:
        name: Human-readable project name (1-255 characters)
        config: Project-specific configuration as dict
        spec_document: Optional project specification as dict
        architecture_document: Optional architecture description as dict
    """

    name: str = Field(..., min_length=1, max_length=255)
    config: dict[str, Any] = Field(default_factory=dict)
    spec_document: dict[str, Any] | None = None
    architecture_document: dict[str, Any] | None = None


class ProjectUpdate(BaseModel):
    """Request schema for updating an existing project.

    All fields are optional. Only provided fields will be updated.

    Attributes:
        name: Human-readable project name
        status: Project lifecycle status
        config: Project-specific configuration as dict
        spec_document: Project specification as dict
        architecture_document: Architecture description as dict
    """

    name: str | None = None
    status: str | None = None
    config: dict[str, Any] | None = None
    spec_document: dict[str, Any] | None = None
    architecture_document: dict[str, Any] | None = None


class ProjectResponse(BaseModel):
    """Response schema for project data.

    Attributes:
        id: UUID primary key
        name: Human-readable project name
        status: Current lifecycle status
        config: Project-specific configuration as dict
        spec_document: Project specification as dict (may be None)
        architecture_document: Architecture description as dict (may be None)
        created_at: Row creation timestamp
        updated_at: Last modification timestamp
    """

    id: UUID
    name: str
    status: str
    config: dict[str, Any]
    spec_document: dict[str, Any] | None
    architecture_document: dict[str, Any] | None
    created_at: Any  # datetime serialized by Pydantic
    updated_at: Any  # datetime serialized by Pydantic

    model_config = {"from_attributes": True}


def get_session_factory(request: Request) -> async_sessionmaker[AsyncSession]:
    """Dependency that retrieves session factory from app state.

    Args:
        request: FastAPI request object

    Returns:
        Session factory from app.state
    """
    return request.app.state.session_factory  # type: ignore[return-value]


def create_projects_router() -> APIRouter:
    """Create projects router with CRUD endpoints.

    Returns:
        Configured APIRouter with project CRUD endpoints.

    Routes:
        GET /projects/ - List all projects with optional status filter
        GET /projects/{project_id} - Get project by ID
        POST /projects/ - Create new project
        PUT /projects/{project_id} - Update project
        DELETE /projects/{project_id} - Delete project
    """
    router = APIRouter(prefix="/projects", tags=["projects"])

    @router.get("/", response_model=list[ProjectResponse])
    async def list_projects(
        status: str | None = None,
        session_factory: async_sessionmaker[AsyncSession] = Depends(  # noqa: B008
            get_session_factory
        ),
    ) -> list[ProjectResponse]:
        """List all projects with optional status filter.

        Args:
            status: Optional status to filter by (draft, active, paused, completed, archived)
            session_factory: Injected session factory from app state

        Returns:
            List of projects matching the filter (or all projects if no filter).

        Raises:
            HTTPException: 400 if status filter is invalid
        """
        # Validate status filter if provided
        status_enum = None
        if status is not None:
            try:
                status_enum = ProjectStatus(status)
            except ValueError:
                logger.warning(
                    "invalid_status_filter",
                    status=status,
                    valid_values=[s.value for s in ProjectStatus],
                )
                raise HTTPException(
                    status_code=http_status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid status: {status}. Valid values: {[s.value for s in ProjectStatus]}",
                ) from None

        async with session_factory() as session:
            projects = await project_queries.list_projects(
                session=session,
                status_filter=status_enum,
            )

        logger.info(
            "projects_listed",
            count=len(projects),
            status_filter=status,
        )

        return [ProjectResponse.model_validate(p) for p in projects]

    @router.get("/{project_id}", response_model=ProjectResponse)
    async def get_project(
        project_id: UUID,
        session_factory: async_sessionmaker[AsyncSession] = Depends(  # noqa: B008
            get_session_factory
        ),
    ) -> ProjectResponse:
        """Get a project by ID.

        Args:
            project_id: UUID of the project to retrieve
            session_factory: Injected session factory from app state

        Returns:
            The requested project.

        Raises:
            HTTPException: 404 if project not found
        """
        async with session_factory() as session:
            project = await project_queries.get_project(
                session=session,
                project_id=project_id,
            )

        if project is None:
            logger.warning("project_not_found", project_id=str(project_id))
            raise HTTPException(
                status_code=http_status.HTTP_404_NOT_FOUND,
                detail=f"Project {project_id} not found",
            )

        logger.info("project_retrieved", project_id=str(project_id))
        return ProjectResponse.model_validate(project)

    @router.post("/", response_model=ProjectResponse, status_code=http_status.HTTP_201_CREATED)
    async def create_project(
        project_data: ProjectCreate,
        session_factory: async_sessionmaker[AsyncSession] = Depends(  # noqa: B008
            get_session_factory
        ),
    ) -> ProjectResponse:
        """Create a new project.

        Args:
            project_data: Project creation data
            session_factory: Injected session factory from app state

        Returns:
            The newly created project.

        Raises:
            HTTPException: 400 if validation fails
        """
        try:
            async with session_factory() as session:
                project = await project_queries.create_project(
                    session=session,
                    name=project_data.name,
                    config=project_data.config,
                    spec_document=project_data.spec_document,
                    architecture_document=project_data.architecture_document,
                )

            logger.info("project_created_via_api", project_id=str(project.id))
            return ProjectResponse.model_validate(project)

        except Exception as exc:
            logger.error(
                "project_creation_failed",
                error=str(exc),
                name=project_data.name,
            )
            raise HTTPException(
                status_code=http_status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to create project: {exc}",
            ) from exc

    @router.put("/{project_id}", response_model=ProjectResponse)
    async def update_project(
        project_id: UUID,
        project_data: ProjectUpdate,
        session_factory: async_sessionmaker[AsyncSession] = Depends(  # noqa: B008
            get_session_factory
        ),
    ) -> ProjectResponse:
        """Update an existing project.

        Only fields provided in the request body will be updated.

        Args:
            project_id: UUID of the project to update
            project_data: Fields to update
            session_factory: Injected session factory from app state

        Returns:
            The updated project.

        Raises:
            HTTPException: 404 if project not found
            HTTPException: 400 if validation fails
        """
        # Build updates dict with only non-None fields
        updates: dict[str, Any] = {}
        if project_data.name is not None:
            updates["name"] = project_data.name
        if project_data.status is not None:
            # Validate status
            try:
                status_enum = ProjectStatus(project_data.status)
                updates["status"] = status_enum
            except ValueError:
                logger.warning(
                    "invalid_status_value",
                    status=project_data.status,
                    valid_values=[s.value for s in ProjectStatus],
                )
                raise HTTPException(
                    status_code=http_status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid status: {project_data.status}. Valid values: {[s.value for s in ProjectStatus]}",
                ) from None
        if project_data.config is not None:
            updates["config"] = project_data.config
        if project_data.spec_document is not None:
            updates["spec_document"] = project_data.spec_document
        if project_data.architecture_document is not None:
            updates["architecture_document"] = project_data.architecture_document

        if not updates:
            logger.warning("no_updates_provided", project_id=str(project_id))
            raise HTTPException(
                status_code=http_status.HTTP_400_BAD_REQUEST,
                detail="No fields to update",
            )

        try:
            async with session_factory() as session:
                project = await project_queries.update_project(
                    session=session,
                    project_id=project_id,
                    **updates,
                )

            logger.info(
                "project_updated_via_api",
                project_id=str(project_id),
                fields_updated=list(updates.keys()),
            )
            return ProjectResponse.model_validate(project)

        except ValueError as exc:
            # Project not found
            logger.warning("project_not_found_for_update", project_id=str(project_id))
            raise HTTPException(
                status_code=http_status.HTTP_404_NOT_FOUND,
                detail=str(exc),
            ) from exc
        except Exception as exc:
            logger.error(
                "project_update_failed",
                project_id=str(project_id),
                error=str(exc),
            )
            raise HTTPException(
                status_code=http_status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to update project: {exc}",
            ) from exc

    @router.delete("/{project_id}", status_code=http_status.HTTP_204_NO_CONTENT)
    async def delete_project(
        project_id: UUID,
        session_factory: async_sessionmaker[AsyncSession] = Depends(  # noqa: B008
            get_session_factory
        ),
    ) -> None:
        """Delete a project.

        Args:
            project_id: UUID of the project to delete
            session_factory: Injected session factory from app state

        Raises:
            HTTPException: 404 if project not found
        """
        async with session_factory() as session:
            deleted = await project_queries.delete_project(
                session=session,
                project_id=project_id,
            )

        if not deleted:
            logger.warning("project_not_found_for_deletion", project_id=str(project_id))
            raise HTTPException(
                status_code=http_status.HTTP_404_NOT_FOUND,
                detail=f"Project {project_id} not found",
            )

        logger.info("project_deleted_via_api", project_id=str(project_id))

    return router
