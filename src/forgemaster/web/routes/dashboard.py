"""Dashboard routes for HTML views.

This module provides FastAPI routes for rendering the web dashboard UI.
All views use htmx for dynamic updates and Tailwind CSS for styling.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from forgemaster.architecture.spec_parser import SpecParser
from forgemaster.database.models.session import SessionStatus
from forgemaster.database.queries import project as project_queries
from forgemaster.database.queries.session import list_sessions
from forgemaster.database.queries.task import list_tasks
from forgemaster.logging import get_logger

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

logger = get_logger(__name__)

# Templates directory relative to this file
TEMPLATES_DIR = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def get_session_factory(request: Request) -> "async_sessionmaker[AsyncSession]":
    """Extract session factory from FastAPI app state."""
    return request.app.state.session_factory  # type: ignore[return-value]


# Status color mappings
TASK_STATUS_COLORS = {
    "pending": {"bg": "bg-gray-100", "text": "text-gray-800", "border": "border-gray-400"},
    "ready": {"bg": "bg-blue-100", "text": "text-blue-800", "border": "border-blue-400"},
    "assigned": {"bg": "bg-yellow-100", "text": "text-yellow-800", "border": "border-yellow-400"},
    "running": {"bg": "bg-green-100", "text": "text-green-800", "border": "border-green-400"},
    "review": {"bg": "bg-purple-100", "text": "text-purple-800", "border": "border-purple-400"},
    "done": {"bg": "bg-emerald-100", "text": "text-emerald-800", "border": "border-emerald-400"},
    "failed": {"bg": "bg-red-100", "text": "text-red-800", "border": "border-red-400"},
    "blocked": {"bg": "bg-orange-100", "text": "text-orange-800", "border": "border-orange-400"},
}

PROJECT_STATUS_COLORS = {
    "draft": {"bg": "bg-gray-100", "text": "text-gray-800"},
    "active": {"bg": "bg-green-100", "text": "text-green-800"},
    "paused": {"bg": "bg-yellow-100", "text": "text-yellow-800"},
    "completed": {"bg": "bg-blue-100", "text": "text-blue-800"},
    "archived": {"bg": "bg-gray-100", "text": "text-gray-800"},
}


def create_dashboard_router() -> APIRouter:
    """Create the dashboard router for HTML views.

    Returns:
        APIRouter configured with dashboard routes.
    """
    router = APIRouter(prefix="/dashboard", tags=["dashboard"])

    @router.get("/", response_class=HTMLResponse)
    async def dashboard_home(request: Request) -> HTMLResponse:
        """Render the main dashboard page."""
        logger.debug("Rendering dashboard home")
        return templates.TemplateResponse("tasks.html", {"request": request})

    @router.get("/tasks", response_class=HTMLResponse)
    async def tasks_view(request: Request) -> HTMLResponse:
        """Render the task board view."""
        logger.debug("Rendering task board view")
        return templates.TemplateResponse("tasks.html", {"request": request})

    @router.get("/sessions", response_class=HTMLResponse)
    async def sessions_view(request: Request) -> HTMLResponse:
        """Render the sessions view."""
        logger.debug("Rendering sessions view")
        return templates.TemplateResponse("sessions.html", {"request": request})

    @router.get("/projects", response_class=HTMLResponse)
    async def projects_view(request: Request) -> HTMLResponse:
        """Render the projects view with creation form."""
        logger.debug("Rendering projects view")
        return templates.TemplateResponse("projects.html", {"request": request})

    # ---- HTMX Fragment Endpoints ----

    @router.get("/htmx/tasks", response_class=HTMLResponse)
    async def htmx_tasks_fragment(
        request: Request,
        session_factory: "async_sessionmaker[AsyncSession]" = Depends(  # noqa: B008
            get_session_factory
        ),
    ) -> HTMLResponse:
        """Return HTML fragment for task board (htmx endpoint)."""
        async with session_factory() as session:
            tasks = await list_tasks(session)

        # Group tasks by status
        task_groups: dict[str, list] = {
            "pending": [],
            "ready": [],
            "running": [],
            "done": [],
        }

        for task in tasks:
            status = task.status.value
            if status in task_groups:
                task_groups[status].append(task)
            elif status in ("assigned", "review"):
                task_groups["running"].append(task)
            elif status in ("failed", "blocked"):
                task_groups["pending"].append(task)

        # Build HTML
        if not tasks:
            html = """
            <div class="col-span-full bg-white rounded-lg shadow-md p-12 text-center">
                <svg class="mx-auto h-12 w-12 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                </svg>
                <h3 class="mt-2 text-sm font-medium text-gray-900">No tasks found</h3>
                <p class="mt-1 text-sm text-gray-500">Create a project and add tasks to get started.</p>
                <p class="mt-4 text-xs text-gray-400">Use CLI: forgemaster project create "My Project"</p>
            </div>
            """
            return HTMLResponse(content=html)

        columns = [
            ("Pending", "pending", "gray"),
            ("Ready", "ready", "blue"),
            ("Running", "running", "green"),
            ("Done", "done", "emerald"),
        ]

        html_parts = []
        for title, status, color in columns:
            group_tasks = task_groups.get(status, [])
            cards_html = ""
            for task in group_tasks:
                task_status = task.status.value
                colors = TASK_STATUS_COLORS.get(task_status, TASK_STATUS_COLORS["pending"])
                cards_html += f"""
                <div class="bg-gray-50 rounded-md p-3 border-l-4 {colors['border']} hover:shadow-md transition-shadow"
                     data-task-id="{task.id}">
                    <div class="flex justify-between items-start mb-2">
                        <h3 class="font-medium text-gray-900 text-sm">{task.title}</h3>
                        <span class="text-xs px-2 py-1 rounded {colors['bg']} {colors['text']}">{task_status}</span>
                    </div>
                    <p class="text-xs text-gray-600 mb-2 line-clamp-2">{task.description or ''}</p>
                    <div class="flex justify-between items-center text-xs text-gray-500">
                        <span>{task.agent_type}</span>
                        <span>P{task.priority}</span>
                    </div>
                </div>
                """

            html_parts.append(f"""
            <div class="bg-white rounded-lg shadow-md p-4">
                <h2 class="text-lg font-semibold mb-4 flex items-center">
                    <span class="inline-block w-3 h-3 rounded-full bg-{color}-500 mr-2"></span>
                    {title}
                    <span class="ml-auto text-sm font-normal text-gray-500">{len(group_tasks)}</span>
                </h2>
                <div class="space-y-3">
                    {cards_html if cards_html else '<p class="text-sm text-gray-400 text-center py-4">No tasks</p>'}
                </div>
            </div>
            """)

        return HTMLResponse(content="".join(html_parts))

    @router.get("/htmx/sessions", response_class=HTMLResponse)
    async def htmx_sessions_fragment(
        request: Request,
        session_factory: "async_sessionmaker[AsyncSession]" = Depends(  # noqa: B008
            get_session_factory
        ),
    ) -> HTMLResponse:
        """Return HTML fragment for sessions list (htmx endpoint)."""
        async with session_factory() as session:
            sessions = await list_sessions(session, status_filter=SessionStatus.running)

        if not sessions:
            html = """
            <div class="bg-white rounded-lg shadow-md p-12 text-center">
                <svg class="mx-auto h-12 w-12 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                <h3 class="mt-2 text-sm font-medium text-gray-900">No active sessions</h3>
                <p class="mt-1 text-sm text-gray-500">Start a new task to create a session.</p>
            </div>
            """
            return HTMLResponse(content=html)

        html_parts = []
        for sess in sessions:
            status_color = "green" if sess.status == SessionStatus.running else "gray"
            html_parts.append(f"""
            <div class="bg-white rounded-lg shadow-md p-6">
                <div class="flex justify-between items-start mb-4">
                    <div>
                        <h2 class="text-xl font-semibold text-gray-900">{sess.agent_type}</h2>
                        <p class="text-sm text-gray-500 mt-1">Session {str(sess.id)[:8]}...</p>
                    </div>
                    <span class="px-3 py-1 rounded-full text-sm font-medium bg-{status_color}-100 text-{status_color}-800">
                        {sess.status.value}
                    </span>
                </div>
                <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                    <div class="border-l-4 border-blue-500 pl-3">
                        <p class="text-xs text-gray-500 uppercase">Model</p>
                        <p class="text-lg font-semibold text-gray-900">{sess.model_tier or 'auto'}</p>
                    </div>
                    <div class="border-l-4 border-green-500 pl-3">
                        <p class="text-xs text-gray-500 uppercase">Token Count</p>
                        <p class="text-lg font-semibold text-gray-900">{sess.token_count or 0:,}</p>
                    </div>
                    <div class="border-l-4 border-purple-500 pl-3">
                        <p class="text-xs text-gray-500 uppercase">Started</p>
                        <p class="text-lg font-semibold text-gray-900">{sess.started_at.strftime('%H:%M:%S') if sess.started_at else '-'}</p>
                    </div>
                </div>
            </div>
            """)

        return HTMLResponse(content="".join(html_parts))

    @router.get("/htmx/projects", response_class=HTMLResponse)
    async def htmx_projects_fragment(
        request: Request,
        session_factory: "async_sessionmaker[AsyncSession]" = Depends(  # noqa: B008
            get_session_factory
        ),
    ) -> HTMLResponse:
        """Return HTML fragment for projects list (htmx endpoint)."""
        async with session_factory() as session:
            projects = await project_queries.list_projects(session)

        if not projects:
            html_content = """
            <div class="bg-white rounded-lg shadow-md p-12 text-center">
                <svg class="mx-auto h-12 w-12 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
                </svg>
                <h3 class="mt-2 text-sm font-medium text-gray-900">No projects yet</h3>
                <p class="mt-1 text-sm text-gray-500">Create your first project to get started.</p>
            </div>
            """
            return HTMLResponse(content=html_content)

        html_parts = []
        for project in projects:
            status = project.status.value
            colors = PROJECT_STATUS_COLORS.get(status, PROJECT_STATUS_COLORS["draft"])
            spec_present = "Yes" if project.spec_document else "No"
            created = project.created_at.strftime("%Y-%m-%d %H:%M") if project.created_at else "-"

            html_parts.append(f"""
            <div class="bg-white rounded-lg shadow-md p-6 hover:shadow-lg transition-shadow">
                <div class="flex justify-between items-start mb-4">
                    <h2 class="text-xl font-semibold text-gray-900">{html.escape(project.name)}</h2>
                    <span class="px-3 py-1 rounded-full text-sm font-medium {colors['bg']} {colors['text']}">
                        {status}
                    </span>
                </div>
                <div class="grid grid-cols-2 gap-4 text-sm">
                    <div>
                        <p class="text-gray-500">Spec</p>
                        <p class="font-medium text-gray-900">{spec_present}</p>
                    </div>
                    <div>
                        <p class="text-gray-500">Created</p>
                        <p class="font-medium text-gray-900">{created}</p>
                    </div>
                </div>
            </div>
            """)

        return HTMLResponse(content="".join(html_parts))

    @router.post("/htmx/projects/validate", response_class=HTMLResponse)
    async def htmx_validate_spec(
        request: Request,
    ) -> HTMLResponse:
        """Validate spec content (htmx endpoint)."""
        from fastapi import Form

        form = await request.form()
        spec_content = form.get("spec_content", "").strip()

        if not spec_content:
            html_content = """
            <div class="text-sm text-gray-500 italic">
                Enter spec content above to validate format and structure
            </div>
            """
            return HTMLResponse(content=html_content)

        # Detect format
        spec_format = "json" if spec_content.startswith("{") else "markdown"

        # Parse spec
        try:
            parser = SpecParser()
            if spec_format == "json":
                spec_doc = parser.parse_json(spec_content)
            else:
                spec_doc = parser.parse_markdown(spec_content)
        except Exception as e:
            html_content = f"""
            <div class="bg-red-50 border-l-4 border-red-500 p-4">
                <div class="flex">
                    <div class="flex-shrink-0">
                        <svg class="h-5 w-5 text-red-400" fill="currentColor" viewBox="0 0 20 20">
                            <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd"/>
                        </svg>
                    </div>
                    <div class="ml-3">
                        <h3 class="text-sm font-medium text-red-800">Parse Error</h3>
                        <p class="text-sm text-red-700 mt-1">{html.escape(str(e))}</p>
                    </div>
                </div>
            </div>
            """
            return HTMLResponse(content=html_content)

        # Validate spec
        validation_result = parser.validate_spec(spec_doc)

        if validation_result.errors:
            errors_html = "".join(
                f'<li class="text-sm text-red-700">{html.escape(err)}</li>'
                for err in validation_result.errors
            )
            html_content = f"""
            <div class="bg-red-50 border-l-4 border-red-500 p-4">
                <div class="flex">
                    <div class="flex-shrink-0">
                        <svg class="h-5 w-5 text-red-400" fill="currentColor" viewBox="0 0 20 20">
                            <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd"/>
                        </svg>
                    </div>
                    <div class="ml-3">
                        <h3 class="text-sm font-medium text-red-800">Validation Errors</h3>
                        <ul class="list-disc list-inside mt-2">{errors_html}</ul>
                    </div>
                </div>
            </div>
            """
            return HTMLResponse(content=html_content)

        if validation_result.warnings:
            warnings_html = "".join(
                f'<li class="text-sm text-yellow-700">{html.escape(warn)}</li>'
                for warn in validation_result.warnings
            )
            html_content = f"""
            <div class="bg-yellow-50 border-l-4 border-yellow-500 p-4">
                <div class="flex">
                    <div class="flex-shrink-0">
                        <svg class="h-5 w-5 text-yellow-400" fill="currentColor" viewBox="0 0 20 20">
                            <path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd"/>
                        </svg>
                    </div>
                    <div class="ml-3">
                        <h3 class="text-sm font-medium text-yellow-800">Validation Warnings</h3>
                        <ul class="list-disc list-inside mt-2">{warnings_html}</ul>
                    </div>
                </div>
            </div>
            """
            return HTMLResponse(content=html_content)

        # Valid spec with no warnings
        html_content = """
        <div class="bg-green-50 border-l-4 border-green-500 p-4">
            <div class="flex">
                <div class="flex-shrink-0">
                    <svg class="h-5 w-5 text-green-400" fill="currentColor" viewBox="0 0 20 20">
                        <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"/>
                    </svg>
                </div>
                <div class="ml-3">
                    <h3 class="text-sm font-medium text-green-800">Spec is valid</h3>
                    <p class="text-sm text-green-700 mt-1">All validation checks passed</p>
                </div>
            </div>
        </div>
        """
        return HTMLResponse(content=html_content)

    @router.post("/htmx/projects/create", response_class=HTMLResponse)
    async def htmx_create_project(
        request: Request,
        session_factory: "async_sessionmaker[AsyncSession]" = Depends(  # noqa: B008
            get_session_factory
        ),
    ) -> HTMLResponse:
        """Create a new project (htmx endpoint)."""
        from fastapi import Form, UploadFile, File

        form = await request.form()
        name = form.get("name", "").strip()
        spec_content = form.get("spec_content", "").strip()

        # Handle file upload if present
        spec_file: UploadFile | None = form.get("spec_file")  # type: ignore[assignment]
        if spec_file and spec_file.filename:
            spec_bytes = await spec_file.read()
            spec_content = spec_bytes.decode("utf-8")

        # Validate name
        if not name:
            html_content = """
            <div class="bg-red-50 border-l-4 border-red-500 p-4">
                <p class="text-sm text-red-700">Project name is required</p>
            </div>
            """
            return HTMLResponse(content=html_content)

        if len(name) > 255:
            html_content = """
            <div class="bg-red-50 border-l-4 border-red-500 p-4">
                <p class="text-sm text-red-700">Project name must be 255 characters or less</p>
            </div>
            """
            return HTMLResponse(content=html_content)

        # Parse spec if provided
        spec_data: dict[str, str] | None = None
        if spec_content:
            try:
                parser = SpecParser()
                spec_format = "json" if spec_content.startswith("{") else "markdown"
                if spec_format == "json":
                    parser.parse_json(spec_content)
                else:
                    parser.parse_markdown(spec_content)
                spec_data = {"format": spec_format, "content": spec_content}
            except Exception as e:
                html_content = f"""
                <div class="bg-red-50 border-l-4 border-red-500 p-4">
                    <p class="text-sm text-red-700">Spec parse error: {html.escape(str(e))}</p>
                </div>
                """
                return HTMLResponse(content=html_content)

        # Create project
        try:
            async with session_factory() as session:
                project = await project_queries.create_project(
                    session,
                    name=name,
                    config={},
                    spec_document=spec_data,
                )

            html_content = f"""
            <div class="bg-green-50 border-l-4 border-green-500 p-4">
                <div class="flex">
                    <div class="flex-shrink-0">
                        <svg class="h-5 w-5 text-green-400" fill="currentColor" viewBox="0 0 20 20">
                            <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"/>
                        </svg>
                    </div>
                    <div class="ml-3">
                        <h3 class="text-sm font-medium text-green-800">Project created successfully</h3>
                        <p class="text-sm text-green-700 mt-1">{html.escape(project.name)}</p>
                    </div>
                </div>
            </div>
            """
            response = HTMLResponse(content=html_content)
            response.headers["HX-Trigger"] = "projectCreated"
            return response

        except Exception as e:
            logger.error("Failed to create project", error=str(e))
            html_content = f"""
            <div class="bg-red-50 border-l-4 border-red-500 p-4">
                <p class="text-sm text-red-700">Failed to create project: {html.escape(str(e))}</p>
            </div>
            """
            return HTMLResponse(content=html_content)

    @router.get("/htmx/projects/example", response_class=HTMLResponse)
    async def htmx_example_spec(request: Request) -> HTMLResponse:
        """Return example spec content (htmx endpoint)."""
        example_content = """# My Project

## Requirements
- User authentication with OAuth2
- RESTful API for data management
- Real-time notifications via WebSocket

## Technology
- Python 3.12
- FastAPI framework
- PostgreSQL database
- Redis for caching

## Architecture
The system follows a layered architecture:
- API Layer: FastAPI routes and request handling
- Service Layer: Business logic and validation
- Data Layer: SQLAlchemy ORM and database queries
- External integrations via adapter pattern"""

        html_content = f"""
        <div class="bg-blue-50 border-l-4 border-blue-500 p-4">
            <h3 class="text-sm font-medium text-blue-800 mb-2">Example Markdown Spec</h3>
            <pre class="text-xs text-blue-900 whitespace-pre-wrap font-mono">{html.escape(example_content)}</pre>
        </div>
        """
        return HTMLResponse(content=html_content)

    return router
