"""Dashboard routes for HTML views.

This module provides FastAPI routes for rendering the web dashboard UI.
All views use htmx for dynamic updates and Tailwind CSS for styling.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from forgemaster.database.models.session import SessionStatus
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

    return router
