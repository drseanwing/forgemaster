"""Dashboard routes for HTML views.

This module provides FastAPI routes for rendering the web dashboard UI.
All views use htmx for dynamic updates and Tailwind CSS for styling.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from forgemaster.logging import get_logger

logger = get_logger(__name__)

# Templates directory relative to this file
TEMPLATES_DIR = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def create_dashboard_router() -> APIRouter:
    """Create the dashboard router for HTML views.

    Returns:
        APIRouter configured with dashboard routes.
    """
    router = APIRouter(prefix="/dashboard", tags=["dashboard"])

    @router.get("/", response_class=HTMLResponse)
    async def dashboard_home(request: Request) -> HTMLResponse:
        """Render the main dashboard page.

        Args:
            request: The incoming request object.

        Returns:
            HTMLResponse with the rendered tasks view.
        """
        logger.debug("Rendering dashboard home")
        return templates.TemplateResponse("tasks.html", {"request": request})

    @router.get("/tasks", response_class=HTMLResponse)
    async def tasks_view(request: Request) -> HTMLResponse:
        """Render the task board view.

        Args:
            request: The incoming request object.

        Returns:
            HTMLResponse with the rendered task board.
        """
        logger.debug("Rendering task board view")
        return templates.TemplateResponse("tasks.html", {"request": request})

    @router.get("/sessions", response_class=HTMLResponse)
    async def sessions_view(request: Request) -> HTMLResponse:
        """Render the sessions view.

        Args:
            request: The incoming request object.

        Returns:
            HTMLResponse with the rendered sessions list.
        """
        logger.debug("Rendering sessions view")
        return templates.TemplateResponse("sessions.html", {"request": request})

    return router
