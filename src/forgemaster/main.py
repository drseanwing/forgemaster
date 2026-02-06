"""Main CLI entry point for Forgemaster.

This module provides the main Typer application with sub-commands for project
management, task operations, and orchestrator control.

Usage:
    forgemaster project create "My Project" --spec spec.json
    forgemaster task create <project-id> "Build API" executor
    forgemaster orchestrator start <project-id>
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Annotated, Optional

import structlog
import typer
from rich.console import Console

from forgemaster.cli import project as project_cli
from forgemaster.cli import task as task_cli
from forgemaster.cli import orchestrator as orchestrator_cli
from forgemaster.config import ForgemasterConfig, load_config
from forgemaster.database.connection import get_engine, get_session_factory

app = typer.Typer(
    name="forgemaster",
    help="Forgemaster: AI-Powered Development Orchestration",
    no_args_is_help=True,
)

# Add sub-apps
app.add_typer(project_cli.app, name="project", help="Manage projects")
app.add_typer(task_cli.app, name="task", help="Manage tasks")
app.add_typer(orchestrator_cli.app, name="orchestrator", help="Control orchestrator")

console = Console()


class AppContext:
    """Application context shared across CLI commands.

    Attributes:
        config: Loaded Forgemaster configuration
        engine: Async SQLAlchemy engine
        session_factory: Factory for creating database sessions
    """

    def __init__(self, config: ForgemasterConfig):
        """Initialize application context.

        Args:
            config: Forgemaster configuration
        """
        self.config = config
        self.engine = get_engine(config.database)
        self.session_factory = get_session_factory(self.engine)


# Global context holder
_app_context: AppContext | None = None


def get_app_context() -> AppContext:
    """Get or create the shared application context.

    Returns:
        AppContext instance with config and database connections

    Raises:
        RuntimeError: If context has not been initialized
    """
    if _app_context is None:
        raise RuntimeError("Application context not initialized. Call initialize_context first.")
    return _app_context


def initialize_context(config: ForgemasterConfig) -> AppContext:
    """Initialize the global application context.

    Args:
        config: Forgemaster configuration

    Returns:
        Initialized AppContext instance
    """
    global _app_context
    _app_context = AppContext(config)
    return _app_context


@app.command()
def serve(
    host: Annotated[
        str,
        typer.Option("--host", "-h", help="Host to bind to"),
    ] = "0.0.0.0",
    port: Annotated[
        int,
        typer.Option("--port", "-p", help="Port to bind to"),
    ] = 8000,
    reload: Annotated[
        bool,
        typer.Option("--reload", "-r", help="Enable auto-reload (development)"),
    ] = False,
) -> None:
    """Start the Forgemaster web server.

    Runs the FastAPI application with uvicorn for serving the dashboard
    and REST API endpoints.

    Args:
        host: Host address to bind to (default: 0.0.0.0)
        port: Port number to bind to (default: 8000)
        reload: Enable auto-reload for development
    """
    import uvicorn

    from forgemaster.web.app import create_app

    console.print("[bold cyan]Starting Forgemaster Web Server[/bold cyan]")
    console.print(f"[dim]Host:[/dim] {host}")
    console.print(f"[dim]Port:[/dim] {port}")
    console.print()

    # Load config and create app
    config = load_config()

    # Create the app
    app_instance = create_app(config)

    # Run with uvicorn
    uvicorn.run(
        app_instance,
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


@app.callback()
def main_callback(
    config_path: Annotated[
        Optional[Path],
        typer.Option(
            "--config",
            "-c",
            help="Path to configuration file (TOML format)",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable debug logging"),
    ] = False,
) -> None:
    """Configure global options and initialize application context.

    Args:
        config_path: Optional path to TOML configuration file
        verbose: Enable debug-level logging
    """
    # Configure logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(message)s",
        level=log_level,
        stream=sys.stderr,
    )

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Load configuration
    try:
        config = load_config(config_path)
    except Exception as e:
        console.print(f"[red]Error loading configuration:[/red] {e}")
        raise typer.Exit(code=1)

    # Initialize application context
    try:
        initialize_context(config)
    except Exception as e:
        console.print(f"[red]Error initializing application:[/red] {e}")
        raise typer.Exit(code=1)

    if verbose:
        console.print("[dim]Debug logging enabled[/dim]")


if __name__ == "__main__":
    app()
