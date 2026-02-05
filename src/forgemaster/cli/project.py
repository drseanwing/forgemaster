"""Project management CLI commands.

This module provides CLI commands for creating, listing, and managing projects.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Optional
from uuid import UUID

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from forgemaster.database.models.project import ProjectStatus
from forgemaster.database.queries.project import create_project, list_projects

app = typer.Typer(help="Project management commands")
console = Console()


@app.command()
def create(
    name: Annotated[str, typer.Argument(help="Project name")],
    spec_file: Annotated[
        Optional[Path],
        typer.Option(
            "--spec",
            "-s",
            help="Path to project specification JSON file",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ] = None,
    config_json: Annotated[
        Optional[str],
        typer.Option(
            "--config",
            "-c",
            help="Project configuration as JSON string",
        ),
    ] = None,
) -> None:
    """Create a new project.

    Args:
        name: Human-readable project name
        spec_file: Optional path to specification JSON file
        config_json: Optional configuration as JSON string
    """
    from forgemaster.main import get_app_context

    ctx = get_app_context()

    # Load spec document if provided
    spec_document = None
    if spec_file is not None:
        try:
            with open(spec_file, "r") as f:
                spec_document = json.load(f)
        except Exception as e:
            console.print(f"[red]Error reading spec file:[/red] {e}")
            raise typer.Exit(code=1)

    # Parse config JSON
    config = {}
    if config_json is not None:
        try:
            config = json.loads(config_json)
        except json.JSONDecodeError as e:
            console.print(f"[red]Invalid JSON in config:[/red] {e}")
            raise typer.Exit(code=1)

    # Create project
    async def _create_project():
        async with ctx.session_factory() as session:
            project = await create_project(
                session=session,
                name=name,
                config=config,
                spec_document=spec_document,
            )
            return project

    import asyncio

    try:
        project = asyncio.run(_create_project())
    except Exception as e:
        console.print(f"[red]Error creating project:[/red] {e}")
        raise typer.Exit(code=1)

    # Display result
    panel = Panel(
        f"[green]Project created successfully![/green]\n\n"
        f"[bold]ID:[/bold] {project.id}\n"
        f"[bold]Name:[/bold] {project.name}\n"
        f"[bold]Status:[/bold] {project.status.value}\n"
        f"[bold]Created:[/bold] {project.created_at.strftime('%Y-%m-%d %H:%M:%S')}",
        title="Project Created",
        border_style="green",
    )
    console.print(panel)


@app.command()
def list(
    status: Annotated[
        Optional[str],
        typer.Option(
            "--status",
            "-s",
            help="Filter by status (draft, active, paused, completed, archived)",
        ),
    ] = None,
    format: Annotated[
        str,
        typer.Option(
            "--format",
            "-f",
            help="Output format (table or json)",
        ),
    ] = "table",
) -> None:
    """List all projects.

    Args:
        status: Optional status filter
        format: Output format (table or json)
    """
    from forgemaster.main import get_app_context

    ctx = get_app_context()

    # Parse status filter
    status_filter = None
    if status is not None:
        try:
            status_filter = ProjectStatus(status)
        except ValueError:
            console.print(
                f"[red]Invalid status:[/red] {status}. "
                f"Valid values: draft, active, paused, completed, archived"
            )
            raise typer.Exit(code=1)

    # Fetch projects
    async def _list_projects():
        async with ctx.session_factory() as session:
            projects = await list_projects(session, status_filter=status_filter)
            return projects

    import asyncio

    try:
        projects = asyncio.run(_list_projects())
    except Exception as e:
        console.print(f"[red]Error listing projects:[/red] {e}")
        raise typer.Exit(code=1)

    # Display results
    if format == "json":
        output = [
            {
                "id": str(p.id),
                "name": p.name,
                "status": p.status.value,
                "created_at": p.created_at.isoformat(),
                "updated_at": p.updated_at.isoformat() if p.updated_at else None,
            }
            for p in projects
        ]
        console.print(json.dumps(output, indent=2))
    else:
        # Table format
        if not projects:
            console.print("[yellow]No projects found[/yellow]")
            return

        table = Table(title="Projects")
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Name", style="bold")
        table.add_column("Status", style="magenta")
        table.add_column("Created", style="dim")

        for p in projects:
            # Color-code status
            status_color = {
                "draft": "dim",
                "active": "green",
                "paused": "yellow",
                "completed": "blue",
                "archived": "dim",
            }.get(p.status.value, "white")

            table.add_row(
                str(p.id),
                p.name,
                f"[{status_color}]{p.status.value}[/{status_color}]",
                p.created_at.strftime("%Y-%m-%d %H:%M"),
            )

        console.print(table)
