"""Task management CLI commands.

This module provides CLI commands for creating, listing, and managing tasks.
"""

from __future__ import annotations

import json
from typing import Annotated, Optional
from uuid import UUID

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from forgemaster.database.models.task import TaskStatus
from forgemaster.database.queries.task import create_task, list_tasks

app = typer.Typer(help="Task management commands")
console = Console()


@app.command()
def create(
    project_id: Annotated[str, typer.Argument(help="Project UUID")],
    title: Annotated[str, typer.Argument(help="Task title")],
    agent_type: Annotated[str, typer.Argument(help="Agent type (executor, architect, etc.)")],
    description: Annotated[
        Optional[str],
        typer.Option("--description", "-d", help="Detailed task description"),
    ] = None,
    priority: Annotated[
        int,
        typer.Option("--priority", "-p", help="Task priority (lower = higher priority)"),
    ] = 100,
    dependencies: Annotated[
        Optional[str],
        typer.Option(
            "--dependencies",
            "-D",
            help="Comma-separated list of dependency task UUIDs",
        ),
    ] = None,
    model_tier: Annotated[
        str,
        typer.Option("--model", "-m", help="Model tier (auto, haiku, sonnet, opus)"),
    ] = "auto",
) -> None:
    """Create a new task.

    Args:
        project_id: UUID of the parent project
        title: Short task description
        agent_type: Type of agent to execute the task
        description: Detailed task description
        priority: Numeric priority (lower = higher priority)
        dependencies: Comma-separated dependency task UUIDs
        model_tier: Model tier preference
    """
    from forgemaster.main import get_app_context

    ctx = get_app_context()

    # Parse project UUID
    try:
        project_uuid = UUID(project_id)
    except ValueError:
        console.print(f"[red]Invalid project UUID:[/red] {project_id}")
        raise typer.Exit(code=1)

    # Parse dependencies
    dependency_list = None
    if dependencies:
        try:
            dependency_list = [UUID(dep.strip()) for dep in dependencies.split(",")]
        except ValueError as e:
            console.print(f"[red]Invalid dependency UUID:[/red] {e}")
            raise typer.Exit(code=1)

    # Create task
    async def _create_task():
        async with ctx.session_factory() as session:
            task = await create_task(
                session=session,
                project_id=project_uuid,
                title=title,
                agent_type=agent_type,
                description=description,
                model_tier=model_tier,
                priority=priority,
                dependencies=dependency_list,
            )
            return task

    import asyncio

    try:
        task = asyncio.run(_create_task())
    except Exception as e:
        console.print(f"[red]Error creating task:[/red] {e}")
        raise typer.Exit(code=1)

    # Display result
    panel = Panel(
        f"[green]Task created successfully![/green]\n\n"
        f"[bold]ID:[/bold] {task.id}\n"
        f"[bold]Title:[/bold] {task.title}\n"
        f"[bold]Agent:[/bold] {task.agent_type}\n"
        f"[bold]Status:[/bold] {task.status.value}\n"
        f"[bold]Priority:[/bold] {task.priority}\n"
        f"[bold]Model:[/bold] {task.model_tier}",
        title="Task Created",
        border_style="green",
    )
    console.print(panel)


@app.command()
def list(
    project_id: Annotated[Optional[str], typer.Option("--project", "-p", help="Project UUID")] = None,
    status: Annotated[
        Optional[str],
        typer.Option(
            "--status",
            "-s",
            help="Filter by status (pending, ready, assigned, running, review, done, failed, blocked)",
        ),
    ] = None,
    format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format (table or json)"),
    ] = "table",
) -> None:
    """List tasks.

    Args:
        project_id: Optional project UUID filter
        status: Optional status filter
        format: Output format (table or json)
    """
    from forgemaster.main import get_app_context

    ctx = get_app_context()

    # Parse project UUID
    project_uuid = None
    if project_id is not None:
        try:
            project_uuid = UUID(project_id)
        except ValueError:
            console.print(f"[red]Invalid project UUID:[/red] {project_id}")
            raise typer.Exit(code=1)

    # Parse status filter
    status_filter = None
    if status is not None:
        try:
            status_filter = TaskStatus(status)
        except ValueError:
            console.print(
                f"[red]Invalid status:[/red] {status}. "
                f"Valid values: pending, ready, assigned, running, review, done, failed, blocked"
            )
            raise typer.Exit(code=1)

    # Fetch tasks
    async def _list_tasks():
        async with ctx.session_factory() as session:
            tasks = await list_tasks(
                session, project_id=project_uuid, status_filter=status_filter
            )
            return tasks

    import asyncio

    try:
        tasks = asyncio.run(_list_tasks())
    except Exception as e:
        console.print(f"[red]Error listing tasks:[/red] {e}")
        raise typer.Exit(code=1)

    # Display results
    if format == "json":
        output = [
            {
                "id": str(t.id),
                "project_id": str(t.project_id),
                "title": t.title,
                "agent_type": t.agent_type,
                "status": t.status.value,
                "priority": t.priority,
                "dependencies": [str(d) for d in (t.dependencies or [])],
                "created_at": t.created_at.isoformat(),
            }
            for t in tasks
        ]
        console.print(json.dumps(output, indent=2))
    else:
        # Table format
        if not tasks:
            console.print("[yellow]No tasks found[/yellow]")
            return

        table = Table(title="Tasks")
        table.add_column("ID", style="cyan", no_wrap=True, overflow="fold")
        table.add_column("Title", style="bold")
        table.add_column("Status", style="magenta")
        table.add_column("Priority", justify="right", style="dim")
        table.add_column("Agent", style="blue")
        table.add_column("Dependencies", style="dim")

        for t in tasks:
            # Color-code status
            status_color = {
                "pending": "dim",
                "ready": "yellow",
                "assigned": "cyan",
                "running": "blue",
                "review": "magenta",
                "done": "green",
                "failed": "red",
                "blocked": "red dim",
            }.get(t.status.value, "white")

            # Format dependencies
            dep_count = len(t.dependencies) if t.dependencies else 0
            dep_str = f"{dep_count} deps" if dep_count > 0 else "-"

            table.add_row(
                str(t.id)[:8] + "...",
                t.title,
                f"[{status_color}]{t.status.value}[/{status_color}]",
                str(t.priority),
                t.agent_type,
                dep_str,
            )

        console.print(table)
