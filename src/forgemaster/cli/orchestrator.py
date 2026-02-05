"""Orchestrator control CLI commands.

This module provides CLI commands for starting and controlling the orchestrator.
"""

from __future__ import annotations

import asyncio
import signal
from typing import Annotated
from uuid import UUID

import typer
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from forgemaster.agents.sdk_wrapper import AgentClient
from forgemaster.agents.session import AgentSessionManager
from forgemaster.context.generator import ContextGenerator
from forgemaster.orchestrator.dispatcher import Dispatcher
from forgemaster.orchestrator.state_machine import TaskStateMachine

app = typer.Typer(help="Orchestrator control commands")
console = Console()


@app.command()
def start(
    project_id: Annotated[str, typer.Argument(help="Project UUID to orchestrate")],
) -> None:
    """Start the orchestrator for a project.

    This command initializes all orchestrator components and starts the dispatch loop.
    The orchestrator will continuously poll for ready tasks, assign them to agents,
    and process results until interrupted with Ctrl+C.

    Args:
        project_id: UUID of the project to orchestrate
    """
    from forgemaster.main import get_app_context

    ctx = get_app_context()

    # Parse project UUID
    try:
        project_uuid = UUID(project_id)
    except ValueError:
        console.print(f"[red]Invalid project UUID:[/red] {project_id}")
        raise typer.Exit(code=1)

    # Display startup banner
    console.print()
    startup_panel = Panel(
        f"[bold cyan]Forgemaster Orchestrator[/bold cyan]\n\n"
        f"[bold]Project ID:[/bold] {project_uuid}\n"
        f"[bold]Max Workers:[/bold] {ctx.config.agent.max_concurrent_workers}\n"
        f"[bold]Poll Interval:[/bold] 5.0 seconds",
        title="Starting Orchestrator",
        border_style="cyan",
    )
    console.print(startup_panel)
    console.print()

    # Initialize components
    console.print("[dim]Initializing components...[/dim]")

    # Create state machine
    state_machine = TaskStateMachine()

    # Create agent client
    agent_client = AgentClient()

    # Create session manager
    session_manager = AgentSessionManager(
        config=ctx.config.agent,
        agent_client=agent_client,
    )

    # Create context generator
    context_generator = ContextGenerator()

    # Create dispatcher
    dispatcher = Dispatcher(
        config=ctx.config.agent,
        session_factory=ctx.session_factory,
        state_machine=state_machine,
        session_manager=session_manager,
        context_generator=context_generator,
        poll_interval=5.0,
    )

    console.print("[green]Components initialized[/green]")
    console.print()

    # Setup signal handler for graceful shutdown
    shutdown_event = asyncio.Event()

    def signal_handler(sig, frame):
        console.print()
        console.print("[yellow]Shutdown signal received. Stopping orchestrator...[/yellow]")
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run orchestrator loop
    async def run_orchestrator():
        try:
            # Start dispatcher
            await dispatcher.start(str(project_uuid))

            console.print("[bold green]Orchestrator running[/bold green]")
            console.print("[dim]Press Ctrl+C to stop[/dim]")
            console.print()

            # Display status table
            with Live(generate_status_table(dispatcher), refresh_per_second=1) as live:
                while not shutdown_event.is_set():
                    await asyncio.sleep(0.5)
                    live.update(generate_status_table(dispatcher))

        except Exception as e:
            console.print(f"[red]Orchestrator error:[/red] {e}")
            raise
        finally:
            # Stop dispatcher
            await dispatcher.stop()
            console.print()
            console.print("[green]Orchestrator stopped[/green]")

    try:
        asyncio.run(run_orchestrator())
    except Exception as e:
        console.print(f"[red]Fatal error:[/red] {e}")
        raise typer.Exit(code=1)


def generate_status_table(dispatcher: Dispatcher) -> Table:
    """Generate a status table for the orchestrator.

    Args:
        dispatcher: Dispatcher instance to get status from

    Returns:
        Rich Table with current orchestrator status
    """
    table = Table(title="Orchestrator Status", show_header=False)
    table.add_column("Field", style="bold cyan")
    table.add_column("Value")

    # Dispatcher status
    status_text = "[green]Running[/green]" if dispatcher.is_running else "[dim]Stopped[/dim]"
    table.add_row("Status", status_text)

    # Current task
    current_task = dispatcher._current_task_id if dispatcher._current_task_id else "-"
    table.add_row("Current Task", current_task)

    # Poll interval
    table.add_row("Poll Interval", f"{dispatcher.poll_interval}s")

    return table
