"""Integration tests for CLI commands.

This module tests the Typer-based CLI interface including project, task,
and orchestrator commands.
"""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest
from typer.testing import CliRunner

from forgemaster.database.models.project import ProjectStatus
from forgemaster.database.models.task import TaskStatus
from forgemaster.database.queries.project import create_project, get_project
from forgemaster.database.queries.task import create_task, get_task
from forgemaster.main import app, initialize_context


@pytest.fixture
def cli_runner():
    """Create a Typer CLI runner.

    Returns:
        CliRunner instance for invoking CLI commands
    """
    return CliRunner()


@pytest.fixture
async def cli_context(test_config, test_session_factory):
    """Initialize CLI application context for testing.

    Args:
        test_config: Test configuration fixture
        test_session_factory: Test session factory fixture

    Yields:
        Initialized application context
    """
    # Initialize the global context
    ctx = initialize_context(test_config)

    # Override session factory with test version
    ctx.session_factory = test_session_factory

    yield ctx


@pytest.mark.integration
class TestProjectCLI:
    """Integration tests for project CLI commands."""

    def test_project_create_minimal(self, cli_runner, cli_context):
        """Test creating a project with minimal arguments.

        Args:
            cli_runner: CLI runner fixture
            cli_context: CLI context fixture
        """
        result = cli_runner.invoke(app, ["project", "create", "Test Project"])

        assert result.exit_code == 0
        assert "Project created successfully" in result.stdout
        assert "Test Project" in result.stdout

    def test_project_create_with_config(self, cli_runner, cli_context):
        """Test creating a project with configuration JSON.

        Args:
            cli_runner: CLI runner fixture
            cli_context: CLI context fixture
        """
        config_json = json.dumps({"key": "value", "setting": 42})

        result = cli_runner.invoke(
            app,
            ["project", "create", "Configured Project", "--config", config_json],
        )

        assert result.exit_code == 0
        assert "Configured Project" in result.stdout

    def test_project_create_with_spec_file(self, cli_runner, cli_context, tmp_path):
        """Test creating a project with a spec file.

        Args:
            cli_runner: CLI runner fixture
            cli_context: CLI context fixture
            tmp_path: Pytest temporary directory fixture
        """
        # Create spec file
        spec_file = tmp_path / "spec.json"
        spec_data = {
            "description": "Test specification",
            "requirements": ["req1", "req2"],
        }
        spec_file.write_text(json.dumps(spec_data))

        result = cli_runner.invoke(
            app,
            ["project", "create", "Spec Project", "--spec", str(spec_file)],
        )

        assert result.exit_code == 0
        assert "Spec Project" in result.stdout

    def test_project_list_table_format(self, cli_runner, cli_context):
        """Test listing projects in table format.

        Args:
            cli_runner: CLI runner fixture
            cli_context: CLI context fixture
        """
        result = cli_runner.invoke(app, ["project", "list"])

        assert result.exit_code == 0
        # Should display table or "No projects found"
        assert "Projects" in result.stdout or "No projects found" in result.stdout

    def test_project_list_json_format(self, cli_runner, cli_context):
        """Test listing projects in JSON format.

        Args:
            cli_runner: CLI runner fixture
            cli_context: CLI context fixture
        """
        result = cli_runner.invoke(app, ["project", "list", "--format", "json"])

        assert result.exit_code == 0
        # Should be valid JSON
        try:
            projects = json.loads(result.stdout)
            assert isinstance(projects, list)
        except json.JSONDecodeError:
            pytest.fail("Output is not valid JSON")

    def test_project_list_status_filter(self, cli_runner, cli_context):
        """Test filtering projects by status.

        Args:
            cli_runner: CLI runner fixture
            cli_context: CLI context fixture
        """
        result = cli_runner.invoke(app, ["project", "list", "--status", "draft"])

        assert result.exit_code == 0

    def test_project_list_invalid_status(self, cli_runner, cli_context):
        """Test error handling for invalid status filter.

        Args:
            cli_runner: CLI runner fixture
            cli_context: CLI context fixture
        """
        result = cli_runner.invoke(app, ["project", "list", "--status", "invalid"])

        assert result.exit_code == 1
        assert "Invalid status" in result.stdout


@pytest.mark.integration
class TestTaskCLI:
    """Integration tests for task CLI commands."""

    @pytest.fixture
    async def test_project(self, test_session_factory):
        """Create a test project for task operations.

        Args:
            test_session_factory: Test session factory fixture

        Returns:
            Created test project
        """
        async with test_session_factory() as session:
            project = await create_project(
                session=session,
                name="Test Project for Tasks",
                config={},
            )
            return project

    def test_task_create_minimal(self, cli_runner, cli_context, test_project):
        """Test creating a task with minimal arguments.

        Args:
            cli_runner: CLI runner fixture
            cli_context: CLI context fixture
            test_project: Test project fixture
        """
        result = cli_runner.invoke(
            app,
            ["task", "create", str(test_project.id), "Test Task", "executor"],
        )

        assert result.exit_code == 0
        assert "Task created successfully" in result.stdout
        assert "Test Task" in result.stdout
        assert "executor" in result.stdout

    def test_task_create_with_options(self, cli_runner, cli_context, test_project):
        """Test creating a task with all options.

        Args:
            cli_runner: CLI runner fixture
            cli_context: CLI context fixture
            test_project: Test project fixture
        """
        result = cli_runner.invoke(
            app,
            [
                "task",
                "create",
                str(test_project.id),
                "Complex Task",
                "architect",
                "--description",
                "A detailed description",
                "--priority",
                "10",
                "--model",
                "sonnet",
            ],
        )

        assert result.exit_code == 0
        assert "Complex Task" in result.stdout
        assert "architect" in result.stdout

    def test_task_create_with_dependencies(
        self, cli_runner, cli_context, test_project, test_session_factory
    ):
        """Test creating a task with dependencies.

        Args:
            cli_runner: CLI runner fixture
            cli_context: CLI context fixture
            test_project: Test project fixture
            test_session_factory: Test session factory fixture
        """
        import asyncio

        # Create a dependency task first
        async def create_dep_task():
            async with test_session_factory() as session:
                dep_task = await create_task(
                    session=session,
                    project_id=test_project.id,
                    title="Dependency Task",
                    agent_type="executor",
                )
                return dep_task

        dep_task = asyncio.run(create_dep_task())

        # Create task with dependency
        result = cli_runner.invoke(
            app,
            [
                "task",
                "create",
                str(test_project.id),
                "Dependent Task",
                "executor",
                "--dependencies",
                str(dep_task.id),
            ],
        )

        assert result.exit_code == 0
        assert "Dependent Task" in result.stdout

    def test_task_list_all(self, cli_runner, cli_context):
        """Test listing all tasks.

        Args:
            cli_runner: CLI runner fixture
            cli_context: CLI context fixture
        """
        result = cli_runner.invoke(app, ["task", "list"])

        assert result.exit_code == 0
        # Should display table or "No tasks found"
        assert "Tasks" in result.stdout or "No tasks found" in result.stdout

    def test_task_list_by_project(self, cli_runner, cli_context, test_project):
        """Test listing tasks filtered by project.

        Args:
            cli_runner: CLI runner fixture
            cli_context: CLI context fixture
            test_project: Test project fixture
        """
        result = cli_runner.invoke(
            app, ["task", "list", "--project", str(test_project.id)]
        )

        assert result.exit_code == 0

    def test_task_list_by_status(self, cli_runner, cli_context):
        """Test listing tasks filtered by status.

        Args:
            cli_runner: CLI runner fixture
            cli_context: CLI context fixture
        """
        result = cli_runner.invoke(app, ["task", "list", "--status", "pending"])

        assert result.exit_code == 0

    def test_task_list_json_format(self, cli_runner, cli_context):
        """Test listing tasks in JSON format.

        Args:
            cli_runner: CLI runner fixture
            cli_context: CLI context fixture
        """
        result = cli_runner.invoke(app, ["task", "list", "--format", "json"])

        assert result.exit_code == 0
        # Should be valid JSON
        try:
            tasks = json.loads(result.stdout)
            assert isinstance(tasks, list)
        except json.JSONDecodeError:
            pytest.fail("Output is not valid JSON")

    def test_task_create_invalid_project_id(self, cli_runner, cli_context):
        """Test error handling for invalid project UUID.

        Args:
            cli_runner: CLI runner fixture
            cli_context: CLI context fixture
        """
        result = cli_runner.invoke(
            app,
            ["task", "create", "not-a-uuid", "Test Task", "executor"],
        )

        assert result.exit_code == 1
        assert "Invalid project UUID" in result.stdout

    def test_task_list_invalid_status(self, cli_runner, cli_context):
        """Test error handling for invalid status filter.

        Args:
            cli_runner: CLI runner fixture
            cli_context: CLI context fixture
        """
        result = cli_runner.invoke(app, ["task", "list", "--status", "invalid"])

        assert result.exit_code == 1
        assert "Invalid status" in result.stdout


@pytest.mark.integration
class TestOrchestratorCLI:
    """Integration tests for orchestrator CLI commands."""

    @pytest.fixture
    async def test_project_with_tasks(self, test_session_factory):
        """Create a test project with tasks for orchestrator testing.

        Args:
            test_session_factory: Test session factory fixture

        Returns:
            Created test project
        """
        async with test_session_factory() as session:
            # Create project
            project = await create_project(
                session=session,
                name="Orchestrator Test Project",
                config={},
            )

            # Create a task
            await create_task(
                session=session,
                project_id=project.id,
                title="Test Task",
                agent_type="executor",
            )

            return project

    def test_orchestrator_start_invalid_uuid(self, cli_runner, cli_context):
        """Test error handling for invalid project UUID.

        Args:
            cli_runner: CLI runner fixture
            cli_context: CLI context fixture
        """
        result = cli_runner.invoke(app, ["orchestrator", "start", "not-a-uuid"])

        assert result.exit_code == 1
        assert "Invalid project UUID" in result.stdout

    @pytest.mark.skip(reason="Requires mocking signal handlers and async event loop")
    def test_orchestrator_start_and_stop(
        self, cli_runner, cli_context, test_project_with_tasks
    ):
        """Test starting and stopping the orchestrator.

        This test is skipped because it requires complex mocking of:
        - Signal handlers (SIGINT, SIGTERM)
        - Async event loop
        - Dispatcher lifecycle

        In a real integration test environment, this would be tested with
        process-level testing or container-based testing.

        Args:
            cli_runner: CLI runner fixture
            cli_context: CLI context fixture
            test_project_with_tasks: Test project with tasks fixture
        """
        # This would require sending SIGINT to the running process
        # and verifying graceful shutdown
        pass
