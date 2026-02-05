"""Unit tests for context generation system.

Tests TemplateLoader, ContextGenerator, and template rendering functionality.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from jinja2 import TemplateNotFound

from forgemaster.context.generator import ContextGenerator
from forgemaster.context.loader import TemplateLoader


@pytest.fixture
def temp_template_dir(tmp_path: Path) -> Path:
    """Create temporary template directories for testing.

    Args:
        tmp_path: Pytest temporary directory fixture

    Returns:
        Base path containing agent-templates/ and context-templates/
    """
    # Create template directories
    agent_dir = tmp_path / "agent-templates"
    context_dir = tmp_path / "context-templates"
    agent_dir.mkdir()
    context_dir.mkdir()

    # Create minimal test templates
    base_template = agent_dir / "base.j2"
    base_template.write_text(
        "Agent: {{ agent_type }}\n"
        "Task: {{ task_content }}\n"
        "Project: {{ project_name }}\n"
    )

    arch_template = context_dir / "architecture.j2"
    arch_template.write_text(
        "Architecture for {{ project_name }}\n"
        "{% for tech in tech_stack %}{{ tech.name }}\n{% endfor %}"
    )

    standards_template = context_dir / "standards.j2"
    standards_template.write_text(
        "Standards for {{ language }}\n"
        "Formatter: {{ formatter }}\n"
    )

    return tmp_path


class TestTemplateLoader:
    """Test suite for TemplateLoader class."""

    def test_template_loader_initialization(self, temp_template_dir: Path) -> None:
        """Test TemplateLoader initializes with correct directories."""
        loader = TemplateLoader(base_path=temp_template_dir)

        assert len(loader.template_dirs) == 2
        assert loader.template_dirs[0] == temp_template_dir / "agent-templates"
        assert loader.template_dirs[1] == temp_template_dir / "context-templates"

    def test_load_template_from_agent_dir(self, temp_template_dir: Path) -> None:
        """Test loading template from agent-templates directory."""
        loader = TemplateLoader(base_path=temp_template_dir)

        template = loader.load_template("base.j2")
        result = template.render(
            agent_type="executor",
            task_content="test task",
            project_name="test-project",
        )

        assert "Agent: executor" in result
        assert "Task: test task" in result
        assert "Project: test-project" in result

    def test_load_template_from_context_dir(self, temp_template_dir: Path) -> None:
        """Test loading template from context-templates directory."""
        loader = TemplateLoader(base_path=temp_template_dir)

        template = loader.load_template("architecture.j2")
        result = template.render(
            project_name="test-project",
            tech_stack=[{"name": "Python"}, {"name": "PostgreSQL"}],
        )

        assert "Architecture for test-project" in result
        assert "Python" in result
        assert "PostgreSQL" in result

    def test_load_template_not_found(self, temp_template_dir: Path) -> None:
        """Test TemplateNotFound exception for missing template."""
        loader = TemplateLoader(base_path=temp_template_dir)

        with pytest.raises(TemplateNotFound):
            loader.load_template("nonexistent.j2")

    def test_list_templates(self, temp_template_dir: Path) -> None:
        """Test listing all available templates."""
        loader = TemplateLoader(base_path=temp_template_dir)

        templates = loader.list_templates()

        assert "base.j2" in templates
        assert "architecture.j2" in templates
        assert "standards.j2" in templates

    def test_template_exists(self, temp_template_dir: Path) -> None:
        """Test checking template existence."""
        loader = TemplateLoader(base_path=temp_template_dir)

        assert loader.template_exists("base.j2")
        assert loader.template_exists("architecture.j2")
        assert not loader.template_exists("nonexistent.j2")


class TestContextGenerator:
    """Test suite for ContextGenerator class."""

    def test_generator_initialization(self, temp_template_dir: Path) -> None:
        """Test ContextGenerator initializes with TemplateLoader."""
        generator = ContextGenerator(base_path=temp_template_dir)

        assert isinstance(generator.loader, TemplateLoader)

    def test_generate_agent_context(self, temp_template_dir: Path) -> None:
        """Test generating complete agent system prompt."""
        generator = ContextGenerator(base_path=temp_template_dir)

        project = {
            "name": "test-project",
            "context": "Test project context",
            "standards": "Follow PEP 8",
        }

        result = generator.generate_agent_context(
            agent_type="executor",
            task="Implement feature X",
            project=project,
            model_tier="sonnet",
        )

        assert "Agent: executor" in result
        assert "Task: Implement feature X" in result
        assert "Project: test-project" in result

    def test_generate_agent_context_with_lessons(
        self, temp_template_dir: Path
    ) -> None:
        """Test generating agent context with lessons learned."""
        generator = ContextGenerator(base_path=temp_template_dir)

        project = {"name": "test-project"}

        result = generator.generate_agent_context(
            agent_type="executor",
            task="Test task",
            project=project,
            lessons="Use async/await for I/O operations",
        )

        assert "Agent: executor" in result

    def test_generate_architecture_context(self, temp_template_dir: Path) -> None:
        """Test generating architecture context document."""
        generator = ContextGenerator(base_path=temp_template_dir)

        project = {
            "name": "test-project",
            "tech_stack": [
                {"name": "Python", "version": "3.12", "purpose": "Backend"},
                {"name": "PostgreSQL", "version": "16", "purpose": "Database"},
            ],
            "components": [],
            "deployment_info": {},
        }

        result = generator.generate_architecture_context(project)

        assert "Architecture for test-project" in result
        assert "Python" in result
        assert "PostgreSQL" in result

    def test_generate_standards_context(self, temp_template_dir: Path) -> None:
        """Test generating coding standards context document."""
        generator = ContextGenerator(base_path=temp_template_dir)

        project = {
            "language": "Python",
            "formatter": "black",
            "linter": "ruff",
            "type_checker": "mypy",
        }

        result = generator.generate_standards_context(project)

        assert "Standards for Python" in result
        assert "Formatter: black" in result

    def test_generate_standards_with_defaults(self, temp_template_dir: Path) -> None:
        """Test generating standards context with default values."""
        generator = ContextGenerator(base_path=temp_template_dir)

        # Empty project dict should use defaults
        result = generator.generate_standards_context({})

        assert "Standards for Python" in result
        assert "Formatter: black" in result

    def test_inject_task_context(self, temp_template_dir: Path) -> None:
        """Test injecting task-specific context into base prompt."""
        generator = ContextGenerator(base_path=temp_template_dir)

        base_context = "Base agent context\n"
        task = {
            "task_id": "FM-001",
            "title": "Test Task",
            "description": "Test task description",
            "files": ["src/module.py", "tests/test_module.py"],
            "domain": "backend",
        }
        dependencies = [
            "Task FM-000 completed: Database schema created",
        ]

        result = generator.inject_task_context(base_context, task, dependencies)

        assert "Base agent context" in result
        assert "FM-001" in result
        assert "Test Task" in result
        assert "src/module.py" in result
        assert "tests/test_module.py" in result
        assert "Database schema created" in result
        assert "backend" in result

    def test_inject_task_context_minimal(self, temp_template_dir: Path) -> None:
        """Test injecting minimal task context without optional fields."""
        generator = ContextGenerator(base_path=temp_template_dir)

        base_context = "Base agent context\n"
        task = {
            "task_id": "FM-002",
            "title": "Minimal Task",
        }

        result = generator.inject_task_context(base_context, task)

        assert "Base agent context" in result
        assert "FM-002" in result
        assert "Minimal Task" in result

    def test_missing_template_raises_error(self, temp_template_dir: Path) -> None:
        """Test that missing templates raise appropriate errors."""
        generator = ContextGenerator(base_path=temp_template_dir)

        # Remove a template to simulate missing file
        (temp_template_dir / "agent-templates" / "base.j2").unlink()

        with pytest.raises(TemplateNotFound):
            generator.generate_agent_context(
                agent_type="executor",
                task="Test",
                project={"name": "test"},
            )


class TestPrivateMethods:
    """Test suite for ContextGenerator private helper methods."""

    def test_get_agent_description(self, temp_template_dir: Path) -> None:
        """Test agent type description lookup."""
        generator = ContextGenerator(base_path=temp_template_dir)

        assert "implement code" in generator._get_agent_description("executor")
        assert "analyze architecture" in generator._get_agent_description("architect")
        assert "create project plans" in generator._get_agent_description("planner")
        assert "specialized" in generator._get_agent_description("unknown_agent")

    def test_default_naming_rules(self, temp_template_dir: Path) -> None:
        """Test default naming rules structure."""
        generator = ContextGenerator(base_path=temp_template_dir)

        rules = generator._default_naming_rules()

        assert "files" in rules
        assert "classes" in rules
        assert "functions" in rules
        assert "variables" in rules
        assert isinstance(rules["files"], list)
        assert len(rules["files"]) > 0

    def test_default_patterns(self, temp_template_dir: Path) -> None:
        """Test default patterns structure."""
        generator = ContextGenerator(base_path=temp_template_dir)

        patterns = generator._default_patterns()

        assert "module_structure" in patterns
        assert "import_order" in patterns
        assert "exceptions" in patterns
        assert "error_handling" in patterns
        assert "logging" in patterns
        assert "type_hints" in patterns
        assert "docstring_format" in patterns
        assert "documentation" in patterns
        assert "testing" in patterns
        assert isinstance(patterns["testing"], dict)
        assert patterns["testing"]["min_coverage"] == 80
