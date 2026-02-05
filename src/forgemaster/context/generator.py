"""Context generator using Jinja2 templates for agent prompts.

This module provides ContextGenerator class that renders agent system prompts,
architecture context, and coding standards using Jinja2 templates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from forgemaster.context.loader import TemplateLoader


class ContextGenerator:
    """Generates agent context using Jinja2 templates.

    This class uses TemplateLoader to load and render templates for:
    - Base agent system prompts
    - Architecture context
    - Coding standards context
    - Task-specific context injection

    Attributes:
        loader: TemplateLoader instance for accessing templates
    """

    def __init__(self, base_path: Path | None = None) -> None:
        """Initialize the context generator.

        Args:
            base_path: Base directory containing template directories.
                      Passed to TemplateLoader. Defaults to project root.
        """
        self.loader = TemplateLoader(base_path=base_path)

    def generate_agent_context(
        self,
        agent_type: str,
        task: str,
        project: dict[str, Any],
        lessons: str | None = None,
        model_tier: str = "sonnet",
        working_directory: str | None = None,
    ) -> str:
        """Generate complete agent system prompt using base template.

        Args:
            agent_type: Type of agent (executor, architect, planner, etc.)
            task: Task description or instructions for the agent
            project: Project metadata dictionary containing:
                - name: Project name
                - context: General project context
                - standards: Coding standards summary
            lessons: Optional lessons learned text to inject
            model_tier: Model tier (haiku, sonnet, opus)
            working_directory: Current working directory path

        Returns:
            Rendered agent system prompt as string

        Raises:
            jinja2.TemplateNotFound: If base.j2 template doesn't exist
        """
        template = self.loader.load_template("base.j2")

        # Build template variables
        context_vars = {
            "agent_type": agent_type,
            "model_tier": model_tier,
            "task_description": self._get_agent_description(agent_type),
            "project_name": project.get("name", "Unknown Project"),
            "working_directory": working_directory or Path.cwd(),
            "task_content": task,
            "project_context": project.get("context", "No context provided."),
            "coding_standards": project.get("standards", "Follow standard best practices."),
            "lessons_learned": lessons,
        }

        return template.render(**context_vars)

    def generate_architecture_context(self, project: dict[str, Any]) -> str:
        """Generate architecture context document.

        Args:
            project: Project metadata dictionary containing:
                - name: Project name
                - tech_stack: List of technology dictionaries
                - components: List of component dictionaries
                - deployment_info: Deployment configuration
                - dev_tools: Optional list of development tools
                - integration_points: Optional list of integrations

        Returns:
            Rendered architecture context as string

        Raises:
            jinja2.TemplateNotFound: If architecture.j2 template doesn't exist
        """
        template = self.loader.load_template("architecture.j2")

        context_vars = {
            "project_name": project.get("name", "Unknown Project"),
            "tech_stack": project.get("tech_stack", []),
            "dev_tools": project.get("dev_tools", []),
            "components": project.get("components", []),
            "deployment_info": project.get("deployment_info", {}),
            "integration_points": project.get("integration_points", []),
        }

        return template.render(**context_vars)

    def generate_standards_context(self, project: dict[str, Any]) -> str:
        """Generate coding standards context document.

        Args:
            project: Project metadata dictionary containing:
                - language: Primary programming language
                - formatter: Code formatter tool name
                - linter: Linting tool name
                - type_checker: Type checking tool name
                - naming_rules: Dictionary of naming conventions
                - patterns: Dictionary of code patterns and practices

        Returns:
            Rendered coding standards context as string

        Raises:
            jinja2.TemplateNotFound: If standards.j2 template doesn't exist
        """
        template = self.loader.load_template("standards.j2")

        context_vars = {
            "language": project.get("language", "Python"),
            "formatter": project.get("formatter", "black"),
            "linter": project.get("linter", "ruff"),
            "type_checker": project.get("type_checker", "mypy"),
            "naming_rules": project.get("naming_rules", self._default_naming_rules()),
            "patterns": project.get("patterns", self._default_patterns()),
        }

        return template.render(**context_vars)

    def inject_task_context(
        self,
        base_context: str,
        task: dict[str, Any],
        dependencies_summary: list[str] | None = None,
    ) -> str:
        """Inject task-specific context into base agent context.

        This method appends task-specific information to an existing agent context,
        including task ID, title, description, files to modify, and dependency summaries.

        Args:
            base_context: Base agent system prompt to augment
            task: Task dictionary containing:
                - task_id: Unique task identifier
                - title: Task title
                - description: Detailed task description
                - files: List of file paths to modify
                - domain: Task domain for filtering relevant lessons
            dependencies_summary: Optional list of dependency completion summaries

        Returns:
            Augmented context with task-specific details appended
        """
        task_section = [
            "\n\n## Task-Specific Context\n",
            f"**Task ID**: {task.get('task_id', 'UNKNOWN')}",
            f"**Title**: {task.get('title', 'No title')}",
            f"\n### Description\n{task.get('description', 'No description provided.')}",
        ]

        # Add files to modify
        files = task.get("files", [])
        if files:
            task_section.append("\n### Files to Modify")
            for file_path in files:
                task_section.append(f"- `{file_path}`")

        # Add dependency completion summaries
        if dependencies_summary:
            task_section.append("\n### Dependency Completion Notes")
            task_section.append(
                "The following tasks were completed before this one. "
                "Their outcomes may inform your work:"
            )
            for summary in dependencies_summary:
                task_section.append(f"\n{summary}")

        # Add domain-specific note for lesson filtering
        domain = task.get("domain")
        if domain:
            task_section.append(
                f"\n**Note**: Focus on lessons learned related to **{domain}** domain."
            )

        return base_context + "\n".join(task_section)

    def _get_agent_description(self, agent_type: str) -> str:
        """Get human-readable description for agent type.

        Args:
            agent_type: Agent type identifier

        Returns:
            Description of the agent's role
        """
        descriptions = {
            "executor": "implement code changes and execute development tasks",
            "architect": "analyze architecture, debug issues, and provide technical guidance",
            "planner": "create project plans and break down complex requirements",
            "designer": "design UI/UX and implement frontend components",
            "qa-tester": "test functionality and verify quality standards",
            "writer": "write documentation and technical content",
        }
        return descriptions.get(agent_type, "perform specialized development tasks")

    def _default_naming_rules(self) -> dict[str, list[str]]:
        """Provide default naming convention rules.

        Returns:
            Dictionary of naming rules by category
        """
        return {
            "files": [
                "Use snake_case for Python module files",
                "Use kebab-case for configuration files",
                "Match file name to primary class/function",
            ],
            "classes": [
                "Use PascalCase for class names",
                "Use descriptive, noun-based names",
                "Suffix exception classes with 'Error'",
            ],
            "functions": [
                "Use snake_case for function and method names",
                "Use verb-based names for actions",
                "Prefix private methods with underscore",
            ],
            "variables": [
                "Use snake_case for variable names",
                "Use UPPER_SNAKE_CASE for constants",
                "Use descriptive names, avoid abbreviations",
            ],
        }

    def _default_patterns(self) -> dict[str, Any]:
        """Provide default code patterns and practices.

        Returns:
            Dictionary of code patterns organized by category
        """
        return {
            "module_structure": (
                "Module docstring\n"
                "Imports (stdlib, third-party, local)\n"
                "Constants\n"
                "Type aliases\n"
                "Classes\n"
                "Functions"
            ),
            "import_order": [
                "Future imports (from __future__ import annotations)",
                "Standard library imports",
                "Third-party library imports",
                "Local application imports",
            ],
            "exceptions": [
                {"name": "ValidationError", "use_case": "Input validation failures"},
                {"name": "ConfigurationError", "use_case": "Configuration issues"},
                {"name": "ResourceNotFoundError", "use_case": "Missing resources"},
            ],
            "error_handling": [
                "Use specific exception types, not bare except",
                "Include context in exception messages",
                "Log exceptions with full traceback at ERROR level",
            ],
            "logging": {
                "levels": [
                    {"name": "DEBUG", "usage": "Detailed diagnostic information"},
                    {"name": "INFO", "usage": "Normal operation events"},
                    {"name": "WARNING", "usage": "Unexpected but handled situations"},
                    {"name": "ERROR", "usage": "Errors requiring attention"},
                ],
                "example": 'logger.info("Processing task", extra={"task_id": task.id})',
            },
            "type_hints": {
                "requirements": [
                    "All function parameters must have type hints",
                    "All function return values must have type hints",
                    "Use None for functions with no return value",
                ],
                "examples": (
                    "def process(data: dict[str, Any]) -> ProcessResult:\n"
                    "    ..."
                ),
            },
            "docstring_format": (
                '"""Brief description.\n\n'
                "Detailed explanation.\n\n"
                "Args:\n"
                "    param: Description\n\n"
                "Returns:\n"
                "    Description\n"
                '"""'
            ),
            "documentation": [
                "All public modules require module docstrings",
                "All public classes require class docstrings",
                "All public functions require function docstrings",
            ],
            "testing": {
                "organization": [
                    "Tests mirror src/ directory structure",
                    "One test file per source file",
                    "Group tests by functionality",
                ],
                "naming": [
                    "Test files: test_<module>.py",
                    "Test classes: Test<ClassName>",
                    "Test functions: test_<behavior>_<condition>",
                ],
                "min_coverage": 80,
            },
            "performance": [
                "Avoid premature optimization",
                "Profile before optimizing hot paths",
                "Use appropriate data structures",
            ],
            "security": [
                "Never log sensitive information",
                "Validate all external inputs",
                "Use parameterized queries for SQL",
            ],
        }
