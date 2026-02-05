"""Repository scaffolding and project template generation for FORGEMASTER.

This module provides tools for scaffolding new projects from architecture documents,
generating directory structures, configuration files, and development environment setup.
It supports multiple languages and frameworks through a template registry system.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import structlog
from pydantic import BaseModel, Field

from forgemaster.architecture.architect import ArchitectureDocument, TechnologyStack

logger = structlog.get_logger(__name__)


class ProjectTemplate(BaseModel):
    """Base model for project templates.

    Attributes:
        name: Template name identifier
        language: Primary programming language
        files: Mapping of relative paths to file content templates
        directories: List of directory paths to create
        config_files: Additional configuration files mapping
    """

    name: str = Field(..., description="Template name")
    language: str = Field(..., description="Primary language")
    files: dict[str, str] = Field(
        default_factory=dict, description="File path to content mapping"
    )
    directories: list[str] = Field(default_factory=list, description="Directories to create")
    config_files: dict[str, str] = Field(
        default_factory=dict, description="Config file path to content mapping"
    )


class ScaffoldResult(BaseModel):
    """Result of repository scaffolding operation.

    Attributes:
        target_path: Path where project was scaffolded
        files_created: List of files that were created
        directories_created: List of directories that were created
        template_used: Name of template that was used
        warnings: Any warnings generated during scaffolding
    """

    target_path: Path = Field(..., description="Target directory path")
    files_created: list[Path] = Field(default_factory=list, description="Created files")
    directories_created: list[Path] = Field(
        default_factory=list, description="Created directories"
    )
    template_used: str = Field(..., description="Template name used")
    warnings: list[str] = Field(default_factory=list, description="Warnings generated")


class TemplateRegistry:
    """Registry for managing project templates.

    This class maintains a collection of project templates and provides
    methods for registering, retrieving, and listing available templates.
    """

    def __init__(self) -> None:
        """Initialize the template registry."""
        self._templates: dict[str, ProjectTemplate] = {}
        self.logger = logger.bind(component="TemplateRegistry")

    def register_template(self, name: str, template: ProjectTemplate) -> None:
        """Register a new project template.

        Args:
            name: Template identifier
            template: Template instance to register

        Example:
            >>> registry = TemplateRegistry()
            >>> template = ProjectTemplate(name="python-basic", language="python")
            >>> registry.register_template("python-basic", template)
        """
        self.logger.info("registering_template", name=name, language=template.language)
        self._templates[name] = template

    def get_template(self, language: str) -> ProjectTemplate:
        """Get template for specified language.

        Args:
            language: Language identifier (python, typescript, etc.)

        Returns:
            ProjectTemplate instance for the language

        Raises:
            ValueError: If no template exists for the language

        Example:
            >>> template = registry.get_template("python")
            >>> print(template.name)
            'python-basic'
        """
        # Try exact match first
        if language in self._templates:
            return self._templates[language]

        # Try language-based lookup
        for name, template in self._templates.items():
            if template.language.lower() == language.lower():
                return template

        raise ValueError(f"No template found for language: {language}")

    def list_templates(self) -> list[str]:
        """List all registered template names.

        Returns:
            List of template names

        Example:
            >>> templates = registry.list_templates()
            >>> print(templates)
            ['python', 'typescript']
        """
        return list(self._templates.keys())


class ClaudeMdGenerator:
    """Generator for CLAUDE.md context files.

    This class generates CLAUDE.md files containing project-specific
    context and conventions for AI agent interactions.
    """

    def __init__(self) -> None:
        """Initialize the CLAUDE.md generator."""
        self.logger = logger.bind(component="ClaudeMdGenerator")

    def generate(self, architecture: ArchitectureDocument, language: str) -> str:
        """Generate CLAUDE.md content for a project.

        Args:
            architecture: Architecture document with project details
            language: Programming language (python or typescript)

        Returns:
            CLAUDE.md content as string

        Example:
            >>> generator = ClaudeMdGenerator()
            >>> content = generator.generate(arch_doc, "python")
            >>> print(content[:50])
            '# MyProject - AI Agent Context'
        """
        self.logger.info(
            "generating_claude_md",
            project=architecture.project_name,
            language=language,
        )

        sections: list[str] = []

        # Header
        sections.append(f"# {architecture.project_name} - AI Agent Context")
        sections.append("")
        sections.append("This file provides context for AI agents working on this project.")
        sections.append("")

        # Overview
        sections.append("## Project Overview")
        sections.append("")
        sections.append(architecture.overview)
        sections.append("")

        # Technology Stack
        sections.append("## Technology Stack")
        sections.append("")
        sections.append(f"**Runtime**: {architecture.technology_stack.runtime}")
        sections.append(
            f"**Frameworks**: {', '.join(architecture.technology_stack.frameworks)}"
        )
        sections.append(f"**Database**: {architecture.technology_stack.database}")
        if architecture.technology_stack.messaging:
            sections.append(f"**Messaging**: {architecture.technology_stack.messaging}")
        if architecture.technology_stack.infrastructure:
            sections.append(
                f"**Infrastructure**: {architecture.technology_stack.infrastructure}"
            )
        sections.append("")

        # Key Commands
        sections.append("## Key Commands")
        sections.append("")
        if language.lower() == "python":
            sections.extend(self._generate_python_commands())
        elif language.lower() == "typescript":
            sections.extend(self._generate_typescript_commands())
        sections.append("")

        # Environment Setup
        sections.append("## Environment Setup")
        sections.append("")
        if language.lower() == "python":
            sections.extend(self._generate_python_setup())
        elif language.lower() == "typescript":
            sections.extend(self._generate_typescript_setup())
        sections.append("")

        # Directory Structure
        sections.append("## Directory Structure")
        sections.append("")
        if language.lower() == "python":
            sections.extend(self._generate_python_structure(architecture.project_name))
        elif language.lower() == "typescript":
            sections.extend(self._generate_typescript_structure())
        sections.append("")

        # Components
        if architecture.components:
            sections.append("## Components")
            sections.append("")
            for component in architecture.components:
                sections.append(f"### {component.name}")
                sections.append(f"{component.description}")
                if component.responsibilities:
                    sections.append("")
                    sections.append("**Responsibilities:**")
                    for resp in component.responsibilities:
                        sections.append(f"- {resp}")
                sections.append("")

        # Coding Conventions
        sections.append("## Coding Conventions")
        sections.append("")
        if language.lower() == "python":
            sections.extend(self._generate_python_conventions())
        elif language.lower() == "typescript":
            sections.extend(self._generate_typescript_conventions())
        sections.append("")

        # Database reference (if applicable)
        if architecture.technology_stack.database.lower() != "none":
            sections.append("## Database")
            sections.append("")
            sections.append(f"Database: {architecture.technology_stack.database}")
            sections.append("")

        # Deployment
        sections.append("## Deployment")
        sections.append("")
        sections.append(f"**Strategy**: {architecture.deployment_model.strategy}")
        sections.append(
            f"**Environments**: {', '.join(architecture.deployment_model.environments)}"
        )
        sections.append(f"**Scaling**: {architecture.deployment_model.scaling}")
        if architecture.deployment_model.monitoring:
            sections.append(f"**Monitoring**: {architecture.deployment_model.monitoring}")
        sections.append("")

        return "\n".join(sections)

    def _generate_python_commands(self) -> list[str]:
        """Generate Python-specific key commands."""
        return [
            "```bash",
            "# Install dependencies",
            "uv sync",
            "",
            "# Run tests",
            "uv run pytest",
            "",
            "# Type checking",
            "uv run mypy src",
            "",
            "# Linting",
            "uv run ruff check src",
            "",
            "# Format code",
            "uv run black src",
            "```",
        ]

    def _generate_typescript_commands(self) -> list[str]:
        """Generate TypeScript-specific key commands."""
        return [
            "```bash",
            "# Install dependencies",
            "npm install",
            "",
            "# Build",
            "npm run build",
            "",
            "# Run tests",
            "npm test",
            "",
            "# Type checking",
            "npm run type-check",
            "",
            "# Linting",
            "npm run lint",
            "```",
        ]

    def _generate_python_setup(self) -> list[str]:
        """Generate Python environment setup instructions."""
        return [
            "1. Install Python 3.12+",
            "2. Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`",
            "3. Create virtual environment: `uv venv`",
            "4. Install dependencies: `uv sync`",
        ]

    def _generate_typescript_setup(self) -> list[str]:
        """Generate TypeScript environment setup instructions."""
        return [
            "1. Install Node.js 20+",
            "2. Install dependencies: `npm install`",
            "3. Build project: `npm run build`",
        ]

    def _generate_python_structure(self, project_name: str) -> list[str]:
        """Generate Python directory structure."""
        pkg_name = project_name.lower().replace("-", "_")
        return [
            "```",
            f"src/{pkg_name}/",
            "  __init__.py",
            "  main.py",
            "tests/",
            "  __init__.py",
            "  conftest.py",
            "docs/",
            "pyproject.toml",
            "README.md",
            "CLAUDE.md",
            "```",
        ]

    def _generate_typescript_structure(self) -> list[str]:
        """Generate TypeScript directory structure."""
        return [
            "```",
            "src/",
            "  index.ts",
            "tests/",
            "  index.test.ts",
            "dist/",
            "package.json",
            "tsconfig.json",
            "README.md",
            "CLAUDE.md",
            "```",
        ]

    def _generate_python_conventions(self) -> list[str]:
        """Generate Python coding conventions."""
        return [
            "- Use `from __future__ import annotations` in all files",
            "- All functions require type hints and docstrings",
            "- Follow Black formatting (line-length 100)",
            "- Use Ruff for linting",
            "- Use mypy strict mode for type checking",
            "- Prefer Pydantic v2 for data models",
            "- Use structlog for logging",
            "- Use asyncio for async operations",
        ]

    def _generate_typescript_conventions(self) -> list[str]:
        """Generate TypeScript coding conventions."""
        return [
            "- Use strict TypeScript mode",
            "- All functions require type annotations and JSDoc",
            "- Follow ESLint rules",
            "- Use async/await for promises",
            "- Prefer interfaces over types for object shapes",
            "- Use Prettier for formatting",
        ]


class RepositoryScaffolder:
    """Scaffolds new repositories from architecture documents.

    This class creates complete project structures including directories,
    configuration files, documentation, and initial code templates.
    """

    def __init__(self, registry: TemplateRegistry | None = None) -> None:
        """Initialize the repository scaffolder.

        Args:
            registry: Template registry to use (creates default if None)
        """
        self.registry = registry or TemplateRegistry()
        self.claude_generator = ClaudeMdGenerator()
        self.logger = logger.bind(component="RepositoryScaffolder")

    def scaffold(
        self, architecture: ArchitectureDocument, target_path: Path
    ) -> ScaffoldResult:
        """Scaffold a complete repository from architecture document.

        Args:
            architecture: Architecture document with project specifications
            target_path: Path where repository should be created

        Returns:
            ScaffoldResult with details of created files and directories

        Raises:
            ValueError: If target_path already exists
            ValueError: If no template found for language

        Example:
            >>> scaffolder = RepositoryScaffolder()
            >>> result = scaffolder.scaffold(arch_doc, Path("/tmp/myproject"))
            >>> print(f"Created {len(result.files_created)} files")
        """
        self.logger.info(
            "scaffolding_repository",
            project=architecture.project_name,
            target=str(target_path),
        )

        warnings: list[str] = []

        # Check target path
        if target_path.exists():
            raise ValueError(f"Target path already exists: {target_path}")

        # Detect language from runtime
        language = self._detect_language(architecture.technology_stack.runtime)
        self.logger.info("detected_language", language=language)

        # Get template
        try:
            template = self.registry.get_template(language)
        except ValueError as e:
            raise ValueError(f"Cannot scaffold: {e}") from e

        # Create directory structure
        directories = self._create_directory_structure(
            target_path, architecture, template, language
        )

        # Write template files
        files_created: list[Path] = []

        # Create files from template
        for rel_path, content in template.files.items():
            # Substitute variables in path
            substituted_path = self._substitute_variables(
                rel_path, architecture, language
            )
            file_path = target_path / substituted_path
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Substitute variables in content
            substituted_content = self._substitute_variables(
                content, architecture, language
            )

            file_path.write_text(substituted_content, encoding="utf-8")
            files_created.append(file_path)
            self.logger.debug("created_file", path=str(file_path))

        # Write config files
        config_files = self._write_config_files(
            target_path, architecture, template, language
        )
        files_created.extend(config_files)

        # Generate CLAUDE.md
        claude_content = self.claude_generator.generate(architecture, language)
        claude_path = target_path / "CLAUDE.md"
        claude_path.write_text(claude_content, encoding="utf-8")
        files_created.append(claude_path)

        # Initialize git
        try:
            self._initialize_git(target_path)
        except Exception as e:
            warnings.append(f"Git initialization failed: {e}")

        result = ScaffoldResult(
            target_path=target_path,
            files_created=files_created,
            directories_created=directories,
            template_used=template.name,
            warnings=warnings,
        )

        self.logger.info(
            "scaffolding_complete",
            files=len(files_created),
            directories=len(directories),
            warnings=len(warnings),
        )

        return result

    def _create_directory_structure(
        self,
        target_path: Path,
        architecture: ArchitectureDocument,
        template: ProjectTemplate,
        language: str,
    ) -> list[Path]:
        """Create directory structure for project.

        Args:
            target_path: Root path for project
            architecture: Architecture document for variable substitution
            template: Project template with directory specifications
            language: Programming language

        Returns:
            List of created directory paths
        """
        self.logger.debug("creating_directory_structure", target=str(target_path))

        directories: list[Path] = []

        # Create root directory
        target_path.mkdir(parents=True, exist_ok=True)
        directories.append(target_path)

        # Create template directories
        for dir_path in template.directories:
            # Substitute variables in directory path
            substituted_path = self._substitute_variables(
                dir_path, architecture, language
            )
            full_path = target_path / substituted_path
            full_path.mkdir(parents=True, exist_ok=True)
            directories.append(full_path)

        return directories

    def _initialize_git(self, target_path: Path) -> None:
        """Initialize git repository at target path.

        Args:
            target_path: Path to repository root

        Raises:
            subprocess.CalledProcessError: If git init fails
        """
        self.logger.info("initializing_git", path=str(target_path))

        # Initialize git repo
        subprocess.run(
            ["git", "init"],
            cwd=target_path,
            check=True,
            capture_output=True,
        )

        # Create initial .gitignore if not exists
        gitignore_path = target_path / ".gitignore"
        if not gitignore_path.exists():
            gitignore_path.write_text("", encoding="utf-8")

    def _write_config_files(
        self,
        target_path: Path,
        architecture: ArchitectureDocument,
        template: ProjectTemplate,
        language: str,
    ) -> list[Path]:
        """Write configuration files based on technology stack.

        Args:
            target_path: Root path for project
            architecture: Architecture document for variable substitution
            template: Project template with config file definitions
            language: Programming language

        Returns:
            List of created config file paths
        """
        self.logger.debug("writing_config_files", target=str(target_path))

        files_created: list[Path] = []

        # Write config files from template
        for rel_path, content in template.config_files.items():
            # Substitute variables in path
            substituted_path = self._substitute_variables(
                rel_path, architecture, language
            )
            file_path = target_path / substituted_path
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Substitute variables in content
            substituted_content = self._substitute_variables(
                content, architecture, language
            )
            file_path.write_text(substituted_content, encoding="utf-8")
            files_created.append(file_path)

        return files_created

    def _detect_language(self, runtime: str) -> str:
        """Detect programming language from runtime string.

        Args:
            runtime: Runtime environment string

        Returns:
            Language identifier (python, typescript, etc.)
        """
        runtime_lower = runtime.lower()

        if "python" in runtime_lower:
            return "python"
        elif "node" in runtime_lower or "typescript" in runtime_lower:
            return "typescript"
        elif "go" in runtime_lower:
            return "go"
        elif "rust" in runtime_lower:
            return "rust"
        else:
            # Default to python if uncertain
            return "python"

    def _substitute_variables(
        self, content: str, architecture: ArchitectureDocument, language: str
    ) -> str:
        """Substitute template variables in content.

        Args:
            content: Template content with variables
            architecture: Architecture document for variable values
            language: Programming language

        Returns:
            Content with variables substituted
        """
        substitutions = {
            "{{PROJECT_NAME}}": architecture.project_name,
            "{{PROJECT_NAME_LOWER}}": architecture.project_name.lower(),
            "{{PROJECT_NAME_SNAKE}}": architecture.project_name.lower().replace(
                "-", "_"
            ),
            "{{PROJECT_VERSION}}": architecture.version,
            "{{PROJECT_DESCRIPTION}}": architecture.overview[:200],
            "{{PYTHON_VERSION}}": self._extract_python_version(
                architecture.technology_stack.runtime
            ),
        }

        result = content
        for var, value in substitutions.items():
            result = result.replace(var, value)

        return result

    def _extract_python_version(self, runtime: str) -> str:
        """Extract Python version from runtime string.

        Args:
            runtime: Runtime string (e.g., "Python 3.12+")

        Returns:
            Version string (e.g., "3.12")
        """
        import re

        match = re.search(r"(\d+\.\d+)", runtime)
        if match:
            return match.group(1)
        return "3.12"
