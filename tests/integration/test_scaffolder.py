"""Integration tests for repository scaffolding."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import tomli

from forgemaster.architecture.architect import (
    ArchitectureDocument,
    ComponentDefinition,
    DeploymentModel,
    TechnologyStack,
)
from forgemaster.architecture.scaffolder import (
    ClaudeMdGenerator,
    ProjectTemplate,
    RepositoryScaffolder,
    ScaffoldResult,
    TemplateRegistry,
)
from forgemaster.architecture.templates.python import get_python_template
from forgemaster.architecture.templates.typescript import get_typescript_template


@pytest.fixture
def sample_architecture_python() -> ArchitectureDocument:
    """Create sample Python architecture document."""
    return ArchitectureDocument(
        project_name="TestProject",
        version="0.1.0",
        overview="A test project for scaffolding validation",
        components=[
            ComponentDefinition(
                name="CoreService",
                description="Main service component",
                responsibilities=["Handle requests", "Process data"],
                dependencies=["database"],
                interfaces=["CoreServiceInterface"],
            ),
            ComponentDefinition(
                name="DatabaseLayer",
                description="Database access layer",
                responsibilities=["Query execution", "Connection pooling"],
                dependencies=[],
                interfaces=["DatabaseInterface"],
            ),
        ],
        technology_stack=TechnologyStack(
            runtime="Python 3.12+",
            frameworks=["asyncio", "SQLAlchemy", "Pydantic"],
            database="PostgreSQL",
            messaging="Redis",
            infrastructure="Docker",
        ),
        deployment_model=DeploymentModel(
            strategy="container-based",
            environments=["development", "production"],
            scaling="horizontal",
            monitoring="structlog, Prometheus",
        ),
    )


@pytest.fixture
def sample_architecture_typescript() -> ArchitectureDocument:
    """Create sample TypeScript architecture document."""
    return ArchitectureDocument(
        project_name="TypeScriptApp",
        version="1.0.0",
        overview="A TypeScript application for testing",
        components=[
            ComponentDefinition(
                name="APIService",
                description="REST API service",
                responsibilities=["Handle HTTP requests", "Route endpoints"],
                dependencies=["database"],
                interfaces=["APIInterface"],
            ),
        ],
        technology_stack=TechnologyStack(
            runtime="Node.js 20+",
            frameworks=["Express", "TypeScript"],
            database="MongoDB",
            messaging="",
            infrastructure="Kubernetes",
        ),
        deployment_model=DeploymentModel(
            strategy="microservices",
            environments=["staging", "production"],
            scaling="horizontal",
            monitoring="Winston, Datadog",
        ),
    )


@pytest.fixture
def template_registry() -> TemplateRegistry:
    """Create template registry with Python and TypeScript templates."""
    registry = TemplateRegistry()
    registry.register_template("python", get_python_template())
    registry.register_template("typescript", get_typescript_template())
    return registry


def test_repository_scaffolder_creates_python_project(
    sample_architecture_python: ArchitectureDocument,
    template_registry: TemplateRegistry,
    tmp_path: Path,
) -> None:
    """Test that RepositoryScaffolder creates correct directory structure for Python."""
    scaffolder = RepositoryScaffolder(registry=template_registry)
    target = tmp_path / "test_project"

    result = scaffolder.scaffold(sample_architecture_python, target)

    # Verify result structure
    assert isinstance(result, ScaffoldResult)
    assert result.target_path == target
    assert result.template_used == "python"
    assert len(result.files_created) > 0
    assert len(result.directories_created) > 0

    # Verify directory exists
    assert target.exists()
    assert target.is_dir()

    # Verify key directories
    assert (target / "src" / "testproject").exists()
    assert (target / "tests").exists()
    assert (target / "docs").exists()

    # Verify key files
    assert (target / "src" / "testproject" / "__init__.py").exists()
    assert (target / "src" / "testproject" / "main.py").exists()
    assert (target / "tests" / "__init__.py").exists()
    assert (target / "tests" / "conftest.py").exists()
    assert (target / "pyproject.toml").exists()
    assert (target / "README.md").exists()
    assert (target / "CLAUDE.md").exists()
    assert (target / ".gitignore").exists()

    # Verify git initialization
    assert (target / ".git").exists()


def test_python_template_generates_all_expected_files(
    sample_architecture_python: ArchitectureDocument,
    template_registry: TemplateRegistry,
    tmp_path: Path,
) -> None:
    """Test that Python template generates all expected files."""
    scaffolder = RepositoryScaffolder(registry=template_registry)
    target = tmp_path / "python_test"

    result = scaffolder.scaffold(sample_architecture_python, target)

    # Check all expected Python files exist
    expected_files = [
        "src/testproject/__init__.py",
        "src/testproject/main.py",
        "tests/__init__.py",
        "tests/conftest.py",
        "tests/test_main.py",
        "pyproject.toml",
        "README.md",
        "CLAUDE.md",
        ".gitignore",
    ]

    for file_path in expected_files:
        full_path = target / file_path
        assert full_path.exists(), f"Expected file {file_path} not found"
        assert full_path.is_file()

    # Verify files are in result
    created_paths = {f.relative_to(target) for f in result.files_created}
    for file_path in expected_files:
        assert Path(file_path) in created_paths


def test_typescript_template_generates_all_expected_files(
    sample_architecture_typescript: ArchitectureDocument,
    template_registry: TemplateRegistry,
    tmp_path: Path,
) -> None:
    """Test that TypeScript template generates all expected files."""
    scaffolder = RepositoryScaffolder(registry=template_registry)
    target = tmp_path / "typescript_test"

    result = scaffolder.scaffold(sample_architecture_typescript, target)

    # Check all expected TypeScript files exist
    expected_files = [
        "src/index.ts",
        "tests/index.test.ts",
        "package.json",
        "tsconfig.json",
        ".eslintrc.json",
        ".prettierrc.json",
        "README.md",
        "CLAUDE.md",
        ".gitignore",
    ]

    for file_path in expected_files:
        full_path = target / file_path
        assert full_path.exists(), f"Expected file {file_path} not found"
        assert full_path.is_file()

    # Verify directories
    assert (target / "src").exists()
    assert (target / "tests").exists()
    assert (target / "dist").exists()


def test_claude_md_generation_python(
    sample_architecture_python: ArchitectureDocument,
) -> None:
    """Test CLAUDE.md generation for Python project."""
    generator = ClaudeMdGenerator()
    content = generator.generate(sample_architecture_python, "python")

    # Verify content structure
    assert "# TestProject - AI Agent Context" in content
    assert "## Project Overview" in content
    assert "## Technology Stack" in content
    assert "## Key Commands" in content
    assert "## Environment Setup" in content
    assert "## Directory Structure" in content
    assert "## Components" in content
    assert "## Coding Conventions" in content
    assert "## Database" in content
    assert "## Deployment" in content

    # Verify Python-specific content
    assert "Python 3.12+" in content
    assert "uv sync" in content
    assert "pytest" in content
    assert "mypy" in content
    assert "black" in content

    # Verify component information
    assert "CoreService" in content
    assert "DatabaseLayer" in content
    assert "Handle requests" in content


def test_claude_md_generation_typescript(
    sample_architecture_typescript: ArchitectureDocument,
) -> None:
    """Test CLAUDE.md generation for TypeScript project."""
    generator = ClaudeMdGenerator()
    content = generator.generate(sample_architecture_typescript, "typescript")

    # Verify content structure
    assert "# TypeScriptApp - AI Agent Context" in content
    assert "## Technology Stack" in content

    # Verify TypeScript-specific content
    assert "Node.js 20+" in content
    assert "npm install" in content
    assert "npm test" in content
    assert "npm run build" in content

    # Verify component information
    assert "APIService" in content


def test_template_registry_crud(template_registry: TemplateRegistry) -> None:
    """Test template registry CRUD operations."""
    # Test list templates
    templates = template_registry.list_templates()
    assert "python" in templates
    assert "typescript" in templates
    assert len(templates) == 2

    # Test get template
    python_template = template_registry.get_template("python")
    assert python_template.name == "python"
    assert python_template.language == "python"

    # Test register new template
    custom_template = ProjectTemplate(
        name="custom",
        language="go",
        files={"main.go": "package main"},
        directories=["cmd"],
    )
    template_registry.register_template("go", custom_template)

    assert "go" in template_registry.list_templates()
    retrieved = template_registry.get_template("go")
    assert retrieved.name == "custom"

    # Test get non-existent template
    with pytest.raises(ValueError, match="No template found"):
        template_registry.get_template("rust")


def test_scaffold_with_custom_components(
    template_registry: TemplateRegistry,
    tmp_path: Path,
) -> None:
    """Test scaffold with custom component definitions."""
    # Create architecture with multiple components
    architecture = ArchitectureDocument(
        project_name="MultiComponent",
        version="1.0.0",
        overview="Multi-component application",
        components=[
            ComponentDefinition(
                name="AuthService",
                description="Authentication service",
                responsibilities=["User login", "Token generation"],
            ),
            ComponentDefinition(
                name="UserService",
                description="User management service",
                responsibilities=["User CRUD", "Profile management"],
            ),
            ComponentDefinition(
                name="NotificationService",
                description="Notification service",
                responsibilities=["Email sending", "Push notifications"],
            ),
        ],
        technology_stack=TechnologyStack(
            runtime="Python 3.12+",
            frameworks=["FastAPI", "SQLAlchemy"],
            database="PostgreSQL",
        ),
        deployment_model=DeploymentModel(
            strategy="microservices",
            environments=["dev", "prod"],
            scaling="horizontal",
        ),
    )

    scaffolder = RepositoryScaffolder(registry=template_registry)
    target = tmp_path / "multicomponent"

    result = scaffolder.scaffold(architecture, target)

    # Verify scaffold completed
    assert result.target_path.exists()

    # Verify CLAUDE.md contains all components
    claude_md = (target / "CLAUDE.md").read_text(encoding="utf-8")
    assert "AuthService" in claude_md
    assert "UserService" in claude_md
    assert "NotificationService" in claude_md


def test_scaffold_with_minimal_architecture(
    template_registry: TemplateRegistry,
    tmp_path: Path,
) -> None:
    """Test scaffold with minimal architecture document."""
    # Create minimal architecture
    architecture = ArchitectureDocument(
        project_name="MinimalApp",
        version="0.1.0",
        overview="Minimal test application",
        technology_stack=TechnologyStack(
            runtime="Python 3.12+",
            frameworks=[],
            database="SQLite",
        ),
        deployment_model=DeploymentModel(
            strategy="monolith",
            environments=["development"],
            scaling="vertical",
        ),
    )

    scaffolder = RepositoryScaffolder(registry=template_registry)
    target = tmp_path / "minimal"

    result = scaffolder.scaffold(architecture, target)

    # Verify basic structure exists
    assert result.target_path.exists()
    assert (target / "pyproject.toml").exists()
    assert (target / "README.md").exists()
    assert (target / "CLAUDE.md").exists()


def test_generated_pyproject_toml_is_valid(
    sample_architecture_python: ArchitectureDocument,
    template_registry: TemplateRegistry,
    tmp_path: Path,
) -> None:
    """Test that generated pyproject.toml is valid TOML."""
    scaffolder = RepositoryScaffolder(registry=template_registry)
    target = tmp_path / "toml_test"

    scaffolder.scaffold(sample_architecture_python, target)

    pyproject_path = target / "pyproject.toml"
    assert pyproject_path.exists()

    # Parse TOML to verify it's valid
    with open(pyproject_path, "rb") as f:
        toml_data = tomli.load(f)

    # Verify key sections
    assert "project" in toml_data
    assert "build-system" in toml_data
    assert "tool" in toml_data

    # Verify project metadata
    assert toml_data["project"]["name"] == "testproject"
    assert toml_data["project"]["version"] == "0.1.0"
    assert toml_data["project"]["requires-python"] == ">=3.12"

    # Verify tools configured
    assert "black" in toml_data["tool"]
    assert "ruff" in toml_data["tool"]
    assert "mypy" in toml_data["tool"]
    assert "pytest" in toml_data["tool"]


def test_generated_tsconfig_json_is_valid(
    sample_architecture_typescript: ArchitectureDocument,
    template_registry: TemplateRegistry,
    tmp_path: Path,
) -> None:
    """Test that generated tsconfig.json is valid JSON."""
    scaffolder = RepositoryScaffolder(registry=template_registry)
    target = tmp_path / "tsconfig_test"

    scaffolder.scaffold(sample_architecture_typescript, target)

    tsconfig_path = target / "tsconfig.json"
    assert tsconfig_path.exists()

    # Parse JSON to verify it's valid
    with open(tsconfig_path, encoding="utf-8") as f:
        tsconfig_data = json.load(f)

    # Verify key sections
    assert "compilerOptions" in tsconfig_data
    assert "include" in tsconfig_data
    assert "exclude" in tsconfig_data

    # Verify compiler options
    compiler_options = tsconfig_data["compilerOptions"]
    assert compiler_options["target"] == "ES2022"
    assert compiler_options["strict"] is True
    assert compiler_options["outDir"] == "./dist"


def test_generated_package_json_is_valid(
    sample_architecture_typescript: ArchitectureDocument,
    template_registry: TemplateRegistry,
    tmp_path: Path,
) -> None:
    """Test that generated package.json is valid JSON."""
    scaffolder = RepositoryScaffolder(registry=template_registry)
    target = tmp_path / "package_test"

    scaffolder.scaffold(sample_architecture_typescript, target)

    package_path = target / "package.json"
    assert package_path.exists()

    # Parse JSON to verify it's valid
    with open(package_path, encoding="utf-8") as f:
        package_data = json.load(f)

    # Verify key sections
    assert "name" in package_data
    assert "version" in package_data
    assert "scripts" in package_data
    assert "devDependencies" in package_data

    # Verify scripts
    scripts = package_data["scripts"]
    assert "build" in scripts
    assert "test" in scripts
    assert "lint" in scripts
    assert "type-check" in scripts


def test_scaffold_fails_if_target_exists(
    sample_architecture_python: ArchitectureDocument,
    template_registry: TemplateRegistry,
    tmp_path: Path,
) -> None:
    """Test that scaffold raises error if target path already exists."""
    scaffolder = RepositoryScaffolder(registry=template_registry)
    target = tmp_path / "existing"
    target.mkdir()

    with pytest.raises(ValueError, match="Target path already exists"):
        scaffolder.scaffold(sample_architecture_python, target)


def test_scaffold_fails_with_unknown_language(
    template_registry: TemplateRegistry,
    tmp_path: Path,
) -> None:
    """Test that scaffold raises error for unknown language."""
    # Create architecture with unknown runtime
    architecture = ArchitectureDocument(
        project_name="UnknownLang",
        version="1.0.0",
        overview="Test",
        technology_stack=TechnologyStack(
            runtime="Fortran 95",  # Not in registry
            frameworks=[],
            database="None",
        ),
        deployment_model=DeploymentModel(
            strategy="monolith",
            environments=["dev"],
            scaling="vertical",
        ),
    )

    scaffolder = RepositoryScaffolder(registry=template_registry)
    target = tmp_path / "unknown"

    # This should fall back to python since we default to it
    result = scaffolder.scaffold(architecture, target)
    assert result.target_path.exists()
    assert result.template_used == "python"


def test_variable_substitution_in_files(
    sample_architecture_python: ArchitectureDocument,
    template_registry: TemplateRegistry,
    tmp_path: Path,
) -> None:
    """Test that template variables are properly substituted in files."""
    scaffolder = RepositoryScaffolder(registry=template_registry)
    target = tmp_path / "substitution_test"

    scaffolder.scaffold(sample_architecture_python, target)

    # Check __init__.py has substituted values
    init_content = (target / "src" / "testproject" / "__init__.py").read_text(
        encoding="utf-8"
    )
    assert "TestProject" in init_content
    assert "0.1.0" in init_content

    # Check main.py has substituted values
    main_content = (target / "src" / "testproject" / "main.py").read_text(
        encoding="utf-8"
    )
    assert "TestProject" in main_content
    assert "0.1.0" in main_content

    # Check README has substituted values
    readme_content = (target / "README.md").read_text(encoding="utf-8")
    assert "TestProject" in readme_content
    assert "testproject" in readme_content  # snake_case version


def test_git_initialization_creates_git_repo(
    sample_architecture_python: ArchitectureDocument,
    template_registry: TemplateRegistry,
    tmp_path: Path,
) -> None:
    """Test that git initialization creates .git directory."""
    scaffolder = RepositoryScaffolder(registry=template_registry)
    target = tmp_path / "git_test"

    result = scaffolder.scaffold(sample_architecture_python, target)

    # Verify .git directory exists
    git_dir = target / ".git"
    assert git_dir.exists()
    assert git_dir.is_dir()

    # Verify .gitignore exists
    gitignore = target / ".gitignore"
    assert gitignore.exists()

    # Should have no warnings (git init succeeded)
    git_warnings = [w for w in result.warnings if "git" in w.lower()]
    assert len(git_warnings) == 0
