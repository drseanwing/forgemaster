"""Python project template for repository scaffolding.

This module provides a comprehensive Python project template with modern
tooling including uv, pytest, mypy, black, ruff, and structlog.
"""

from __future__ import annotations

from forgemaster.architecture.scaffolder import ProjectTemplate


def get_python_template() -> ProjectTemplate:
    """Get Python project template.

    Returns:
        ProjectTemplate configured for Python projects

    Example:
        >>> template = get_python_template()
        >>> print(template.language)
        'python'
    """
    # Define directory structure
    directories = [
        "src/{{PROJECT_NAME_SNAKE}}",
        "tests",
        "docs",
    ]

    # Define file templates
    files = {
        "src/{{PROJECT_NAME_SNAKE}}/__init__.py": _get_init_template(),
        "src/{{PROJECT_NAME_SNAKE}}/main.py": _get_main_template(),
        "tests/__init__.py": "",
        "tests/conftest.py": _get_conftest_template(),
        "tests/test_main.py": _get_test_main_template(),
        "README.md": _get_readme_template(),
        ".gitignore": _get_gitignore_template(),
    }

    # Define config files
    config_files = {
        "pyproject.toml": _get_pyproject_template(),
        "uv.lock": "",
    }

    return ProjectTemplate(
        name="python",
        language="python",
        files=files,
        directories=directories,
        config_files=config_files,
    )


def _get_init_template() -> str:
    """Get __init__.py template."""
    return '''"""{{PROJECT_NAME}} - {{PROJECT_DESCRIPTION}}"""

from __future__ import annotations

__version__ = "{{PROJECT_VERSION}}"

__all__ = [
    "__version__",
]
'''


def _get_main_template() -> str:
    """Get main.py template."""
    return '''"""Main entry point for {{PROJECT_NAME}}."""

from __future__ import annotations

import structlog

logger = structlog.get_logger(__name__)


def main() -> None:
    """Main entry point for the application.

    Example:
        >>> main()
    """
    logger.info("starting_application", version="{{PROJECT_VERSION}}")
    print("Hello from {{PROJECT_NAME}}!")


if __name__ == "__main__":
    main()
'''


def _get_conftest_template() -> str:
    """Get conftest.py template."""
    return '''"""Pytest configuration and fixtures for {{PROJECT_NAME}}."""

from __future__ import annotations

import pytest


@pytest.fixture
def sample_data() -> dict[str, str]:
    """Provide sample test data.

    Returns:
        Sample data dictionary
    """
    return {
        "name": "test",
        "value": "example",
    }
'''


def _get_test_main_template() -> str:
    """Get test_main.py template."""
    return '''"""Tests for main module."""

from __future__ import annotations

from {{PROJECT_NAME_SNAKE}}.main import main


def test_main() -> None:
    """Test main function runs without error."""
    main()  # Should not raise
'''


def _get_readme_template() -> str:
    """Get README.md template."""
    return '''# {{PROJECT_NAME}}

{{PROJECT_DESCRIPTION}}

## Installation

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync
```

## Usage

```bash
# Run the application
uv run python -m {{PROJECT_NAME_SNAKE}}.main

# Run tests
uv run pytest

# Type checking
uv run mypy src

# Linting
uv run ruff check src

# Format code
uv run black src
```

## Development

This project uses:
- **uv** for dependency management
- **pytest** for testing
- **mypy** for type checking
- **black** for code formatting
- **ruff** for linting
- **structlog** for logging

## Project Structure

```
src/{{PROJECT_NAME_SNAKE}}/
  __init__.py
  main.py
tests/
  __init__.py
  conftest.py
  test_main.py
docs/
pyproject.toml
README.md
CLAUDE.md
```

## License

Copyright (c) 2026. All rights reserved.
'''


def _get_gitignore_template() -> str:
    """Get .gitignore template."""
    return '''# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
.venv/
venv/
ENV/
env/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/

# Type checking
.mypy_cache/
.pytype/
.dmypy.json
dmypy.json

# Logs
*.log

# OS
.DS_Store
Thumbs.db

# uv
uv.lock
'''


def _get_pyproject_template() -> str:
    """Get pyproject.toml template."""
    return '''[project]
name = "{{PROJECT_NAME_SNAKE}}"
version = "{{PROJECT_VERSION}}"
description = "{{PROJECT_DESCRIPTION}}"
requires-python = ">=3.12"
dependencies = [
    "structlog>=24.1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "mypy>=1.8.0",
    "black>=24.1.0",
    "ruff>=0.2.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.black]
line-length = 100
target-version = ["py312"]

[tool.ruff]
line-length = 100
target-version = "py312"
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # pyflakes
    "I",   # isort
    "N",   # pep8-naming
    "UP",  # pyupgrade
    "B",   # flake8-bugbear
    "C4",  # flake8-comprehensions
    "SIM", # flake8-simplify
]
ignore = []

[tool.mypy]
python_version = "{{PYTHON_VERSION}}"
strict = true
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_any_unimported = false
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
check_untyped_defs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-v --strict-markers"
'''
