"""Project templates for repository scaffolding.

This package contains template definitions for different programming languages
and frameworks, used by the RepositoryScaffolder to generate new projects.
"""

from __future__ import annotations

from forgemaster.architecture.templates.python import get_python_template
from forgemaster.architecture.templates.typescript import get_typescript_template

__all__ = [
    "get_python_template",
    "get_typescript_template",
]
