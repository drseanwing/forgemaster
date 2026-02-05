"""Template loader for Jinja2-based context generation.

This module provides a TemplateLoader class that discovers and loads Jinja2 templates
from both agent-templates/ and context-templates/ directories.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from jinja2 import Environment, FileSystemLoader, select_autoescape

if TYPE_CHECKING:
    from jinja2 import Template


class TemplateLoader:
    """Loads and caches Jinja2 templates for agent context generation.

    The loader searches for templates in two directories:
    - agent-templates/: Base system prompt templates for agent types
    - context-templates/: Context injection templates (architecture, standards, etc.)

    Templates are cached after first load for performance.

    Attributes:
        template_dirs: List of Path objects pointing to template directories
        env: Jinja2 Environment with configured loaders and caching
    """

    def __init__(self, base_path: Path | None = None) -> None:
        """Initialize the template loader.

        Args:
            base_path: Base directory containing template directories.
                      Defaults to project root (3 levels up from this file).
        """
        if base_path is None:
            # Default to project root: src/forgemaster/context -> ../../../
            base_path = Path(__file__).parent.parent.parent.parent

        self.template_dirs = [
            base_path / "agent-templates",
            base_path / "context-templates",
        ]

        # Create Jinja2 environment with both template directories
        loader = FileSystemLoader([str(d) for d in self.template_dirs])
        self.env = Environment(
            loader=loader,
            autoescape=select_autoescape(),
            trim_blocks=True,
            lstrip_blocks=True,
            # Enable template caching for performance
            cache_size=50,
            auto_reload=False,  # Templates are static at runtime
        )

    def load_template(self, template_name: str) -> Template:
        """Load a template by name.

        The loader searches all configured template directories in order.
        Templates are cached after first load.

        Args:
            template_name: Name of the template file (e.g., "base.j2", "architecture.j2")

        Returns:
            Template: Loaded Jinja2 Template object

        Raises:
            jinja2.TemplateNotFound: If template doesn't exist in any directory
        """
        return self.env.get_template(template_name)

    def list_templates(self) -> list[str]:
        """List all available template names.

        Returns:
            List of template filenames found in all template directories
        """
        return sorted(self.env.list_templates())

    def template_exists(self, template_name: str) -> bool:
        """Check if a template exists.

        Args:
            template_name: Name of the template file

        Returns:
            True if template exists in any directory, False otherwise
        """
        try:
            self.env.get_template(template_name)
            return True
        except Exception:
            return False
