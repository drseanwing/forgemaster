"""Context generation subsystem for Forgemaster.

This module uses Jinja2 templates to generate agent context files including
CLAUDE.md projections, task briefs, and environment configurations.
"""

from __future__ import annotations

from forgemaster.context.generator import ContextGenerator
from forgemaster.context.loader import TemplateLoader

__all__ = [
    "ContextGenerator",
    "TemplateLoader",
]
