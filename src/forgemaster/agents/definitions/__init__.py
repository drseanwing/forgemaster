"""Agent type definitions for Forgemaster.

This module contains agent definitions including system prompts,
tool permissions, model routing, and specialisation configurations.
"""

from __future__ import annotations

__all__ = [
    "ArchitectConfig",
    "get_architect_config",
    "FrontendReviewerConfig",
    "get_frontend_reviewer_config",
    "BackendReviewerConfig",
    "get_backend_reviewer_config",
    "DatabaseReviewerConfig",
    "get_database_reviewer_config",
    "SpecReviewerConfig",
    "get_spec_reviewer_config",
    "SecurityReviewerConfig",
    "get_security_reviewer_config",
    "AccessibilityReviewerConfig",
    "get_accessibility_reviewer_config",
    "IntegrationReviewerConfig",
    "get_integration_reviewer_config",
    "DependencyReviewerConfig",
    "get_dependency_reviewer_config",
    "DockerReviewerConfig",
    "get_docker_reviewer_config",
    "ScmReviewerConfig",
    "get_scm_reviewer_config",
    "ErrorsReviewerConfig",
    "get_errors_reviewer_config",
    "DocsReviewerConfig",
    "get_docs_reviewer_config",
]

from forgemaster.agents.definitions.architect import (
    ArchitectConfig,
    get_architect_config,
)
from forgemaster.agents.definitions.reviewer_accessibility import (
    AccessibilityReviewerConfig,
    get_accessibility_reviewer_config,
)
from forgemaster.agents.definitions.reviewer_backend import (
    BackendReviewerConfig,
    get_backend_reviewer_config,
)
from forgemaster.agents.definitions.reviewer_database import (
    DatabaseReviewerConfig,
    get_database_reviewer_config,
)
from forgemaster.agents.definitions.reviewer_dependency import (
    DependencyReviewerConfig,
    get_dependency_reviewer_config,
)
from forgemaster.agents.definitions.reviewer_docker import (
    DockerReviewerConfig,
    get_docker_reviewer_config,
)
from forgemaster.agents.definitions.reviewer_docs import (
    DocsReviewerConfig,
    get_docs_reviewer_config,
)
from forgemaster.agents.definitions.reviewer_errors import (
    ErrorsReviewerConfig,
    get_errors_reviewer_config,
)
from forgemaster.agents.definitions.reviewer_frontend import (
    FrontendReviewerConfig,
    get_frontend_reviewer_config,
)
from forgemaster.agents.definitions.reviewer_integration import (
    IntegrationReviewerConfig,
    get_integration_reviewer_config,
)
from forgemaster.agents.definitions.reviewer_scm import (
    ScmReviewerConfig,
    get_scm_reviewer_config,
)
from forgemaster.agents.definitions.reviewer_security import (
    SecurityReviewerConfig,
    get_security_reviewer_config,
)
from forgemaster.agents.definitions.reviewer_spec import (
    SpecReviewerConfig,
    get_spec_reviewer_config,
)
