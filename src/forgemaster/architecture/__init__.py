"""Architecture pipeline for FORGEMASTER.

This module handles the ingestion, parsing, and validation of project specifications
in multiple formats (Markdown, JSON). The architecture pipeline transforms raw
specification documents into structured data models suitable for planning and execution.

The pipeline consists of:
1. Specification ingestion (SpecParser)
2. Validation and completeness checking (ValidationResult)
3. Structured representation (SpecDocument, SpecSection)
4. Architecture document generation (ArchitectureGenerator, TechnologyEvaluator)
5. Component modeling (ComponentDefinition, InterfaceDefinition, etc.)
6. Task decomposition and planning (TaskDecomposer, DependencyGraphGenerator, ParallelGroupAssigner)

Example:
    ```python
    from forgemaster.architecture import SpecParser, ArchitectureGenerator, TaskDecomposer

    parser = SpecParser()
    spec = parser.parse_file(Path("specification.md"))
    validation = parser.validate_spec(spec)
    if validation.is_valid:
        generator = ArchitectureGenerator()
        arch = generator.generate_architecture(spec)
        decomposer = TaskDecomposer()
        plan = decomposer.decompose(arch)
        print(f"Generated {plan.total_tasks} tasks")
    ```
"""

from __future__ import annotations

__all__ = [
    "SpecParser",
    "SpecDocument",
    "SpecSection",
    "ValidationResult",
    "ArchitectureGenerator",
    "TechnologyEvaluator",
    "ArchitectureDocument",
    "ComponentDefinition",
    "InterfaceDefinition",
    "DataFlowDefinition",
    "TechnologyStack",
    "DeploymentModel",
    "ArchitectureDecision",
    "TechnologyEvaluation",
    "ComparisonResult",
    "TaskDecomposer",
    "TaskDefinition",
    "ComponentTasks",
    "TaskPhase",
    "TaskPlan",
    "DependencyGraphGenerator",
    "DependencyGraph",
    "TaskNode",
    "DependencyEdge",
    "ParallelGroupAssigner",
    "ParallelGroup",
    "FileConflict",
]

from forgemaster.architecture.architect import (
    ArchitectureDecision,
    ArchitectureDocument,
    ArchitectureGenerator,
    ComparisonResult,
    ComponentDefinition,
    DataFlowDefinition,
    DeploymentModel,
    InterfaceDefinition,
    TechnologyEvaluation,
    TechnologyEvaluator,
    TechnologyStack,
)
from forgemaster.architecture.planner import (
    ComponentTasks,
    DependencyEdge,
    DependencyGraph,
    DependencyGraphGenerator,
    FileConflict,
    ParallelGroup,
    ParallelGroupAssigner,
    TaskDefinition,
    TaskDecomposer,
    TaskNode,
    TaskPhase,
    TaskPlan,
)
from forgemaster.architecture.spec_parser import (
    SpecDocument,
    SpecParser,
    SpecSection,
    ValidationResult,
)
