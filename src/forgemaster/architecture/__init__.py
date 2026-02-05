"""Architecture pipeline for FORGEMASTER.

This module handles the ingestion, parsing, and validation of project specifications
in multiple formats (Markdown, JSON). The architecture pipeline transforms raw
specification documents into structured data models suitable for planning and execution.

The pipeline consists of:
1. Specification ingestion (SpecParser)
2. Validation and completeness checking (ValidationResult)
3. Structured representation (SpecDocument, SpecSection)
4. Interview orchestration (InterviewOrchestrator, InterviewSession)
5. Architecture document generation (ArchitectureGenerator, TechnologyEvaluator)
6. Component modeling (ComponentDefinition, InterfaceDefinition, etc.)
7. Task decomposition and planning (TaskDecomposer, DependencyGraphGenerator, ParallelGroupAssigner)

Example:
    ```python
    from forgemaster.architecture import (
        SpecParser,
        InterviewOrchestrator,
        InterviewSession,
        ArchitectureGenerator,
        TaskDecomposer,
    )

    parser = SpecParser()
    spec = parser.parse_file(Path("specification.md"))
    validation = parser.validate_spec(spec)
    if validation.is_valid:
        # Conduct interview for clarification
        orchestrator = InterviewOrchestrator()
        session = InterviewSession(spec, orchestrator)
        questions = session.start_interview()
        # ... process responses and finalize
        clarified = session.finalize_interview()

        # Generate architecture
        generator = ArchitectureGenerator()
        arch = generator.generate_architecture(clarified.original_spec)
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
    "InterviewOrchestrator",
    "InterviewSession",
    "InterviewQuestion",
    "InterviewResponse",
    "InterviewRoundResult",
    "ClarifiedSpec",
    "QuestionCategory",
    "QuestionImportance",
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
    "RepositoryScaffolder",
    "ScaffoldResult",
    "ProjectTemplate",
    "TemplateRegistry",
    "ClaudeMdGenerator",
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
from forgemaster.architecture.interviewer import (
    ClarifiedSpec,
    InterviewOrchestrator,
    InterviewQuestion,
    InterviewResponse,
    InterviewRoundResult,
    InterviewSession,
    QuestionCategory,
    QuestionImportance,
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
from forgemaster.architecture.scaffolder import (
    ClaudeMdGenerator,
    ProjectTemplate,
    RepositoryScaffolder,
    ScaffoldResult,
    TemplateRegistry,
)
from forgemaster.architecture.spec_parser import (
    SpecDocument,
    SpecParser,
    SpecSection,
    ValidationResult,
)
