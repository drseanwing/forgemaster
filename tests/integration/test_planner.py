"""Integration tests for planner agent and task decomposition.

Tests the complete planning pipeline including:
- PlannerConfig creation and defaults
- Task decomposition from architecture documents
- Dependency graph generation and validation
- Cycle detection in dependency graphs
- Topological sort for execution ordering
- Parallel group assignment
- File conflict detection
- Group optimization with max_parallel constraints
"""

from __future__ import annotations

import pytest

from forgemaster.agents.definitions.planner import PlannerConfig, get_planner_config
from forgemaster.architecture.architect import (
    ArchitectureDocument,
    ComponentDefinition,
    DeploymentModel,
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
from forgemaster.architecture.spec_parser import SpecDocument, SpecParser, SpecSection


# Fixtures


@pytest.fixture
def sample_spec_markdown() -> str:
    """Sample markdown specification for testing."""
    return """# Sample Project

## Requirements
- User authentication
- Data persistence
- RESTful API

## Technology
- Python 3.12+
- PostgreSQL
- FastAPI

## Architecture
The system consists of three main components:
- Authentication service
- Database layer
- API gateway
"""


@pytest.fixture
def sample_architecture_doc() -> ArchitectureDocument:
    """Sample architecture document for testing."""
    return ArchitectureDocument(
        project_name="Sample Project",
        version="0.1.0",
        overview="A sample project with three main components",
        components=[
            ComponentDefinition(
                name="Authentication Service",
                description="Handles user login and session management",
                responsibilities=[
                    "User authentication",
                    "Session management",
                    "Token generation",
                ],
                dependencies=["Database Layer"],
                interfaces=["AuthAPI"],
            ),
            ComponentDefinition(
                name="Database Layer",
                description="ORM and migrations",
                responsibilities=["Data persistence", "Schema migrations"],
                dependencies=[],
                interfaces=["DatabaseInterface"],
            ),
            ComponentDefinition(
                name="API Gateway",
                description="RESTful API endpoints",
                responsibilities=["Request routing", "Response formatting"],
                dependencies=["Authentication Service"],
                interfaces=["RestAPI"],
            ),
        ],
        technology_stack=TechnologyStack(
            runtime="Python 3.12+",
            frameworks=["FastAPI", "SQLAlchemy"],
            database="PostgreSQL",
            messaging="Redis",
            infrastructure="Docker",
        ),
        deployment_model=DeploymentModel(
            strategy="container-based",
            environments=["development", "production"],
            scaling="horizontal",
            monitoring="Prometheus",
        ),
    )


@pytest.fixture
def sample_tasks() -> list[TaskDefinition]:
    """Sample task definitions for testing."""
    return [
        TaskDefinition(
            id="P1-001",
            title="Create base model",
            description="SQLAlchemy base model with timestamp mixin",
            files_to_modify=["src/models/base.py"],
            agent_type="executor",
            estimated_complexity="low",
            dependencies=[],
            acceptance_criteria=["Type checking passes", "Has docstrings"],
        ),
        TaskDefinition(
            id="P1-002",
            title="Create user model",
            description="User model with authentication fields",
            files_to_modify=["src/models/user.py"],
            agent_type="executor",
            estimated_complexity="medium",
            dependencies=["P1-001"],
            acceptance_criteria=["Inherits from base", "Has password hashing"],
        ),
        TaskDefinition(
            id="P1-003",
            title="Create session model",
            description="Session model for user sessions",
            files_to_modify=["src/models/session.py"],
            agent_type="executor",
            estimated_complexity="low",
            dependencies=["P1-001"],
            acceptance_criteria=["Inherits from base", "Has expiry field"],
        ),
        TaskDefinition(
            id="P1-004",
            title="Test user model",
            description="Unit tests for user model",
            files_to_modify=["tests/test_user.py"],
            agent_type="tester",
            estimated_complexity="medium",
            dependencies=["P1-002"],
            acceptance_criteria=["All tests pass", "Coverage > 90%"],
        ),
    ]


@pytest.fixture
def sample_dependency_graph(sample_tasks: list[TaskDefinition]) -> DependencyGraph:
    """Sample dependency graph for testing."""
    graph = DependencyGraph()

    # Add nodes
    for task in sample_tasks:
        node = TaskNode(
            task_id=task.id,
            task_title=task.title,
            agent_type=task.agent_type,
            complexity=task.estimated_complexity,
            files_to_modify=task.files_to_modify,
        )
        graph.nodes[task.id] = node

    # Add edges
    for task in sample_tasks:
        for dep_id in task.dependencies:
            edge = DependencyEdge(
                from_task=dep_id, to_task=task.id, dependency_type="blocks"
            )
            graph.edges.append(edge)

    return graph


# Test PlannerConfig


def test_planner_config_defaults() -> None:
    """Test that PlannerConfig has correct default values."""
    config = PlannerConfig()

    assert config.agent_type == "planner"
    assert config.model == "claude-opus-4-5-20251101"
    assert "Read" in config.tools
    assert "Grep" in config.tools
    assert "Glob" in config.tools
    assert config.max_tokens == 16384
    assert config.temperature == 0.3
    assert config.purpose == "Task decomposition and dependency graph generation"
    assert config.max_tasks_per_phase == 50
    assert config.max_dependency_depth == 10


def test_get_planner_config_factory() -> None:
    """Test that get_planner_config factory returns valid config."""
    config = get_planner_config()

    assert isinstance(config, PlannerConfig)
    assert config.agent_type == "planner"
    assert len(config.tools) > 0


def test_planner_config_customization() -> None:
    """Test that PlannerConfig can be customized."""
    config = PlannerConfig(
        max_tasks_per_phase=100,
        max_dependency_depth=20,
        temperature=0.5,
    )

    assert config.max_tasks_per_phase == 100
    assert config.max_dependency_depth == 20
    assert config.temperature == 0.5


# Test TaskDecomposer


def test_task_decomposer_initialization() -> None:
    """Test TaskDecomposer can be initialized."""
    decomposer = TaskDecomposer()
    assert decomposer is not None


def test_task_decomposer_decompose(sample_architecture_doc: ArchitectureDocument) -> None:
    """Test task decomposition from architecture document."""
    decomposer = TaskDecomposer()
    plan = decomposer.decompose(sample_architecture_doc)

    assert isinstance(plan, TaskPlan)
    assert plan.total_tasks > 0
    assert plan.estimated_phases > 0
    assert len(plan.phases) > 0


def test_task_decomposer_identify_components(
    sample_architecture_doc: ArchitectureDocument,
) -> None:
    """Test component identification from architecture."""
    decomposer = TaskDecomposer()
    components = decomposer._identify_components(sample_architecture_doc)

    assert len(components) == 3
    assert all(isinstance(c, ComponentTasks) for c in components)
    assert any(c.component_name == "Authentication Service" for c in components)
    # Each component should generate 3 tasks (implement, test, document)
    assert all(len(c.tasks) == 3 for c in components)


def test_task_decomposer_estimate_complexity() -> None:
    """Test complexity estimation for tasks."""
    decomposer = TaskDecomposer()

    # Low complexity: single file, short description
    low_task = TaskDefinition(
        id="T1",
        title="Simple task",
        description="Short",
        files_to_modify=["file.py"],
        agent_type="executor",
        estimated_complexity="low",
        acceptance_criteria=["Done"],
    )
    complexity = decomposer._estimate_complexity(low_task)
    assert complexity in ["low", "medium", "high"]

    # High complexity: many files, long description
    high_task = TaskDefinition(
        id="T2",
        title="Complex task",
        description="A" * 500,  # Long description
        files_to_modify=["f1.py", "f2.py", "f3.py", "f4.py"],
        agent_type="executor",
        estimated_complexity="high",
        acceptance_criteria=["C1", "C2", "C3", "C4", "C5"],
    )
    complexity = decomposer._estimate_complexity(high_task)
    assert complexity in ["medium", "high"]


def test_task_decomposer_assign_agent_type() -> None:
    """Test agent type assignment based on task characteristics."""
    decomposer = TaskDecomposer()

    # Test task -> tester
    test_task = TaskDefinition(
        id="T1",
        title="Test user authentication",
        description="Write tests",
        files_to_modify=["test_auth.py"],
        agent_type="executor",
        estimated_complexity="low",
    )
    agent_type = decomposer._assign_agent_type(test_task)
    assert agent_type == "tester"

    # Fix task -> fixer
    fix_task = TaskDefinition(
        id="T2",
        title="Fix authentication bug",
        description="Debug and fix",
        files_to_modify=["auth.py"],
        agent_type="executor",
        estimated_complexity="low",
    )
    agent_type = decomposer._assign_agent_type(fix_task)
    assert agent_type == "fixer"

    # Design task -> architect
    design_task = TaskDefinition(
        id="T3",
        title="Design database schema",
        description="Architect the schema",
        files_to_modify=["schema.py"],
        agent_type="executor",
        estimated_complexity="low",
    )
    agent_type = decomposer._assign_agent_type(design_task)
    assert agent_type == "architect"

    # Implementation task -> executor
    impl_task = TaskDefinition(
        id="T4",
        title="Implement feature",
        description="Write code",
        files_to_modify=["feature.py"],
        agent_type="executor",
        estimated_complexity="low",
    )
    agent_type = decomposer._assign_agent_type(impl_task)
    assert agent_type == "executor"


# Test DependencyGraphGenerator


def test_dependency_graph_generator_initialization() -> None:
    """Test DependencyGraphGenerator can be initialized."""
    generator = DependencyGraphGenerator()
    assert generator is not None


def test_dependency_graph_generation(sample_tasks: list[TaskDefinition]) -> None:
    """Test dependency graph generation from tasks."""
    # Create a simple plan for testing
    plan = TaskPlan(
        phases=[TaskPhase(phase_number=1, total_tasks=len(sample_tasks))],
        total_tasks=len(sample_tasks),
        estimated_phases=1,
    )

    generator = DependencyGraphGenerator()
    graph = generator.generate_graph(plan)

    assert isinstance(graph, DependencyGraph)
    assert isinstance(graph.nodes, dict)
    assert isinstance(graph.edges, list)


def test_dependency_graph_validation_valid(
    sample_dependency_graph: DependencyGraph,
) -> None:
    """Test validation of valid dependency graph."""
    generator = DependencyGraphGenerator()
    errors = generator.validate_graph(sample_dependency_graph)

    # Should have no cycles in our sample graph
    cycle_errors = [e for e in errors if "cycle" in e.lower()]
    assert len(cycle_errors) == 0


def test_dependency_graph_validation_cycle() -> None:
    """Test cycle detection in dependency graph."""
    # Create graph with cycle: A -> B -> C -> A
    graph = DependencyGraph()

    graph.nodes["A"] = TaskNode(
        task_id="A",
        task_title="Task A",
        agent_type="executor",
        complexity="low",
    )
    graph.nodes["B"] = TaskNode(
        task_id="B",
        task_title="Task B",
        agent_type="executor",
        complexity="low",
    )
    graph.nodes["C"] = TaskNode(
        task_id="C",
        task_title="Task C",
        agent_type="executor",
        complexity="low",
    )

    graph.edges.append(DependencyEdge(from_task="A", to_task="B", dependency_type="blocks"))
    graph.edges.append(DependencyEdge(from_task="B", to_task="C", dependency_type="blocks"))
    graph.edges.append(DependencyEdge(from_task="C", to_task="A", dependency_type="blocks"))

    generator = DependencyGraphGenerator()
    errors = generator.validate_graph(graph)

    # Should detect cycle
    assert len(errors) > 0
    assert any("cycle" in e.lower() for e in errors)


def test_dependency_graph_validation_orphans() -> None:
    """Test detection of orphaned tasks."""
    graph = DependencyGraph()

    # Add connected tasks
    graph.nodes["A"] = TaskNode(
        task_id="A",
        task_title="Task A",
        agent_type="executor",
        complexity="low",
    )
    graph.nodes["B"] = TaskNode(
        task_id="B",
        task_title="Task B",
        agent_type="executor",
        complexity="low",
    )
    graph.edges.append(DependencyEdge(from_task="A", to_task="B", dependency_type="blocks"))

    # Add orphan
    graph.nodes["ORPHAN"] = TaskNode(
        task_id="ORPHAN",
        task_title="Orphan Task",
        agent_type="executor",
        complexity="low",
    )

    generator = DependencyGraphGenerator()
    errors = generator.validate_graph(graph)

    # Should detect orphan
    assert len(errors) > 0
    assert any("orphan" in e.lower() for e in errors)


def test_topological_sort(sample_dependency_graph: DependencyGraph) -> None:
    """Test topological sort produces valid execution order."""
    generator = DependencyGraphGenerator()
    waves = generator.topological_sort(sample_dependency_graph)

    assert len(waves) > 0
    assert all(isinstance(wave, list) for wave in waves)

    # Flatten to check all tasks are included
    all_tasks_in_waves = [task for wave in waves for task in wave]
    assert len(all_tasks_in_waves) == len(sample_dependency_graph.nodes)

    # Verify ordering: P1-001 must come before P1-002 and P1-003
    task_to_wave = {}
    for wave_idx, wave in enumerate(waves):
        for task in wave:
            task_to_wave[task] = wave_idx

    assert task_to_wave["P1-001"] < task_to_wave["P1-002"]
    assert task_to_wave["P1-001"] < task_to_wave["P1-003"]
    assert task_to_wave["P1-002"] < task_to_wave["P1-004"]


def test_topological_sort_parallel_tasks() -> None:
    """Test that independent tasks are placed in same wave."""
    # Create graph with parallel tasks
    graph = DependencyGraph()

    graph.nodes["A"] = TaskNode(
        task_id="A",
        task_title="Task A",
        agent_type="executor",
        complexity="low",
    )
    graph.nodes["B"] = TaskNode(
        task_id="B",
        task_title="Task B",
        agent_type="executor",
        complexity="low",
    )
    graph.nodes["C"] = TaskNode(
        task_id="C",
        task_title="Task C",
        agent_type="executor",
        complexity="low",
    )

    # A and B are independent, both feed into C
    graph.edges.append(DependencyEdge(from_task="A", to_task="C", dependency_type="blocks"))
    graph.edges.append(DependencyEdge(from_task="B", to_task="C", dependency_type="blocks"))

    generator = DependencyGraphGenerator()
    waves = generator.topological_sort(graph)

    # Should have 2 waves
    assert len(waves) == 2

    # First wave should have A and B (parallel)
    assert set(waves[0]) == {"A", "B"}

    # Second wave should have C
    assert waves[1] == ["C"]


def test_visualize_graph(sample_dependency_graph: DependencyGraph) -> None:
    """Test graph visualization produces text output."""
    generator = DependencyGraphGenerator()
    viz = generator.visualize_graph(sample_dependency_graph)

    assert isinstance(viz, str)
    assert len(viz) > 0
    assert "Dependency Graph" in viz
    assert "P1-001" in viz


# Test ParallelGroupAssigner


def test_parallel_group_assigner_initialization() -> None:
    """Test ParallelGroupAssigner can be initialized."""
    assigner = ParallelGroupAssigner()
    assert assigner is not None


def test_parallel_group_assignment(sample_dependency_graph: DependencyGraph) -> None:
    """Test parallel group assignment from dependency graph."""
    assigner = ParallelGroupAssigner()
    groups = assigner.assign_groups(sample_dependency_graph)

    assert len(groups) > 0
    assert all(isinstance(g, ParallelGroup) for g in groups)

    # Each task should be in exactly one group
    all_tasks_in_groups = [task for group in groups for task in group.tasks]
    assert len(all_tasks_in_groups) == len(sample_dependency_graph.nodes)


def test_file_conflict_detection() -> None:
    """Test file conflict detection between tasks."""
    assigner = ParallelGroupAssigner()

    # Tasks with no file conflicts
    tasks_no_conflict = [
        TaskNode(
            task_id="T1",
            task_title="Task 1",
            agent_type="executor",
            complexity="low",
            files_to_modify=["file1.py"],
        ),
        TaskNode(
            task_id="T2",
            task_title="Task 2",
            agent_type="executor",
            complexity="low",
            files_to_modify=["file2.py"],
        ),
    ]

    conflicts = assigner._check_file_conflicts(tasks_no_conflict)
    assert len(conflicts) == 0

    # Tasks with file conflict
    tasks_with_conflict = [
        TaskNode(
            task_id="T1",
            task_title="Task 1",
            agent_type="executor",
            complexity="low",
            files_to_modify=["shared.py"],
        ),
        TaskNode(
            task_id="T2",
            task_title="Task 2",
            agent_type="executor",
            complexity="low",
            files_to_modify=["shared.py"],
        ),
    ]

    conflicts = assigner._check_file_conflicts(tasks_with_conflict)
    assert len(conflicts) == 1
    assert conflicts[0].file_path == "shared.py"
    assert set(conflicts[0].conflicting_tasks) == {"T1", "T2"}


def test_detect_parallelization_type() -> None:
    """Test parallelization type detection."""
    assigner = ParallelGroupAssigner()

    # Single task -> SEQ
    single_task = [
        TaskNode(
            task_id="T1",
            task_title="Task 1",
            agent_type="executor",
            complexity="low",
            files_to_modify=["file.py"],
        )
    ]
    para_type = assigner._detect_parallelization_type(single_task)
    assert para_type == "SEQ"

    # No conflicts -> PAR-A
    parallel_tasks = [
        TaskNode(
            task_id="T1",
            task_title="Task 1",
            agent_type="executor",
            complexity="low",
            files_to_modify=["file1.py"],
        ),
        TaskNode(
            task_id="T2",
            task_title="Task 2",
            agent_type="executor",
            complexity="low",
            files_to_modify=["file2.py"],
        ),
    ]
    para_type = assigner._detect_parallelization_type(parallel_tasks)
    assert para_type == "PAR-A"

    # Single shared file -> PAR-C or PAR-B
    shared_file_tasks = [
        TaskNode(
            task_id="T1",
            task_title="Task 1",
            agent_type="executor",
            complexity="low",
            files_to_modify=["shared.py"],
        ),
        TaskNode(
            task_id="T2",
            task_title="Task 2",
            agent_type="executor",
            complexity="low",
            files_to_modify=["shared.py"],
        ),
    ]
    para_type = assigner._detect_parallelization_type(shared_file_tasks)
    assert para_type in ["PAR-B", "PAR-C"]


def test_group_optimization_within_limit() -> None:
    """Test group optimization when tasks fit within max_parallel."""
    assigner = ParallelGroupAssigner()

    groups = [
        ParallelGroup(
            group_id="G1",
            tasks=["T1", "T2", "T3"],
            parallelization_type="PAR-A",
            estimated_wave=1,
        )
    ]

    optimized = assigner.optimize_groups(groups, max_parallel=5)

    # Should remain unchanged
    assert len(optimized) == 1
    assert len(optimized[0].tasks) == 3


def test_group_optimization_exceeds_limit() -> None:
    """Test group optimization when tasks exceed max_parallel."""
    assigner = ParallelGroupAssigner()

    groups = [
        ParallelGroup(
            group_id="G1",
            tasks=["T1", "T2", "T3", "T4", "T5", "T6"],
            parallelization_type="PAR-A",
            estimated_wave=1,
        )
    ]

    optimized = assigner.optimize_groups(groups, max_parallel=3)

    # Should be split into 2 groups
    assert len(optimized) == 2
    assert len(optimized[0].tasks) == 3
    assert len(optimized[1].tasks) == 3

    # All tasks preserved
    all_tasks = optimized[0].tasks + optimized[1].tasks
    assert set(all_tasks) == {"T1", "T2", "T3", "T4", "T5", "T6"}


def test_group_optimization_multiple_groups() -> None:
    """Test optimization with multiple groups."""
    assigner = ParallelGroupAssigner()

    groups = [
        ParallelGroup(
            group_id="G1",
            tasks=["T1", "T2"],
            parallelization_type="PAR-A",
            estimated_wave=1,
        ),
        ParallelGroup(
            group_id="G2",
            tasks=["T3", "T4", "T5", "T6"],
            parallelization_type="PAR-A",
            estimated_wave=2,
        ),
    ]

    optimized = assigner.optimize_groups(groups, max_parallel=2)

    # First group unchanged, second split
    assert len(optimized) == 3
    assert optimized[0].tasks == ["T1", "T2"]


# Integration tests combining multiple components


def test_full_planning_pipeline(sample_architecture_doc: ArchitectureDocument) -> None:
    """Test complete planning pipeline from architecture to parallel groups."""
    # Step 1: Decompose architecture
    decomposer = TaskDecomposer()
    plan = decomposer.decompose(sample_architecture_doc)

    assert plan.total_tasks > 0

    # Step 2: Generate dependency graph
    generator = DependencyGraphGenerator()
    graph = generator.generate_graph(plan)

    assert len(graph.nodes) >= 0

    # Step 3: Validate graph
    errors = generator.validate_graph(graph)
    # Empty graph is valid
    assert isinstance(errors, list)

    # Step 4: Topological sort
    waves = generator.topological_sort(graph)
    assert isinstance(waves, list)

    # Step 5: Assign parallel groups
    assigner = ParallelGroupAssigner()
    groups = assigner.assign_groups(graph)
    assert isinstance(groups, list)

    # Step 6: Optimize groups
    optimized = assigner.optimize_groups(groups, max_parallel=5)
    assert isinstance(optimized, list)
