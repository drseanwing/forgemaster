"""Task decomposition and planning for Forgemaster.

This module provides core planning functionality including:
- Task decomposition from architecture documents
- Complexity estimation for work units
- Agent type assignment based on task characteristics
- Dependency graph generation and validation
- Parallel group assignment for concurrent execution

The planner transforms high-level architecture specifications into
executable task plans with precise ordering and parallelization strategies.
"""

from __future__ import annotations

from typing import Literal

import structlog
from pydantic import BaseModel, Field

from forgemaster.architecture.architect import ArchitectureDocument

logger = structlog.get_logger(__name__)


# Type aliases
ComplexityLevel = Literal["low", "medium", "high"]
AgentType = Literal["executor", "architect", "tester", "fixer"]
DependencyType = Literal["blocks", "informs", "tests"]
ParallelizationType = Literal["SEQ", "PAR-A", "PAR-B", "PAR-C", "PAR-D"]


class TaskDefinition(BaseModel):
    """Definition of a single atomic task.

    Attributes:
        id: Unique task identifier (format: P{phase}-{number}).
        title: Short description of the task.
        description: Detailed task description and instructions.
        files_to_modify: List of file paths to create or modify.
        agent_type: Type of agent required for execution.
        estimated_complexity: Relative effort estimation.
        dependencies: List of task IDs this task depends on.
        acceptance_criteria: Measurable completion criteria.
    """

    id: str
    title: str
    description: str
    files_to_modify: list[str] = Field(default_factory=list)
    agent_type: AgentType
    estimated_complexity: ComplexityLevel
    dependencies: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)


class ComponentTasks(BaseModel):
    """Tasks grouped by architectural component.

    Attributes:
        component_name: Name of the component or feature.
        tasks: List of tasks for this component.
    """

    component_name: str
    tasks: list[TaskDefinition] = Field(default_factory=list)


class FileConflict(BaseModel):
    """Represents a file modification conflict between tasks.

    Attributes:
        file_path: Path to the conflicting file.
        conflicting_tasks: List of task IDs that modify this file.
        resolution_strategy: How to resolve the conflict.
    """

    file_path: str
    conflicting_tasks: list[str] = Field(default_factory=list)
    resolution_strategy: str = "sequential"


class ParallelGroup(BaseModel):
    """Group of tasks that can execute in parallel.

    Attributes:
        group_id: Unique group identifier.
        tasks: List of task IDs in this group.
        parallelization_type: Type of parallelization strategy.
        shared_files: Files shared across tasks (coordination points).
        estimated_wave: Execution wave number within phase.
    """

    group_id: str
    tasks: list[str] = Field(default_factory=list)
    parallelization_type: ParallelizationType
    shared_files: list[str] = Field(default_factory=list)
    estimated_wave: int = 1


class TaskPhase(BaseModel):
    """A phase in the execution plan containing parallel groups.

    Attributes:
        phase_number: Sequential phase identifier.
        groups: Parallel execution groups in this phase.
        total_tasks: Total number of tasks in this phase.
    """

    phase_number: int
    groups: list[ParallelGroup] = Field(default_factory=list)
    total_tasks: int = 0


class TaskPlan(BaseModel):
    """Complete task execution plan.

    Attributes:
        phases: Sequential phases of execution.
        total_tasks: Total number of tasks across all phases.
        estimated_phases: Total number of phases.
        tasks: Mapping from task_id to TaskDefinition for lookup.
    """

    phases: list[TaskPhase] = Field(default_factory=list)
    total_tasks: int = 0
    estimated_phases: int = 0
    tasks: dict[str, TaskDefinition] = Field(default_factory=dict)


class TaskNode(BaseModel):
    """Node in the dependency graph representing a task.

    Attributes:
        task_id: Unique task identifier.
        task_title: Task title for display.
        agent_type: Agent type assigned to this task.
        complexity: Estimated complexity level.
        parallelization_group: Parallel group identifier if assigned.
        files_to_modify: Files this task will modify.
    """

    task_id: str
    task_title: str
    agent_type: AgentType
    complexity: ComplexityLevel
    parallelization_group: str | None = None
    files_to_modify: list[str] = Field(default_factory=list)


class DependencyEdge(BaseModel):
    """Edge in the dependency graph representing a dependency relationship.

    Attributes:
        from_task: Source task ID.
        to_task: Target task ID (depends on from_task).
        dependency_type: Nature of the dependency.
    """

    from_task: str
    to_task: str
    dependency_type: DependencyType


class DependencyGraph(BaseModel):
    """Directed acyclic graph of task dependencies.

    Attributes:
        nodes: Mapping from task_id to TaskNode.
        edges: List of dependency edges.
    """

    nodes: dict[str, TaskNode] = Field(default_factory=dict)
    edges: list[DependencyEdge] = Field(default_factory=list)


class TaskDecomposer:
    """Decomposes architecture documents into executable tasks.

    This class analyzes architecture specifications and generates
    atomic task definitions with proper scoping and agent assignment.
    """

    def __init__(self) -> None:
        """Initialize the task decomposer."""
        self.logger = logger.bind(component="TaskDecomposer")

    def decompose(self, architecture: ArchitectureDocument) -> TaskPlan:
        """Decompose architecture document into a structured task plan.

        Args:
            architecture: High-level architecture specification.

        Returns:
            TaskPlan with phases, groups, and individual tasks.

        Example:
            ```python
            decomposer = TaskDecomposer()
            plan = decomposer.decompose(architecture_doc)
            print(f"Generated {plan.total_tasks} tasks in {plan.estimated_phases} phases")
            ```
        """
        self.logger.info("decomposing_architecture", project=architecture.project_name)

        # Identify components from architecture
        components = self._identify_components(architecture)

        # Generate task plan
        all_tasks: list[TaskDefinition] = []
        task_registry: dict[str, TaskDefinition] = {}
        for component in components:
            all_tasks.extend(component.tasks)
            for task in component.tasks:
                task_registry[task.id] = task

        # Create initial single-phase plan
        # (Phases will be properly generated by dependency graph analysis)
        plan = TaskPlan(
            phases=[
                TaskPhase(
                    phase_number=1,
                    groups=[],  # Will be populated by parallel group assignment
                    total_tasks=len(all_tasks),
                )
            ],
            total_tasks=len(all_tasks),
            estimated_phases=1,
            tasks=task_registry,
        )

        self.logger.info(
            "decomposition_complete",
            total_tasks=len(all_tasks),
            components_count=len(components),
        )

        return plan

    def _identify_components(
        self, architecture: ArchitectureDocument
    ) -> list[ComponentTasks]:
        """Identify architectural components and generate tasks for each.

        Args:
            architecture: Architecture document to analyze.

        Returns:
            List of ComponentTasks with generated task definitions.
        """
        self.logger.debug("identifying_components")

        components: list[ComponentTasks] = []

        # Parse components from architecture document
        # Each ComponentDefinition generates multiple tasks
        for idx, component in enumerate(architecture.components, start=1):
            component_tasks: list[TaskDefinition] = []

            # Generate implementation task
            impl_task = TaskDefinition(
                id=f"P1-{idx * 3 - 2:03d}",
                title=f"Implement {component.name}",
                description=component.description,
                files_to_modify=[],
                agent_type="executor",
                estimated_complexity=self._estimate_complexity_from_responsibilities(
                    component.responsibilities
                ),
                dependencies=[],
                acceptance_criteria=[
                    "Component implements specified functionality",
                    "All responsibilities addressed",
                ],
            )
            component_tasks.append(impl_task)

            # Generate test task
            test_task = TaskDefinition(
                id=f"P1-{idx * 3 - 1:03d}",
                title=f"Test {component.name}",
                description=f"Write tests for {component.name}",
                files_to_modify=[],
                agent_type="tester",
                estimated_complexity="medium",
                dependencies=[impl_task.id],
                acceptance_criteria=[
                    "All tests pass",
                    "Coverage > 80%",
                ],
            )
            component_tasks.append(test_task)

            # Generate documentation task
            doc_task = TaskDefinition(
                id=f"P1-{idx * 3:03d}",
                title=f"Document {component.name}",
                description=f"Add documentation for {component.name}",
                files_to_modify=[],
                agent_type="executor",
                estimated_complexity="low",
                dependencies=[impl_task.id],
                acceptance_criteria=[
                    "Docstrings added",
                    "Usage examples included",
                ],
            )
            component_tasks.append(doc_task)

            components.append(
                ComponentTasks(component_name=component.name, tasks=component_tasks)
            )

        self.logger.debug("components_identified", count=len(components))

        return components

    def _estimate_complexity(self, task: TaskDefinition) -> ComplexityLevel:
        """Estimate relative complexity of a task.

        Args:
            task: Task definition to analyze.

        Returns:
            Complexity level (low/medium/high).

        Example:
            ```python
            complexity = decomposer._estimate_complexity(task_def)
            print(f"Task complexity: {complexity}")
            ```
        """
        # Heuristics for complexity estimation
        file_count = len(task.files_to_modify)
        description_length = len(task.description)
        criteria_count = len(task.acceptance_criteria)

        # Score based on multiple factors
        score = 0

        # File count contributes to complexity
        if file_count == 1:
            score += 1
        elif file_count <= 3:
            score += 2
        else:
            score += 3

        # Description length suggests scope
        if description_length < 100:
            score += 1
        elif description_length < 300:
            score += 2
        else:
            score += 3

        # More acceptance criteria = more complex validation
        if criteria_count <= 2:
            score += 1
        elif criteria_count <= 4:
            score += 2
        else:
            score += 3

        # Map score to complexity level
        if score <= 4:
            return "low"
        elif score <= 7:
            return "medium"
        else:
            return "high"

    def _estimate_complexity_from_content(self, content: str) -> ComplexityLevel:
        """Estimate complexity from content length and keywords.

        Args:
            content: Content to analyze.

        Returns:
            Complexity level.
        """
        length = len(content)
        if length < 100:
            return "low"
        elif length < 300:
            return "medium"
        else:
            return "high"

    def _estimate_complexity_from_responsibilities(
        self, responsibilities: list[str]
    ) -> ComplexityLevel:
        """Estimate complexity from number of responsibilities.

        Args:
            responsibilities: List of component responsibilities.

        Returns:
            Complexity level.
        """
        count = len(responsibilities)
        if count <= 2:
            return "low"
        elif count <= 4:
            return "medium"
        else:
            return "high"

    def _assign_agent_type(self, task: TaskDefinition) -> AgentType:
        """Assign appropriate agent type based on task characteristics.

        Args:
            task: Task definition to analyze.

        Returns:
            Agent type identifier.

        Example:
            ```python
            agent_type = decomposer._assign_agent_type(task_def)
            print(f"Task requires {agent_type} agent")
            ```
        """
        # Heuristics for agent assignment based on title/description keywords
        title_lower = task.title.lower()
        desc_lower = task.description.lower()

        # Test-related tasks go to tester
        if "test" in title_lower or "test" in desc_lower:
            return "tester"

        # Fix/debug tasks go to fixer
        if any(keyword in title_lower for keyword in ["fix", "debug", "repair"]):
            return "fixer"

        # Design/architecture tasks go to architect
        if any(
            keyword in title_lower for keyword in ["design", "architect", "analyze", "plan"]
        ):
            return "architect"

        # Default to executor for implementation tasks
        return "executor"


class DependencyGraphGenerator:
    """Generates and validates task dependency graphs.

    This class builds directed acyclic graphs (DAGs) of task dependencies,
    validates graph properties, and produces execution orderings.
    """

    def __init__(self) -> None:
        """Initialize the dependency graph generator."""
        self.logger = logger.bind(component="DependencyGraphGenerator")

    def generate_graph(self, plan: TaskPlan) -> DependencyGraph:
        """Generate dependency graph from task plan.

        Args:
            plan: Task plan containing tasks with dependencies.

        Returns:
            DependencyGraph with nodes and edges.

        Example:
            ```python
            generator = DependencyGraphGenerator()
            graph = generator.generate_graph(task_plan)
            print(f"Graph has {len(graph.nodes)} nodes")
            ```
        """
        self.logger.info("generating_dependency_graph", phases=len(plan.phases))

        graph = DependencyGraph()

        # Use the task registry from the plan
        task_registry = plan.tasks

        # Build nodes from all tasks in the registry
        for task_id, task in task_registry.items():
            node = TaskNode(
                task_id=task.id,
                task_title=task.title,
                agent_type=task.agent_type,
                complexity=task.estimated_complexity,
                files_to_modify=task.files_to_modify,
            )
            graph.nodes[task.id] = node

        # Build edges from task dependencies
        for task_id, task in task_registry.items():
            for dep_id in task.dependencies:
                edge = DependencyEdge(
                    from_task=dep_id, to_task=task.id, dependency_type="blocks"
                )
                graph.edges.append(edge)

        # Add phase-based dependencies
        # Tasks in phase N+1 depend on completion of phase N
        for phase_idx, phase in enumerate(plan.phases):
            if phase_idx + 1 < len(plan.phases):
                # Get task IDs from current phase
                current_phase_tasks: set[str] = set()
                for group in phase.groups:
                    current_phase_tasks.update(group.tasks)

                # Get task IDs from next phase
                next_phase = plan.phases[phase_idx + 1]
                next_phase_tasks: set[str] = set()
                for group in next_phase.groups:
                    next_phase_tasks.update(group.tasks)

                # Create edges from current phase to next phase
                # (Only if not already connected via explicit dependencies)
                for current_task_id in current_phase_tasks:
                    for next_task_id in next_phase_tasks:
                        # Check if edge already exists
                        edge_exists = any(
                            e.from_task == current_task_id and e.to_task == next_task_id
                            for e in graph.edges
                        )
                        if not edge_exists:
                            edge = DependencyEdge(
                                from_task=current_task_id,
                                to_task=next_task_id,
                                dependency_type="blocks",
                            )
                            graph.edges.append(edge)

        # Add edges for file conflicts within parallel groups
        for phase in plan.phases:
            for group in phase.groups:
                # Get task nodes for this group
                group_tasks = [
                    graph.nodes[task_id]
                    for task_id in group.tasks
                    if task_id in graph.nodes
                ]

                # Check for file conflicts
                file_to_tasks: dict[str, list[str]] = {}
                for task_node in group_tasks:
                    for file_path in task_node.files_to_modify:
                        if file_path not in file_to_tasks:
                            file_to_tasks[file_path] = []
                        file_to_tasks[file_path].append(task_node.task_id)

                # Create sequential dependencies for conflicting tasks
                for file_path, conflicting_task_ids in file_to_tasks.items():
                    if len(conflicting_task_ids) > 1:
                        # Make tasks sequential
                        for i in range(len(conflicting_task_ids) - 1):
                            edge = DependencyEdge(
                                from_task=conflicting_task_ids[i],
                                to_task=conflicting_task_ids[i + 1],
                                dependency_type="blocks",
                            )
                            graph.edges.append(edge)

        self.logger.info(
            "graph_generated", nodes=len(graph.nodes), edges=len(graph.edges)
        )

        return graph

    def validate_graph(self, graph: DependencyGraph) -> list[str]:
        """Validate dependency graph for cycles and orphans.

        Args:
            graph: Dependency graph to validate.

        Returns:
            List of validation error messages (empty if valid).

        Example:
            ```python
            errors = generator.validate_graph(graph)
            if errors:
                print(f"Validation failed: {errors}")
            ```
        """
        self.logger.info("validating_graph", nodes=len(graph.nodes))

        errors: list[str] = []

        # Check for cycles using DFS
        visited: set[str] = set()
        rec_stack: set[str] = set()

        def has_cycle(node_id: str) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)

            # Get outgoing edges
            for edge in graph.edges:
                if edge.from_task == node_id:
                    neighbor = edge.to_task
                    if neighbor not in visited:
                        if has_cycle(neighbor):
                            return True
                    elif neighbor in rec_stack:
                        errors.append(f"Cycle detected involving task {neighbor}")
                        return True

            rec_stack.remove(node_id)
            return False

        # Check all nodes for cycles
        for node_id in graph.nodes:
            if node_id not in visited:
                has_cycle(node_id)

        # Check for orphaned tasks (no incoming or outgoing edges)
        connected_nodes: set[str] = set()
        for edge in graph.edges:
            connected_nodes.add(edge.from_task)
            connected_nodes.add(edge.to_task)

        orphans = set(graph.nodes.keys()) - connected_nodes
        if orphans and len(graph.nodes) > 1:
            for orphan in orphans:
                errors.append(f"Orphaned task with no dependencies: {orphan}")

        self.logger.info("validation_complete", errors_count=len(errors))

        return errors

    def topological_sort(self, graph: DependencyGraph) -> list[list[str]]:
        """Produce topological ordering of tasks in execution waves.

        Args:
            graph: Dependency graph to sort.

        Returns:
            List of waves, where each wave contains task IDs that can execute in parallel.

        Example:
            ```python
            waves = generator.topological_sort(graph)
            for i, wave in enumerate(waves):
                print(f"Wave {i+1}: {wave}")
            ```
        """
        self.logger.info("topological_sorting", nodes=len(graph.nodes))

        # Build adjacency list for incoming edges
        in_degree: dict[str, int] = {node_id: 0 for node_id in graph.nodes}
        adjacency: dict[str, list[str]] = {node_id: [] for node_id in graph.nodes}

        for edge in graph.edges:
            in_degree[edge.to_task] += 1
            adjacency[edge.from_task].append(edge.to_task)

        # Start with nodes that have no dependencies
        waves: list[list[str]] = []
        current_wave = [
            node_id for node_id, degree in in_degree.items() if degree == 0
        ]

        while current_wave:
            waves.append(current_wave[:])

            next_wave: list[str] = []
            for node_id in current_wave:
                for neighbor in adjacency[node_id]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        next_wave.append(neighbor)

            current_wave = next_wave

        self.logger.info("topological_sort_complete", waves_count=len(waves))

        return waves

    def visualize_graph(self, graph: DependencyGraph) -> str:
        """Generate text-based ASCII visualization of dependency graph.

        Args:
            graph: Dependency graph to visualize.

        Returns:
            ASCII art representation of the graph.

        Example:
            ```python
            viz = generator.visualize_graph(graph)
            print(viz)
            ```
        """
        lines: list[str] = ["Dependency Graph:", ""]

        # Sort nodes for consistent output
        sorted_nodes = sorted(graph.nodes.items(), key=lambda x: x[0])

        for node_id, node in sorted_nodes:
            # Show node
            lines.append(
                f"[{node_id}] {node.task_title} "
                f"({node.agent_type}, {node.complexity})"
            )

            # Show outgoing edges
            outgoing = [edge for edge in graph.edges if edge.from_task == node_id]
            for edge in outgoing:
                lines.append(f"  └─> {edge.to_task} ({edge.dependency_type})")

            lines.append("")

        return "\n".join(lines)


class ParallelGroupAssigner:
    """Assigns tasks to parallel execution groups.

    This class analyzes task dependencies and file conflicts to determine
    which tasks can safely execute concurrently and generates parallel
    execution groups with appropriate coordination strategies.
    """

    def __init__(self) -> None:
        """Initialize the parallel group assigner."""
        self.logger = logger.bind(component="ParallelGroupAssigner")

    def assign_groups(self, graph: DependencyGraph) -> list[ParallelGroup]:
        """Assign tasks to parallel execution groups.

        Args:
            graph: Dependency graph with tasks and dependencies.

        Returns:
            List of parallel groups with tasks and coordination strategies.

        Example:
            ```python
            assigner = ParallelGroupAssigner()
            groups = assigner.assign_groups(dependency_graph)
            for group in groups:
                print(f"Group {group.group_id}: {len(group.tasks)} tasks")
            ```
        """
        self.logger.info("assigning_parallel_groups", nodes=len(graph.nodes))

        groups: list[ParallelGroup] = []

        # Get topological waves
        generator = DependencyGraphGenerator()
        waves = generator.topological_sort(graph)

        # Create groups from waves
        for wave_idx, wave in enumerate(waves, start=1):
            # Check for file conflicts within wave
            conflicts = self._check_file_conflicts(
                [graph.nodes[task_id] for task_id in wave]
            )

            if not conflicts:
                # No conflicts - fully parallel (PAR-A)
                group = ParallelGroup(
                    group_id=f"wave{wave_idx}_group1",
                    tasks=wave,
                    parallelization_type="PAR-A",
                    shared_files=[],
                    estimated_wave=wave_idx,
                )
                groups.append(group)
            else:
                # Has conflicts - analyze for PAR-B/C or fall back to sequential
                shared_files = list({c.file_path for c in conflicts})

                if len(shared_files) == 1:
                    # Single shared file - PAR-C
                    group = ParallelGroup(
                        group_id=f"wave{wave_idx}_group1",
                        tasks=wave,
                        parallelization_type="PAR-C",
                        shared_files=shared_files,
                        estimated_wave=wave_idx,
                    )
                    groups.append(group)
                else:
                    # Multiple conflicts - create sequential groups
                    for task_idx, task_id in enumerate(wave, start=1):
                        group = ParallelGroup(
                            group_id=f"wave{wave_idx}_group{task_idx}",
                            tasks=[task_id],
                            parallelization_type="SEQ",
                            shared_files=[],
                            estimated_wave=wave_idx,
                        )
                        groups.append(group)

        self.logger.info("parallel_groups_assigned", groups_count=len(groups))

        return groups

    def _detect_parallelization_type(self, tasks: list[TaskNode]) -> ParallelizationType:
        """Detect parallelization type for a group of tasks.

        Args:
            tasks: List of task nodes to analyze.

        Returns:
            Parallelization type (SEQ/PAR-A/PAR-B/PAR-C/PAR-D).
        """
        if len(tasks) == 1:
            return "SEQ"

        # Check for file conflicts
        conflicts = self._check_file_conflicts(tasks)

        if not conflicts:
            return "PAR-A"

        # Analyze conflict patterns
        shared_files = list({c.file_path for c in conflicts})

        if len(shared_files) == 1:
            # Check if it's an interface file (heuristic: has "protocol" or "interface")
            file_path_lower = shared_files[0].lower()
            if "protocol" in file_path_lower or "interface" in file_path_lower:
                return "PAR-B"
            else:
                return "PAR-C"

        # Multiple conflicts - sequential
        return "SEQ"

    def _check_file_conflicts(self, tasks: list[TaskNode]) -> list[FileConflict]:
        """Check for file modification conflicts between tasks.

        Args:
            tasks: List of task nodes to check.

        Returns:
            List of file conflicts found.

        Example:
            ```python
            conflicts = assigner._check_file_conflicts(task_nodes)
            for conflict in conflicts:
                print(f"Conflict in {conflict.file_path}: {conflict.conflicting_tasks}")
            ```
        """
        # Build mapping of file -> tasks that modify it
        file_to_tasks: dict[str, list[str]] = {}

        for task in tasks:
            for file_path in task.files_to_modify:
                if file_path not in file_to_tasks:
                    file_to_tasks[file_path] = []
                file_to_tasks[file_path].append(task.task_id)

        # Identify conflicts (files modified by multiple tasks)
        conflicts: list[FileConflict] = []

        for file_path, task_ids in file_to_tasks.items():
            if len(task_ids) > 1:
                conflict = FileConflict(
                    file_path=file_path,
                    conflicting_tasks=task_ids,
                    resolution_strategy="sequential",
                )
                conflicts.append(conflict)

        return conflicts

    def optimize_groups(
        self, groups: list[ParallelGroup], max_parallel: int
    ) -> list[ParallelGroup]:
        """Optimize parallel groups to respect max_parallel constraint.

        Args:
            groups: Initial parallel groups.
            max_parallel: Maximum number of concurrent tasks allowed.

        Returns:
            Optimized list of parallel groups.

        Example:
            ```python
            optimized = assigner.optimize_groups(groups, max_parallel=5)
            ```
        """
        self.logger.info(
            "optimizing_groups", initial_count=len(groups), max_parallel=max_parallel
        )

        optimized: list[ParallelGroup] = []

        for group in groups:
            if len(group.tasks) <= max_parallel:
                # Group fits within limit
                optimized.append(group)
            else:
                # Split into multiple sequential groups
                for i in range(0, len(group.tasks), max_parallel):
                    chunk = group.tasks[i : i + max_parallel]
                    split_group = ParallelGroup(
                        group_id=f"{group.group_id}_split{i // max_parallel + 1}",
                        tasks=chunk,
                        parallelization_type=group.parallelization_type,
                        shared_files=group.shared_files,
                        estimated_wave=group.estimated_wave,
                    )
                    optimized.append(split_group)

        self.logger.info("optimization_complete", optimized_count=len(optimized))

        return optimized
