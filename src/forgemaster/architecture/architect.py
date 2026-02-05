"""Architecture document generation and technology decision framework for FORGEMASTER.

This module provides tools for generating structured architecture documents from
specifications, evaluating technology choices, and documenting architectural decisions.
It includes models for components, interfaces, data flows, and deployment configurations.
"""

from __future__ import annotations

from typing import Any

import structlog
from pydantic import BaseModel, Field

from forgemaster.architecture.spec_parser import SpecDocument

logger = structlog.get_logger(__name__)


class ComponentDefinition(BaseModel):
    """Definition of a system component.

    Attributes:
        name: Component name (e.g., "UserService", "DatabaseLayer")
        description: What this component does
        responsibilities: List of component responsibilities
        dependencies: Other components and external libraries this depends on
        interfaces: Names of interfaces this component implements or exposes
    """

    name: str = Field(..., description="Component name")
    description: str = Field(..., description="Component description")
    responsibilities: list[str] = Field(
        default_factory=list, description="Component responsibilities"
    )
    dependencies: list[str] = Field(
        default_factory=list, description="Component dependencies"
    )
    interfaces: list[str] = Field(
        default_factory=list, description="Implemented/exposed interfaces"
    )


class InterfaceDefinition(BaseModel):
    """Definition of an interface between components.

    Attributes:
        name: Interface name (e.g., "UserAPI", "EventBus")
        protocol: Communication protocol (HTTP, gRPC, MessageQueue, etc.)
        endpoints: List of available endpoints or methods
        data_format: Data serialization format (JSON, Protobuf, etc.)
    """

    name: str = Field(..., description="Interface name")
    protocol: str = Field(..., description="Communication protocol")
    endpoints: list[str] = Field(default_factory=list, description="Available endpoints")
    data_format: str = Field(..., description="Data serialization format")


class DataFlowDefinition(BaseModel):
    """Definition of data flow between components.

    Attributes:
        source: Source component name
        destination: Destination component name
        data_type: Type of data being transferred
        protocol: Protocol used for transfer
    """

    source: str = Field(..., description="Source component")
    destination: str = Field(..., description="Destination component")
    data_type: str = Field(..., description="Data type being transferred")
    protocol: str = Field(..., description="Transfer protocol")


class TechnologyStack(BaseModel):
    """Technology stack specification.

    Attributes:
        runtime: Runtime environment (e.g., "Python 3.12+", "Node.js 20+")
        frameworks: List of frameworks used
        database: Database system(s) used
        messaging: Message queue or event bus system
        infrastructure: Infrastructure and deployment tools
    """

    runtime: str = Field(..., description="Runtime environment")
    frameworks: list[str] = Field(default_factory=list, description="Frameworks used")
    database: str = Field(..., description="Database system")
    messaging: str = Field(default="", description="Messaging system")
    infrastructure: str = Field(default="", description="Infrastructure tools")


class DeploymentModel(BaseModel):
    """Deployment model specification.

    Attributes:
        strategy: Deployment strategy (container-based, serverless, monolith, microservices)
        environments: List of deployment environments
        scaling: Scaling approach (horizontal, vertical, both)
        monitoring: Monitoring and observability tools
    """

    strategy: str = Field(..., description="Deployment strategy")
    environments: list[str] = Field(
        default_factory=list, description="Deployment environments"
    )
    scaling: str = Field(..., description="Scaling approach")
    monitoring: str = Field(default="", description="Monitoring tools")


class ArchitectureDecision(BaseModel):
    """Architecture Decision Record (ADR).

    Attributes:
        id: Decision identifier (e.g., "ADR-001")
        title: Short decision title
        context: Why this decision was needed
        decision: What was decided
        rationale: Why this choice was made
        alternatives: Other options that were considered
        consequences: Expected outcomes and impacts
    """

    id: str = Field(..., description="Decision identifier")
    title: str = Field(..., description="Decision title")
    context: str = Field(..., description="Decision context")
    decision: str = Field(..., description="The decision made")
    rationale: str = Field(..., description="Rationale for the decision")
    alternatives: list[str] = Field(
        default_factory=list, description="Alternatives considered"
    )
    consequences: list[str] = Field(
        default_factory=list, description="Decision consequences"
    )


class ArchitectureDocument(BaseModel):
    """Complete architecture document.

    Attributes:
        project_name: Name of the project
        version: Document version (semver format)
        overview: High-level system description
        components: List of system components
        interfaces: List of component interfaces
        data_flow: List of data flow definitions
        technology_stack: Technology stack specification
        deployment_model: Deployment model specification
        decisions: List of architecture decisions
    """

    project_name: str = Field(..., description="Project name")
    version: str = Field(..., description="Document version")
    overview: str = Field(..., description="System overview")
    components: list[ComponentDefinition] = Field(
        default_factory=list, description="System components"
    )
    interfaces: list[InterfaceDefinition] = Field(
        default_factory=list, description="Component interfaces"
    )
    data_flow: list[DataFlowDefinition] = Field(
        default_factory=list, description="Data flow definitions"
    )
    technology_stack: TechnologyStack = Field(..., description="Technology stack")
    deployment_model: DeploymentModel = Field(..., description="Deployment model")
    decisions: list[ArchitectureDecision] = Field(
        default_factory=list, description="Architecture decisions"
    )


class TechnologyEvaluation(BaseModel):
    """Evaluation of a single technology option.

    Attributes:
        name: Technology name
        scores: Scores for each evaluation criterion (0.0-1.0)
        strengths: Identified strengths
        weaknesses: Identified weaknesses
        recommendation: Overall recommendation (recommended, acceptable, not_recommended)
    """

    name: str = Field(..., description="Technology name")
    scores: dict[str, float] = Field(
        default_factory=dict, description="Criterion scores (0.0-1.0)"
    )
    strengths: list[str] = Field(default_factory=list, description="Strengths")
    weaknesses: list[str] = Field(default_factory=list, description="Weaknesses")
    recommendation: str = Field(
        ..., description="Recommendation: recommended, acceptable, not_recommended"
    )


class ComparisonResult(BaseModel):
    """Result of technology comparison.

    Attributes:
        options: Technologies being compared
        criteria_weights: Weight for each criterion (0.0-1.0, sum to 1.0)
        scores: Overall weighted scores for each option
        winner: Name of recommended option
        rationale: Explanation of why winner was selected
    """

    options: list[str] = Field(..., description="Options being compared")
    criteria_weights: dict[str, float] = Field(
        default_factory=dict, description="Criterion weights"
    )
    scores: dict[str, float] = Field(default_factory=dict, description="Overall scores")
    winner: str = Field(..., description="Recommended option")
    rationale: str = Field(..., description="Selection rationale")


class ArchitectureGenerator:
    """Generator for architecture documents from specifications.

    This class transforms specification documents into structured architecture
    documents with components, interfaces, data flows, and deployment models.
    """

    def generate_architecture(
        self, spec: SpecDocument, constraints: dict[str, Any] | None = None
    ) -> ArchitectureDocument:
        """Generate architecture document from specification.

        Args:
            spec: Parsed specification document
            constraints: Optional technology or design constraints

        Returns:
            Complete architecture document

        Example:
            >>> generator = ArchitectureGenerator()
            >>> spec = SpecDocument(title="MyProject", sections=[...])
            >>> arch = generator.generate_architecture(spec)
            >>> print(arch.project_name)
            'MyProject'
        """
        logger.info(
            "generating_architecture",
            project=spec.title,
            sections=len(spec.sections),
        )

        constraints = constraints or {}

        # Extract basic info
        project_name = spec.title
        version = constraints.get("version", "0.1.0")
        overview = self._extract_overview(spec)

        # Generate components from spec sections
        components = self._extract_components(spec)
        interfaces = self._generate_interfaces(components)
        data_flow = self._generate_data_flow(components, interfaces)

        # Build technology stack from constraints or defaults
        technology_stack = self._build_technology_stack(spec, constraints)
        deployment_model = self._build_deployment_model(constraints)

        # Generate initial architectural decisions
        decisions = self._generate_initial_decisions(
            spec, technology_stack, deployment_model
        )

        arch_doc = ArchitectureDocument(
            project_name=project_name,
            version=version,
            overview=overview,
            components=components,
            interfaces=interfaces,
            data_flow=data_flow,
            technology_stack=technology_stack,
            deployment_model=deployment_model,
            decisions=decisions,
        )

        logger.info(
            "architecture_generated",
            components=len(components),
            interfaces=len(interfaces),
            decisions=len(decisions),
        )

        return arch_doc

    def generate_component_diagram(self, arch: ArchitectureDocument) -> str:
        """Generate text-based component diagram.

        Args:
            arch: Architecture document

        Returns:
            Text representation of component relationships

        Example:
            >>> diagram = generator.generate_component_diagram(arch)
            >>> print(diagram)
            Component Diagram: MyProject
            ...
        """
        logger.info("generating_component_diagram", project=arch.project_name)

        lines = [
            f"Component Diagram: {arch.project_name}",
            "=" * (len(arch.project_name) + 19),
            "",
        ]

        # List components
        lines.append("Components:")
        for comp in arch.components:
            lines.append(f"  [{comp.name}]")
            lines.append(f"    {comp.description}")
            if comp.responsibilities:
                lines.append(f"    Responsibilities: {', '.join(comp.responsibilities)}")
            lines.append("")

        # Show dependencies
        lines.append("Dependencies:")
        for comp in arch.components:
            if comp.dependencies:
                lines.append(f"  {comp.name} -> {', '.join(comp.dependencies)}")
        lines.append("")

        # Show data flows
        if arch.data_flow:
            lines.append("Data Flow:")
            for flow in arch.data_flow:
                lines.append(
                    f"  {flow.source} --[{flow.data_type} via {flow.protocol}]--> "
                    f"{flow.destination}"
                )

        return "\n".join(lines)

    def _extract_overview(self, spec: SpecDocument) -> str:
        """Extract overview from specification."""
        # Look for overview, summary, or description sections
        for section in spec.sections:
            heading_lower = section.heading.lower()
            if any(
                keyword in heading_lower
                for keyword in ("overview", "summary", "description", "introduction")
            ):
                return section.content

        # Fallback: use first section content
        if spec.sections and spec.sections[0].content:
            return spec.sections[0].content

        return "No overview available"

    def _extract_components(self, spec: SpecDocument) -> list[ComponentDefinition]:
        """Extract component definitions from specification."""
        components: list[ComponentDefinition] = []

        # Look for architecture, components, or modules section
        for section in spec.sections:
            heading_lower = section.heading.lower()
            if any(
                keyword in heading_lower
                for keyword in ("component", "module", "service", "layer")
            ):
                # Each subsection is a component
                for subsection in section.subsections:
                    component = ComponentDefinition(
                        name=subsection.heading,
                        description=subsection.content or "Component description",
                        responsibilities=self._parse_responsibilities(subsection.content),
                        dependencies=[],
                        interfaces=[],
                    )
                    components.append(component)

        # If no components found, create placeholder
        if not components:
            components.append(
                ComponentDefinition(
                    name="CoreService",
                    description="Main application component",
                    responsibilities=["Primary application logic"],
                )
            )

        return components

    def _parse_responsibilities(self, content: str) -> list[str]:
        """Parse responsibilities from content."""
        responsibilities: list[str] = []

        # Look for bullet points or numbered lists
        for line in content.split("\n"):
            line = line.strip()
            if line.startswith(("-", "*", "•")):
                responsibilities.append(line[1:].strip())
            elif line and line[0].isdigit() and "." in line[:3]:
                responsibilities.append(line.split(".", 1)[1].strip())

        return responsibilities

    def _generate_interfaces(
        self, components: list[ComponentDefinition]
    ) -> list[InterfaceDefinition]:
        """Generate interface definitions for components."""
        interfaces: list[InterfaceDefinition] = []

        for comp in components:
            # Create a basic interface for each component
            interface = InterfaceDefinition(
                name=f"{comp.name}Interface",
                protocol="HTTP",
                endpoints=[f"/{comp.name.lower()}"],
                data_format="JSON",
            )
            interfaces.append(interface)
            comp.interfaces.append(interface.name)

        return interfaces

    def _generate_data_flow(
        self,
        components: list[ComponentDefinition],
        interfaces: list[InterfaceDefinition],
    ) -> list[DataFlowDefinition]:
        """Generate data flow definitions."""
        flows: list[DataFlowDefinition] = []

        # Create flows based on component adjacency
        # Note: This does NOT modify component dependencies
        for i in range(len(components) - 1):
            flow = DataFlowDefinition(
                source=components[i].name,
                destination=components[i + 1].name,
                data_type="Request",
                protocol="HTTP",
            )
            flows.append(flow)

        return flows

    def _build_technology_stack(
        self, spec: SpecDocument, constraints: dict[str, Any]
    ) -> TechnologyStack:
        """Build technology stack from spec and constraints."""
        # Extract from spec or use defaults
        runtime = constraints.get("runtime", "Python 3.12+")
        frameworks = constraints.get("frameworks", ["asyncio", "SQLAlchemy", "Pydantic"])
        database = constraints.get("database", "PostgreSQL")
        messaging = constraints.get("messaging", "Redis")
        infrastructure = constraints.get("infrastructure", "Docker")

        return TechnologyStack(
            runtime=runtime,
            frameworks=frameworks,
            database=database,
            messaging=messaging,
            infrastructure=infrastructure,
        )

    def _build_deployment_model(
        self, constraints: dict[str, Any]
    ) -> DeploymentModel:
        """Build deployment model from constraints."""
        return DeploymentModel(
            strategy=constraints.get("deployment_strategy", "container-based"),
            environments=constraints.get("environments", ["development", "production"]),
            scaling=constraints.get("scaling", "horizontal"),
            monitoring=constraints.get("monitoring", "structlog, Prometheus"),
        )

    def _generate_initial_decisions(
        self,
        spec: SpecDocument,
        tech_stack: TechnologyStack,
        deployment: DeploymentModel,
    ) -> list[ArchitectureDecision]:
        """Generate initial architectural decisions."""
        decisions: list[ArchitectureDecision] = []

        # Decision about runtime
        decisions.append(
            ArchitectureDecision(
                id="ADR-001",
                title=f"Use {tech_stack.runtime} as runtime",
                context="Need to select runtime environment for the project",
                decision=f"Use {tech_stack.runtime}",
                rationale="Modern Python with async support and strong typing",
                alternatives=["Python 3.10", "Python 3.11"],
                consequences=[
                    "Requires Python 3.12+ for deployment",
                    "Access to latest language features",
                ],
            )
        )

        # Decision about deployment strategy
        decisions.append(
            ArchitectureDecision(
                id="ADR-002",
                title=f"Deploy using {deployment.strategy}",
                context="Need to define deployment and scaling strategy",
                decision=f"Use {deployment.strategy} deployment",
                rationale="Provides isolation, reproducibility, and scalability",
                alternatives=["Serverless", "Traditional VMs"],
                consequences=[
                    "Requires container orchestration knowledge",
                    "Better resource utilization",
                ],
            )
        )

        return decisions


class TechnologyEvaluator:
    """Evaluator for technology choices and decision documentation.

    This class provides methods for evaluating technologies against criteria,
    comparing multiple options, and generating Architecture Decision Records.
    """

    # Standard evaluation criteria
    CRITERIA = [
        "maturity",
        "community_support",
        "performance",
        "learning_curve",
        "integration_ease",
        "license",
    ]

    def evaluate_technology(
        self, name: str, criteria: dict[str, float] | None = None
    ) -> TechnologyEvaluation:
        """Evaluate a technology against standard criteria.

        Args:
            name: Technology name
            criteria: Optional custom criteria scores (0.0-1.0)

        Returns:
            Technology evaluation with scores and recommendation

        Example:
            >>> evaluator = TechnologyEvaluator()
            >>> eval = evaluator.evaluate_technology("FastAPI", {
            ...     "maturity": 0.9,
            ...     "performance": 0.95
            ... })
            >>> print(eval.recommendation)
            'recommended'
        """
        logger.info("evaluating_technology", name=name)

        criteria = criteria or {}
        scores: dict[str, float] = {}

        # Populate scores with provided values or defaults
        for criterion in self.CRITERIA:
            scores[criterion] = criteria.get(criterion, 0.5)

        # Analyze scores to determine strengths and weaknesses
        strengths: list[str] = []
        weaknesses: list[str] = []

        for criterion, score in scores.items():
            if score >= 0.8:
                strengths.append(f"Strong {criterion.replace('_', ' ')}")
            elif score <= 0.3:
                weaknesses.append(f"Weak {criterion.replace('_', ' ')}")

        # Calculate recommendation based on average score
        avg_score = sum(scores.values()) / len(scores)

        if avg_score >= 0.7:
            recommendation = "recommended"
        elif avg_score >= 0.4:
            recommendation = "acceptable"
        else:
            recommendation = "not_recommended"

        return TechnologyEvaluation(
            name=name,
            scores=scores,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendation=recommendation,
        )

    def compare_technologies(
        self, options: list[str], criteria: dict[str, dict[str, float]]
    ) -> ComparisonResult:
        """Compare multiple technology options.

        Args:
            options: List of technology names
            criteria: Nested dict with criteria scores for each option
                     Format: {option_name: {criterion: score}}

        Returns:
            Comparison result with winner and rationale

        Example:
            >>> evaluator = TechnologyEvaluator()
            >>> result = evaluator.compare_technologies(
            ...     ["FastAPI", "Flask"],
            ...     {
            ...         "FastAPI": {"performance": 0.9, "maturity": 0.8},
            ...         "Flask": {"performance": 0.7, "maturity": 0.95}
            ...     }
            ... )
            >>> print(result.winner)
            'FastAPI'
        """
        logger.info("comparing_technologies", options=options)

        # Equal weights for all criteria by default
        criteria_weights: dict[str, float] = {
            criterion: 1.0 / len(self.CRITERIA) for criterion in self.CRITERIA
        }

        # Calculate weighted scores for each option
        scores: dict[str, float] = {}

        for option in options:
            option_criteria = criteria.get(option, {})
            weighted_score = 0.0

            for criterion in self.CRITERIA:
                score = option_criteria.get(criterion, 0.5)
                weight = criteria_weights.get(criterion, 0.0)
                weighted_score += score * weight

            scores[option] = weighted_score

        # Determine winner
        winner = max(scores, key=lambda k: scores[k])
        winner_score = scores[winner]

        # Generate rationale
        winner_criteria = criteria.get(winner, {})
        strong_points = [
            criterion
            for criterion, score in winner_criteria.items()
            if score >= 0.8
        ]

        rationale = (
            f"{winner} selected with score {winner_score:.2f}. "
            f"Strong in: {', '.join(strong_points) if strong_points else 'balanced performance'}."
        )

        return ComparisonResult(
            options=options,
            criteria_weights=criteria_weights,
            scores=scores,
            winner=winner,
            rationale=rationale,
        )

    def generate_adr(self, decision: ArchitectureDecision) -> str:
        """Generate Architecture Decision Record in standard format.

        Args:
            decision: Architecture decision to document

        Returns:
            Formatted ADR as markdown

        Example:
            >>> decision = ArchitectureDecision(
            ...     id="ADR-001",
            ...     title="Use PostgreSQL",
            ...     context="Need database",
            ...     decision="PostgreSQL",
            ...     rationale="ACID compliance",
            ...     alternatives=["MySQL"],
            ...     consequences=["Need PostgreSQL hosting"]
            ... )
            >>> adr = evaluator.generate_adr(decision)
        """
        logger.info("generating_adr", decision_id=decision.id)

        lines = [
            f"# {decision.id}: {decision.title}",
            "",
            "## Status",
            "Accepted",
            "",
            "## Context",
            decision.context,
            "",
            "## Decision",
            decision.decision,
            "",
            "## Rationale",
            decision.rationale,
            "",
        ]

        if decision.alternatives:
            lines.extend([
                "## Alternatives Considered",
                "",
            ])
            for alt in decision.alternatives:
                lines.append(f"- {alt}")
            lines.append("")

        if decision.consequences:
            lines.extend([
                "## Consequences",
                "",
            ])
            for cons in decision.consequences:
                lines.append(f"- {cons}")

        return "\n".join(lines)
