"""Integration tests for architect agent and architecture generation.

Tests cover agent configuration, architecture document generation, component diagrams,
technology evaluation, technology comparison, and ADR generation.
"""

from __future__ import annotations

import pytest

from forgemaster.agents.definitions.architect import ArchitectConfig, get_architect_config
from forgemaster.architecture import (
    ArchitectureDecision,
    ArchitectureDocument,
    ArchitectureGenerator,
    ComponentDefinition,
    ComparisonResult,
    DeploymentModel,
    InterfaceDefinition,
    SpecDocument,
    SpecSection,
    TechnologyEvaluation,
    TechnologyEvaluator,
    TechnologyStack,
)


class TestArchitectConfig:
    """Tests for ArchitectConfig creation and defaults."""

    def test_architect_config_defaults(self) -> None:
        """Test that ArchitectConfig has correct default values."""
        config = ArchitectConfig()

        assert config.agent_type == "architect"
        assert config.model == "claude-opus-4-5-20251101"
        assert config.tools == ["Read", "Write", "Bash", "Grep", "Glob"]
        assert config.max_tokens == 16384
        assert config.temperature == 0.3
        assert config.purpose == "System architecture design and technical decision-making"
        assert config.output_format == "architecture_document"

    def test_get_architect_config(self) -> None:
        """Test factory function for creating architect config."""
        config = get_architect_config()

        assert isinstance(config, ArchitectConfig)
        assert config.agent_type == "architect"

    def test_architect_config_immutable(self) -> None:
        """Test that ArchitectConfig is frozen (immutable)."""
        config = ArchitectConfig()

        with pytest.raises(Exception):  # FrozenInstanceError in Python 3.12+
            config.agent_type = "modified"  # type: ignore


class TestArchitectureGeneration:
    """Tests for architecture document generation from specifications."""

    @pytest.fixture
    def sample_spec(self) -> SpecDocument:
        """Create a sample specification document for testing."""
        sections = [
            SpecSection(
                heading="Overview",
                level=1,
                content="A system for managing tasks and projects",
            ),
            SpecSection(
                heading="Components",
                level=1,
                content="",
                subsections=[
                    SpecSection(
                        heading="TaskService",
                        level=2,
                        content="Manages task CRUD operations\n- Create tasks\n- Update tasks",
                    ),
                    SpecSection(
                        heading="ProjectService",
                        level=2,
                        content="Manages projects\n- Create projects\n- List projects",
                    ),
                ],
            ),
        ]

        return SpecDocument(
            title="TaskManager",
            sections=sections,
            raw_content="# TaskManager\n...",
        )

    @pytest.fixture
    def minimal_spec(self) -> SpecDocument:
        """Create a minimal specification document."""
        return SpecDocument(
            title="MinimalProject",
            sections=[
                SpecSection(
                    heading="Description",
                    level=1,
                    content="A minimal project",
                )
            ],
            raw_content="# MinimalProject\n...",
        )

    @pytest.fixture
    def over_specified_spec(self) -> SpecDocument:
        """Create an over-specified specification document."""
        sections = [
            SpecSection(heading=f"Section{i}", level=1, content=f"Content {i}")
            for i in range(20)
        ]

        return SpecDocument(
            title="OverSpecified",
            sections=sections,
            raw_content="# OverSpecified\n...",
        )

    def test_generate_architecture_basic(self, sample_spec: SpecDocument) -> None:
        """Test basic architecture generation from specification."""
        generator = ArchitectureGenerator()
        arch = generator.generate_architecture(sample_spec)

        assert isinstance(arch, ArchitectureDocument)
        assert arch.project_name == "TaskManager"
        assert arch.version == "0.1.0"
        assert arch.overview != ""
        assert len(arch.components) >= 2
        assert len(arch.interfaces) >= 2
        assert len(arch.decisions) >= 2

    def test_generate_architecture_with_constraints(
        self, sample_spec: SpecDocument
    ) -> None:
        """Test architecture generation with custom constraints."""
        generator = ArchitectureGenerator()
        constraints = {
            "version": "1.0.0",
            "runtime": "Python 3.13+",
            "database": "MongoDB",
            "deployment_strategy": "serverless",
        }

        arch = generator.generate_architecture(sample_spec, constraints)

        assert arch.version == "1.0.0"
        assert arch.technology_stack.runtime == "Python 3.13+"
        assert arch.technology_stack.database == "MongoDB"
        assert arch.deployment_model.strategy == "serverless"

    def test_generate_architecture_minimal_spec(
        self, minimal_spec: SpecDocument
    ) -> None:
        """Test architecture generation from minimal specification."""
        generator = ArchitectureGenerator()
        arch = generator.generate_architecture(minimal_spec)

        assert arch.project_name == "MinimalProject"
        # Should still have at least placeholder component
        assert len(arch.components) >= 1
        assert arch.components[0].name == "CoreService"

    def test_generate_architecture_over_specified(
        self, over_specified_spec: SpecDocument
    ) -> None:
        """Test architecture generation handles over-specified input."""
        generator = ArchitectureGenerator()
        arch = generator.generate_architecture(over_specified_spec)

        # Should handle gracefully without errors
        assert arch.project_name == "OverSpecified"
        assert isinstance(arch.components, list)

    def test_components_have_required_fields(self, sample_spec: SpecDocument) -> None:
        """Test that generated components have all required fields."""
        generator = ArchitectureGenerator()
        arch = generator.generate_architecture(sample_spec)

        for comp in arch.components:
            assert comp.name != ""
            assert comp.description != ""
            assert isinstance(comp.responsibilities, list)
            assert isinstance(comp.dependencies, list)
            assert isinstance(comp.interfaces, list)

    def test_interfaces_have_required_fields(self, sample_spec: SpecDocument) -> None:
        """Test that generated interfaces have all required fields."""
        generator = ArchitectureGenerator()
        arch = generator.generate_architecture(sample_spec)

        for interface in arch.interfaces:
            assert interface.name != ""
            assert interface.protocol != ""
            assert isinstance(interface.endpoints, list)
            assert interface.data_format != ""

    def test_technology_stack_structure(self, sample_spec: SpecDocument) -> None:
        """Test that technology stack has correct structure."""
        generator = ArchitectureGenerator()
        arch = generator.generate_architecture(sample_spec)

        tech = arch.technology_stack
        assert isinstance(tech, TechnologyStack)
        assert tech.runtime != ""
        assert isinstance(tech.frameworks, list)
        assert tech.database != ""

    def test_deployment_model_structure(self, sample_spec: SpecDocument) -> None:
        """Test that deployment model has correct structure."""
        generator = ArchitectureGenerator()
        arch = generator.generate_architecture(sample_spec)

        deploy = arch.deployment_model
        assert isinstance(deploy, DeploymentModel)
        assert deploy.strategy != ""
        assert isinstance(deploy.environments, list)
        assert len(deploy.environments) > 0


class TestComponentDiagram:
    """Tests for component diagram generation."""

    @pytest.fixture
    def sample_architecture(self) -> ArchitectureDocument:
        """Create a sample architecture document for testing."""
        components = [
            ComponentDefinition(
                name="APIGateway",
                description="Handles HTTP requests",
                responsibilities=["Request routing", "Authentication"],
                dependencies=["UserService"],
                interfaces=["APIGatewayInterface"],
            ),
            ComponentDefinition(
                name="UserService",
                description="Manages users",
                responsibilities=["User CRUD"],
                dependencies=[],
                interfaces=["UserServiceInterface"],
            ),
        ]

        return ArchitectureDocument(
            project_name="TestProject",
            version="1.0.0",
            overview="Test architecture",
            components=components,
            interfaces=[],
            data_flow=[],
            technology_stack=TechnologyStack(
                runtime="Python 3.12+",
                frameworks=["FastAPI"],
                database="PostgreSQL",
            ),
            deployment_model=DeploymentModel(
                strategy="container-based",
                environments=["dev", "prod"],
                scaling="horizontal",
            ),
            decisions=[],
        )

    def test_generate_component_diagram(
        self, sample_architecture: ArchitectureDocument
    ) -> None:
        """Test component diagram generation."""
        generator = ArchitectureGenerator()
        diagram = generator.generate_component_diagram(sample_architecture)

        assert isinstance(diagram, str)
        assert "Component Diagram: TestProject" in diagram
        assert "APIGateway" in diagram
        assert "UserService" in diagram

    def test_diagram_includes_components(
        self, sample_architecture: ArchitectureDocument
    ) -> None:
        """Test that diagram includes all components."""
        generator = ArchitectureGenerator()
        diagram = generator.generate_component_diagram(sample_architecture)

        for comp in sample_architecture.components:
            assert comp.name in diagram
            assert comp.description in diagram

    def test_diagram_shows_dependencies(
        self, sample_architecture: ArchitectureDocument
    ) -> None:
        """Test that diagram shows component dependencies."""
        generator = ArchitectureGenerator()
        diagram = generator.generate_component_diagram(sample_architecture)

        assert "APIGateway -> UserService" in diagram


class TestTechnologyEvaluation:
    """Tests for technology evaluation framework."""

    @pytest.fixture
    def evaluator(self) -> TechnologyEvaluator:
        """Create a technology evaluator instance."""
        return TechnologyEvaluator()

    def test_evaluate_technology_basic(self, evaluator: TechnologyEvaluator) -> None:
        """Test basic technology evaluation."""
        criteria = {
            "maturity": 0.9,
            "community_support": 0.85,
            "performance": 0.8,
            "learning_curve": 0.7,
            "integration_ease": 0.75,
            "license": 1.0,
        }

        evaluation = evaluator.evaluate_technology("FastAPI", criteria)

        assert isinstance(evaluation, TechnologyEvaluation)
        assert evaluation.name == "FastAPI"
        assert evaluation.recommendation == "recommended"
        assert len(evaluation.scores) == len(evaluator.CRITERIA)

    def test_evaluate_technology_with_defaults(
        self, evaluator: TechnologyEvaluator
    ) -> None:
        """Test technology evaluation with default criteria scores."""
        evaluation = evaluator.evaluate_technology("SomeTech")

        assert evaluation.name == "SomeTech"
        assert evaluation.recommendation == "acceptable"
        # All criteria should have default score of 0.5
        assert all(score == 0.5 for score in evaluation.scores.values())

    def test_evaluate_technology_recommended(
        self, evaluator: TechnologyEvaluator
    ) -> None:
        """Test that high scores result in 'recommended' status."""
        criteria = {criterion: 0.8 for criterion in evaluator.CRITERIA}
        evaluation = evaluator.evaluate_technology("HighScore", criteria)

        assert evaluation.recommendation == "recommended"

    def test_evaluate_technology_acceptable(
        self, evaluator: TechnologyEvaluator
    ) -> None:
        """Test that medium scores result in 'acceptable' status."""
        criteria = {criterion: 0.5 for criterion in evaluator.CRITERIA}
        evaluation = evaluator.evaluate_technology("MediumScore", criteria)

        assert evaluation.recommendation == "acceptable"

    def test_evaluate_technology_not_recommended(
        self, evaluator: TechnologyEvaluator
    ) -> None:
        """Test that low scores result in 'not_recommended' status."""
        criteria = {criterion: 0.2 for criterion in evaluator.CRITERIA}
        evaluation = evaluator.evaluate_technology("LowScore", criteria)

        assert evaluation.recommendation == "not_recommended"

    def test_evaluation_identifies_strengths(
        self, evaluator: TechnologyEvaluator
    ) -> None:
        """Test that evaluation identifies strengths (score >= 0.8)."""
        criteria = {
            "maturity": 0.9,
            "community_support": 0.5,
            "performance": 0.85,
            "learning_curve": 0.3,
            "integration_ease": 0.5,
            "license": 0.5,
        }

        evaluation = evaluator.evaluate_technology("TestTech", criteria)

        assert len(evaluation.strengths) >= 2
        assert any("maturity" in s.lower() for s in evaluation.strengths)
        assert any("performance" in s.lower() for s in evaluation.strengths)

    def test_evaluation_identifies_weaknesses(
        self, evaluator: TechnologyEvaluator
    ) -> None:
        """Test that evaluation identifies weaknesses (score <= 0.3)."""
        criteria = {
            "maturity": 0.5,
            "community_support": 0.2,
            "performance": 0.5,
            "learning_curve": 0.1,
            "integration_ease": 0.5,
            "license": 0.5,
        }

        evaluation = evaluator.evaluate_technology("TestTech", criteria)

        assert len(evaluation.weaknesses) >= 2


class TestTechnologyComparison:
    """Tests for technology comparison functionality."""

    @pytest.fixture
    def evaluator(self) -> TechnologyEvaluator:
        """Create a technology evaluator instance."""
        return TechnologyEvaluator()

    def test_compare_technologies_basic(self, evaluator: TechnologyEvaluator) -> None:
        """Test basic technology comparison."""
        options = ["FastAPI", "Flask"]
        criteria = {
            "FastAPI": {
                "maturity": 0.8,
                "performance": 0.95,
                "learning_curve": 0.7,
            },
            "Flask": {
                "maturity": 0.95,
                "performance": 0.7,
                "learning_curve": 0.9,
            },
        }

        result = evaluator.compare_technologies(options, criteria)

        assert isinstance(result, ComparisonResult)
        assert result.winner in options
        assert len(result.scores) == len(options)
        assert result.rationale != ""

    def test_comparison_has_all_options(self, evaluator: TechnologyEvaluator) -> None:
        """Test that comparison includes all options."""
        options = ["Option1", "Option2", "Option3"]
        criteria = {opt: {"maturity": 0.5} for opt in options}

        result = evaluator.compare_technologies(options, criteria)

        assert result.options == options
        assert len(result.scores) == len(options)

    def test_comparison_selects_highest_score(
        self, evaluator: TechnologyEvaluator
    ) -> None:
        """Test that comparison selects option with highest score."""
        options = ["Low", "High", "Medium"]
        criteria = {
            "Low": {criterion: 0.3 for criterion in evaluator.CRITERIA},
            "High": {criterion: 0.9 for criterion in evaluator.CRITERIA},
            "Medium": {criterion: 0.5 for criterion in evaluator.CRITERIA},
        }

        result = evaluator.compare_technologies(options, criteria)

        assert result.winner == "High"
        assert result.scores["High"] > result.scores["Low"]
        assert result.scores["High"] > result.scores["Medium"]

    def test_comparison_criteria_weights(
        self, evaluator: TechnologyEvaluator
    ) -> None:
        """Test that comparison includes criteria weights."""
        options = ["Option1", "Option2"]
        criteria = {opt: {"maturity": 0.5} for opt in options}

        result = evaluator.compare_technologies(options, criteria)

        assert len(result.criteria_weights) > 0
        # Weights should sum to approximately 1.0
        total_weight = sum(result.criteria_weights.values())
        assert abs(total_weight - 1.0) < 0.01


class TestADRGeneration:
    """Tests for Architecture Decision Record generation."""

    @pytest.fixture
    def evaluator(self) -> TechnologyEvaluator:
        """Create a technology evaluator instance."""
        return TechnologyEvaluator()

    @pytest.fixture
    def sample_decision(self) -> ArchitectureDecision:
        """Create a sample architecture decision for testing."""
        return ArchitectureDecision(
            id="ADR-001",
            title="Use PostgreSQL for data storage",
            context="Need a reliable relational database with ACID guarantees",
            decision="Use PostgreSQL 15+",
            rationale="Mature, reliable, strong community support, excellent performance",
            alternatives=["MySQL", "MongoDB", "SQLite"],
            consequences=[
                "Need PostgreSQL hosting infrastructure",
                "Team needs PostgreSQL expertise",
                "Excellent query performance",
            ],
        )

    def test_generate_adr_basic(
        self, evaluator: TechnologyEvaluator, sample_decision: ArchitectureDecision
    ) -> None:
        """Test basic ADR generation."""
        adr = evaluator.generate_adr(sample_decision)

        assert isinstance(adr, str)
        assert sample_decision.id in adr
        assert sample_decision.title in adr

    def test_adr_includes_all_sections(
        self, evaluator: TechnologyEvaluator, sample_decision: ArchitectureDecision
    ) -> None:
        """Test that ADR includes all required sections."""
        adr = evaluator.generate_adr(sample_decision)

        assert "## Status" in adr
        assert "## Context" in adr
        assert "## Decision" in adr
        assert "## Rationale" in adr
        assert "## Alternatives Considered" in adr
        assert "## Consequences" in adr

    def test_adr_includes_content(
        self, evaluator: TechnologyEvaluator, sample_decision: ArchitectureDecision
    ) -> None:
        """Test that ADR includes decision content."""
        adr = evaluator.generate_adr(sample_decision)

        assert sample_decision.context in adr
        assert sample_decision.decision in adr
        assert sample_decision.rationale in adr

    def test_adr_lists_alternatives(
        self, evaluator: TechnologyEvaluator, sample_decision: ArchitectureDecision
    ) -> None:
        """Test that ADR lists all alternatives."""
        adr = evaluator.generate_adr(sample_decision)

        for alternative in sample_decision.alternatives:
            assert alternative in adr

    def test_adr_lists_consequences(
        self, evaluator: TechnologyEvaluator, sample_decision: ArchitectureDecision
    ) -> None:
        """Test that ADR lists all consequences."""
        adr = evaluator.generate_adr(sample_decision)

        for consequence in sample_decision.consequences:
            assert consequence in adr

    def test_adr_markdown_format(
        self, evaluator: TechnologyEvaluator, sample_decision: ArchitectureDecision
    ) -> None:
        """Test that ADR is formatted as valid markdown."""
        adr = evaluator.generate_adr(sample_decision)

        # Should start with H1 heading
        assert adr.startswith("#")
        # Should have H2 sections
        assert "##" in adr
        # Should use bullet points for lists
        assert "- " in adr
