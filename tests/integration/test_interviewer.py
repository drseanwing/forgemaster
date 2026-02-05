"""Integration tests for interviewer agent and interview workflow.

Tests the InterviewerConfig, InterviewOrchestrator, and InterviewSession classes
for specification clarification through structured interviews.
"""

from __future__ import annotations

import pytest

from forgemaster.agents.definitions.interviewer import (
    InterviewerConfig,
    get_interviewer_config,
)
from forgemaster.architecture.interviewer import (
    InterviewOrchestrator,
    InterviewQuestion,
    InterviewResponse,
    InterviewSession,
    QuestionCategory,
    QuestionImportance,
)
from forgemaster.architecture.spec_parser import SpecDocument, SpecSection


class TestInterviewerConfig:
    """Test suite for InterviewerConfig."""

    def test_config_creation(self) -> None:
        """Test creating InterviewerConfig with defaults."""
        config = InterviewerConfig()

        assert config.agent_type == "interviewer"
        assert config.model == "claude-opus-4-5-20251101"
        assert config.tools == ["Read"]
        assert config.max_tokens == 8192
        assert config.temperature == 0.7
        assert config.purpose == "Specification clarification and requirements gathering"
        assert config.max_questions == 10
        assert config.max_rounds == 3

    def test_config_factory(self) -> None:
        """Test get_interviewer_config factory function."""
        config = get_interviewer_config()

        assert isinstance(config, InterviewerConfig)
        assert config.agent_type == "interviewer"

    def test_config_customization(self) -> None:
        """Test creating InterviewerConfig with custom values."""
        config = InterviewerConfig(
            max_questions=5,
            max_rounds=2,
            temperature=0.5,
        )

        assert config.max_questions == 5
        assert config.max_rounds == 2
        assert config.temperature == 0.5
        # Defaults still apply
        assert config.agent_type == "interviewer"


class TestQuestionGeneration:
    """Test suite for question generation."""

    def test_generate_questions_from_sample_spec(self) -> None:
        """Test generating questions from a minimal specification."""
        # Create minimal spec
        spec = SpecDocument(
            title="Test Project",
            sections=[
                SpecSection(heading="Overview", level=1, content="A test project")
            ],
            raw_content="# Test Project\n\n## Overview\nA test project",
        )

        orchestrator = InterviewOrchestrator(max_questions_per_round=10)
        questions = orchestrator.generate_questions(spec)

        # Should generate questions about missing sections
        assert len(questions) > 0
        assert any("requirements" in q.question.lower() for q in questions)
        assert any("architecture" in q.question.lower() for q in questions)

        # Check question structure
        for question in questions:
            assert question.id.startswith("Q")
            assert question.question
            assert isinstance(question.category, QuestionCategory)
            assert isinstance(question.importance, QuestionImportance)

    def test_generate_questions_empty_sections(self) -> None:
        """Test generating questions when sections are empty."""
        spec = SpecDocument(
            title="Empty Sections Project",
            sections=[
                SpecSection(heading="Requirements", level=1, content=""),
                SpecSection(heading="Architecture", level=1, content=""),
            ],
            raw_content="# Empty Sections Project\n\n## Requirements\n\n## Architecture",
        )

        orchestrator = InterviewOrchestrator(max_questions_per_round=10)
        questions = orchestrator.generate_questions(spec)

        # Should ask about empty sections
        assert len(questions) > 0
        assert any("requirements" in q.question.lower() for q in questions)

    def test_generate_questions_complete_spec(self) -> None:
        """Test generating questions from a well-formed spec."""
        spec = SpecDocument(
            title="Complete Project",
            sections=[
                SpecSection(
                    heading="Requirements",
                    level=1,
                    content="User authentication, data storage, API endpoints",
                ),
                SpecSection(
                    heading="Architecture",
                    level=1,
                    content="Microservices with PostgreSQL and Redis",
                ),
                SpecSection(
                    heading="Technology",
                    level=1,
                    content="Python 3.12, FastAPI, SQLAlchemy",
                ),
                SpecSection(
                    heading="Deployment",
                    level=1,
                    content="Docker containers on Kubernetes",
                ),
            ],
            raw_content="# Complete Project\n...",
        )

        orchestrator = InterviewOrchestrator(max_questions_per_round=10)
        questions = orchestrator.generate_questions(spec)

        # Should generate fewer questions for complete spec
        # May still ask about scope boundaries
        assert len(questions) <= 5

    def test_question_prioritization(self) -> None:
        """Test that questions are prioritized by importance."""
        spec = SpecDocument(
            title="Test Project",
            sections=[],
            raw_content="# Test Project",
        )

        orchestrator = InterviewOrchestrator(max_questions_per_round=10)
        questions = orchestrator.generate_questions(spec)

        # High importance questions should come first
        if len(questions) > 1:
            first_importance = questions[0].importance
            last_importance = questions[-1].importance

            # First should be high or medium
            assert first_importance in [QuestionImportance.HIGH, QuestionImportance.MEDIUM]


class TestInterviewRoundProcessing:
    """Test suite for interview round processing."""

    def test_start_interview(self) -> None:
        """Test starting an interview session."""
        spec = SpecDocument(
            title="Test Project",
            sections=[],
            raw_content="# Test Project",
        )

        orchestrator = InterviewOrchestrator()
        session = InterviewSession(spec, orchestrator, max_rounds=3)

        questions = session.start_interview()

        assert len(questions) > 0
        assert session.get_current_round() is not None
        assert session.get_current_round().round_number == 1

    def test_process_responses(self) -> None:
        """Test processing responses to interview questions."""
        spec = SpecDocument(
            title="Test Project",
            sections=[],
            raw_content="# Test Project",
        )

        orchestrator = InterviewOrchestrator()
        session = InterviewSession(spec, orchestrator)

        questions = session.start_interview()

        # Create mock responses
        responses = [
            InterviewResponse(
                question_id=questions[0].id,
                answer="The project requires user authentication via OAuth2",
                follow_up_needed=False,
            )
        ]

        if len(questions) > 1:
            responses.append(
                InterviewResponse(
                    question_id=questions[1].id,
                    answer="Not sure yet",
                    follow_up_needed=True,
                )
            )

        result = session.process_responses(responses)

        assert result.round_number == 1
        assert len(result.responses) == len(responses)
        assert len(result.clarifications_gained) > 0

    def test_generate_follow_ups(self) -> None:
        """Test generating follow-up questions."""
        spec = SpecDocument(
            title="Test Project",
            sections=[],
            raw_content="# Test Project",
        )

        orchestrator = InterviewOrchestrator()
        session = InterviewSession(spec, orchestrator, max_rounds=3)

        questions = session.start_interview()

        # Create response that needs follow-up
        responses = [
            InterviewResponse(
                question_id=questions[0].id,
                answer="Not completely clear",
                follow_up_needed=True,
            )
        ]

        round_result = session.process_responses(responses)
        follow_ups = session.generate_follow_ups(round_result)

        # Should generate follow-up questions
        assert len(follow_ups) > 0
        assert any("more detail" in q.question.lower() for q in follow_ups)

        # Round number should increment
        assert session.get_current_round().round_number == 2

    def test_max_rounds_limit(self) -> None:
        """Test that interview respects max_rounds limit."""
        spec = SpecDocument(
            title="Test Project",
            sections=[],
            raw_content="# Test Project",
        )

        orchestrator = InterviewOrchestrator()
        session = InterviewSession(spec, orchestrator, max_rounds=2)

        # Round 1
        questions = session.start_interview()
        responses = [
            InterviewResponse(
                question_id=questions[0].id,
                answer="Incomplete",
                follow_up_needed=True,
            )
        ]
        round1 = session.process_responses(responses)

        # Round 2
        follow_ups = session.generate_follow_ups(round1)
        assert len(follow_ups) > 0

        # Try to generate round 3 (should be empty due to max_rounds=2)
        responses2 = [
            InterviewResponse(
                question_id=follow_ups[0].id,
                answer="Still unclear",
                follow_up_needed=True,
            )
        ]
        round2 = session.process_responses(responses2)
        follow_ups_2 = session.generate_follow_ups(round2)

        assert len(follow_ups_2) == 0


class TestInterviewFinalization:
    """Test suite for interview finalization."""

    def test_finalize_interview(self) -> None:
        """Test finalizing an interview and producing clarified spec."""
        spec = SpecDocument(
            title="Test Project",
            sections=[],
            raw_content="# Test Project",
        )

        orchestrator = InterviewOrchestrator()
        session = InterviewSession(spec, orchestrator)

        questions = session.start_interview()
        responses = [
            InterviewResponse(
                question_id=questions[0].id,
                answer="OAuth2 authentication with JWT tokens",
                follow_up_needed=False,
            )
        ]

        session.process_responses(responses)
        clarified = session.finalize_interview()

        assert clarified.original_spec == spec
        assert len(clarified.clarifications) > 0
        assert len(clarified.resolved_ambiguities) > 0
        assert clarified.interview_rounds == 1

    def test_finalize_with_remaining_unknowns(self) -> None:
        """Test finalization when some questions remain unanswered."""
        spec = SpecDocument(
            title="Test Project",
            sections=[],
            raw_content="# Test Project",
        )

        orchestrator = InterviewOrchestrator()
        session = InterviewSession(spec, orchestrator, max_rounds=1)

        questions = session.start_interview()
        responses = [
            InterviewResponse(
                question_id=questions[0].id,
                answer="Partially answered",
                follow_up_needed=True,
            )
        ]

        round_result = session.process_responses(responses)
        clarified = session.finalize_interview()

        # Should track remaining unknowns
        assert len(clarified.remaining_unknowns) > 0


class TestEdgeCases:
    """Test suite for edge cases."""

    def test_empty_spec(self) -> None:
        """Test handling completely empty specification."""
        spec = SpecDocument(
            title="Empty Project",
            sections=[],
            raw_content="",
        )

        orchestrator = InterviewOrchestrator()
        questions = orchestrator.generate_questions(spec)

        # Should generate questions about all missing sections
        assert len(questions) > 0

    def test_already_clear_spec(self) -> None:
        """Test handling specification that's already clear."""
        spec = SpecDocument(
            title="Clear Project",
            sections=[
                SpecSection(
                    heading="Requirements",
                    level=1,
                    content="Detailed requirements here with 50+ characters of content",
                ),
                SpecSection(
                    heading="Architecture",
                    level=1,
                    content="Detailed architecture description with 50+ characters",
                ),
                SpecSection(
                    heading="Technology",
                    level=1,
                    content="Python 3.12, FastAPI, PostgreSQL with asyncpg driver",
                ),
                SpecSection(
                    heading="Deployment",
                    level=1,
                    content="Docker containers deployed to Kubernetes cluster",
                ),
                SpecSection(
                    heading="Scope",
                    level=1,
                    content="In scope: API and database. Out of scope: frontend",
                ),
            ],
            raw_content="# Clear Project\n...",
        )

        orchestrator = InterviewOrchestrator()
        questions = orchestrator.generate_questions(spec)

        # Should generate minimal or no questions
        assert len(questions) <= 2

    def test_process_responses_before_start(self) -> None:
        """Test that processing responses before starting raises error."""
        spec = SpecDocument(
            title="Test Project",
            sections=[],
            raw_content="# Test Project",
        )

        orchestrator = InterviewOrchestrator()
        session = InterviewSession(spec, orchestrator)

        responses = [
            InterviewResponse(
                question_id="Q1",
                answer="Test answer",
                follow_up_needed=False,
            )
        ]

        with pytest.raises(ValueError, match="Interview not started"):
            session.process_responses(responses)
