"""Interviewer orchestration for FORGEMASTER.

This module provides the InterviewOrchestrator and InterviewSession classes for
conducting multi-round specification interviews, generating clarifying questions,
and producing clarified specifications.
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any

import structlog
from pydantic import BaseModel, Field

from forgemaster.architecture.spec_parser import SpecDocument

logger = structlog.get_logger(__name__)


class QuestionCategory(str, Enum):
    """Categories for interview questions.

    Attributes:
        REQUIREMENTS: Missing or unclear functional/non-functional requirements
        TECHNICAL: Technology choices, architecture, constraints
        SCOPE: Boundary conditions, what's in/out of scope
        PRIORITY: Urgency, sequencing, what to tackle first
    """

    REQUIREMENTS = "requirements"
    TECHNICAL = "technical"
    SCOPE = "scope"
    PRIORITY = "priority"


class QuestionImportance(str, Enum):
    """Importance levels for interview questions.

    Attributes:
        HIGH: Critical, blocks planning or success
        MEDIUM: Important but has reasonable defaults
        LOW: Nice-to-have clarification
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class InterviewQuestion(BaseModel):
    """A single interview question.

    Attributes:
        id: Unique question identifier (e.g., "Q1", "Q2")
        question: The question text
        category: Question category
        importance: Importance level
        context: Optional context explaining why this question matters
    """

    id: str = Field(..., description="Unique question identifier")
    question: str = Field(..., description="Question text")
    category: QuestionCategory = Field(..., description="Question category")
    importance: QuestionImportance = Field(..., description="Importance level")
    context: str | None = Field(None, description="Context explaining importance")


class InterviewResponse(BaseModel):
    """Response to an interview question.

    Attributes:
        question_id: ID of the question being answered
        answer: The answer text
        follow_up_needed: Whether this answer requires follow-up
    """

    question_id: str = Field(..., description="Question ID being answered")
    answer: str = Field(..., description="Answer text")
    follow_up_needed: bool = Field(
        default=False, description="Whether follow-up is needed"
    )


class InterviewRoundResult(BaseModel):
    """Result from a single interview round.

    Attributes:
        round_number: Which round this is (1-indexed)
        questions_asked: Questions asked in this round
        responses: Responses received
        clarifications_gained: Clarifications extracted from responses
        remaining_gaps: Gaps that still need addressing
    """

    round_number: int = Field(..., ge=1, description="Round number")
    questions_asked: list[InterviewQuestion] = Field(
        default_factory=list, description="Questions asked"
    )
    responses: list[InterviewResponse] = Field(
        default_factory=list, description="Responses received"
    )
    clarifications_gained: dict[str, str] = Field(
        default_factory=dict, description="Clarifications gained (topic -> detail)"
    )
    remaining_gaps: list[str] = Field(
        default_factory=list, description="Gaps still remaining"
    )


class ClarifiedSpec(BaseModel):
    """Specification with clarifications applied.

    Attributes:
        original_spec: The original specification document
        clarifications: Map of topics to clarifications
        resolved_ambiguities: List of ambiguities that were resolved
        remaining_unknowns: List of items still unclear
        interview_rounds: Number of interview rounds conducted
    """

    original_spec: SpecDocument
    clarifications: dict[str, str] = Field(
        default_factory=dict, description="Topic to clarification mapping"
    )
    resolved_ambiguities: list[str] = Field(
        default_factory=list, description="Ambiguities that were resolved"
    )
    remaining_unknowns: list[str] = Field(
        default_factory=list, description="Items still unclear"
    )
    interview_rounds: int = Field(default=0, ge=0, description="Rounds conducted")


class InterviewOrchestrator:
    """Orchestrates the interview process for specification clarification.

    This class manages question generation, response processing, and the overall
    interview lifecycle.

    Attributes:
        max_questions_per_round: Maximum questions to ask per round
    """

    def __init__(self, max_questions_per_round: int = 10) -> None:
        """Initialize the interviewer orchestrator.

        Args:
            max_questions_per_round: Maximum questions per round
        """
        self.max_questions_per_round = max_questions_per_round
        self._logger = structlog.get_logger(__name__)

    def generate_questions(
        self, spec: SpecDocument, context: dict[str, Any] | None = None
    ) -> list[InterviewQuestion]:
        """Generate initial interview questions from a specification.

        This analyzes the specification for completeness and generates targeted
        questions to fill gaps.

        Args:
            spec: Specification document to analyze
            context: Optional additional context (e.g., project type, tech stack)

        Returns:
            List of interview questions, prioritized by importance
        """
        self._logger.info(
            "generating_questions", spec_title=spec.title, sections=len(spec.sections)
        )

        questions: list[InterviewQuestion] = []
        ctx = context or {}

        # Check for missing required sections
        section_headings = {s.heading.lower() for s in spec.sections}
        required_topics = {"requirements", "architecture", "technology", "deployment"}

        missing_topics = required_topics - section_headings
        for topic in missing_topics:
            questions.append(
                InterviewQuestion(
                    id=f"Q{len(questions) + 1}",
                    question=f"What are the {topic} details for this project?",
                    category=QuestionCategory.REQUIREMENTS
                    if topic == "requirements"
                    else QuestionCategory.TECHNICAL,
                    importance=QuestionImportance.HIGH,
                    context=f"No {topic} section found in specification",
                )
            )

        # Check for empty or minimal sections
        for section in spec.sections:
            if not section.content.strip() and not section.subsections:
                questions.append(
                    InterviewQuestion(
                        id=f"Q{len(questions) + 1}",
                        question=f"Can you provide details for the '{section.heading}' section?",
                        category=self._infer_category(section.heading),
                        importance=QuestionImportance.MEDIUM,
                        context=f"Section '{section.heading}' is empty",
                    )
                )

        # Check for technology constraints
        if "technology" not in section_headings:
            questions.append(
                InterviewQuestion(
                    id=f"Q{len(questions) + 1}",
                    question="What are the technology stack requirements and constraints?",
                    category=QuestionCategory.TECHNICAL,
                    importance=QuestionImportance.HIGH,
                    context="No technology constraints specified",
                )
            )

        # Check for deployment requirements
        if "deployment" not in section_headings and "infrastructure" not in section_headings:
            questions.append(
                InterviewQuestion(
                    id=f"Q{len(questions) + 1}",
                    question="What are the deployment and infrastructure requirements?",
                    category=QuestionCategory.TECHNICAL,
                    importance=QuestionImportance.MEDIUM,
                    context="No deployment information found",
                )
            )

        # Check for scope boundaries
        if "scope" not in section_headings:
            questions.append(
                InterviewQuestion(
                    id=f"Q{len(questions) + 1}",
                    question="What is explicitly out of scope for this project?",
                    category=QuestionCategory.SCOPE,
                    importance=QuestionImportance.MEDIUM,
                    context="No explicit scope boundaries defined",
                )
            )

        # Sort by importance and limit
        questions.sort(
            key=lambda q: (
                0 if q.importance == QuestionImportance.HIGH else 1 if q.importance == QuestionImportance.MEDIUM else 2
            )
        )

        limited_questions = questions[: self.max_questions_per_round]

        self._logger.info("questions_generated", count=len(limited_questions))
        return limited_questions

    def _infer_category(self, section_heading: str) -> QuestionCategory:
        """Infer question category from section heading.

        Args:
            section_heading: Section heading text

        Returns:
            Inferred question category
        """
        heading_lower = section_heading.lower()

        if any(
            keyword in heading_lower
            for keyword in ["requirement", "feature", "functionality", "user"]
        ):
            return QuestionCategory.REQUIREMENTS
        elif any(
            keyword in heading_lower
            for keyword in ["technology", "architecture", "technical", "infrastructure"]
        ):
            return QuestionCategory.TECHNICAL
        elif any(keyword in heading_lower for keyword in ["scope", "boundary", "limit"]):
            return QuestionCategory.SCOPE
        elif any(
            keyword in heading_lower for keyword in ["priority", "timeline", "schedule"]
        ):
            return QuestionCategory.PRIORITY
        else:
            return QuestionCategory.REQUIREMENTS


class InterviewSession:
    """Manages a multi-round interview session.

    This class coordinates multiple rounds of questions and responses, tracks
    clarifications, and produces a final clarified specification.

    Attributes:
        spec: The specification being clarified
        orchestrator: The interview orchestrator
        max_rounds: Maximum interview rounds
    """

    def __init__(
        self,
        spec: SpecDocument,
        orchestrator: InterviewOrchestrator,
        max_rounds: int = 3,
    ) -> None:
        """Initialize an interview session.

        Args:
            spec: Specification document to clarify
            orchestrator: Interview orchestrator instance
            max_rounds: Maximum interview rounds
        """
        self.spec = spec
        self.orchestrator = orchestrator
        self.max_rounds = max_rounds
        self._rounds: list[InterviewRoundResult] = []
        self._clarifications: dict[str, str] = {}
        self._logger = structlog.get_logger(__name__)

    def start_interview(
        self, context: dict[str, Any] | None = None
    ) -> list[InterviewQuestion]:
        """Start the interview and generate initial questions.

        Args:
            context: Optional context for question generation

        Returns:
            Initial list of interview questions
        """
        self._logger.info("interview_starting", spec_title=self.spec.title)

        questions = self.orchestrator.generate_questions(self.spec, context)

        # Create first round result
        round_result = InterviewRoundResult(
            round_number=1, questions_asked=questions, responses=[], remaining_gaps=[]
        )
        self._rounds.append(round_result)

        return questions

    def process_responses(
        self, responses: list[InterviewResponse]
    ) -> InterviewRoundResult:
        """Process responses from the current round.

        Args:
            responses: List of responses to questions

        Returns:
            Result from processing this round
        """
        if not self._rounds:
            raise ValueError("Interview not started. Call start_interview() first.")

        current_round = self._rounds[-1]
        current_round.responses = responses

        self._logger.info(
            "processing_responses",
            round_number=current_round.round_number,
            response_count=len(responses),
        )

        # Extract clarifications from responses
        for response in responses:
            # Find the corresponding question
            question = next(
                (q for q in current_round.questions_asked if q.id == response.question_id),
                None,
            )
            if question:
                topic = question.category.value
                self._clarifications[topic] = response.answer
                current_round.clarifications_gained[topic] = response.answer

        # Track remaining gaps (responses that need follow-up)
        for response in responses:
            if response.follow_up_needed:
                question = next(
                    (
                        q
                        for q in current_round.questions_asked
                        if q.id == response.question_id
                    ),
                    None,
                )
                if question:
                    current_round.remaining_gaps.append(question.question)

        self._logger.info(
            "responses_processed",
            clarifications=len(current_round.clarifications_gained),
            remaining_gaps=len(current_round.remaining_gaps),
        )

        return current_round

    def generate_follow_ups(
        self, round_result: InterviewRoundResult
    ) -> list[InterviewQuestion]:
        """Generate follow-up questions based on previous round results.

        Args:
            round_result: Result from the previous round

        Returns:
            List of follow-up questions
        """
        if round_result.round_number >= self.max_rounds:
            self._logger.info("max_rounds_reached", max_rounds=self.max_rounds)
            return []

        follow_ups: list[InterviewQuestion] = []

        # Generate follow-ups for responses that need more detail
        for response in round_result.responses:
            if response.follow_up_needed:
                question = next(
                    (
                        q
                        for q in round_result.questions_asked
                        if q.id == response.question_id
                    ),
                    None,
                )
                if question:
                    follow_ups.append(
                        InterviewQuestion(
                            id=f"Q{len(follow_ups) + 1}",
                            question=f"Can you provide more detail about: {question.question}",
                            category=question.category,
                            importance=question.importance,
                            context=f"Follow-up to incomplete answer: {response.answer[:50]}...",
                        )
                    )

        # Limit follow-ups
        limited_follow_ups = follow_ups[: self.orchestrator.max_questions_per_round]

        if limited_follow_ups:
            # Create next round
            next_round = InterviewRoundResult(
                round_number=round_result.round_number + 1,
                questions_asked=limited_follow_ups,
                responses=[],
                remaining_gaps=round_result.remaining_gaps.copy(),
            )
            self._rounds.append(next_round)

        self._logger.info("follow_ups_generated", count=len(limited_follow_ups))
        return limited_follow_ups

    def finalize_interview(self) -> ClarifiedSpec:
        """Finalize the interview and produce clarified specification.

        Returns:
            ClarifiedSpec with all clarifications applied
        """
        self._logger.info("finalizing_interview", rounds_conducted=len(self._rounds))

        # Collect resolved ambiguities
        resolved: list[str] = []
        for round_result in self._rounds:
            for question in round_result.questions_asked:
                if question.id in [r.question_id for r in round_result.responses]:
                    resolved.append(question.question)

        # Collect remaining unknowns
        remaining_unknowns: list[str] = []
        if self._rounds:
            last_round = self._rounds[-1]
            remaining_unknowns = last_round.remaining_gaps.copy()

        clarified = ClarifiedSpec(
            original_spec=self.spec,
            clarifications=self._clarifications,
            resolved_ambiguities=resolved,
            remaining_unknowns=remaining_unknowns,
            interview_rounds=len(self._rounds),
        )

        self._logger.info(
            "interview_finalized",
            clarifications=len(clarified.clarifications),
            resolved=len(clarified.resolved_ambiguities),
            remaining=len(clarified.remaining_unknowns),
        )

        return clarified

    def get_current_round(self) -> InterviewRoundResult | None:
        """Get the current interview round result.

        Returns:
            Current round result or None if interview not started
        """
        return self._rounds[-1] if self._rounds else None
