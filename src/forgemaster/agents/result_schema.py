"""Agent result schema validation and parsing for Forgemaster.

This module defines Pydantic models for validating agent outputs and provides
parsing logic to extract structured results from agent responses. It handles
both complete and partial results, with graceful fallback for malformed outputs.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field, field_validator


class TestResult(BaseModel):
    """Individual test result from agent execution.

    Attributes:
        name: Test name or identifier
        status: Test status (passed, failed, skipped)
        duration_seconds: Test execution duration in seconds
        message: Optional message (error message for failures)
    """

    name: str = Field(..., description="Test name or identifier")
    status: str = Field(..., description="Test status: passed, failed, skipped")
    duration_seconds: float | None = Field(
        None, description="Test execution duration in seconds"
    )
    message: str | None = Field(None, description="Optional test message or error")

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        """Validate test status is recognized.

        Args:
            v: Status string to validate

        Returns:
            Lowercase status string

        Raises:
            ValueError: If status is not recognized
        """
        valid_statuses = {"passed", "failed", "skipped"}
        v_lower = v.lower()
        if v_lower not in valid_statuses:
            raise ValueError(f"Invalid test status: {v}. Must be one of {valid_statuses}")
        return v_lower


class IssueDiscovered(BaseModel):
    """Issue discovered by agent during task execution.

    Attributes:
        description: Description of the issue
        severity: Issue severity (critical, high, medium, low)
        location: Optional file/line location reference
        suggested_fix: Optional suggestion for fixing the issue
    """

    description: str = Field(..., description="Description of the issue")
    severity: str = Field(..., description="Issue severity: critical, high, medium, low")
    location: str | None = Field(None, description="Optional file:line location")
    suggested_fix: str | None = Field(None, description="Optional suggested fix")

    @field_validator("severity")
    @classmethod
    def validate_severity(cls, v: str) -> str:
        """Validate severity level is recognized.

        Args:
            v: Severity string to validate

        Returns:
            Lowercase severity string

        Raises:
            ValueError: If severity is not recognized
        """
        valid_severities = {"critical", "high", "medium", "low"}
        v_lower = v.lower()
        if v_lower not in valid_severities:
            raise ValueError(
                f"Invalid severity: {v}. Must be one of {valid_severities}"
            )
        return v_lower


class LessonLearned(BaseModel):
    """Lesson learned by agent during task execution.

    Attributes:
        context: Context or situation where lesson applies
        observation: What was observed or learned
        recommendation: Recommended action or best practice
    """

    context: str = Field(..., description="Context where lesson applies")
    observation: str = Field(..., description="What was observed or learned")
    recommendation: str = Field(..., description="Recommended action or best practice")


class AgentResult(BaseModel):
    """Complete agent execution result.

    This is the top-level schema for agent outputs, containing status,
    summary, detailed results, and metadata about the execution.

    Attributes:
        status: Execution status (success, partial, failed)
        summary: Brief summary of what was accomplished
        details: Detailed description of work performed
        tests_run: List of test results (if tests were executed)
        issues_discovered: List of issues discovered during execution
        lessons_learned: List of lessons learned during execution
        files_modified: List of file paths that were modified
        confidence_score: Agent's confidence in result quality (0.0-1.0)
    """

    status: str = Field(
        ..., description="Execution status: success, partial, failed"
    )
    summary: str = Field(..., description="Brief summary of accomplishments")
    details: str = Field(..., description="Detailed description of work performed")
    tests_run: list[TestResult] = Field(
        default_factory=list, description="Test results from execution"
    )
    issues_discovered: list[IssueDiscovered] = Field(
        default_factory=list, description="Issues discovered during execution"
    )
    lessons_learned: list[LessonLearned] = Field(
        default_factory=list, description="Lessons learned during execution"
    )
    files_modified: list[str] = Field(
        default_factory=list, description="File paths that were modified"
    )
    confidence_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Agent's confidence in result quality (0.0-1.0)",
    )

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        """Validate execution status is recognized.

        Args:
            v: Status string to validate

        Returns:
            Lowercase status string

        Raises:
            ValueError: If status is not recognized
        """
        valid_statuses = {"success", "partial", "failed"}
        v_lower = v.lower()
        if v_lower not in valid_statuses:
            raise ValueError(f"Invalid status: {v}. Must be one of {valid_statuses}")
        return v_lower


def parse_agent_result(raw_output: str) -> AgentResult:
    """Parse agent output and validate against result schema.

    This function attempts to extract JSON from agent output, which may be
    embedded in markdown code blocks or mixed with explanatory text. It
    validates the extracted JSON against the AgentResult schema.

    Args:
        raw_output: Raw text output from agent

    Returns:
        Validated AgentResult instance

    Raises:
        ValueError: If no valid JSON can be extracted
        ValidationError: If JSON doesn't match schema

    Example:
        >>> output = '''
        ... Here's my result:
        ... ```json
        ... {"status": "success", "summary": "Task complete", "details": "..."}
        ... ```
        ... '''
        >>> result = parse_agent_result(output)
        >>> print(result.status)
        'success'
    """
    # Try to extract JSON from the output
    json_str = _extract_json(raw_output)

    if json_str is None:
        raise ValueError("No JSON found in agent output")

    # Parse JSON
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in agent output: {e}") from e

    # Validate against schema
    return AgentResult.model_validate(data)


def parse_agent_result_safe(raw_output: str) -> AgentResult:
    """Parse agent output with fallback for malformed results.

    This function is a safe wrapper around parse_agent_result that returns
    a minimal valid result if parsing fails. This is useful for handling
    agent outputs that don't conform to the expected schema.

    Args:
        raw_output: Raw text output from agent

    Returns:
        AgentResult instance (may be fallback if parsing failed)

    Example:
        >>> output = "I completed the task but forgot to format as JSON"
        >>> result = parse_agent_result_safe(output)
        >>> print(result.status)
        'partial'
        >>> print(result.summary)
        'Agent output could not be parsed'
    """
    try:
        return parse_agent_result(raw_output)
    except Exception as e:
        # Return fallback result with error information
        return AgentResult(
            status="partial",
            summary="Agent output could not be parsed",
            details=f"Failed to parse agent output: {str(e)}\n\nRaw output:\n{raw_output[:500]}",
            confidence_score=0.0,
        )


def _extract_json(text: str) -> str | None:
    """Extract JSON from text that may contain markdown or other content.

    This function attempts multiple strategies to extract JSON:
    1. Look for JSON in markdown code blocks (```json...```)
    2. Look for JSON-like structures with curly braces
    3. Try to find the first complete JSON object

    Args:
        text: Text that may contain JSON

    Returns:
        Extracted JSON string or None if not found
    """
    # Strategy 1: Look for markdown code blocks with json tag
    markdown_pattern = r"```json\s*\n(.*?)\n```"
    markdown_match = re.search(markdown_pattern, text, re.DOTALL | re.IGNORECASE)
    if markdown_match:
        return markdown_match.group(1).strip()

    # Strategy 2: Look for any markdown code blocks
    code_block_pattern = r"```\s*\n(.*?)\n```"
    code_block_match = re.search(code_block_pattern, text, re.DOTALL)
    if code_block_match:
        potential_json = code_block_match.group(1).strip()
        if potential_json.startswith("{") and potential_json.endswith("}"):
            return potential_json

    # Strategy 3: Look for JSON-like structure with curly braces
    # Find the first { and last } that could be a complete JSON object
    first_brace = text.find("{")
    if first_brace == -1:
        return None

    # Find matching closing brace
    brace_count = 0
    in_string = False
    escape_next = False

    for i in range(first_brace, len(text)):
        char = text[i]

        if escape_next:
            escape_next = False
            continue

        if char == "\\":
            escape_next = True
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            continue

        if not in_string:
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    # Found complete JSON object
                    return text[first_brace : i + 1]

    return None


def create_minimal_result(
    status: str, summary: str, details: str = "", confidence: float = 1.0
) -> AgentResult:
    """Create a minimal valid AgentResult for testing or fallback.

    Args:
        status: Execution status (success, partial, failed)
        summary: Brief summary
        details: Optional detailed description
        confidence: Confidence score (0.0-1.0)

    Returns:
        Minimal AgentResult instance

    Example:
        >>> result = create_minimal_result("success", "Task completed")
        >>> print(result.status)
        'success'
    """
    return AgentResult(
        status=status,
        summary=summary,
        details=details or summary,
        confidence_score=confidence,
    )
