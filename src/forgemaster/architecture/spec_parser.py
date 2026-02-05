"""Specification parser for FORGEMASTER.

Handles ingestion, parsing, and validation of project specifications in multiple formats.
Supports Markdown (headings, code blocks, tables, lists) and JSON (flat/nested structures).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class SpecSection(BaseModel):
    """A section within a specification document.

    Attributes:
        heading: Section heading text
        level: Heading level (1-6 for markdown H1-H6)
        content: Text content of the section
        subsections: Nested subsections
    """

    heading: str
    level: int = Field(ge=1, le=6)
    content: str = ""
    subsections: list[SpecSection] = Field(default_factory=list)


class SpecDocument(BaseModel):
    """Structured representation of a specification document.

    Attributes:
        title: Document title (typically first H1 heading)
        sections: Top-level sections
        metadata: Additional metadata (e.g., author, version)
        raw_content: Original unparsed content
    """

    title: str
    sections: list[SpecSection] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    raw_content: str = ""


class ValidationResult(BaseModel):
    """Result of specification validation.

    Attributes:
        is_valid: Whether the specification is valid
        warnings: Non-critical issues
        errors: Critical validation errors
        missing_sections: Required sections that are missing
    """

    is_valid: bool
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    missing_sections: list[str] = Field(default_factory=list)


class SpecParser:
    """Parser for project specification documents.

    Handles multiple input formats (Markdown, JSON) and provides validation
    to ensure specifications contain required information.
    """

    REQUIRED_SECTIONS = {"requirements", "technology", "architecture"}
    MIN_CONTENT_LENGTH = 10

    def parse_markdown(self, content: str) -> SpecDocument:
        """Parse markdown specification into structured format.

        Handles heading hierarchy (H1-H6), code blocks, tables, and bullet lists.
        Uses regex-based parsing without external markdown libraries.

        Args:
            content: Raw markdown content

        Returns:
            Structured SpecDocument

        Example:
            ```python
            parser = SpecParser()
            spec = parser.parse_markdown("# My Project\\n\\n## Requirements\\n...")
            ```
        """
        logger.info("parsing_markdown_spec", content_length=len(content))

        if not content.strip():
            logger.warning("empty_markdown_content")
            return SpecDocument(title="Untitled", raw_content=content)

        # Find all code block ranges to exclude from heading detection
        code_block_pattern = re.compile(r"```[\s\S]*?```", re.MULTILINE)
        code_blocks = [(m.start(), m.end()) for m in code_block_pattern.finditer(content)]

        def _is_in_code_block(pos: int) -> bool:
            """Check if position is inside a code block."""
            return any(start <= pos < end for start, end in code_blocks)

        # Parse all headings with regex
        heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
        all_matches = list(heading_pattern.finditer(content))

        # Filter out headings inside code blocks
        matches = [m for m in all_matches if not _is_in_code_block(m.start())]

        # Extract title (first H1 heading not in code block)
        title = "Untitled"
        for match in matches:
            if len(match.group(1)) == 1:
                title = match.group(2).strip()
                break

        sections: list[SpecSection] = []
        section_stack: list[tuple[int, SpecSection]] = []

        # Skip first H1 if it matches the title
        start_idx = 0
        if matches and len(matches[0].group(1)) == 1 and matches[0].group(2).strip() == title:
            start_idx = 1

        for i in range(start_idx, len(matches)):
            match = matches[i]
            level = len(match.group(1))
            heading_text = match.group(2).strip()

            # Extract content between this heading and the next
            start_pos = match.end()
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            section_content = content[start_pos:end_pos].strip()

            section = SpecSection(
                heading=heading_text,
                level=level,
                content=section_content,
            )

            # Build hierarchy based on heading levels
            while section_stack and section_stack[-1][0] >= level:
                section_stack.pop()

            if section_stack:
                parent_section = section_stack[-1][1]
                parent_section.subsections.append(section)
            else:
                sections.append(section)

            section_stack.append((level, section))

        logger.info("markdown_parsed", sections_count=len(sections), title=title)

        return SpecDocument(
            title=title,
            sections=sections,
            raw_content=content,
        )

    def parse_json(self, content: str) -> SpecDocument:
        """Parse JSON specification into structured format.

        Supports both flat and nested JSON structures. Maps JSON keys to SpecSection
        fields, handling arrays and object values appropriately.

        Args:
            content: Raw JSON content

        Returns:
            Structured SpecDocument

        Raises:
            json.JSONDecodeError: If JSON is malformed

        Example:
            ```python
            parser = SpecParser()
            spec = parser.parse_json('{"title": "My Project", "requirements": [...]}')
            ```
        """
        logger.info("parsing_json_spec", content_length=len(content))

        data = json.loads(content)

        if not isinstance(data, dict):
            logger.error("json_root_not_dict", type=type(data).__name__)
            raise ValueError("JSON root must be a dictionary")

        title = data.get("title", "Untitled")
        metadata = data.get("metadata", {})

        sections: list[SpecSection] = []

        def _json_to_sections(obj: dict[str, Any], level: int = 1) -> list[SpecSection]:
            """Recursively convert JSON object to sections."""
            result: list[SpecSection] = []

            for key, value in obj.items():
                if key in ("title", "metadata"):
                    continue

                if isinstance(value, dict):
                    # Nested object becomes section with subsections
                    subsections = _json_to_sections(value, level + 1)
                    section = SpecSection(
                        heading=key.replace("_", " ").title(),
                        level=level,
                        content="",
                        subsections=subsections,
                    )
                    result.append(section)
                elif isinstance(value, list):
                    # Array becomes section with bullet points
                    content_lines = [f"- {item}" for item in value]
                    section = SpecSection(
                        heading=key.replace("_", " ").title(),
                        level=level,
                        content="\n".join(content_lines),
                    )
                    result.append(section)
                else:
                    # Scalar value becomes section with content
                    section = SpecSection(
                        heading=key.replace("_", " ").title(),
                        level=level,
                        content=str(value),
                    )
                    result.append(section)

            return result

        sections = _json_to_sections(data)

        logger.info("json_parsed", sections_count=len(sections), title=title)

        return SpecDocument(
            title=title,
            sections=sections,
            metadata=metadata,
            raw_content=content,
        )

    def validate_spec(self, spec: SpecDocument) -> ValidationResult:
        """Validate specification completeness.

        Checks for required sections, empty sections, and minimum content length.

        Args:
            spec: Specification document to validate

        Returns:
            ValidationResult with errors, warnings, and missing sections

        Example:
            ```python
            parser = SpecParser()
            spec = parser.parse_markdown(content)
            result = parser.validate_spec(spec)
            if not result.is_valid:
                print(f"Errors: {result.errors}")
            ```
        """
        logger.info("validating_spec", title=spec.title, sections_count=len(spec.sections))

        warnings: list[str] = []
        errors: list[str] = []
        missing_sections: list[str] = []

        # Check for empty document
        if not spec.sections:
            errors.append("Specification has no sections")
            logger.warning("spec_no_sections", title=spec.title)
            return ValidationResult(
                is_valid=False,
                errors=errors,
                warnings=warnings,
                missing_sections=list(self.REQUIRED_SECTIONS),
            )

        # Build set of all section headings (case-insensitive)
        def _collect_headings(sections: list[SpecSection]) -> set[str]:
            headings: set[str] = set()
            for section in sections:
                headings.add(section.heading.lower())
                headings.update(_collect_headings(section.subsections))
            return headings

        present_headings = _collect_headings(spec.sections)

        # Check for required sections
        for required in self.REQUIRED_SECTIONS:
            if not any(required in heading for heading in present_headings):
                missing_sections.append(required)
                errors.append(f"Missing required section: {required}")

        # Check for empty sections
        def _check_empty_sections(sections: list[SpecSection], path: str = "") -> None:
            for section in sections:
                section_path = f"{path}/{section.heading}" if path else section.heading
                if not section.content.strip() and not section.subsections:
                    warnings.append(f"Empty section: {section_path}")
                elif len(section.content.strip()) < self.MIN_CONTENT_LENGTH:
                    if section.subsections:
                        # Has subsections, so empty content is acceptable
                        pass
                    else:
                        warnings.append(
                            f"Section has minimal content ({len(section.content)} chars): "
                            f"{section_path}"
                        )
                _check_empty_sections(section.subsections, section_path)

        _check_empty_sections(spec.sections)

        is_valid = len(errors) == 0

        logger.info(
            "validation_complete",
            is_valid=is_valid,
            errors_count=len(errors),
            warnings_count=len(warnings),
        )

        return ValidationResult(
            is_valid=is_valid,
            warnings=warnings,
            errors=errors,
            missing_sections=missing_sections,
        )

    def parse_file(self, file_path: Path) -> SpecDocument:
        """Parse specification file with auto-detection of format.

        Detects format based on file extension (.md, .json) and delegates
        to appropriate parser.

        Args:
            file_path: Path to specification file

        Returns:
            Parsed SpecDocument

        Raises:
            ValueError: If file format is unsupported
            FileNotFoundError: If file does not exist
            json.JSONDecodeError: If JSON is malformed

        Example:
            ```python
            parser = SpecParser()
            spec = parser.parse_file(Path("specification.md"))
            ```
        """
        logger.info("parsing_file", file_path=str(file_path))

        if not file_path.exists():
            logger.error("file_not_found", file_path=str(file_path))
            raise FileNotFoundError(f"Specification file not found: {file_path}")

        content = file_path.read_text(encoding="utf-8")
        extension = file_path.suffix.lower()

        if extension in (".md", ".markdown"):
            return self.parse_markdown(content)
        elif extension == ".json":
            return self.parse_json(content)
        else:
            logger.error("unsupported_format", extension=extension)
            raise ValueError(f"Unsupported specification format: {extension}")
