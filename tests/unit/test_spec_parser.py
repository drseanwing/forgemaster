"""Unit tests for specification parser."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from forgemaster.architecture.spec_parser import SpecParser, SpecDocument, ValidationResult


@pytest.fixture
def parser() -> SpecParser:
    """Create a SpecParser instance."""
    return SpecParser()


@pytest.fixture
def sample_markdown() -> str:
    """Sample markdown specification."""
    return """# FORGEMASTER Project

## Requirements

The system must handle autonomous development orchestration.

### Functional Requirements

- Parse specifications
- Generate plans
- Execute tasks

### Non-Functional Requirements

- Performance: < 100ms response time
- Scalability: Handle 1000+ concurrent tasks

## Technology

### Stack

Python 3.12+, SQLAlchemy 2.0, FastAPI

### Tools

```python
# Example configuration
config = {"database": "postgresql"}
```

## Architecture

### Components

| Component | Responsibility |
|-----------|---------------|
| Parser | Spec ingestion |
| Planner | Task planning |

- Agent orchestration
- Task execution
- Result aggregation
"""


@pytest.fixture
def sample_json() -> str:
    """Sample JSON specification."""
    return json.dumps({
        "title": "FORGEMASTER Project",
        "metadata": {
            "version": "1.0",
            "author": "System"
        },
        "requirements": {
            "functional": [
                "Parse specifications",
                "Generate plans",
                "Execute tasks"
            ],
            "non_functional": {
                "performance": "< 100ms response time",
                "scalability": "Handle 1000+ concurrent tasks"
            }
        },
        "technology": {
            "stack": "Python 3.12+, SQLAlchemy 2.0, FastAPI",
            "tools": ["pytest", "mypy", "ruff"]
        },
        "architecture": [
            "Agent orchestration",
            "Task execution",
            "Result aggregation"
        ]
    })


class TestMarkdownParsing:
    """Tests for markdown specification parsing."""

    def test_parse_simple_markdown(self, parser: SpecParser) -> None:
        """Test parsing simple markdown with headings."""
        content = """# My Project

## Introduction

This is a test project.

## Features

- Feature 1
- Feature 2
"""
        spec = parser.parse_markdown(content)

        assert spec.title == "My Project"
        assert len(spec.sections) == 2
        assert spec.sections[0].heading == "Introduction"
        assert spec.sections[0].level == 2
        assert "test project" in spec.sections[0].content
        assert spec.sections[1].heading == "Features"

    def test_parse_nested_headings(self, parser: SpecParser) -> None:
        """Test parsing markdown with nested heading hierarchy."""
        content = """# Project

## Section 1

Content 1

### Subsection 1.1

Content 1.1

### Subsection 1.2

Content 1.2

## Section 2

Content 2
"""
        spec = parser.parse_markdown(content)

        assert spec.title == "Project"
        assert len(spec.sections) == 2
        assert spec.sections[0].heading == "Section 1"
        assert len(spec.sections[0].subsections) == 2
        assert spec.sections[0].subsections[0].heading == "Subsection 1.1"
        assert spec.sections[0].subsections[0].level == 3

    def test_parse_code_blocks(self, parser: SpecParser) -> None:
        """Test parsing markdown with code blocks."""
        content = """# Project

## Code Example

```python
def hello():
    return "world"
```
"""
        spec = parser.parse_markdown(content)

        assert "```python" in spec.sections[0].content
        assert "def hello()" in spec.sections[0].content

    def test_parse_tables(self, parser: SpecParser) -> None:
        """Test parsing markdown with tables."""
        content = """# Project

## Components

| Name | Type |
|------|------|
| Parser | Core |
"""
        spec = parser.parse_markdown(content)

        assert "| Name | Type |" in spec.sections[0].content
        assert "Parser" in spec.sections[0].content

    def test_parse_bullet_lists(self, parser: SpecParser) -> None:
        """Test parsing markdown with bullet lists."""
        content = """# Project

## Features

- Feature A
- Feature B
  - Sub-feature B1
- Feature C
"""
        spec = parser.parse_markdown(content)

        assert "- Feature A" in spec.sections[0].content
        assert "- Feature B" in spec.sections[0].content
        assert "Sub-feature B1" in spec.sections[0].content

    def test_parse_empty_markdown(self, parser: SpecParser) -> None:
        """Test parsing empty markdown content."""
        spec = parser.parse_markdown("")

        assert spec.title == "Untitled"
        assert len(spec.sections) == 0
        assert spec.raw_content == ""

    def test_parse_markdown_without_title(self, parser: SpecParser) -> None:
        """Test parsing markdown without H1 heading."""
        content = """## Section 1

Content here.
"""
        spec = parser.parse_markdown(content)

        assert spec.title == "Untitled"
        assert len(spec.sections) == 1

    def test_full_markdown_sample(self, parser: SpecParser, sample_markdown: str) -> None:
        """Test parsing full markdown sample with complex structure."""
        spec = parser.parse_markdown(sample_markdown)

        assert spec.title == "FORGEMASTER Project"
        assert len(spec.sections) == 3
        assert spec.sections[0].heading == "Requirements"
        assert len(spec.sections[0].subsections) == 2
        assert spec.sections[1].heading == "Technology"
        assert spec.sections[2].heading == "Architecture"


class TestJSONParsing:
    """Tests for JSON specification parsing."""

    def test_parse_flat_json(self, parser: SpecParser) -> None:
        """Test parsing flat JSON structure."""
        content = json.dumps({
            "title": "Test Project",
            "description": "A test project",
            "version": "1.0.0"
        })

        spec = parser.parse_json(content)

        assert spec.title == "Test Project"
        assert len(spec.sections) == 2  # description, version (title excluded)
        assert any(s.heading == "Description" for s in spec.sections)

    def test_parse_nested_json(self, parser: SpecParser) -> None:
        """Test parsing nested JSON structure."""
        content = json.dumps({
            "title": "Project",
            "config": {
                "database": "postgresql",
                "cache": "redis"
            }
        })

        spec = parser.parse_json(content)

        assert spec.title == "Project"
        assert len(spec.sections) == 1
        assert spec.sections[0].heading == "Config"
        assert len(spec.sections[0].subsections) == 2

    def test_parse_json_with_arrays(self, parser: SpecParser) -> None:
        """Test parsing JSON with array values."""
        content = json.dumps({
            "title": "Project",
            "features": ["Feature 1", "Feature 2", "Feature 3"]
        })

        spec = parser.parse_json(content)

        assert spec.title == "Project"
        assert len(spec.sections) == 1
        assert spec.sections[0].heading == "Features"
        assert "- Feature 1" in spec.sections[0].content
        assert "- Feature 2" in spec.sections[0].content

    def test_parse_json_with_metadata(self, parser: SpecParser) -> None:
        """Test parsing JSON with metadata field."""
        content = json.dumps({
            "title": "Project",
            "metadata": {
                "author": "Test",
                "version": "1.0"
            },
            "description": "Test description"
        })

        spec = parser.parse_json(content)

        assert spec.title == "Project"
        assert spec.metadata == {"author": "Test", "version": "1.0"}
        assert len(spec.sections) == 1

    def test_parse_invalid_json(self, parser: SpecParser) -> None:
        """Test parsing invalid JSON raises error."""
        with pytest.raises(json.JSONDecodeError):
            parser.parse_json("{invalid json")

    def test_parse_json_non_dict_root(self, parser: SpecParser) -> None:
        """Test parsing JSON with non-dict root raises error."""
        with pytest.raises(ValueError, match="JSON root must be a dictionary"):
            parser.parse_json("[]")

    def test_full_json_sample(self, parser: SpecParser, sample_json: str) -> None:
        """Test parsing full JSON sample with complex structure."""
        spec = parser.parse_json(sample_json)

        assert spec.title == "FORGEMASTER Project"
        assert spec.metadata == {"version": "1.0", "author": "System"}
        assert len(spec.sections) == 3

        # Check requirements section
        req_section = next(s for s in spec.sections if s.heading == "Requirements")
        assert len(req_section.subsections) == 2

        # Check technology section
        tech_section = next(s for s in spec.sections if s.heading == "Technology")
        assert len(tech_section.subsections) == 2


class TestValidation:
    """Tests for specification validation."""

    def test_validate_valid_spec(self, parser: SpecParser, sample_markdown: str) -> None:
        """Test validation of valid specification."""
        spec = parser.parse_markdown(sample_markdown)
        result = parser.validate_spec(spec)

        assert result.is_valid
        assert len(result.errors) == 0
        assert len(result.missing_sections) == 0

    def test_validate_missing_required_sections(self, parser: SpecParser) -> None:
        """Test validation detects missing required sections."""
        content = """# Project

## Introduction

Just an intro.
"""
        spec = parser.parse_markdown(content)
        result = parser.validate_spec(spec)

        assert not result.is_valid
        assert len(result.errors) > 0
        assert len(result.missing_sections) == 3
        assert "requirements" in result.missing_sections
        assert "technology" in result.missing_sections
        assert "architecture" in result.missing_sections

    def test_validate_partial_spec(self, parser: SpecParser) -> None:
        """Test validation of spec with some required sections."""
        content = """# Project

## Requirements

Some requirements here.

## Introduction

Not a required section.
"""
        spec = parser.parse_markdown(content)
        result = parser.validate_spec(spec)

        assert not result.is_valid
        assert "technology" in result.missing_sections
        assert "architecture" in result.missing_sections
        assert "requirements" not in result.missing_sections

    def test_validate_empty_sections_warning(self, parser: SpecParser) -> None:
        """Test validation warns about empty sections."""
        content = """# Project

## Requirements

## Technology

Some tech stack.

## Architecture
"""
        spec = parser.parse_markdown(content)
        result = parser.validate_spec(spec)

        assert len(result.warnings) >= 1
        assert any("Empty section" in w for w in result.warnings)

    def test_validate_minimal_content_warning(self, parser: SpecParser) -> None:
        """Test validation warns about minimal content."""
        content = """# Project

## Requirements

Short.

## Technology

Tech.

## Architecture

Arch.
"""
        spec = parser.parse_markdown(content)
        result = parser.validate_spec(spec)

        assert len(result.warnings) >= 1
        assert any("minimal content" in w for w in result.warnings)

    def test_validate_empty_spec(self, parser: SpecParser) -> None:
        """Test validation of empty specification."""
        spec = parser.parse_markdown("")
        result = parser.validate_spec(spec)

        assert not result.is_valid
        assert "no sections" in result.errors[0]
        assert len(result.missing_sections) == 3

    def test_validate_case_insensitive_sections(self, parser: SpecParser) -> None:
        """Test validation recognizes sections case-insensitively."""
        content = """# Project

## REQUIREMENTS

Requirements here.

## Technology Stack

Tech here.

## System Architecture

Architecture here.
"""
        spec = parser.parse_markdown(content)
        result = parser.validate_spec(spec)

        assert result.is_valid
        assert len(result.missing_sections) == 0


class TestFileOperations:
    """Tests for file parsing operations."""

    def test_parse_markdown_file(
        self, parser: SpecParser, tmp_path: Path, sample_markdown: str
    ) -> None:
        """Test parsing markdown file."""
        file_path = tmp_path / "spec.md"
        file_path.write_text(sample_markdown, encoding="utf-8")

        spec = parser.parse_file(file_path)

        assert spec.title == "FORGEMASTER Project"
        assert len(spec.sections) == 3

    def test_parse_json_file(
        self, parser: SpecParser, tmp_path: Path, sample_json: str
    ) -> None:
        """Test parsing JSON file."""
        file_path = tmp_path / "spec.json"
        file_path.write_text(sample_json, encoding="utf-8")

        spec = parser.parse_file(file_path)

        assert spec.title == "FORGEMASTER Project"
        assert len(spec.sections) == 3

    def test_parse_file_auto_detect_markdown(
        self, parser: SpecParser, tmp_path: Path
    ) -> None:
        """Test auto-detection of markdown format."""
        file_path = tmp_path / "spec.markdown"
        file_path.write_text("# Test\n\n## Section\n\nContent", encoding="utf-8")

        spec = parser.parse_file(file_path)

        assert spec.title == "Test"

    def test_parse_file_not_found(self, parser: SpecParser, tmp_path: Path) -> None:
        """Test parsing non-existent file raises error."""
        file_path = tmp_path / "nonexistent.md"

        with pytest.raises(FileNotFoundError):
            parser.parse_file(file_path)

    def test_parse_file_unsupported_format(
        self, parser: SpecParser, tmp_path: Path
    ) -> None:
        """Test parsing unsupported file format raises error."""
        file_path = tmp_path / "spec.txt"
        file_path.write_text("Some content", encoding="utf-8")

        with pytest.raises(ValueError, match="Unsupported specification format"):
            parser.parse_file(file_path)

    def test_parse_file_malformed_json(
        self, parser: SpecParser, tmp_path: Path
    ) -> None:
        """Test parsing malformed JSON file raises error."""
        file_path = tmp_path / "spec.json"
        file_path.write_text("{invalid json content}", encoding="utf-8")

        with pytest.raises(json.JSONDecodeError):
            parser.parse_file(file_path)


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_parse_markdown_with_special_characters(self, parser: SpecParser) -> None:
        """Test parsing markdown with special characters."""
        content = """# Project™

## Requirements & Goals

- Goal #1: Parse **all** formats
- Goal #2: Handle _edge_ cases
"""
        spec = parser.parse_markdown(content)

        assert spec.title == "Project™"
        assert spec.sections[0].heading == "Requirements & Goals"

    def test_parse_very_deep_nesting(self, parser: SpecParser) -> None:
        """Test parsing deeply nested heading hierarchy."""
        content = """# L1
## L2
### L3
#### L4
##### L5
###### L6
Content at max depth.
"""
        spec = parser.parse_markdown(content)

        # Navigate to deepest level
        current = spec.sections[0]
        depth = 1
        while current.subsections:
            current = current.subsections[0]
            depth += 1

        assert depth == 5  # L2 through L6
        assert current.level == 6

    def test_parse_json_with_empty_values(self, parser: SpecParser) -> None:
        """Test parsing JSON with empty values."""
        content = json.dumps({
            "title": "Project",
            "description": "",
            "features": [],
            "config": {}
        })

        spec = parser.parse_json(content)

        assert spec.title == "Project"
        # Empty values still create sections
        assert len(spec.sections) >= 1

    def test_parse_markdown_multiple_h1(self, parser: SpecParser) -> None:
        """Test parsing markdown with multiple H1 headings."""
        content = """# First Title

Content 1

# Second Title

Content 2
"""
        spec = parser.parse_markdown(content)

        # First H1 becomes title
        assert spec.title == "First Title"
        # Second H1 becomes a section
        assert len(spec.sections) == 1

    def test_validate_subsection_content(self, parser: SpecParser) -> None:
        """Test validation doesn't flag empty parent sections with subsections."""
        content = """# Project

## Requirements

### Functional

Functional requirements here.

### Non-Functional

Non-functional requirements here.

## Technology

Tech stack.

## Architecture

Architecture details.
"""
        spec = parser.parse_markdown(content)
        result = parser.validate_spec(spec)

        # Parent "Requirements" section is empty but has subsections - should be OK
        assert result.is_valid
        # Should not warn about Requirements being empty
        assert not any("Requirements" in w and "Empty" in w for w in result.warnings)
