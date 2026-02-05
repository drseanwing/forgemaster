"""Unit tests for context search service.

Tests the context search module which provides semantic, keyword, and file
overlap search strategies for retrieving relevant lessons learned.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from forgemaster.intelligence.context_search import (
    DEFAULT_WEIGHTS,
    VERIFICATION_BONUS,
    ContextSearchService,
    SearchQuery,
    SearchResult,
    SearchStrategy,
    _lesson_to_result,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_embedding_service() -> AsyncMock:
    """Create a mock embedding service."""
    mock = AsyncMock()
    mock.generate = AsyncMock(return_value=[0.1, 0.2, 0.3] * 256)  # 768-dim vector
    return mock


@pytest.fixture
def mock_session_factory() -> MagicMock:
    """Create a mock session factory that produces async context manager sessions."""
    mock_session = AsyncMock()
    mock_session.execute = AsyncMock()

    factory = MagicMock()
    factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
    factory.return_value.__aexit__ = AsyncMock(return_value=False)

    return factory


@pytest.fixture
def sample_lesson() -> Any:
    """Create a sample LessonLearned mock object."""
    lesson = MagicMock()
    lesson.id = uuid.uuid4()
    lesson.symptom = "Import error in module initialization"
    lesson.root_cause = "Circular import between modules"
    lesson.fix_applied = "Refactored to use lazy imports"
    lesson.files_affected = ["src/app/main.py", "src/app/utils.py"]
    lesson.pattern_tags = ["import-error", "circular-dependency"]
    lesson.verification_status = "verified"
    lesson.confidence_score = 0.85
    lesson.created_at = datetime.now(timezone.utc)
    return lesson


@pytest.fixture
def sample_unverified_lesson() -> Any:
    """Create a sample unverified LessonLearned mock object."""
    lesson = MagicMock()
    lesson.id = uuid.uuid4()
    lesson.symptom = "Build failure"
    lesson.root_cause = "Missing dependency"
    lesson.fix_applied = "Added package to requirements.txt"
    lesson.files_affected = ["requirements.txt"]
    lesson.pattern_tags = ["build-error"]
    lesson.verification_status = "unverified"
    lesson.confidence_score = 0.60
    lesson.created_at = datetime.now(timezone.utc)
    return lesson


# =============================================================================
# Enum Tests
# =============================================================================


def test_search_strategy_enum_values() -> None:
    """Test SearchStrategy enum has expected values."""
    assert SearchStrategy.SEMANTIC.value == "semantic"
    assert SearchStrategy.KEYWORD.value == "keyword"
    assert SearchStrategy.FILE_OVERLAP.value == "file_overlap"
    assert SearchStrategy.COMBINED.value == "combined"


def test_search_strategy_enum_count() -> None:
    """Test SearchStrategy enum has exactly 4 values."""
    strategies = list(SearchStrategy)
    assert len(strategies) == 4


# =============================================================================
# Model Tests
# =============================================================================


def test_search_result_construction() -> None:
    """Test SearchResult model construction with all fields."""
    result = SearchResult(
        lesson_id="123e4567-e89b-12d3-a456-426614174000",
        symptom="Test symptom",
        root_cause="Test root cause",
        fix_applied="Test fix",
        files_affected=["file1.py", "file2.py"],
        pattern_tags=["tag1", "tag2"],
        score=0.85,
        strategy=SearchStrategy.SEMANTIC,
        verification_status="verified",
    )

    assert result.lesson_id == "123e4567-e89b-12d3-a456-426614174000"
    assert result.symptom == "Test symptom"
    assert result.root_cause == "Test root cause"
    assert result.fix_applied == "Test fix"
    assert result.files_affected == ["file1.py", "file2.py"]
    assert result.pattern_tags == ["tag1", "tag2"]
    assert result.score == 0.85
    assert result.strategy == SearchStrategy.SEMANTIC
    assert result.verification_status == "verified"


def test_search_result_default_lists() -> None:
    """Test SearchResult defaults for lists."""
    result = SearchResult(
        lesson_id="123e4567-e89b-12d3-a456-426614174000",
        symptom="Test",
        root_cause="Test",
        fix_applied="Test",
        score=0.5,
        strategy=SearchStrategy.KEYWORD,
        verification_status="unverified",
    )

    assert result.files_affected == []
    assert result.pattern_tags == []


def test_search_result_score_validation() -> None:
    """Test SearchResult score validation enforces 0-1 range."""
    # Valid scores
    SearchResult(
        lesson_id="123",
        symptom="Test",
        root_cause="Test",
        fix_applied="Test",
        score=0.0,
        strategy=SearchStrategy.SEMANTIC,
        verification_status="verified",
    )

    SearchResult(
        lesson_id="123",
        symptom="Test",
        root_cause="Test",
        fix_applied="Test",
        score=1.0,
        strategy=SearchStrategy.SEMANTIC,
        verification_status="verified",
    )

    # Invalid scores should raise validation error
    with pytest.raises(Exception):  # pydantic.ValidationError
        SearchResult(
            lesson_id="123",
            symptom="Test",
            root_cause="Test",
            fix_applied="Test",
            score=1.5,
            strategy=SearchStrategy.SEMANTIC,
            verification_status="verified",
        )


def test_search_query_defaults() -> None:
    """Test SearchQuery default values."""
    query = SearchQuery()

    assert query.text is None
    assert query.files == []
    assert query.pattern_tags == []
    assert query.max_results == 10
    assert query.min_score == 0.3
    assert query.strategies == [SearchStrategy.COMBINED]
    assert query.include_unverified is True


def test_search_query_custom_values() -> None:
    """Test SearchQuery with custom values."""
    query = SearchQuery(
        text="test query",
        files=["file1.py"],
        pattern_tags=["tag1"],
        max_results=5,
        min_score=0.5,
        strategies=[SearchStrategy.SEMANTIC, SearchStrategy.KEYWORD],
        include_unverified=False,
    )

    assert query.text == "test query"
    assert query.files == ["file1.py"]
    assert query.pattern_tags == ["tag1"]
    assert query.max_results == 5
    assert query.min_score == 0.5
    assert query.strategies == [SearchStrategy.SEMANTIC, SearchStrategy.KEYWORD]
    assert query.include_unverified is False


def test_search_query_validation() -> None:
    """Test SearchQuery field validation."""
    # max_results must be between 1 and 100
    with pytest.raises(Exception):  # pydantic.ValidationError
        SearchQuery(max_results=0)

    with pytest.raises(Exception):  # pydantic.ValidationError
        SearchQuery(max_results=101)

    # min_score must be between 0 and 1
    with pytest.raises(Exception):  # pydantic.ValidationError
        SearchQuery(min_score=-0.1)

    with pytest.raises(Exception):  # pydantic.ValidationError
        SearchQuery(min_score=1.1)


# =============================================================================
# Helper Function Tests
# =============================================================================


def test_lesson_to_result_basic(sample_lesson: Any) -> None:
    """Test _lesson_to_result converts ORM instance correctly."""
    result = _lesson_to_result(sample_lesson, 0.75, SearchStrategy.SEMANTIC)

    assert result.lesson_id == str(sample_lesson.id)
    assert result.symptom == sample_lesson.symptom
    assert result.root_cause == sample_lesson.root_cause
    assert result.fix_applied == sample_lesson.fix_applied
    assert result.files_affected == sample_lesson.files_affected
    assert result.pattern_tags == sample_lesson.pattern_tags
    assert result.score == 0.75
    assert result.strategy == SearchStrategy.SEMANTIC
    assert result.verification_status == sample_lesson.verification_status


def test_lesson_to_result_score_clamping() -> None:
    """Test _lesson_to_result clamps scores to [0, 1] range."""
    lesson = MagicMock()
    lesson.id = uuid.uuid4()
    lesson.symptom = "Test"
    lesson.root_cause = "Test"
    lesson.fix_applied = "Test"
    lesson.files_affected = []
    lesson.pattern_tags = []
    lesson.verification_status = "verified"

    # Score above 1.0 should be clamped
    result = _lesson_to_result(lesson, 1.5, SearchStrategy.SEMANTIC)
    assert result.score == 1.0

    # Score below 0.0 should be clamped
    result = _lesson_to_result(lesson, -0.5, SearchStrategy.SEMANTIC)
    assert result.score == 0.0


def test_lesson_to_result_none_lists(sample_lesson: Any) -> None:
    """Test _lesson_to_result handles None for list fields."""
    sample_lesson.files_affected = None
    sample_lesson.pattern_tags = None

    result = _lesson_to_result(sample_lesson, 0.5, SearchStrategy.KEYWORD)

    assert result.files_affected == []
    assert result.pattern_tags == []


# =============================================================================
# Service Initialization Tests
# =============================================================================


def test_service_init_with_embedding(
    mock_session_factory: MagicMock,
    mock_embedding_service: AsyncMock,
) -> None:
    """Test service initialization with embedding service."""
    service = ContextSearchService(
        session_factory=mock_session_factory,
        embedding_service=mock_embedding_service,
    )

    assert service.session_factory is mock_session_factory
    assert service.embedding_service is mock_embedding_service


def test_service_init_without_embedding(mock_session_factory: MagicMock) -> None:
    """Test service initialization without embedding service."""
    service = ContextSearchService(
        session_factory=mock_session_factory,
        embedding_service=None,
    )

    assert service.session_factory is mock_session_factory
    assert service.embedding_service is None


# =============================================================================
# Search Dispatch Tests
# =============================================================================


@pytest.mark.asyncio
async def test_search_combined_expands_strategies(
    mock_session_factory: MagicMock,
    mock_embedding_service: AsyncMock,
) -> None:
    """Test COMBINED strategy expands to all three individual strategies."""
    service = ContextSearchService(
        session_factory=mock_session_factory,
        embedding_service=mock_embedding_service,
    )

    # Mock _run_strategy to track calls
    called_strategies: list[SearchStrategy] = []

    async def track_strategy(strategy: SearchStrategy, query: SearchQuery) -> list[SearchResult]:
        called_strategies.append(strategy)
        return []

    service._run_strategy = track_strategy  # type: ignore

    query = SearchQuery(text="test", strategies=[SearchStrategy.COMBINED])
    await service.search(query)

    assert SearchStrategy.SEMANTIC in called_strategies
    assert SearchStrategy.KEYWORD in called_strategies
    assert SearchStrategy.FILE_OVERLAP in called_strategies
    assert SearchStrategy.COMBINED not in called_strategies


@pytest.mark.asyncio
async def test_search_individual_strategies(
    mock_session_factory: MagicMock,
    mock_embedding_service: AsyncMock,
) -> None:
    """Test individual strategies are called correctly."""
    service = ContextSearchService(
        session_factory=mock_session_factory,
        embedding_service=mock_embedding_service,
    )

    called_strategies: list[SearchStrategy] = []

    async def track_strategy(strategy: SearchStrategy, query: SearchQuery) -> list[SearchResult]:
        called_strategies.append(strategy)
        return []

    service._run_strategy = track_strategy  # type: ignore

    query = SearchQuery(
        text="test",
        strategies=[SearchStrategy.SEMANTIC, SearchStrategy.KEYWORD],
    )
    await service.search(query)

    assert called_strategies == [SearchStrategy.SEMANTIC, SearchStrategy.KEYWORD]


@pytest.mark.asyncio
async def test_search_empty_results(
    mock_session_factory: MagicMock,
    mock_embedding_service: AsyncMock,
) -> None:
    """Test search returns empty list when no results found."""
    service = ContextSearchService(
        session_factory=mock_session_factory,
        embedding_service=mock_embedding_service,
    )

    async def return_empty(strategy: SearchStrategy, query: SearchQuery) -> list[SearchResult]:
        return []

    service._run_strategy = return_empty  # type: ignore

    query = SearchQuery(text="test", strategies=[SearchStrategy.SEMANTIC])
    results = await service.search(query)

    assert results == []


# =============================================================================
# Strategy Runner Tests
# =============================================================================


@pytest.mark.asyncio
async def test_run_strategy_semantic_with_text_and_service(
    mock_session_factory: MagicMock,
    mock_embedding_service: AsyncMock,
) -> None:
    """Test _run_strategy dispatches to semantic_search when conditions met."""
    service = ContextSearchService(
        session_factory=mock_session_factory,
        embedding_service=mock_embedding_service,
    )

    # Mock semantic_search
    expected_results = [
        SearchResult(
            lesson_id="123",
            symptom="Test",
            root_cause="Test",
            fix_applied="Test",
            score=0.8,
            strategy=SearchStrategy.SEMANTIC,
            verification_status="verified",
        )
    ]
    service.semantic_search = AsyncMock(return_value=expected_results)  # type: ignore

    query = SearchQuery(text="test query", max_results=5, min_score=0.3)
    results = await service._run_strategy(SearchStrategy.SEMANTIC, query)

    assert results == expected_results
    service.semantic_search.assert_called_once_with(  # type: ignore
        text="test query",
        max_results=5,
        min_score=0.3,
    )


@pytest.mark.asyncio
async def test_run_strategy_semantic_without_text(
    mock_session_factory: MagicMock,
    mock_embedding_service: AsyncMock,
) -> None:
    """Test _run_strategy returns empty for semantic without text."""
    service = ContextSearchService(
        session_factory=mock_session_factory,
        embedding_service=mock_embedding_service,
    )

    query = SearchQuery(text=None)
    results = await service._run_strategy(SearchStrategy.SEMANTIC, query)

    assert results == []


@pytest.mark.asyncio
async def test_run_strategy_semantic_without_service(
    mock_session_factory: MagicMock,
) -> None:
    """Test _run_strategy returns empty for semantic without embedding service."""
    service = ContextSearchService(
        session_factory=mock_session_factory,
        embedding_service=None,
    )

    query = SearchQuery(text="test")
    results = await service._run_strategy(SearchStrategy.SEMANTIC, query)

    assert results == []


@pytest.mark.asyncio
async def test_run_strategy_keyword_with_text(
    mock_session_factory: MagicMock,
) -> None:
    """Test _run_strategy dispatches to keyword_search when text provided."""
    service = ContextSearchService(
        session_factory=mock_session_factory,
        embedding_service=None,
    )

    expected_results = [
        SearchResult(
            lesson_id="123",
            symptom="Test",
            root_cause="Test",
            fix_applied="Test",
            score=0.7,
            strategy=SearchStrategy.KEYWORD,
            verification_status="verified",
        )
    ]
    service.keyword_search = AsyncMock(return_value=expected_results)  # type: ignore

    query = SearchQuery(text="test query", max_results=5)
    results = await service._run_strategy(SearchStrategy.KEYWORD, query)

    assert results == expected_results
    service.keyword_search.assert_called_once_with(  # type: ignore
        text="test query",
        max_results=5,
    )


@pytest.mark.asyncio
async def test_run_strategy_keyword_without_text(
    mock_session_factory: MagicMock,
) -> None:
    """Test _run_strategy returns empty for keyword without text."""
    service = ContextSearchService(
        session_factory=mock_session_factory,
        embedding_service=None,
    )

    query = SearchQuery(text=None)
    results = await service._run_strategy(SearchStrategy.KEYWORD, query)

    assert results == []


@pytest.mark.asyncio
async def test_run_strategy_file_overlap_with_files(
    mock_session_factory: MagicMock,
) -> None:
    """Test _run_strategy dispatches to file_overlap_search when files provided."""
    service = ContextSearchService(
        session_factory=mock_session_factory,
        embedding_service=None,
    )

    expected_results = [
        SearchResult(
            lesson_id="123",
            symptom="Test",
            root_cause="Test",
            fix_applied="Test",
            score=0.6,
            strategy=SearchStrategy.FILE_OVERLAP,
            verification_status="verified",
        )
    ]
    service.file_overlap_search = AsyncMock(return_value=expected_results)  # type: ignore

    query = SearchQuery(files=["file1.py", "file2.py"], max_results=5)
    results = await service._run_strategy(SearchStrategy.FILE_OVERLAP, query)

    assert results == expected_results
    service.file_overlap_search.assert_called_once_with(  # type: ignore
        files=["file1.py", "file2.py"],
        max_results=5,
    )


@pytest.mark.asyncio
async def test_run_strategy_file_overlap_without_files(
    mock_session_factory: MagicMock,
) -> None:
    """Test _run_strategy returns empty for file_overlap without files."""
    service = ContextSearchService(
        session_factory=mock_session_factory,
        embedding_service=None,
    )

    query = SearchQuery(files=[])
    results = await service._run_strategy(SearchStrategy.FILE_OVERLAP, query)

    assert results == []


# =============================================================================
# Semantic Search Tests
# =============================================================================


@pytest.mark.asyncio
async def test_semantic_search_success(
    mock_session_factory: MagicMock,
    mock_embedding_service: AsyncMock,
    sample_lesson: Any,
) -> None:
    """Test semantic_search returns results with scores."""
    # Mock session execute to return lesson with score
    mock_result = AsyncMock()
    mock_result.all = MagicMock(return_value=[(sample_lesson, 0.85)])

    mock_session = await mock_session_factory.return_value.__aenter__()
    mock_session.execute.return_value = mock_result

    service = ContextSearchService(
        session_factory=mock_session_factory,
        embedding_service=mock_embedding_service,
    )

    results = await service.semantic_search(text="test query", max_results=10, min_score=0.3)

    assert len(results) == 1
    assert results[0].lesson_id == str(sample_lesson.id)
    assert results[0].score == 0.85
    assert results[0].strategy == SearchStrategy.SEMANTIC

    # Verify embedding was generated
    mock_embedding_service.generate.assert_called_once_with("test query")


@pytest.mark.asyncio
async def test_semantic_search_min_score_filter(
    mock_session_factory: MagicMock,
    mock_embedding_service: AsyncMock,
    sample_lesson: Any,
) -> None:
    """Test semantic_search filters results below min_score."""
    # Return a lesson with score below min_score
    mock_result = AsyncMock()
    mock_result.all = MagicMock(return_value=[(sample_lesson, 0.2)])

    mock_session = await mock_session_factory.return_value.__aenter__()
    mock_session.execute.return_value = mock_result

    service = ContextSearchService(
        session_factory=mock_session_factory,
        embedding_service=mock_embedding_service,
    )

    results = await service.semantic_search(text="test query", max_results=10, min_score=0.3)

    assert len(results) == 0


@pytest.mark.asyncio
async def test_semantic_search_no_embedding_service(
    mock_session_factory: MagicMock,
) -> None:
    """Test semantic_search raises RuntimeError without embedding service."""
    service = ContextSearchService(
        session_factory=mock_session_factory,
        embedding_service=None,
    )

    with pytest.raises(RuntimeError, match="Cannot perform semantic search"):
        await service.semantic_search(text="test query")


@pytest.mark.asyncio
async def test_semantic_search_empty_results(
    mock_session_factory: MagicMock,
    mock_embedding_service: AsyncMock,
) -> None:
    """Test semantic_search handles empty database results."""
    mock_result = AsyncMock()
    mock_result.all = MagicMock(return_value=[])

    mock_session = await mock_session_factory.return_value.__aenter__()
    mock_session.execute.return_value = mock_result

    service = ContextSearchService(
        session_factory=mock_session_factory,
        embedding_service=mock_embedding_service,
    )

    results = await service.semantic_search(text="test query")

    assert results == []


# =============================================================================
# Keyword Search Tests
# =============================================================================


@pytest.mark.asyncio
async def test_keyword_search_success(
    mock_session_factory: MagicMock,
    sample_lesson: Any,
) -> None:
    """Test keyword_search returns results with normalized scores."""
    # Mock session execute to return lesson with ts_rank score
    mock_result = AsyncMock()
    mock_result.all = MagicMock(return_value=[(sample_lesson, 0.65)])

    mock_session = await mock_session_factory.return_value.__aenter__()
    mock_session.execute.return_value = mock_result

    service = ContextSearchService(
        session_factory=mock_session_factory,
        embedding_service=None,
    )

    results = await service.keyword_search(text="import error", max_results=10)

    assert len(results) == 1
    assert results[0].lesson_id == str(sample_lesson.id)
    assert results[0].score == 0.65
    assert results[0].strategy == SearchStrategy.KEYWORD


@pytest.mark.asyncio
async def test_keyword_search_empty_results(
    mock_session_factory: MagicMock,
) -> None:
    """Test keyword_search handles empty database results."""
    mock_result = AsyncMock()
    mock_result.all = MagicMock(return_value=[])

    mock_session = await mock_session_factory.return_value.__aenter__()
    mock_session.execute.return_value = mock_result

    service = ContextSearchService(
        session_factory=mock_session_factory,
        embedding_service=None,
    )

    results = await service.keyword_search(text="nonexistent")

    assert results == []


# =============================================================================
# File Overlap Search Tests
# =============================================================================


@pytest.mark.asyncio
async def test_file_overlap_search_success(
    mock_session_factory: MagicMock,
    sample_lesson: Any,
) -> None:
    """Test file_overlap_search calculates scores correctly."""
    # sample_lesson has files_affected = ["src/app/main.py", "src/app/utils.py"]
    mock_result = AsyncMock()
    mock_result.scalars = MagicMock(
        return_value=MagicMock(all=MagicMock(return_value=[sample_lesson]))
    )

    mock_session = await mock_session_factory.return_value.__aenter__()
    mock_session.execute.return_value = mock_result

    service = ContextSearchService(
        session_factory=mock_session_factory,
        embedding_service=None,
    )

    # Query with 3 files, 2 overlap with lesson
    results = await service.file_overlap_search(
        files=["src/app/main.py", "src/app/utils.py", "src/app/other.py"],
        max_results=10,
    )

    assert len(results) == 1
    assert results[0].lesson_id == str(sample_lesson.id)
    # Score = overlap_count / query_files = 2 / 3 = 0.6667
    assert abs(results[0].score - 0.6667) < 0.001
    assert results[0].strategy == SearchStrategy.FILE_OVERLAP


@pytest.mark.asyncio
async def test_file_overlap_search_empty_files(
    mock_session_factory: MagicMock,
) -> None:
    """Test file_overlap_search returns empty for empty files list."""
    service = ContextSearchService(
        session_factory=mock_session_factory,
        embedding_service=None,
    )

    results = await service.file_overlap_search(files=[], max_results=10)

    assert results == []


@pytest.mark.asyncio
async def test_file_overlap_search_sorted_by_score(
    mock_session_factory: MagicMock,
) -> None:
    """Test file_overlap_search sorts results by score descending."""
    # Create two lessons with different overlaps
    lesson1 = MagicMock()
    lesson1.id = uuid.uuid4()
    lesson1.symptom = "Issue 1"
    lesson1.root_cause = "Cause 1"
    lesson1.fix_applied = "Fix 1"
    lesson1.files_affected = ["file1.py"]  # 1 overlap
    lesson1.pattern_tags = []
    lesson1.verification_status = "verified"

    lesson2 = MagicMock()
    lesson2.id = uuid.uuid4()
    lesson2.symptom = "Issue 2"
    lesson2.root_cause = "Cause 2"
    lesson2.fix_applied = "Fix 2"
    lesson2.files_affected = ["file1.py", "file2.py"]  # 2 overlaps
    lesson2.pattern_tags = []
    lesson2.verification_status = "verified"

    mock_result = AsyncMock()
    mock_result.scalars = MagicMock(
        return_value=MagicMock(all=MagicMock(return_value=[lesson1, lesson2]))
    )

    mock_session = await mock_session_factory.return_value.__aenter__()
    mock_session.execute.return_value = mock_result

    service = ContextSearchService(
        session_factory=mock_session_factory,
        embedding_service=None,
    )

    results = await service.file_overlap_search(
        files=["file1.py", "file2.py"],
        max_results=10,
    )

    assert len(results) == 2
    # lesson2 should be first (score 1.0), lesson1 second (score 0.5)
    assert results[0].lesson_id == str(lesson2.id)
    assert results[0].score == 1.0
    assert results[1].lesson_id == str(lesson1.id)
    assert results[1].score == 0.5


# =============================================================================
# Merge Algorithm Tests
# =============================================================================


def test_merge_results_single_set() -> None:
    """Test merge_results passes through a single result set."""
    service = ContextSearchService(
        session_factory=MagicMock(),
        embedding_service=None,
    )

    results = [
        SearchResult(
            lesson_id="123",
            symptom="Test",
            root_cause="Test",
            fix_applied="Test",
            score=0.8,
            strategy=SearchStrategy.SEMANTIC,
            verification_status="verified",
        ),
        SearchResult(
            lesson_id="456",
            symptom="Test 2",
            root_cause="Test 2",
            fix_applied="Test 2",
            score=0.6,
            strategy=SearchStrategy.SEMANTIC,
            verification_status="unverified",
        ),
    ]

    merged = service.merge_results(results, max_results=10)

    assert len(merged) == 2
    assert merged[0].lesson_id == "123"
    # Verified lesson gets verification bonus: 0.8 + 0.1 = 0.9
    assert merged[0].score == 0.9
    assert merged[1].lesson_id == "456"
    assert merged[1].score == 0.6


def test_merge_results_duplicate_lesson_weighted_average() -> None:
    """Test merge_results combines duplicate lessons with weighted average."""
    service = ContextSearchService(
        session_factory=MagicMock(),
        embedding_service=None,
    )

    # Same lesson found by two strategies
    result1 = SearchResult(
        lesson_id="123",
        symptom="Test",
        root_cause="Test",
        fix_applied="Test",
        score=0.9,
        strategy=SearchStrategy.SEMANTIC,
        verification_status="unverified",
    )

    result2 = SearchResult(
        lesson_id="123",
        symptom="Test",
        root_cause="Test",
        fix_applied="Test",
        score=0.6,
        strategy=SearchStrategy.KEYWORD,
        verification_status="unverified",
    )

    merged = service.merge_results([result1], [result2], max_results=10)

    assert len(merged) == 1
    # Weighted average: (0.9 * 0.40 + 0.6 * 0.35) / (0.40 + 0.35)
    expected_score = (0.9 * 0.40 + 0.6 * 0.35) / (0.40 + 0.35)
    assert abs(merged[0].score - expected_score) < 0.001
    assert merged[0].strategy == SearchStrategy.COMBINED


def test_merge_results_verification_bonus() -> None:
    """Test merge_results applies verification bonus to verified lessons."""
    service = ContextSearchService(
        session_factory=MagicMock(),
        embedding_service=None,
    )

    verified = SearchResult(
        lesson_id="123",
        symptom="Test",
        root_cause="Test",
        fix_applied="Test",
        score=0.7,
        strategy=SearchStrategy.SEMANTIC,
        verification_status="verified",
    )

    unverified = SearchResult(
        lesson_id="456",
        symptom="Test 2",
        root_cause="Test 2",
        fix_applied="Test 2",
        score=0.7,
        strategy=SearchStrategy.SEMANTIC,
        verification_status="unverified",
    )

    merged = service.merge_results([verified, unverified], max_results=10)

    assert len(merged) == 2
    # Verified should have bonus applied
    assert merged[0].lesson_id == "123"
    assert merged[0].score == 0.7 + VERIFICATION_BONUS
    # Unverified should not
    assert merged[1].lesson_id == "456"
    assert merged[1].score == 0.7


def test_merge_results_score_clamped() -> None:
    """Test merge_results clamps final scores to [0, 1]."""
    service = ContextSearchService(
        session_factory=MagicMock(),
        embedding_service=None,
    )

    # High score + verification bonus should be clamped at 1.0
    result = SearchResult(
        lesson_id="123",
        symptom="Test",
        root_cause="Test",
        fix_applied="Test",
        score=0.95,
        strategy=SearchStrategy.SEMANTIC,
        verification_status="verified",
    )

    merged = service.merge_results([result], max_results=10)

    assert merged[0].score == 1.0  # Clamped from 0.95 + 0.1


def test_merge_results_max_results_limit() -> None:
    """Test merge_results limits output to max_results."""
    service = ContextSearchService(
        session_factory=MagicMock(),
        embedding_service=None,
    )

    # Create results with scores that stay valid (>= 0.0)
    results = [
        SearchResult(
            lesson_id=str(i),
            symptom=f"Test {i}",
            root_cause="Test",
            fix_applied="Test",
            score=max(0.0, 1.0 - (i * 0.05)),  # Ensure non-negative scores
            strategy=SearchStrategy.SEMANTIC,
            verification_status="unverified",  # Use unverified to avoid bonus
        )
        for i in range(20)
    ]

    merged = service.merge_results(results, max_results=5)

    assert len(merged) == 5


def test_merge_results_custom_weights() -> None:
    """Test merge_results respects custom weight overrides."""
    service = ContextSearchService(
        session_factory=MagicMock(),
        embedding_service=None,
    )

    result1 = SearchResult(
        lesson_id="123",
        symptom="Test",
        root_cause="Test",
        fix_applied="Test",
        score=0.8,
        strategy=SearchStrategy.SEMANTIC,
        verification_status="unverified",
    )

    result2 = SearchResult(
        lesson_id="123",
        symptom="Test",
        root_cause="Test",
        fix_applied="Test",
        score=0.4,
        strategy=SearchStrategy.KEYWORD,
        verification_status="unverified",
    )

    # Custom weights: semantic=0.8, keyword=0.2
    custom_weights = {
        SearchStrategy.SEMANTIC: 0.8,
        SearchStrategy.KEYWORD: 0.2,
    }

    merged = service.merge_results(
        [result1], [result2], max_results=10, weights=custom_weights
    )

    # Weighted average: (0.8 * 0.8 + 0.4 * 0.2) / (0.8 + 0.2) = 0.72
    expected_score = (0.8 * 0.8 + 0.4 * 0.2) / (0.8 + 0.2)
    assert abs(merged[0].score - expected_score) < 0.001


def test_merge_results_sorted_by_score() -> None:
    """Test merge_results sorts by final score descending."""
    service = ContextSearchService(
        session_factory=MagicMock(),
        embedding_service=None,
    )

    results = [
        SearchResult(
            lesson_id="1",
            symptom="Test",
            root_cause="Test",
            fix_applied="Test",
            score=0.5,
            strategy=SearchStrategy.SEMANTIC,
            verification_status="unverified",
        ),
        SearchResult(
            lesson_id="2",
            symptom="Test",
            root_cause="Test",
            fix_applied="Test",
            score=0.9,
            strategy=SearchStrategy.SEMANTIC,
            verification_status="unverified",
        ),
        SearchResult(
            lesson_id="3",
            symptom="Test",
            root_cause="Test",
            fix_applied="Test",
            score=0.7,
            strategy=SearchStrategy.SEMANTIC,
            verification_status="unverified",
        ),
    ]

    merged = service.merge_results(results, max_results=10)

    assert merged[0].lesson_id == "2"  # score 0.9
    assert merged[1].lesson_id == "3"  # score 0.7
    assert merged[2].lesson_id == "1"  # score 0.5


# =============================================================================
# Format Tests
# =============================================================================


def test_format_for_context_empty_results() -> None:
    """Test format_for_context returns empty string for empty results."""
    service = ContextSearchService(
        session_factory=MagicMock(),
        embedding_service=None,
    )

    formatted = service.format_for_context([])

    assert formatted == ""


def test_format_for_context_basic() -> None:
    """Test format_for_context produces expected output structure."""
    service = ContextSearchService(
        session_factory=MagicMock(),
        embedding_service=None,
    )

    results = [
        SearchResult(
            lesson_id="123",
            symptom="Import error",
            root_cause="Circular import",
            fix_applied="Use lazy imports",
            files_affected=["main.py", "utils.py"],
            pattern_tags=["import-error"],
            score=0.85,
            strategy=SearchStrategy.SEMANTIC,
            verification_status="verified",
        )
    ]

    formatted = service.format_for_context(results)

    assert "=== Relevant Lessons Learned ===" in formatted
    assert "Lesson 1 (score: 0.85, status: verified)" in formatted
    assert "Symptom: Import error" in formatted
    assert "Root Cause: Circular import" in formatted
    assert "Fix: Use lazy imports" in formatted
    assert "Files: main.py, utils.py" in formatted
    assert "Tags: import-error" in formatted


def test_format_for_context_truncation() -> None:
    """Test format_for_context truncates output at max_chars."""
    service = ContextSearchService(
        session_factory=MagicMock(),
        embedding_service=None,
    )

    results = [
        SearchResult(
            lesson_id=str(i),
            symptom=f"Symptom {i}" * 50,  # Long symptom
            root_cause=f"Cause {i}" * 50,
            fix_applied=f"Fix {i}" * 50,
            score=0.8,
            strategy=SearchStrategy.SEMANTIC,
            verification_status="verified",
        )
        for i in range(10)
    ]

    formatted = service.format_for_context(results, max_chars=500)

    assert len(formatted) <= 500
    assert "more lessons omitted" in formatted


def test_format_for_context_no_files_or_tags() -> None:
    """Test format_for_context handles results without files or tags."""
    service = ContextSearchService(
        session_factory=MagicMock(),
        embedding_service=None,
    )

    results = [
        SearchResult(
            lesson_id="123",
            symptom="Test",
            root_cause="Test",
            fix_applied="Test",
            files_affected=[],
            pattern_tags=[],
            score=0.7,
            strategy=SearchStrategy.SEMANTIC,
            verification_status="unverified",
        )
    ]

    formatted = service.format_for_context(results)

    assert "Files:" not in formatted
    assert "Tags:" not in formatted


# =============================================================================
# Strategy Failure Handling Tests
# =============================================================================


@pytest.mark.asyncio
async def test_search_strategy_failure_doesnt_fail_others(
    mock_session_factory: MagicMock,
    mock_embedding_service: AsyncMock,
) -> None:
    """Test that exception in one strategy doesn't prevent others from running."""
    service = ContextSearchService(
        session_factory=mock_session_factory,
        embedding_service=mock_embedding_service,
    )

    # Mock semantic_search to raise exception
    service.semantic_search = AsyncMock(side_effect=RuntimeError("DB error"))  # type: ignore

    # Mock keyword_search to return results
    expected_results = [
        SearchResult(
            lesson_id="123",
            symptom="Test",
            root_cause="Test",
            fix_applied="Test",
            score=0.7,
            strategy=SearchStrategy.KEYWORD,
            verification_status="verified",
        )
    ]
    service.keyword_search = AsyncMock(return_value=expected_results)  # type: ignore
    service.file_overlap_search = AsyncMock(return_value=[])  # type: ignore

    query = SearchQuery(
        text="test",
        files=["test.py"],
        strategies=[SearchStrategy.COMBINED],
    )
    results = await service.search(query)

    # Should get keyword results despite semantic failure
    assert len(results) == 1
    assert results[0].strategy == SearchStrategy.KEYWORD


# =============================================================================
# Filter Tests
# =============================================================================


@pytest.mark.asyncio
async def test_search_min_score_filter(
    mock_session_factory: MagicMock,
    mock_embedding_service: AsyncMock,
) -> None:
    """Test search filters results below min_score."""
    service = ContextSearchService(
        session_factory=mock_session_factory,
        embedding_service=mock_embedding_service,
    )

    # Mock results with varying scores
    mock_results = [
        SearchResult(
            lesson_id="1",
            symptom="Test",
            root_cause="Test",
            fix_applied="Test",
            score=0.8,
            strategy=SearchStrategy.SEMANTIC,
            verification_status="verified",
        ),
        SearchResult(
            lesson_id="2",
            symptom="Test",
            root_cause="Test",
            fix_applied="Test",
            score=0.2,  # Below min_score
            strategy=SearchStrategy.SEMANTIC,
            verification_status="verified",
        ),
    ]

    service._run_strategy = AsyncMock(return_value=mock_results)  # type: ignore

    query = SearchQuery(text="test", min_score=0.5)
    results = await service.search(query)

    # Should only get the result with score >= 0.5
    assert len(results) == 1
    assert results[0].lesson_id == "1"


@pytest.mark.asyncio
async def test_search_include_unverified_false(
    mock_session_factory: MagicMock,
    mock_embedding_service: AsyncMock,
) -> None:
    """Test search filters unverified lessons when include_unverified=False."""
    service = ContextSearchService(
        session_factory=mock_session_factory,
        embedding_service=mock_embedding_service,
    )

    mock_results = [
        SearchResult(
            lesson_id="1",
            symptom="Test",
            root_cause="Test",
            fix_applied="Test",
            score=0.8,
            strategy=SearchStrategy.SEMANTIC,
            verification_status="verified",
        ),
        SearchResult(
            lesson_id="2",
            symptom="Test",
            root_cause="Test",
            fix_applied="Test",
            score=0.7,
            strategy=SearchStrategy.SEMANTIC,
            verification_status="unverified",
        ),
    ]

    service._run_strategy = AsyncMock(return_value=mock_results)  # type: ignore

    query = SearchQuery(text="test", include_unverified=False)
    results = await service.search(query)

    # Should only get verified result
    assert len(results) == 1
    assert results[0].lesson_id == "1"
    assert results[0].verification_status == "verified"


@pytest.mark.asyncio
async def test_search_pattern_tags_filter(
    mock_session_factory: MagicMock,
    mock_embedding_service: AsyncMock,
) -> None:
    """Test search filters by pattern_tags."""
    service = ContextSearchService(
        session_factory=mock_session_factory,
        embedding_service=mock_embedding_service,
    )

    mock_results = [
        SearchResult(
            lesson_id="1",
            symptom="Test",
            root_cause="Test",
            fix_applied="Test",
            pattern_tags=["import-error", "python"],
            score=0.8,
            strategy=SearchStrategy.SEMANTIC,
            verification_status="verified",
        ),
        SearchResult(
            lesson_id="2",
            symptom="Test",
            root_cause="Test",
            fix_applied="Test",
            pattern_tags=["build-error"],
            score=0.7,
            strategy=SearchStrategy.SEMANTIC,
            verification_status="verified",
        ),
        SearchResult(
            lesson_id="3",
            symptom="Test",
            root_cause="Test",
            fix_applied="Test",
            pattern_tags=["import-error"],
            score=0.6,
            strategy=SearchStrategy.SEMANTIC,
            verification_status="verified",
        ),
    ]

    service._run_strategy = AsyncMock(return_value=mock_results)  # type: ignore

    query = SearchQuery(text="test", pattern_tags=["import-error"])
    results = await service.search(query)

    # Should only get results with "import-error" tag
    assert len(results) == 2
    assert results[0].lesson_id == "1"
    assert results[1].lesson_id == "3"


# =============================================================================
# Constants Tests
# =============================================================================


def test_default_weights_values() -> None:
    """Test DEFAULT_WEIGHTS has expected values."""
    assert DEFAULT_WEIGHTS["semantic"] == 0.40
    assert DEFAULT_WEIGHTS["keyword"] == 0.35
    assert DEFAULT_WEIGHTS["file_overlap"] == 0.25
    assert sum(DEFAULT_WEIGHTS.values()) == 1.0


def test_verification_bonus_value() -> None:
    """Test VERIFICATION_BONUS has expected value."""
    assert VERIFICATION_BONUS == 0.1
