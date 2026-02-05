"""Semantic context pre-selection for agent context injection.

This module retrieves relevant lessons learned for agent context injection
using multiple search strategies: semantic similarity (pgvector cosine distance),
full-text keyword search (PostgreSQL tsvector), and file overlap matching.

Results from all strategies are merged, deduplicated, and ranked using a
weighted scoring algorithm with a verification bonus for proven lessons.

Example usage:
    >>> from forgemaster.intelligence.context_search import (
    ...     ContextSearchService, SearchQuery, SearchStrategy,
    ... )
    >>>
    >>> service = ContextSearchService(
    ...     session_factory=session_factory,
    ...     embedding_service=embedding_service,
    ... )
    >>> results = await service.search(SearchQuery(
    ...     text="import error in module initialization",
    ...     files=["src/app/main.py"],
    ...     strategies=[SearchStrategy.COMBINED],
    ... ))
"""

from __future__ import annotations

from collections import defaultdict
from enum import Enum
from typing import Callable

import structlog
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from forgemaster.database.models.lesson import LessonLearned
from forgemaster.intelligence.embeddings import EmbeddingService

logger = structlog.get_logger(__name__)

# Type alias matching project convention from connection.py
SessionFactory = Callable[[], AsyncSession]

# Default strategy weights for result merging
DEFAULT_WEIGHTS: dict[SearchStrategy, float] = {
    "semantic": 0.40,
    "keyword": 0.35,
    "file_overlap": 0.25,
}

# Verification bonus added to verified lesson scores
VERIFICATION_BONUS = 0.1


class SearchStrategy(str, Enum):
    """Strategy for searching lessons learned."""

    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    FILE_OVERLAP = "file_overlap"
    COMBINED = "combined"


class SearchResult(BaseModel):
    """A single search result with ranking score.

    Attributes:
        lesson_id: UUID string of the lesson.
        symptom: Description of the observed problem.
        root_cause: Identified root cause.
        fix_applied: Description of the fix.
        files_affected: List of file paths involved.
        pattern_tags: Classification tags.
        score: Relevance score between 0.0 and 1.0.
        strategy: Which search strategy found this result.
        verification_status: Whether the lesson has been verified.
    """

    lesson_id: str
    symptom: str
    root_cause: str
    fix_applied: str
    files_affected: list[str] = Field(default_factory=list)
    pattern_tags: list[str] = Field(default_factory=list)
    score: float = Field(ge=0.0, le=1.0, description="Relevance score")
    strategy: SearchStrategy
    verification_status: str


class SearchQuery(BaseModel):
    """Search parameters for context pre-selection.

    Attributes:
        text: Text for semantic and keyword search.
        files: File paths for overlap search.
        pattern_tags: Tags to filter results by.
        max_results: Maximum number of results to return.
        min_score: Minimum relevance score threshold.
        strategies: List of search strategies to apply.
        include_unverified: Whether to include unverified lessons.
    """

    text: str | None = None
    files: list[str] = Field(default_factory=list)
    pattern_tags: list[str] = Field(default_factory=list)
    max_results: int = Field(default=10, ge=1, le=100)
    min_score: float = Field(default=0.3, ge=0.0, le=1.0)
    strategies: list[SearchStrategy] = Field(
        default_factory=lambda: [SearchStrategy.COMBINED],
    )
    include_unverified: bool = Field(default=True)


def _lesson_to_result(
    lesson: LessonLearned,
    score: float,
    strategy: SearchStrategy,
) -> SearchResult:
    """Convert a LessonLearned ORM instance to a SearchResult.

    Args:
        lesson: ORM model instance.
        score: Computed relevance score.
        strategy: The strategy that produced this result.

    Returns:
        SearchResult with fields copied from the lesson.
    """
    return SearchResult(
        lesson_id=str(lesson.id),
        symptom=lesson.symptom,
        root_cause=lesson.root_cause,
        fix_applied=lesson.fix_applied,
        files_affected=lesson.files_affected or [],
        pattern_tags=lesson.pattern_tags or [],
        score=max(0.0, min(1.0, score)),
        strategy=strategy,
        verification_status=lesson.verification_status,
    )


class ContextSearchService:
    """Pre-selects relevant lessons for agent context injection.

    Uses multiple search strategies to find the most relevant lessons
    learned for a given task context, then merges and ranks results.

    Attributes:
        session_factory: Callable that produces AsyncSession instances.
        embedding_service: Optional service for generating query embeddings.
    """

    def __init__(
        self,
        session_factory: SessionFactory,
        embedding_service: EmbeddingService | None = None,
    ) -> None:
        """Initialize context search service.

        Args:
            session_factory: Factory for creating async database sessions.
            embedding_service: Optional embedding service for semantic search.
                If None, semantic search will be skipped gracefully.
        """
        self.session_factory = session_factory
        self.embedding_service = embedding_service

        logger.info(
            "context_search_service_initialized",
            has_embedding_service=embedding_service is not None,
        )

    # ------------------------------------------------------------------
    # Main search interface
    # ------------------------------------------------------------------

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        """Execute search using specified strategies and return merged results.

        Dispatches to individual strategy methods based on the query's
        ``strategies`` list. When ``COMBINED`` is specified, all applicable
        strategies are run and their results are merged.

        Args:
            query: Search parameters including text, files, and strategy config.

        Returns:
            Merged and ranked list of SearchResult instances.
        """
        strategies = query.strategies
        if SearchStrategy.COMBINED in strategies:
            strategies = [
                SearchStrategy.SEMANTIC,
                SearchStrategy.KEYWORD,
                SearchStrategy.FILE_OVERLAP,
            ]

        result_sets: list[list[SearchResult]] = []

        for strategy in strategies:
            try:
                results = await self._run_strategy(strategy, query)
                if results:
                    result_sets.append(results)
            except Exception as exc:
                logger.warning(
                    "context_search_strategy_failed",
                    strategy=strategy.value,
                    error=str(exc),
                    error_type=type(exc).__name__,
                )

        if not result_sets:
            logger.info("context_search_no_results", query_text=query.text)
            return []

        merged = self.merge_results(
            *result_sets,
            max_results=query.max_results,
        )

        # Apply min_score filter
        merged = [r for r in merged if r.score >= query.min_score]

        # Filter unverified if requested
        if not query.include_unverified:
            merged = [r for r in merged if r.verification_status == "verified"]

        # Apply pattern_tags filter
        if query.pattern_tags:
            tag_set = set(query.pattern_tags)
            merged = [
                r for r in merged
                if set(r.pattern_tags) & tag_set
            ]

        logger.info(
            "context_search_completed",
            strategies=[s.value for s in strategies],
            result_count=len(merged),
        )

        return merged[: query.max_results]

    async def _run_strategy(
        self,
        strategy: SearchStrategy,
        query: SearchQuery,
    ) -> list[SearchResult]:
        """Dispatch to the appropriate search strategy method.

        Args:
            strategy: The strategy to execute.
            query: Search parameters.

        Returns:
            List of results from the selected strategy.
        """
        if strategy == SearchStrategy.SEMANTIC:
            if query.text and self.embedding_service is not None:
                return await self.semantic_search(
                    text=query.text,
                    max_results=query.max_results,
                    min_score=query.min_score,
                )
            return []

        if strategy == SearchStrategy.KEYWORD:
            if query.text:
                return await self.keyword_search(
                    text=query.text,
                    max_results=query.max_results,
                )
            return []

        if strategy == SearchStrategy.FILE_OVERLAP:
            if query.files:
                return await self.file_overlap_search(
                    files=query.files,
                    max_results=query.max_results,
                )
            return []

        return []

    # ------------------------------------------------------------------
    # Individual search strategies
    # ------------------------------------------------------------------

    async def semantic_search(
        self,
        text: str,
        max_results: int = 10,
        min_score: float = 0.3,
    ) -> list[SearchResult]:
        """Find lessons by vector cosine similarity.

        Generates an embedding for the query text, then uses pgvector's
        cosine distance operator to find the closest lesson embeddings.
        Score is computed as ``1 - cosine_distance``.

        Args:
            text: Query text to embed and search for.
            max_results: Maximum number of results.
            min_score: Minimum cosine similarity score.

        Returns:
            List of SearchResult ordered by similarity descending.

        Raises:
            RuntimeError: If no embedding service is configured.
        """
        if self.embedding_service is None:
            raise RuntimeError(
                "Cannot perform semantic search without an embedding service"
            )

        logger.debug("semantic_search_started", text_length=len(text))

        query_embedding = await self.embedding_service.generate(text)

        async with self.session_factory() as session:
            score_expr = (
                1 - LessonLearned.content_embedding.cosine_distance(query_embedding)
            ).label("score")

            stmt = (
                select(LessonLearned, score_expr)
                .where(LessonLearned.content_embedding.isnot(None))
                .order_by(
                    LessonLearned.content_embedding.cosine_distance(query_embedding)
                )
                .limit(max_results)
            )

            result = await session.execute(stmt)
            rows = result.all()

        results: list[SearchResult] = []
        for row in rows:
            lesson: LessonLearned = row[0]
            score: float = float(row[1])
            if score >= min_score:
                results.append(
                    _lesson_to_result(lesson, score, SearchStrategy.SEMANTIC)
                )

        logger.info(
            "semantic_search_completed",
            result_count=len(results),
            max_results=max_results,
        )
        return results

    async def keyword_search(
        self,
        text: str,
        max_results: int = 10,
    ) -> list[SearchResult]:
        """Find lessons by PostgreSQL full-text search.

        Uses ``ts_rank`` with the lesson's ``content_tsv`` column and
        ``plainto_tsquery`` to rank relevance. Scores are normalized to
        the 0-1 range by dividing by ``(1 + ts_rank)`` so the result
        always falls in [0, 1).

        Args:
            text: Query text for full-text search.
            max_results: Maximum number of results.

        Returns:
            List of SearchResult ordered by text-search rank descending.
        """
        logger.debug("keyword_search_started", text_length=len(text))

        query_tsv = func.plainto_tsquery("english", text)
        raw_rank = func.ts_rank(LessonLearned.content_tsv, query_tsv)
        # Normalize: ts_rank returns unbounded floats; map to [0, 1)
        normalized_score = (raw_rank / (1 + raw_rank)).label("score")

        async with self.session_factory() as session:
            stmt = (
                select(LessonLearned, normalized_score)
                .where(LessonLearned.content_tsv.op("@@")(query_tsv))
                .order_by(raw_rank.desc())
                .limit(max_results)
            )

            result = await session.execute(stmt)
            rows = result.all()

        results: list[SearchResult] = []
        for row in rows:
            lesson: LessonLearned = row[0]
            score: float = float(row[1])
            results.append(
                _lesson_to_result(lesson, score, SearchStrategy.KEYWORD)
            )

        logger.info(
            "keyword_search_completed",
            result_count=len(results),
            max_results=max_results,
        )
        return results

    async def file_overlap_search(
        self,
        files: list[str],
        max_results: int = 10,
    ) -> list[SearchResult]:
        """Find lessons affecting the same files.

        Uses PostgreSQL's ``&&`` array overlap operator to find lessons
        whose ``files_affected`` intersect with the query file list.
        Score is computed as ``overlap_count / len(query_files)``.

        Args:
            files: List of file paths to match against.
            max_results: Maximum number of results.

        Returns:
            List of SearchResult ordered by overlap score descending.
        """
        if not files:
            return []

        logger.debug("file_overlap_search_started", file_count=len(files))

        async with self.session_factory() as session:
            stmt = (
                select(LessonLearned)
                .where(LessonLearned.files_affected.isnot(None))
                .where(LessonLearned.files_affected.overlap(files))
                .limit(max_results)
            )

            result = await session.execute(stmt)
            lessons = list(result.scalars().all())

        query_file_set = set(files)
        total_query_files = len(query_file_set)

        results: list[SearchResult] = []
        for lesson in lessons:
            lesson_files = set(lesson.files_affected or [])
            overlap_count = len(query_file_set & lesson_files)
            score = overlap_count / total_query_files
            results.append(
                _lesson_to_result(lesson, score, SearchStrategy.FILE_OVERLAP)
            )

        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)

        logger.info(
            "file_overlap_search_completed",
            result_count=len(results),
            file_count=len(files),
        )
        return results[: max_results]

    # ------------------------------------------------------------------
    # Result merging
    # ------------------------------------------------------------------

    def merge_results(
        self,
        *result_sets: list[SearchResult],
        max_results: int = 10,
        weights: dict[SearchStrategy, float] | None = None,
    ) -> list[SearchResult]:
        """Merge results from multiple strategies.

        Algorithm:
            1. Deduplicate by lesson_id.
            2. For duplicate lessons found by multiple strategies, combine
               scores using a weighted average.
            3. Apply a verification bonus (+0.1) to verified lessons.
            4. Sort by final score descending.
            5. Limit to ``max_results``.

        Args:
            *result_sets: Variable number of result lists to merge.
            max_results: Maximum number of merged results.
            weights: Optional per-strategy weight overrides.
                Defaults to semantic=0.4, keyword=0.35, file_overlap=0.25.

        Returns:
            Merged and re-ranked list of SearchResult.
        """
        effective_weights = dict(DEFAULT_WEIGHTS)
        if weights is not None:
            effective_weights.update(weights)

        # Collect all results grouped by lesson_id
        grouped: dict[str, list[SearchResult]] = defaultdict(list)
        for result_set in result_sets:
            for result in result_set:
                grouped[result.lesson_id].append(result)

        merged: list[SearchResult] = []
        for lesson_id, results in grouped.items():
            if len(results) == 1:
                # Single strategy hit: use its score directly
                base = results[0]
                final_score = base.score
            else:
                # Multiple strategy hits: weighted average
                weighted_sum = 0.0
                weight_sum = 0.0
                for r in results:
                    w = effective_weights.get(r.strategy, 0.3)
                    weighted_sum += r.score * w
                    weight_sum += w

                final_score = weighted_sum / weight_sum if weight_sum > 0 else 0.0
                base = results[0]

            # Apply verification bonus
            if base.verification_status == "verified":
                final_score = min(1.0, final_score + VERIFICATION_BONUS)

            merged.append(
                SearchResult(
                    lesson_id=base.lesson_id,
                    symptom=base.symptom,
                    root_cause=base.root_cause,
                    fix_applied=base.fix_applied,
                    files_affected=base.files_affected,
                    pattern_tags=base.pattern_tags,
                    score=max(0.0, min(1.0, final_score)),
                    strategy=(
                        results[0].strategy
                        if len(results) == 1
                        else SearchStrategy.COMBINED
                    ),
                    verification_status=base.verification_status,
                )
            )

        merged.sort(key=lambda r: r.score, reverse=True)
        return merged[: max_results]

    # ------------------------------------------------------------------
    # Context formatting
    # ------------------------------------------------------------------

    def format_for_context(
        self,
        results: list[SearchResult],
        max_chars: int = 4000,
    ) -> str:
        """Format search results as text for agent context injection.

        Produces a human-readable block of text listing lessons in order
        of relevance, truncated to ``max_chars`` to stay within token
        budgets.

        Args:
            results: Ranked search results to format.
            max_chars: Maximum character length of the output string.

        Returns:
            Formatted string suitable for injection into agent system
            prompts. Returns an empty string if no results are provided.
        """
        if not results:
            return ""

        sections: list[str] = [
            "=== Relevant Lessons Learned ===",
            "",
        ]

        for i, result in enumerate(results, start=1):
            section = (
                f"--- Lesson {i} (score: {result.score:.2f}, "
                f"status: {result.verification_status}) ---\n"
                f"Symptom: {result.symptom}\n"
                f"Root Cause: {result.root_cause}\n"
                f"Fix: {result.fix_applied}"
            )

            if result.files_affected:
                section += f"\nFiles: {', '.join(result.files_affected)}"

            if result.pattern_tags:
                section += f"\nTags: {', '.join(result.pattern_tags)}"

            section += "\n"

            # Check if adding this section would exceed the limit
            current_text = "\n".join(sections)
            if len(current_text) + len(section) + 1 > max_chars:
                # Add truncation notice and stop
                sections.append(
                    f"... ({len(results) - i + 1} more lessons omitted "
                    f"due to size limit)"
                )
                break

            sections.append(section)

        output = "\n".join(sections).strip()
        return output[:max_chars]
