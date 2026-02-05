"""Unit tests for the advisory file locking system.

Tests cover:
- Lock acquisition (exclusive and shared)
- Atomic all-or-nothing acquisition semantics
- Conflict detection between exclusive and shared locks
- Lock release by task ID
- Lock release by worker ID (crash recovery)
- Stale lock cleanup
- Path normalisation
- Active lock queries
- Task-specific lock queries
- Shared lock compatibility (multiple shared locks allowed)
- Lock conflict matrix (exclusive vs shared)
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from forgemaster.database.models.file_lock import FileLock, LockType
from forgemaster.orchestrator.file_locker import FileLocker, LockConflict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_file_lock(
    file_path: str = "src/main.py",
    task_id: str | None = None,
    worker_id: str = "worker-1",
    lock_type: LockType = LockType.EXCLUSIVE,
    acquired_at: datetime | None = None,
    released_at: datetime | None = None,
) -> FileLock:
    """Create a FileLock instance for testing."""
    lock = FileLock(
        id=uuid.uuid4(),
        file_path=file_path,
        task_id=task_id or str(uuid.uuid4()),
        worker_id=worker_id,
        lock_type=lock_type,
        acquired_at=acquired_at or datetime.now(timezone.utc),
        released_at=released_at,
    )
    return lock


class FakeResult:
    """Fake SQLAlchemy result object for update statements."""

    def __init__(self, rowcount: int = 0) -> None:
        self.rowcount = rowcount


class FakeScalars:
    """Fake scalars() return for select queries."""

    def __init__(self, items: list) -> None:
        self._items = items

    def all(self) -> list:
        return self._items


class FakeSelectResult:
    """Fake result from session.execute(select(...))."""

    def __init__(self, items: list) -> None:
        self._items = items

    def scalars(self) -> FakeScalars:
        return FakeScalars(self._items)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_session() -> AsyncMock:
    """Create a mock AsyncSession."""
    session = AsyncMock()
    session.add = MagicMock()
    session.commit = AsyncMock()
    session.execute = AsyncMock()
    return session


@pytest.fixture
def session_factory(mock_session: AsyncMock):
    """Create a session factory that yields the mock session."""
    @asynccontextmanager
    async def factory():
        yield mock_session

    return factory


@pytest.fixture
def locker(session_factory) -> FileLocker:
    """Create a FileLocker instance with mock session factory."""
    return FileLocker(session_factory)


# ---------------------------------------------------------------------------
# Path Normalisation
# ---------------------------------------------------------------------------


class TestPathNormalisation:
    """Tests for internal path normalisation."""

    def test_strips_whitespace(self) -> None:
        assert FileLocker._normalise_path("  src/main.py  ") == "src/main.py"

    def test_normalises_backslashes(self) -> None:
        assert FileLocker._normalise_path("src\\models\\user.py") == "src/models/user.py"

    def test_removes_dot_slash_prefix(self) -> None:
        assert FileLocker._normalise_path("./src/main.py") == "src/main.py"

    def test_preserves_normal_path(self) -> None:
        assert FileLocker._normalise_path("src/main.py") == "src/main.py"

    def test_combined_normalisation(self) -> None:
        assert FileLocker._normalise_path(" .\\src\\main.py ") == "src/main.py"


# ---------------------------------------------------------------------------
# Lock Conflict Matrix
# ---------------------------------------------------------------------------


class TestLockConflictMatrix:
    """Tests for the _locks_conflict static method."""

    def test_exclusive_vs_exclusive_conflicts(self) -> None:
        assert FileLocker._locks_conflict(LockType.EXCLUSIVE, LockType.EXCLUSIVE) is True

    def test_exclusive_vs_shared_conflicts(self) -> None:
        assert FileLocker._locks_conflict(LockType.EXCLUSIVE, LockType.SHARED) is True

    def test_shared_vs_exclusive_conflicts(self) -> None:
        assert FileLocker._locks_conflict(LockType.SHARED, LockType.EXCLUSIVE) is True

    def test_shared_vs_shared_no_conflict(self) -> None:
        assert FileLocker._locks_conflict(LockType.SHARED, LockType.SHARED) is False


# ---------------------------------------------------------------------------
# Lock Acquisition
# ---------------------------------------------------------------------------


class TestAcquireLocks:
    """Tests for acquire_locks method."""

    @pytest.mark.asyncio
    async def test_acquire_empty_paths_returns_true(
        self, locker: FileLocker
    ) -> None:
        """Acquiring locks with no files should succeed trivially."""
        result = await locker.acquire_locks("task-1", "worker-1", [])
        assert result is True

    @pytest.mark.asyncio
    async def test_acquire_exclusive_no_conflicts(
        self, locker: FileLocker, mock_session: AsyncMock
    ) -> None:
        """Should acquire locks when no conflicts exist."""
        # No existing locks
        mock_session.execute.return_value = FakeSelectResult([])

        result = await locker.acquire_locks(
            "task-1", "worker-1", ["src/main.py", "src/utils.py"]
        )

        assert result is True
        # Should have added 2 locks and committed
        assert mock_session.add.call_count == 2
        mock_session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_acquire_denied_on_conflict(
        self, locker: FileLocker, mock_session: AsyncMock
    ) -> None:
        """Should deny acquisition when a conflicting lock exists."""
        conflicting_lock = _make_file_lock(
            file_path="src/main.py",
            task_id="task-other",
            lock_type=LockType.EXCLUSIVE,
        )
        mock_session.execute.return_value = FakeSelectResult([conflicting_lock])

        result = await locker.acquire_locks(
            "task-1", "worker-1", ["src/main.py"]
        )

        assert result is False
        # Should NOT have added any locks
        mock_session.add.assert_not_called()
        mock_session.commit.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_acquire_all_or_nothing(
        self, locker: FileLocker, mock_session: AsyncMock
    ) -> None:
        """If any file has a conflict, no locks should be acquired."""
        # Only one file conflicts, but we request two
        conflicting_lock = _make_file_lock(
            file_path="src/main.py",
            task_id="task-other",
        )
        mock_session.execute.return_value = FakeSelectResult([conflicting_lock])

        result = await locker.acquire_locks(
            "task-1", "worker-1", ["src/main.py", "src/utils.py"]
        )

        assert result is False
        mock_session.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_acquire_same_task_no_conflict(
        self, locker: FileLocker, mock_session: AsyncMock
    ) -> None:
        """Locks held by the same task should not conflict."""
        # Lock held by the requesting task itself
        same_task_lock = _make_file_lock(
            file_path="src/main.py",
            task_id="task-1",
        )
        # The query filters out same-task locks (task_id != task_id),
        # so this lock won't be in the results
        mock_session.execute.return_value = FakeSelectResult([])

        result = await locker.acquire_locks(
            "task-1", "worker-1", ["src/main.py"]
        )

        assert result is True
        assert mock_session.add.call_count == 1

    @pytest.mark.asyncio
    async def test_acquire_shared_with_existing_shared_succeeds(
        self, locker: FileLocker, mock_session: AsyncMock
    ) -> None:
        """Shared lock acquisition should succeed if only shared locks exist."""
        existing_shared = _make_file_lock(
            file_path="src/main.py",
            task_id="task-other",
            lock_type=LockType.SHARED,
        )
        mock_session.execute.return_value = FakeSelectResult([existing_shared])

        result = await locker.acquire_locks(
            "task-1", "worker-1", ["src/main.py"],
            lock_type=LockType.SHARED,
        )

        assert result is True
        assert mock_session.add.call_count == 1

    @pytest.mark.asyncio
    async def test_acquire_shared_denied_by_exclusive(
        self, locker: FileLocker, mock_session: AsyncMock
    ) -> None:
        """Shared lock acquisition should be denied if exclusive lock exists."""
        existing_exclusive = _make_file_lock(
            file_path="src/main.py",
            task_id="task-other",
            lock_type=LockType.EXCLUSIVE,
        )
        mock_session.execute.return_value = FakeSelectResult([existing_exclusive])

        result = await locker.acquire_locks(
            "task-1", "worker-1", ["src/main.py"],
            lock_type=LockType.SHARED,
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_acquire_normalises_paths(
        self, locker: FileLocker, mock_session: AsyncMock
    ) -> None:
        """Paths should be normalised before acquisition."""
        mock_session.execute.return_value = FakeSelectResult([])

        await locker.acquire_locks(
            "task-1", "worker-1",
            [".\\src\\main.py", " ./tests/test.py "],
        )

        # Check that the locks added have normalised paths
        added_locks = [call.args[0] for call in mock_session.add.call_args_list]
        paths = [lock.file_path for lock in added_locks]
        assert "src/main.py" in paths
        assert "tests/test.py" in paths

    @pytest.mark.asyncio
    async def test_acquire_sets_correct_fields(
        self, locker: FileLocker, mock_session: AsyncMock
    ) -> None:
        """Acquired locks should have correct task_id, worker_id, and lock_type."""
        mock_session.execute.return_value = FakeSelectResult([])

        await locker.acquire_locks(
            "task-abc", "worker-2", ["src/main.py"],
            lock_type=LockType.SHARED,
        )

        added_lock = mock_session.add.call_args_list[0].args[0]
        assert added_lock.task_id == "task-abc"
        assert added_lock.worker_id == "worker-2"
        assert added_lock.lock_type == LockType.SHARED
        assert added_lock.file_path == "src/main.py"
        assert added_lock.acquired_at is not None


# ---------------------------------------------------------------------------
# Conflict Detection
# ---------------------------------------------------------------------------


class TestDetectConflicts:
    """Tests for detect_conflicts method."""

    @pytest.mark.asyncio
    async def test_no_conflicts_empty_paths(
        self, locker: FileLocker
    ) -> None:
        """Empty file list should return no conflicts."""
        result = await locker.detect_conflicts("task-1", [])
        assert result == []

    @pytest.mark.asyncio
    async def test_no_conflicts_when_no_locks_exist(
        self, locker: FileLocker, mock_session: AsyncMock
    ) -> None:
        """No conflicts when no active locks exist."""
        mock_session.execute.return_value = FakeSelectResult([])

        result = await locker.detect_conflicts(
            "task-1", ["src/main.py"]
        )

        assert result == []

    @pytest.mark.asyncio
    async def test_detects_exclusive_conflict(
        self, locker: FileLocker, mock_session: AsyncMock
    ) -> None:
        """Should detect conflict with existing exclusive lock."""
        existing = _make_file_lock(
            file_path="src/main.py",
            task_id="task-other",
            worker_id="worker-3",
            lock_type=LockType.EXCLUSIVE,
        )
        mock_session.execute.return_value = FakeSelectResult([existing])

        conflicts = await locker.detect_conflicts(
            "task-1", ["src/main.py"]
        )

        assert len(conflicts) == 1
        conflict = conflicts[0]
        assert conflict.file_path == "src/main.py"
        assert conflict.held_by_task_id == str(existing.task_id)
        assert conflict.held_by_worker_id == "worker-3"
        assert conflict.requesting_task_id == "task-1"
        assert conflict.lock_type == LockType.EXCLUSIVE

    @pytest.mark.asyncio
    async def test_shared_no_conflict_with_shared(
        self, locker: FileLocker, mock_session: AsyncMock
    ) -> None:
        """Shared lock request should not conflict with existing shared lock."""
        existing = _make_file_lock(
            file_path="src/main.py",
            task_id="task-other",
            lock_type=LockType.SHARED,
        )
        mock_session.execute.return_value = FakeSelectResult([existing])

        conflicts = await locker.detect_conflicts(
            "task-1", ["src/main.py"],
            lock_type=LockType.SHARED,
        )

        assert len(conflicts) == 0

    @pytest.mark.asyncio
    async def test_shared_conflicts_with_exclusive(
        self, locker: FileLocker, mock_session: AsyncMock
    ) -> None:
        """Shared lock request should conflict with existing exclusive lock."""
        existing = _make_file_lock(
            file_path="src/main.py",
            task_id="task-other",
            lock_type=LockType.EXCLUSIVE,
        )
        mock_session.execute.return_value = FakeSelectResult([existing])

        conflicts = await locker.detect_conflicts(
            "task-1", ["src/main.py"],
            lock_type=LockType.SHARED,
        )

        assert len(conflicts) == 1

    @pytest.mark.asyncio
    async def test_detects_multiple_conflicts(
        self, locker: FileLocker, mock_session: AsyncMock
    ) -> None:
        """Should report multiple conflicting files."""
        locks = [
            _make_file_lock(file_path="src/a.py", task_id="task-x"),
            _make_file_lock(file_path="src/b.py", task_id="task-y"),
        ]
        mock_session.execute.return_value = FakeSelectResult(locks)

        conflicts = await locker.detect_conflicts(
            "task-1", ["src/a.py", "src/b.py", "src/c.py"]
        )

        assert len(conflicts) == 2
        conflict_files = {c.file_path for c in conflicts}
        assert conflict_files == {"src/a.py", "src/b.py"}


# ---------------------------------------------------------------------------
# Lock Release
# ---------------------------------------------------------------------------


class TestReleaseLocks:
    """Tests for release_locks and release_worker_locks methods."""

    @pytest.mark.asyncio
    async def test_release_by_task_id(
        self, locker: FileLocker, mock_session: AsyncMock
    ) -> None:
        """Should release all active locks for a task."""
        mock_session.execute.return_value = FakeResult(rowcount=3)

        count = await locker.release_locks("task-1")

        assert count == 3
        mock_session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_release_no_locks_returns_zero(
        self, locker: FileLocker, mock_session: AsyncMock
    ) -> None:
        """Should return 0 when no active locks exist for the task."""
        mock_session.execute.return_value = FakeResult(rowcount=0)

        count = await locker.release_locks("task-nonexistent")

        assert count == 0

    @pytest.mark.asyncio
    async def test_release_by_worker_id(
        self, locker: FileLocker, mock_session: AsyncMock
    ) -> None:
        """Should release all active locks for a worker (crash recovery)."""
        mock_session.execute.return_value = FakeResult(rowcount=5)

        count = await locker.release_worker_locks("worker-1")

        assert count == 5
        mock_session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_release_worker_no_locks(
        self, locker: FileLocker, mock_session: AsyncMock
    ) -> None:
        """Should return 0 when worker has no active locks."""
        mock_session.execute.return_value = FakeResult(rowcount=0)

        count = await locker.release_worker_locks("worker-absent")

        assert count == 0


# ---------------------------------------------------------------------------
# Active Lock Queries
# ---------------------------------------------------------------------------


class TestGetActiveLocks:
    """Tests for get_active_locks and get_locks_for_task."""

    @pytest.mark.asyncio
    async def test_get_active_locks_returns_all(
        self, locker: FileLocker, mock_session: AsyncMock
    ) -> None:
        """Should return all active locks."""
        locks = [
            _make_file_lock(file_path="src/a.py"),
            _make_file_lock(file_path="src/b.py"),
        ]
        mock_session.execute.return_value = FakeSelectResult(locks)

        result = await locker.get_active_locks()

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_get_active_locks_empty(
        self, locker: FileLocker, mock_session: AsyncMock
    ) -> None:
        """Should return empty list when no active locks."""
        mock_session.execute.return_value = FakeSelectResult([])

        result = await locker.get_active_locks()

        assert result == []

    @pytest.mark.asyncio
    async def test_get_locks_for_task(
        self, locker: FileLocker, mock_session: AsyncMock
    ) -> None:
        """Should return locks for a specific task."""
        task_id = str(uuid.uuid4())
        locks = [
            _make_file_lock(file_path="src/a.py", task_id=task_id),
        ]
        mock_session.execute.return_value = FakeSelectResult(locks)

        result = await locker.get_locks_for_task(task_id)

        assert len(result) == 1
        assert result[0].task_id == task_id


# ---------------------------------------------------------------------------
# Stale Lock Cleanup
# ---------------------------------------------------------------------------


class TestCleanupStaleLocks:
    """Tests for cleanup_stale_locks method."""

    @pytest.mark.asyncio
    async def test_cleanup_removes_stale_locks(
        self, locker: FileLocker, mock_session: AsyncMock
    ) -> None:
        """Should release locks older than max_age_seconds."""
        mock_session.execute.return_value = FakeResult(rowcount=2)

        count = await locker.cleanup_stale_locks(max_age_seconds=1800)

        assert count == 2
        mock_session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_cleanup_no_stale_locks(
        self, locker: FileLocker, mock_session: AsyncMock
    ) -> None:
        """Should return 0 when no stale locks exist."""
        mock_session.execute.return_value = FakeResult(rowcount=0)

        count = await locker.cleanup_stale_locks(max_age_seconds=3600)

        assert count == 0

    @pytest.mark.asyncio
    async def test_cleanup_uses_custom_max_age(
        self, locker: FileLocker, mock_session: AsyncMock
    ) -> None:
        """Should respect the custom max_age_seconds parameter."""
        mock_session.execute.return_value = FakeResult(rowcount=1)

        count = await locker.cleanup_stale_locks(max_age_seconds=60)

        assert count == 1


# ---------------------------------------------------------------------------
# LockConflict Model
# ---------------------------------------------------------------------------


class TestLockConflictModel:
    """Tests for the LockConflict pydantic model."""

    def test_lock_conflict_fields(self) -> None:
        """LockConflict should store all required fields."""
        conflict = LockConflict(
            file_path="src/main.py",
            held_by_task_id="task-1",
            held_by_worker_id="worker-1",
            requesting_task_id="task-2",
            lock_type=LockType.EXCLUSIVE,
        )
        assert conflict.file_path == "src/main.py"
        assert conflict.held_by_task_id == "task-1"
        assert conflict.held_by_worker_id == "worker-1"
        assert conflict.requesting_task_id == "task-2"
        assert conflict.lock_type == LockType.EXCLUSIVE

    def test_lock_conflict_shared_type(self) -> None:
        """LockConflict should accept SHARED lock type."""
        conflict = LockConflict(
            file_path="src/main.py",
            held_by_task_id="task-1",
            held_by_worker_id="worker-1",
            requesting_task_id="task-2",
            lock_type=LockType.SHARED,
        )
        assert conflict.lock_type == LockType.SHARED


# ---------------------------------------------------------------------------
# LockType Enum
# ---------------------------------------------------------------------------


class TestLockTypeEnum:
    """Tests for the LockType enum."""

    def test_exclusive_value(self) -> None:
        assert LockType.EXCLUSIVE.value == "exclusive"

    def test_shared_value(self) -> None:
        assert LockType.SHARED.value == "shared"

    def test_is_string_enum(self) -> None:
        assert isinstance(LockType.EXCLUSIVE, str)
        assert isinstance(LockType.SHARED, str)
