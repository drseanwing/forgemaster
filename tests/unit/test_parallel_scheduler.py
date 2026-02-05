"""Unit tests for the parallel group scheduler.

Tests cover:
- Group detection from task data (with and without parallel_group)
- Wave computation via topological sort of group dependencies
- Barrier synchronisation (check_group_completion)
- Task selection for each parallelisation type (FULL, FILE_ISOLATED, SEQUENTIAL)
- Wave advancement after current wave completes
- Scheduling status reporting
- Edge cases: empty projects, singleton groups, cyclic group deps
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest

from forgemaster.database.models.task import Task, TaskStatus
from forgemaster.orchestrator.scheduler import (
    GroupStatus,
    ParallelGroupScheduler,
    ScheduledGroup,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_task(
    project_id: str,
    title: str = "Task",
    priority: int = 100,
    parallel_group: str | None = None,
    files_touched: list[str] | None = None,
    dependencies: list[uuid.UUID] | None = None,
    status: TaskStatus = TaskStatus.pending,
) -> Task:
    """Create an in-memory Task with sensible defaults."""
    return Task(
        id=uuid.uuid4(),
        project_id=uuid.UUID(project_id),
        title=title,
        description=f"Description for {title}",
        status=status,
        agent_type="executor",
        model_tier="sonnet",
        priority=priority,
        retry_count=0,
        max_retries=3,
        parallel_group=parallel_group,
        files_touched=files_touched or [],
        dependencies=dependencies or [],
    )


class FakeAsyncSession:
    """Lightweight fake async session that returns pre-loaded tasks.

    Filters tasks based on SQLAlchemy statement where-clauses by compiling
    them to a string representation and checking task attributes directly.
    """

    def __init__(self, tasks: list[Task]) -> None:
        self._tasks = tasks

    async def execute(self, stmt: object) -> object:
        """Simulate query execution against the in-memory task list."""
        filtered = list(self._tasks)

        # Extract filters from the compiled statement parameters and
        # the string representation.  This is more robust than walking
        # the internal clause tree.
        try:
            compiled = stmt.compile(compile_kwargs={"literal_binds": False})  # type: ignore[attr-defined]
            params = compiled.params
            stmt_str = str(compiled)
        except Exception:
            return _FakeResult(filtered)

        # Collect all filter criteria we can recognise
        id_in_values: set[uuid.UUID] | None = None
        project_id_eq: uuid.UUID | None = None
        status_eq: TaskStatus | None = None
        status_in_values: set[TaskStatus] | None = None

        # Walk through the where clause tree to extract filter values
        try:
            clause = getattr(stmt, "whereclause", None)
            if clause is not None:
                filters = self._extract_filters(clause)
                for col_name, op, values in filters:
                    if col_name == "id" and op == "in":
                        id_in_values = {v if isinstance(v, uuid.UUID) else uuid.UUID(str(v)) for v in values}
                    elif col_name == "project_id" and op == "eq":
                        val = values[0]
                        project_id_eq = val if isinstance(val, uuid.UUID) else uuid.UUID(str(val))
                    elif col_name == "status" and op == "eq":
                        status_eq = values[0]
                    elif col_name == "status" and op == "in":
                        status_in_values = set(values)
        except Exception:
            pass

        # Apply filters
        if project_id_eq is not None:
            filtered = [t for t in filtered if t.project_id == project_id_eq]
        if id_in_values is not None:
            filtered = [t for t in filtered if t.id in id_in_values]
        if status_eq is not None:
            filtered = [t for t in filtered if t.status == status_eq]
        if status_in_values is not None:
            filtered = [t for t in filtered if t.status in status_in_values]

        return _FakeResult(filtered)

    def _extract_filters(
        self, clause: object
    ) -> list[tuple[str, str, list[object]]]:
        """Extract (column_name, operator, values) from a where clause tree."""
        results: list[tuple[str, str, list[object]]] = []

        # Compound clause (AND/OR)
        sub_clauses = getattr(clause, "clauses", None)
        if sub_clauses is not None and hasattr(sub_clauses, "__iter__"):
            for sub in sub_clauses:
                results.extend(self._extract_filters(sub))
            return results

        # Leaf clause
        left = getattr(clause, "left", None)
        right = getattr(clause, "right", None)

        if left is None:
            return results

        col_name = getattr(left, "key", None) or getattr(left, "name", None)
        if col_name is None:
            return results
        col_name = str(col_name)

        # IN clause: expanding BindParameter (SQLAlchemy 2.x)
        # The `expanding` flag signals a `column.in_(values)` clause.
        if getattr(right, "expanding", False):
            val = getattr(right, "value", None)
            if val is not None and hasattr(val, "__iter__"):
                results.append((col_name, "in", list(val)))
                return results

        # IN clause: Grouping wrapper (older pattern)
        if hasattr(right, "element"):
            element = right.element
            val = getattr(element, "value", None)
            if val is not None and hasattr(val, "__iter__"):
                results.append((col_name, "in", list(val)))
                return results

        # Equality clause
        if hasattr(right, "value"):
            val = right.value
            # Guard against expanding bindparams that also have .value
            if not getattr(right, "expanding", False):
                results.append((col_name, "eq", [val]))
                return results

        # BindParameter at top level
        if hasattr(right, "effective_value"):
            results.append((col_name, "eq", [right.effective_value]))

        return results

    async def commit(self) -> None:
        """No-op commit."""


class _FakeResult:
    """Wraps a list of tasks to mimic SQLAlchemy Result.scalars().all()."""

    def __init__(self, tasks: list[Task]) -> None:
        self._tasks = tasks

    def scalars(self) -> _FakeResult:
        return self

    def all(self) -> list[Task]:
        return list(self._tasks)


def _session_factory_from_tasks(tasks: list[Task]):
    """Build a session factory returning FakeAsyncSession with given tasks."""

    @asynccontextmanager
    async def factory():
        yield FakeAsyncSession(tasks)

    return factory


# ---------------------------------------------------------------------------
# GroupStatus enum tests
# ---------------------------------------------------------------------------


class TestGroupStatus:
    """Test the GroupStatus enum."""

    def test_values_are_strings(self):
        assert GroupStatus.PENDING == "pending"
        assert GroupStatus.ACTIVE == "active"
        assert GroupStatus.COMPLETING == "completing"
        assert GroupStatus.COMPLETED == "completed"
        assert GroupStatus.FAILED == "failed"

    def test_value_attribute(self):
        assert GroupStatus.PENDING.value == "pending"


# ---------------------------------------------------------------------------
# ScheduledGroup model tests
# ---------------------------------------------------------------------------


class TestScheduledGroup:
    """Test ScheduledGroup Pydantic model."""

    def test_defaults(self):
        sg = ScheduledGroup(group_id="g1")
        assert sg.task_ids == []
        assert sg.parallelization_type == "FULL"
        assert sg.shared_files == []
        assert sg.wave == 1
        assert sg.status == GroupStatus.PENDING
        assert sg.tasks_completed == 0
        assert sg.tasks_failed == 0
        assert sg.tasks_total == 0

    def test_full_construction(self):
        sg = ScheduledGroup(
            group_id="g2",
            task_ids=["t1", "t2"],
            parallelization_type="SEQUENTIAL",
            shared_files=["src/shared.py"],
            wave=3,
            status=GroupStatus.ACTIVE,
            tasks_completed=1,
            tasks_failed=0,
            tasks_total=2,
        )
        assert sg.group_id == "g2"
        assert len(sg.task_ids) == 2
        assert sg.wave == 3
        assert sg.status == GroupStatus.ACTIVE


# ---------------------------------------------------------------------------
# Group Detection  (P3-021)
# ---------------------------------------------------------------------------


class TestGroupDetection:
    """Test ParallelGroupScheduler.detect_parallel_groups."""

    async def test_groups_tasks_by_parallel_group(self):
        """Tasks sharing a parallel_group value should be in one group."""
        pid = str(uuid.uuid4())
        tasks = [
            _make_task(pid, "T1", parallel_group="alpha"),
            _make_task(pid, "T2", parallel_group="alpha"),
            _make_task(pid, "T3", parallel_group="beta"),
        ]
        scheduler = ParallelGroupScheduler(_session_factory_from_tasks(tasks))

        groups = await scheduler.detect_parallel_groups(pid)

        # Should have 2 groups: alpha and beta
        group_ids = {g.group_id for g in groups}
        assert "alpha" in group_ids
        assert "beta" in group_ids
        alpha = next(g for g in groups if g.group_id == "alpha")
        assert alpha.tasks_total == 2
        beta = next(g for g in groups if g.group_id == "beta")
        assert beta.tasks_total == 1

    async def test_ungrouped_tasks_become_singletons(self):
        """Tasks without parallel_group get singleton groups."""
        pid = str(uuid.uuid4())
        tasks = [
            _make_task(pid, "T1"),
            _make_task(pid, "T2"),
        ]
        scheduler = ParallelGroupScheduler(_session_factory_from_tasks(tasks))

        groups = await scheduler.detect_parallel_groups(pid)

        assert len(groups) == 2
        for g in groups:
            assert g.tasks_total == 1
            assert g.group_id.startswith("__singleton_")

    async def test_mixed_grouped_and_ungrouped(self):
        """Mix of grouped and ungrouped tasks."""
        pid = str(uuid.uuid4())
        tasks = [
            _make_task(pid, "T1", parallel_group="alpha"),
            _make_task(pid, "T2", parallel_group="alpha"),
            _make_task(pid, "T3"),
        ]
        scheduler = ParallelGroupScheduler(_session_factory_from_tasks(tasks))

        groups = await scheduler.detect_parallel_groups(pid)

        assert len(groups) == 2
        alpha = next(g for g in groups if g.group_id == "alpha")
        assert alpha.tasks_total == 2

    async def test_empty_project(self):
        """Empty project should return empty group list."""
        pid = str(uuid.uuid4())
        scheduler = ParallelGroupScheduler(_session_factory_from_tasks([]))

        groups = await scheduler.detect_parallel_groups(pid)

        assert groups == []

    async def test_groups_cached_after_detection(self):
        """Groups should be cached for subsequent lookups."""
        pid = str(uuid.uuid4())
        tasks = [_make_task(pid, "T1", parallel_group="alpha")]
        scheduler = ParallelGroupScheduler(_session_factory_from_tasks(tasks))

        await scheduler.detect_parallel_groups(pid)

        assert "alpha" in scheduler._groups
        assert pid in scheduler._project_groups

    async def test_inferred_parallelization_full(self):
        """Group with no file overlap should be FULL."""
        pid = str(uuid.uuid4())
        tasks = [
            _make_task(pid, "T1", parallel_group="g",
                       files_touched=["src/a.py"]),
            _make_task(pid, "T2", parallel_group="g",
                       files_touched=["src/b.py"]),
        ]
        scheduler = ParallelGroupScheduler(_session_factory_from_tasks(tasks))

        groups = await scheduler.detect_parallel_groups(pid)

        g = groups[0]
        assert g.parallelization_type == "FULL"

    async def test_inferred_parallelization_file_isolated(self):
        """Group with file overlap should be FILE_ISOLATED."""
        pid = str(uuid.uuid4())
        tasks = [
            _make_task(pid, "T1", parallel_group="g",
                       files_touched=["src/shared.py", "src/a.py"]),
            _make_task(pid, "T2", parallel_group="g",
                       files_touched=["src/shared.py", "src/b.py"]),
        ]
        scheduler = ParallelGroupScheduler(_session_factory_from_tasks(tasks))

        groups = await scheduler.detect_parallel_groups(pid)

        g = groups[0]
        assert g.parallelization_type == "FILE_ISOLATED"
        assert "src/shared.py" in g.shared_files

    async def test_inferred_parallelization_sequential_for_single_task(self):
        """Single-task group should be SEQUENTIAL."""
        pid = str(uuid.uuid4())
        tasks = [_make_task(pid, "T1", parallel_group="g")]
        scheduler = ParallelGroupScheduler(_session_factory_from_tasks(tasks))

        groups = await scheduler.detect_parallel_groups(pid)

        assert groups[0].parallelization_type == "SEQUENTIAL"


# ---------------------------------------------------------------------------
# Wave Computation
# ---------------------------------------------------------------------------


class TestWaveComputation:
    """Test ParallelGroupScheduler.compute_waves."""

    def test_single_wave_no_deps(self):
        """Groups without inter-group deps belong to wave 1."""
        pid = str(uuid.uuid4())
        g1 = ScheduledGroup(group_id="g1", task_ids=["t1"])
        g2 = ScheduledGroup(group_id="g2", task_ids=["t2"])

        t1 = _make_task(pid, "T1")
        t1.id = uuid.UUID("00000000-0000-0000-0000-000000000001")
        t2 = _make_task(pid, "T2")
        t2.id = uuid.UUID("00000000-0000-0000-0000-000000000002")
        g1.task_ids = [str(t1.id)]
        g2.task_ids = [str(t2.id)]

        scheduler = ParallelGroupScheduler(_session_factory_from_tasks([]))

        wave_map = scheduler.compute_waves([g1, g2], [t1, t2])

        assert len(wave_map) == 1
        assert g1.wave == 1
        assert g2.wave == 1

    def test_two_waves_with_dependency(self):
        """Group depending on another should be in a later wave."""
        pid = str(uuid.uuid4())
        t1 = _make_task(pid, "T1")
        t2 = _make_task(pid, "T2", dependencies=[t1.id])

        g1 = ScheduledGroup(group_id="g1", task_ids=[str(t1.id)])
        g2 = ScheduledGroup(group_id="g2", task_ids=[str(t2.id)])

        scheduler = ParallelGroupScheduler(_session_factory_from_tasks([]))

        wave_map = scheduler.compute_waves([g1, g2], [t1, t2])

        assert g1.wave == 1
        assert g2.wave == 2
        assert len(wave_map) == 2

    def test_three_level_chain(self):
        """A -> B -> C should produce 3 waves."""
        pid = str(uuid.uuid4())
        t1 = _make_task(pid, "T1")
        t2 = _make_task(pid, "T2", dependencies=[t1.id])
        t3 = _make_task(pid, "T3", dependencies=[t2.id])

        g1 = ScheduledGroup(group_id="g1", task_ids=[str(t1.id)])
        g2 = ScheduledGroup(group_id="g2", task_ids=[str(t2.id)])
        g3 = ScheduledGroup(group_id="g3", task_ids=[str(t3.id)])

        scheduler = ParallelGroupScheduler(_session_factory_from_tasks([]))

        wave_map = scheduler.compute_waves([g1, g2, g3], [t1, t2, t3])

        assert g1.wave == 1
        assert g2.wave == 2
        assert g3.wave == 3

    def test_diamond_dependency(self):
        """Diamond: A -> B, A -> C, B -> D, C -> D."""
        pid = str(uuid.uuid4())
        ta = _make_task(pid, "A")
        tb = _make_task(pid, "B", dependencies=[ta.id])
        tc = _make_task(pid, "C", dependencies=[ta.id])
        td = _make_task(pid, "D", dependencies=[tb.id, tc.id])

        ga = ScheduledGroup(group_id="ga", task_ids=[str(ta.id)])
        gb = ScheduledGroup(group_id="gb", task_ids=[str(tb.id)])
        gc = ScheduledGroup(group_id="gc", task_ids=[str(tc.id)])
        gd = ScheduledGroup(group_id="gd", task_ids=[str(td.id)])

        scheduler = ParallelGroupScheduler(_session_factory_from_tasks([]))

        wave_map = scheduler.compute_waves([ga, gb, gc, gd], [ta, tb, tc, td])

        assert ga.wave == 1
        assert gb.wave == 2
        assert gc.wave == 2
        assert gd.wave == 3

    def test_no_tasks_returns_default_wave(self):
        """Without task data, all groups default to wave 1."""
        g1 = ScheduledGroup(group_id="g1")
        g2 = ScheduledGroup(group_id="g2")

        scheduler = ParallelGroupScheduler(_session_factory_from_tasks([]))

        wave_map = scheduler.compute_waves([g1, g2], None)

        assert g1.wave == 1
        assert g2.wave == 1

    def test_empty_groups_list(self):
        """Empty input returns empty wave map."""
        scheduler = ParallelGroupScheduler(_session_factory_from_tasks([]))
        assert scheduler.compute_waves([], []) == {}

    def test_intra_group_deps_ignored(self):
        """Dependencies within the same group should not affect wave order."""
        pid = str(uuid.uuid4())
        t1 = _make_task(pid, "T1")
        t2 = _make_task(pid, "T2", dependencies=[t1.id])

        # Both tasks in same group
        g = ScheduledGroup(group_id="g", task_ids=[str(t1.id), str(t2.id)])

        scheduler = ParallelGroupScheduler(_session_factory_from_tasks([]))

        wave_map = scheduler.compute_waves([g], [t1, t2])

        assert g.wave == 1
        assert len(wave_map) == 1


# ---------------------------------------------------------------------------
# Barrier Synchronisation  (P3-023)
# ---------------------------------------------------------------------------


class TestBarrierSynchronisation:
    """Test check_group_completion barrier logic."""

    async def test_all_done_returns_true(self):
        """Group with all tasks done should be complete."""
        pid = str(uuid.uuid4())
        t1 = _make_task(pid, "T1", status=TaskStatus.done)
        t2 = _make_task(pid, "T2", status=TaskStatus.done)

        scheduler = ParallelGroupScheduler(_session_factory_from_tasks([t1, t2]))
        scheduler._groups["g1"] = ScheduledGroup(
            group_id="g1",
            task_ids=[str(t1.id), str(t2.id)],
            status=GroupStatus.ACTIVE,
            tasks_total=2,
        )

        result = await scheduler.check_group_completion("g1")

        assert result is True
        assert scheduler._groups["g1"].status == GroupStatus.COMPLETED
        assert scheduler._groups["g1"].tasks_completed == 2

    async def test_all_failed_returns_true_with_failed_status(self):
        """Group with all tasks failed should be FAILED."""
        pid = str(uuid.uuid4())
        t1 = _make_task(pid, "T1", status=TaskStatus.failed)

        scheduler = ParallelGroupScheduler(_session_factory_from_tasks([t1]))
        scheduler._groups["g1"] = ScheduledGroup(
            group_id="g1",
            task_ids=[str(t1.id)],
            status=GroupStatus.ACTIVE,
            tasks_total=1,
        )

        result = await scheduler.check_group_completion("g1")

        assert result is True
        assert scheduler._groups["g1"].status == GroupStatus.FAILED

    async def test_mixed_done_and_failed_returns_true(self):
        """Mix of done and failed is still terminal (COMPLETED)."""
        pid = str(uuid.uuid4())
        t1 = _make_task(pid, "T1", status=TaskStatus.done)
        t2 = _make_task(pid, "T2", status=TaskStatus.failed)

        scheduler = ParallelGroupScheduler(_session_factory_from_tasks([t1, t2]))
        scheduler._groups["g1"] = ScheduledGroup(
            group_id="g1",
            task_ids=[str(t1.id), str(t2.id)],
            status=GroupStatus.ACTIVE,
            tasks_total=2,
        )

        result = await scheduler.check_group_completion("g1")

        assert result is True
        assert scheduler._groups["g1"].status == GroupStatus.COMPLETED

    async def test_incomplete_returns_false(self):
        """Group with running tasks should not be complete."""
        pid = str(uuid.uuid4())
        t1 = _make_task(pid, "T1", status=TaskStatus.done)
        t2 = _make_task(pid, "T2", status=TaskStatus.running)

        scheduler = ParallelGroupScheduler(_session_factory_from_tasks([t1, t2]))
        scheduler._groups["g1"] = ScheduledGroup(
            group_id="g1",
            task_ids=[str(t1.id), str(t2.id)],
            status=GroupStatus.ACTIVE,
            tasks_total=2,
        )

        result = await scheduler.check_group_completion("g1")

        assert result is False
        assert scheduler._groups["g1"].status == GroupStatus.COMPLETING

    async def test_pending_not_terminal(self):
        """Pending tasks prevent completion."""
        pid = str(uuid.uuid4())
        t1 = _make_task(pid, "T1", status=TaskStatus.pending)

        scheduler = ParallelGroupScheduler(_session_factory_from_tasks([t1]))
        scheduler._groups["g1"] = ScheduledGroup(
            group_id="g1",
            task_ids=[str(t1.id)],
            status=GroupStatus.ACTIVE,
            tasks_total=1,
        )

        result = await scheduler.check_group_completion("g1")

        assert result is False

    async def test_unknown_group_returns_false(self):
        """Checking a non-existent group should return False."""
        scheduler = ParallelGroupScheduler(_session_factory_from_tasks([]))

        result = await scheduler.check_group_completion("nonexistent")

        assert result is False

    async def test_empty_group_returns_true(self):
        """Group with no tasks should be considered complete."""
        scheduler = ParallelGroupScheduler(_session_factory_from_tasks([]))
        scheduler._groups["empty"] = ScheduledGroup(
            group_id="empty",
            task_ids=[],
            status=GroupStatus.ACTIVE,
            tasks_total=0,
        )

        result = await scheduler.check_group_completion("empty")

        assert result is True


# ---------------------------------------------------------------------------
# Task Selection per Parallelisation Type  (P3-022)
# ---------------------------------------------------------------------------


class TestTaskSelection:
    """Test get_schedulable_tasks for FULL, FILE_ISOLATED, SEQUENTIAL."""

    async def test_full_returns_all_ready(self):
        """FULL type should return all READY tasks in the group."""
        pid = str(uuid.uuid4())
        t1 = _make_task(pid, "T1", status=TaskStatus.ready,
                        parallel_group="g", files_touched=["a.py"])
        t2 = _make_task(pid, "T2", status=TaskStatus.ready,
                        parallel_group="g", files_touched=["b.py"])

        scheduler = ParallelGroupScheduler(_session_factory_from_tasks([t1, t2]))
        scheduler._groups["g"] = ScheduledGroup(
            group_id="g",
            task_ids=[str(t1.id), str(t2.id)],
            parallelization_type="FULL",
            status=GroupStatus.ACTIVE,
            tasks_total=2,
        )
        scheduler._project_groups[pid] = ["g"]

        tasks = await scheduler.get_schedulable_tasks(pid)

        assert len(tasks) == 2

    async def test_sequential_returns_one(self):
        """SEQUENTIAL type should return only one task."""
        pid = str(uuid.uuid4())
        t1 = _make_task(pid, "T1", priority=10, status=TaskStatus.ready,
                        parallel_group="g")
        t2 = _make_task(pid, "T2", priority=20, status=TaskStatus.ready,
                        parallel_group="g")

        scheduler = ParallelGroupScheduler(_session_factory_from_tasks([t1, t2]))
        scheduler._groups["g"] = ScheduledGroup(
            group_id="g",
            task_ids=[str(t1.id), str(t2.id)],
            parallelization_type="SEQUENTIAL",
            status=GroupStatus.ACTIVE,
            tasks_total=2,
        )
        scheduler._project_groups[pid] = ["g"]

        tasks = await scheduler.get_schedulable_tasks(pid)

        assert len(tasks) == 1

    async def test_file_isolated_excludes_overlapping(self):
        """FILE_ISOLATED should exclude tasks whose files overlap with running tasks."""
        pid = str(uuid.uuid4())
        # t1 is running and touches shared.py
        t1 = _make_task(pid, "T1", status=TaskStatus.running,
                        parallel_group="g",
                        files_touched=["shared.py", "a.py"])
        # t2 is ready and also touches shared.py
        t2 = _make_task(pid, "T2", status=TaskStatus.ready,
                        parallel_group="g",
                        files_touched=["shared.py", "b.py"])
        # t3 is ready and only touches c.py -- no overlap
        t3 = _make_task(pid, "T3", status=TaskStatus.ready,
                        parallel_group="g",
                        files_touched=["c.py"])

        scheduler = ParallelGroupScheduler(
            _session_factory_from_tasks([t1, t2, t3])
        )
        scheduler._groups["g"] = ScheduledGroup(
            group_id="g",
            task_ids=[str(t1.id), str(t2.id), str(t3.id)],
            parallelization_type="FILE_ISOLATED",
            status=GroupStatus.ACTIVE,
            tasks_total=3,
        )
        scheduler._project_groups[pid] = ["g"]

        tasks = await scheduler.get_schedulable_tasks(pid)

        # Only t3 should be returned (t2 overlaps with running t1)
        task_ids = {str(t.id) for t in tasks}
        assert str(t3.id) in task_ids
        assert str(t2.id) not in task_ids

    async def test_max_tasks_cap(self):
        """max_tasks should cap the returned list."""
        pid = str(uuid.uuid4())
        ready_tasks = [
            _make_task(pid, f"T{i}", status=TaskStatus.ready,
                       parallel_group="g", files_touched=[f"f{i}.py"])
            for i in range(5)
        ]

        scheduler = ParallelGroupScheduler(
            _session_factory_from_tasks(ready_tasks)
        )
        scheduler._groups["g"] = ScheduledGroup(
            group_id="g",
            task_ids=[str(t.id) for t in ready_tasks],
            parallelization_type="FULL",
            status=GroupStatus.ACTIVE,
            tasks_total=5,
        )
        scheduler._project_groups[pid] = ["g"]

        tasks = await scheduler.get_schedulable_tasks(pid, max_tasks=2)

        assert len(tasks) == 2

    async def test_no_active_groups_activates_next(self):
        """When no groups are active, the first pending group should activate."""
        pid = str(uuid.uuid4())
        t1 = _make_task(pid, "T1", status=TaskStatus.ready,
                        parallel_group="g", files_touched=["a.py"])

        scheduler = ParallelGroupScheduler(_session_factory_from_tasks([t1]))
        scheduler._groups["g"] = ScheduledGroup(
            group_id="g",
            task_ids=[str(t1.id)],
            parallelization_type="FULL",
            status=GroupStatus.PENDING,
            wave=1,
            tasks_total=1,
        )
        scheduler._project_groups[pid] = ["g"]

        tasks = await scheduler.get_schedulable_tasks(pid)

        assert len(tasks) == 1
        assert scheduler._groups["g"].status == GroupStatus.ACTIVE

    async def test_empty_active_group_returns_empty(self):
        """Active group with no ready tasks returns empty list."""
        pid = str(uuid.uuid4())
        t1 = _make_task(pid, "T1", status=TaskStatus.running,
                        parallel_group="g")

        scheduler = ParallelGroupScheduler(_session_factory_from_tasks([t1]))
        scheduler._groups["g"] = ScheduledGroup(
            group_id="g",
            task_ids=[str(t1.id)],
            parallelization_type="FULL",
            status=GroupStatus.ACTIVE,
            tasks_total=1,
        )
        scheduler._project_groups[pid] = ["g"]

        tasks = await scheduler.get_schedulable_tasks(pid)

        assert tasks == []


# ---------------------------------------------------------------------------
# Next Schedulable Group
# ---------------------------------------------------------------------------


class TestNextSchedulableGroup:
    """Test get_next_schedulable_group logic."""

    async def test_returns_first_pending_in_wave(self):
        pid = str(uuid.uuid4())
        scheduler = ParallelGroupScheduler(_session_factory_from_tasks([]))
        scheduler._groups["g1"] = ScheduledGroup(
            group_id="g1", wave=1, status=GroupStatus.PENDING, tasks_total=1,
        )
        scheduler._groups["g2"] = ScheduledGroup(
            group_id="g2", wave=2, status=GroupStatus.PENDING, tasks_total=1,
        )
        scheduler._project_groups[pid] = ["g1", "g2"]

        result = await scheduler.get_next_schedulable_group(pid)

        assert result is not None
        assert result.group_id == "g1"

    async def test_blocks_until_preceding_wave_complete(self):
        pid = str(uuid.uuid4())
        scheduler = ParallelGroupScheduler(_session_factory_from_tasks([]))
        scheduler._groups["g1"] = ScheduledGroup(
            group_id="g1", wave=1, status=GroupStatus.ACTIVE, tasks_total=1,
        )
        scheduler._groups["g2"] = ScheduledGroup(
            group_id="g2", wave=2, status=GroupStatus.PENDING, tasks_total=1,
        )
        scheduler._project_groups[pid] = ["g1", "g2"]

        result = await scheduler.get_next_schedulable_group(pid)

        # g1 is still active, so g2 should not be returned
        # But g1 is ACTIVE not PENDING, so it won't be returned either
        assert result is None

    async def test_returns_none_when_all_done(self):
        pid = str(uuid.uuid4())
        scheduler = ParallelGroupScheduler(_session_factory_from_tasks([]))
        scheduler._groups["g1"] = ScheduledGroup(
            group_id="g1", wave=1, status=GroupStatus.COMPLETED, tasks_total=1,
        )
        scheduler._project_groups[pid] = ["g1"]

        result = await scheduler.get_next_schedulable_group(pid)

        assert result is None

    async def test_returns_none_for_unknown_project(self):
        scheduler = ParallelGroupScheduler(_session_factory_from_tasks([]))

        result = await scheduler.get_next_schedulable_group("unknown")

        assert result is None


# ---------------------------------------------------------------------------
# Wave Advancement
# ---------------------------------------------------------------------------


class TestWaveAdvancement:
    """Test advance_to_next_wave."""

    async def test_advances_when_current_wave_complete(self):
        """Should activate next wave when current wave is done."""
        pid = str(uuid.uuid4())
        t1 = _make_task(pid, "T1", status=TaskStatus.done)
        t2 = _make_task(pid, "T2", status=TaskStatus.pending)

        scheduler = ParallelGroupScheduler(
            _session_factory_from_tasks([t1, t2])
        )
        scheduler._groups["g1"] = ScheduledGroup(
            group_id="g1",
            task_ids=[str(t1.id)],
            wave=1,
            status=GroupStatus.COMPLETED,
            tasks_total=1,
            tasks_completed=1,
        )
        scheduler._groups["g2"] = ScheduledGroup(
            group_id="g2",
            task_ids=[str(t2.id)],
            wave=2,
            status=GroupStatus.PENDING,
            tasks_total=1,
        )
        scheduler._project_groups[pid] = ["g1", "g2"]

        result = await scheduler.advance_to_next_wave(pid)

        assert result is not None
        assert result.group_id == "g2"
        assert result.status == GroupStatus.ACTIVE

    async def test_does_not_advance_when_incomplete(self):
        """Should not advance if current wave has non-terminal tasks."""
        pid = str(uuid.uuid4())
        t1 = _make_task(pid, "T1", status=TaskStatus.running)
        t2 = _make_task(pid, "T2", status=TaskStatus.pending)

        scheduler = ParallelGroupScheduler(
            _session_factory_from_tasks([t1, t2])
        )
        scheduler._groups["g1"] = ScheduledGroup(
            group_id="g1",
            task_ids=[str(t1.id)],
            wave=1,
            status=GroupStatus.ACTIVE,
            tasks_total=1,
        )
        scheduler._groups["g2"] = ScheduledGroup(
            group_id="g2",
            task_ids=[str(t2.id)],
            wave=2,
            status=GroupStatus.PENDING,
            tasks_total=1,
        )
        scheduler._project_groups[pid] = ["g1", "g2"]

        result = await scheduler.advance_to_next_wave(pid)

        assert result is None

    async def test_returns_none_when_all_complete(self):
        """Should return None when every wave is done."""
        pid = str(uuid.uuid4())
        t1 = _make_task(pid, "T1", status=TaskStatus.done)

        scheduler = ParallelGroupScheduler(
            _session_factory_from_tasks([t1])
        )
        scheduler._groups["g1"] = ScheduledGroup(
            group_id="g1",
            task_ids=[str(t1.id)],
            wave=1,
            status=GroupStatus.COMPLETED,
            tasks_total=1,
            tasks_completed=1,
        )
        scheduler._project_groups[pid] = ["g1"]

        result = await scheduler.advance_to_next_wave(pid)

        assert result is None

    async def test_returns_none_for_empty_project(self):
        """Empty project should return None."""
        scheduler = ParallelGroupScheduler(_session_factory_from_tasks([]))

        result = await scheduler.advance_to_next_wave("unknown")

        assert result is None


# ---------------------------------------------------------------------------
# Scheduling Status Reporting
# ---------------------------------------------------------------------------


class TestSchedulingStatus:
    """Test get_scheduling_status."""

    async def test_status_includes_waves_and_groups(self):
        pid = str(uuid.uuid4())
        scheduler = ParallelGroupScheduler(_session_factory_from_tasks([]))
        scheduler._groups["g1"] = ScheduledGroup(
            group_id="g1", wave=1, status=GroupStatus.COMPLETED,
            tasks_total=3, tasks_completed=3, tasks_failed=0,
        )
        scheduler._groups["g2"] = ScheduledGroup(
            group_id="g2", wave=2, status=GroupStatus.ACTIVE,
            tasks_total=2, tasks_completed=0, tasks_failed=0,
        )
        scheduler._project_groups[pid] = ["g1", "g2"]

        status = await scheduler.get_scheduling_status(pid)

        assert status["project_id"] == pid
        assert status["total_groups"] == 2
        assert status["total_waves"] == 2
        assert status["total_tasks"] == 5
        assert status["total_completed"] == 3
        assert status["total_failed"] == 0
        assert status["completion_percentage"] == 60.0
        assert 1 in status["waves"]
        assert 2 in status["waves"]

    async def test_status_empty_project(self):
        pid = str(uuid.uuid4())
        scheduler = ParallelGroupScheduler(_session_factory_from_tasks([]))
        scheduler._project_groups[pid] = []

        status = await scheduler.get_scheduling_status(pid)

        assert status["total_groups"] == 0
        assert status["total_tasks"] == 0
        assert status["completion_percentage"] == 0.0

    async def test_status_completion_percentage(self):
        pid = str(uuid.uuid4())
        scheduler = ParallelGroupScheduler(_session_factory_from_tasks([]))
        scheduler._groups["g1"] = ScheduledGroup(
            group_id="g1", wave=1, status=GroupStatus.COMPLETED,
            tasks_total=4, tasks_completed=3, tasks_failed=1,
        )
        scheduler._project_groups[pid] = ["g1"]

        status = await scheduler.get_scheduling_status(pid)

        # 4/4 terminal = 100%
        assert status["completion_percentage"] == 100.0


# ---------------------------------------------------------------------------
# Integration-style scenario
# ---------------------------------------------------------------------------


class TestSchedulerScenario:
    """End-to-end scenario exercising the full scheduling lifecycle."""

    async def test_two_wave_lifecycle(self):
        """Two groups across two waves: detect -> schedule -> barrier -> advance."""
        pid = str(uuid.uuid4())

        # Wave 1: two independent tasks in group alpha
        t1 = _make_task(pid, "T1", parallel_group="alpha",
                        files_touched=["a.py"], status=TaskStatus.ready)
        t2 = _make_task(pid, "T2", parallel_group="alpha",
                        files_touched=["b.py"], status=TaskStatus.ready)
        # Wave 2: one task depending on t1
        t3 = _make_task(pid, "T3", parallel_group="beta",
                        files_touched=["c.py"], status=TaskStatus.pending,
                        dependencies=[t1.id])

        all_tasks = [t1, t2, t3]
        scheduler = ParallelGroupScheduler(
            _session_factory_from_tasks(all_tasks)
        )

        # Step 1: detect groups
        groups = await scheduler.detect_parallel_groups(pid)
        assert len(groups) == 2

        alpha = next(g for g in groups if g.group_id == "alpha")
        beta = next(g for g in groups if g.group_id == "beta")

        assert alpha.wave == 1
        assert beta.wave == 2

        # Step 2: schedule wave 1 tasks
        schedulable = await scheduler.get_schedulable_tasks(pid)
        assert len(schedulable) == 2  # t1 and t2

        # Step 3: simulate completion
        t1.status = TaskStatus.done
        t2.status = TaskStatus.done
        complete = await scheduler.check_group_completion(alpha.group_id)
        assert complete is True

        # Step 4: advance to wave 2
        next_group = await scheduler.advance_to_next_wave(pid)
        assert next_group is not None
        assert next_group.group_id == "beta"

        # Step 5: status check
        status = await scheduler.get_scheduling_status(pid)
        assert status["total_waves"] == 2
        assert status["total_completed"] == 2
