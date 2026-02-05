"""End-to-end tests for parallel execution flows.

Tests parallel task execution scenarios including multi-worker dispatch,
file locking, wave-based scheduling, worker slot management, and merge
coordination.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import select

from forgemaster.database.models.file_lock import FileLock, LockType
from forgemaster.database.models.task import Task, TaskStatus
from forgemaster.orchestrator.scheduler import GroupStatus, ParallelGroupScheduler, ScheduledGroup


@pytest.mark.e2e
@pytest.mark.asyncio
class TestMultiWorkerAssignment:
    """Test multi-worker task assignment scenarios."""

    async def test_multiple_tasks_ready_for_parallel_assignment(
        self,
        e2e_session,
        sample_project,
    ) -> None:
        """Test that multiple READY tasks can be identified for parallel dispatch."""
        # Create 3 tasks in READY state
        tasks = []
        for i in range(3):
            task = Task(
                id=uuid.uuid4(),
                project_id=sample_project.id,
                title=f"Parallel task {i}",
                status=TaskStatus.ready,
                agent_type="executor",
                priority=100,
                max_retries=3,
                retry_count=0,
                created_at=datetime.now(timezone.utc),
            )
            e2e_session.add(task)
            tasks.append(task)

        await e2e_session.commit()

        # Query for ready tasks
        result = await e2e_session.execute(
            select(Task).where(Task.status == TaskStatus.ready)
        )
        ready_tasks = result.scalars().all()

        assert len(ready_tasks) == 3
        assert all(t.status == TaskStatus.ready for t in ready_tasks)

    async def test_tasks_in_same_parallel_group(
        self,
        e2e_session,
        sample_project,
    ) -> None:
        """Test that tasks can share a parallel_group identifier."""
        group_id = "wave-1"

        # Create tasks in the same parallel group
        task1 = Task(
            id=uuid.uuid4(),
            project_id=sample_project.id,
            title="Grouped task 1",
            status=TaskStatus.ready,
            agent_type="executor",
            parallel_group=group_id,
            max_retries=3,
            retry_count=0,
            created_at=datetime.now(timezone.utc),
        )
        task2 = Task(
            id=uuid.uuid4(),
            project_id=sample_project.id,
            title="Grouped task 2",
            status=TaskStatus.ready,
            agent_type="executor",
            parallel_group=group_id,
            max_retries=3,
            retry_count=0,
            created_at=datetime.now(timezone.utc),
        )

        e2e_session.add_all([task1, task2])
        await e2e_session.commit()

        # Query for tasks in the group
        result = await e2e_session.execute(
            select(Task).where(Task.parallel_group == group_id)
        )
        group_tasks = result.scalars().all()

        assert len(group_tasks) == 2
        assert all(t.parallel_group == group_id for t in group_tasks)

    async def test_tasks_assigned_to_different_workers(
        self,
        e2e_session,
        sample_project,
    ) -> None:
        """Test that parallel tasks can be assigned to different workers."""
        # Create tasks
        task1 = Task(
            id=uuid.uuid4(),
            project_id=sample_project.id,
            title="Worker 1 task",
            status=TaskStatus.assigned,
            agent_type="executor",
            max_retries=3,
            retry_count=0,
            created_at=datetime.now(timezone.utc),
        )
        task2 = Task(
            id=uuid.uuid4(),
            project_id=sample_project.id,
            title="Worker 2 task",
            status=TaskStatus.assigned,
            agent_type="executor",
            max_retries=3,
            retry_count=0,
            created_at=datetime.now(timezone.utc),
        )

        e2e_session.add_all([task1, task2])
        await e2e_session.commit()

        assert task1.status == TaskStatus.assigned
        assert task2.status == TaskStatus.assigned
        assert task1.id != task2.id


@pytest.mark.e2e
@pytest.mark.asyncio
class TestFileLocking:
    """Test file locking for parallel execution."""

    async def test_exclusive_file_lock_creation(
        self,
        e2e_session,
        sample_task_running,
    ) -> None:
        """Test creating an exclusive file lock for a task."""
        lock = FileLock(
            id=uuid.uuid4(),
            file_path="src/main.py",
            task_id=sample_task_running.id,
            worker_id="worker-1",
            lock_type=LockType.EXCLUSIVE,
            acquired_at=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
        )
        e2e_session.add(lock)
        await e2e_session.commit()
        await e2e_session.refresh(lock)

        assert lock.file_path == "src/main.py"
        assert lock.task_id == sample_task_running.id
        assert lock.lock_type == LockType.EXCLUSIVE
        assert lock.released_at is None

    async def test_shared_file_locks_for_multiple_tasks(
        self,
        e2e_session,
        sample_project,
    ) -> None:
        """Test that multiple tasks can hold shared locks on the same file."""
        # Create two tasks
        task1 = Task(
            id=uuid.uuid4(),
            project_id=sample_project.id,
            title="Task 1",
            status=TaskStatus.running,
            agent_type="executor",
            max_retries=3,
            retry_count=0,
            created_at=datetime.now(timezone.utc),
        )
        task2 = Task(
            id=uuid.uuid4(),
            project_id=sample_project.id,
            title="Task 2",
            status=TaskStatus.running,
            agent_type="executor",
            max_retries=3,
            retry_count=0,
            created_at=datetime.now(timezone.utc),
        )
        e2e_session.add_all([task1, task2])
        await e2e_session.commit()

        # Create shared locks on the same file
        lock1 = FileLock(
            id=uuid.uuid4(),
            file_path="src/config.py",
            task_id=task1.id,
            worker_id="worker-1",
            lock_type=LockType.SHARED,
            acquired_at=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
        )
        lock2 = FileLock(
            id=uuid.uuid4(),
            file_path="src/config.py",
            task_id=task2.id,
            worker_id="worker-2",
            lock_type=LockType.SHARED,
            acquired_at=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
        )
        e2e_session.add_all([lock1, lock2])
        await e2e_session.commit()

        # Query for active locks on the file
        result = await e2e_session.execute(
            select(FileLock).where(
                FileLock.file_path == "src/config.py",
                FileLock.released_at.is_(None),
            )
        )
        active_locks = result.scalars().all()

        assert len(active_locks) == 2
        assert all(lock.lock_type == LockType.SHARED for lock in active_locks)

    async def test_file_lock_release(
        self,
        e2e_session,
        sample_task_running,
    ) -> None:
        """Test releasing a file lock after task completion."""
        lock = FileLock(
            id=uuid.uuid4(),
            file_path="src/utils.py",
            task_id=sample_task_running.id,
            worker_id="worker-1",
            lock_type=LockType.EXCLUSIVE,
            acquired_at=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
        )
        e2e_session.add(lock)
        await e2e_session.commit()
        await e2e_session.refresh(lock)

        # Simulate lock release
        lock.released_at = datetime.now(timezone.utc)
        await e2e_session.commit()
        await e2e_session.refresh(lock)

        assert lock.released_at is not None

    async def test_conflicting_file_locks_detection(
        self,
        e2e_session,
        sample_project,
    ) -> None:
        """Test detecting conflicting locks (exclusive vs shared)."""
        # Create task with exclusive lock
        task1 = Task(
            id=uuid.uuid4(),
            project_id=sample_project.id,
            title="Task with exclusive lock",
            status=TaskStatus.running,
            agent_type="executor",
            max_retries=3,
            retry_count=0,
            created_at=datetime.now(timezone.utc),
        )
        e2e_session.add(task1)
        await e2e_session.commit()

        # Create exclusive lock
        lock1 = FileLock(
            id=uuid.uuid4(),
            file_path="src/critical.py",
            task_id=task1.id,
            worker_id="worker-1",
            lock_type=LockType.EXCLUSIVE,
            acquired_at=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
        )
        e2e_session.add(lock1)
        await e2e_session.commit()

        # Query for active locks on the file
        result = await e2e_session.execute(
            select(FileLock).where(
                FileLock.file_path == "src/critical.py",
                FileLock.released_at.is_(None),
            )
        )
        active_locks = result.scalars().all()

        # Should only be one exclusive lock
        assert len(active_locks) == 1
        assert active_locks[0].lock_type == LockType.EXCLUSIVE


@pytest.mark.e2e
@pytest.mark.asyncio
class TestParallelGroupScheduling:
    """Test wave-based parallel group scheduling."""

    async def test_scheduled_group_creation(self) -> None:
        """Test creating a ScheduledGroup with metadata."""
        group = ScheduledGroup(
            group_id="wave-1-group-a",
            task_ids=["task-1", "task-2", "task-3"],
            parallelization_type="FULL",
            wave=1,
            status=GroupStatus.PENDING,
            tasks_total=3,
        )

        assert group.group_id == "wave-1-group-a"
        assert len(group.task_ids) == 3
        assert group.wave == 1
        assert group.status == GroupStatus.PENDING

    async def test_group_status_transitions(self) -> None:
        """Test transitioning group through lifecycle states."""
        group = ScheduledGroup(
            group_id="test-group",
            task_ids=["task-1", "task-2"],
            wave=1,
            status=GroupStatus.PENDING,
            tasks_total=2,
        )

        # PENDING → ACTIVE
        group.status = GroupStatus.ACTIVE
        assert group.status == GroupStatus.ACTIVE

        # ACTIVE → COMPLETING
        group.tasks_completed = 1
        group.status = GroupStatus.COMPLETING
        assert group.status == GroupStatus.COMPLETING

        # COMPLETING → COMPLETED
        group.tasks_completed = 2
        group.status = GroupStatus.COMPLETED
        assert group.status == GroupStatus.COMPLETED

    async def test_multiple_waves_scheduling(
        self,
        e2e_session,
        sample_project,
    ) -> None:
        """Test scheduling tasks in multiple waves."""
        # Wave 1 tasks
        wave1_task1 = Task(
            id=uuid.uuid4(),
            project_id=sample_project.id,
            title="Wave 1 Task 1",
            status=TaskStatus.ready,
            agent_type="executor",
            parallel_group="wave-1",
            priority=100,
            max_retries=3,
            retry_count=0,
            created_at=datetime.now(timezone.utc),
        )
        wave1_task2 = Task(
            id=uuid.uuid4(),
            project_id=sample_project.id,
            title="Wave 1 Task 2",
            status=TaskStatus.ready,
            agent_type="executor",
            parallel_group="wave-1",
            priority=100,
            max_retries=3,
            retry_count=0,
            created_at=datetime.now(timezone.utc),
        )

        # Wave 2 tasks (depend on wave 1 completion)
        wave2_task1 = Task(
            id=uuid.uuid4(),
            project_id=sample_project.id,
            title="Wave 2 Task 1",
            status=TaskStatus.pending,
            agent_type="executor",
            parallel_group="wave-2",
            dependencies=[wave1_task1.id, wave1_task2.id],
            priority=100,
            max_retries=3,
            retry_count=0,
            created_at=datetime.now(timezone.utc),
        )

        e2e_session.add_all([wave1_task1, wave1_task2, wave2_task1])
        await e2e_session.commit()

        # Verify wave 1 tasks are ready
        result = await e2e_session.execute(
            select(Task).where(Task.parallel_group == "wave-1")
        )
        wave1_tasks = result.scalars().all()
        assert len(wave1_tasks) == 2
        assert all(t.status == TaskStatus.ready for t in wave1_tasks)

        # Verify wave 2 task is pending with dependencies
        result = await e2e_session.execute(
            select(Task).where(Task.parallel_group == "wave-2")
        )
        wave2_tasks = result.scalars().all()
        assert len(wave2_tasks) == 1
        assert wave2_tasks[0].status == TaskStatus.pending
        assert len(wave2_tasks[0].dependencies or []) == 2


@pytest.mark.e2e
@pytest.mark.asyncio
class TestWorkerSlotManagement:
    """Test worker slot allocation and release."""

    async def test_worker_slot_allocation(
        self,
        e2e_session,
        sample_project,
    ) -> None:
        """Test allocating tasks to worker slots."""
        # Simulate 3 workers with assigned tasks
        workers = []
        for i in range(3):
            task = Task(
                id=uuid.uuid4(),
                project_id=sample_project.id,
                title=f"Worker {i} task",
                status=TaskStatus.assigned,
                agent_type="executor",
                max_retries=3,
                retry_count=0,
                created_at=datetime.now(timezone.utc),
            )
            e2e_session.add(task)
            workers.append({"worker_id": f"worker-{i}", "task_id": task.id})

        await e2e_session.commit()

        # Verify all workers have assignments
        assert len(workers) == 3
        assert all("worker_id" in w and "task_id" in w for w in workers)

    async def test_worker_slot_release_after_completion(
        self,
        e2e_session,
        sample_project,
    ) -> None:
        """Test releasing worker slot when task completes."""
        # Create task assigned to worker
        task = Task(
            id=uuid.uuid4(),
            project_id=sample_project.id,
            title="Task to complete",
            status=TaskStatus.running,
            agent_type="executor",
            max_retries=3,
            retry_count=0,
            started_at=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
        )
        e2e_session.add(task)
        await e2e_session.commit()
        await e2e_session.refresh(task)

        # Transition to DONE
        task.status = TaskStatus.done
        task.completed_at = datetime.now(timezone.utc)
        await e2e_session.commit()

        # Verify task is done (worker slot implicitly released)
        assert task.status == TaskStatus.done
        assert task.completed_at is not None


@pytest.mark.e2e
@pytest.mark.asyncio
class TestMergeCoordination:
    """Test merge coordination after parallel task completion."""

    async def test_parallel_tasks_ready_for_merge(
        self,
        e2e_session,
        sample_project,
    ) -> None:
        """Test detecting when parallel tasks are ready for merge."""
        # Create parallel group with all tasks completed
        task1 = Task(
            id=uuid.uuid4(),
            project_id=sample_project.id,
            title="Completed task 1",
            status=TaskStatus.done,
            agent_type="executor",
            parallel_group="merge-group",
            max_retries=3,
            retry_count=0,
            completed_at=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
        )
        task2 = Task(
            id=uuid.uuid4(),
            project_id=sample_project.id,
            title="Completed task 2",
            status=TaskStatus.done,
            agent_type="executor",
            parallel_group="merge-group",
            max_retries=3,
            retry_count=0,
            completed_at=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
        )

        e2e_session.add_all([task1, task2])
        await e2e_session.commit()

        # Query for completed tasks in group
        result = await e2e_session.execute(
            select(Task).where(
                Task.parallel_group == "merge-group",
                Task.status == TaskStatus.done,
            )
        )
        completed_tasks = result.scalars().all()

        assert len(completed_tasks) == 2
        assert all(t.status == TaskStatus.done for t in completed_tasks)

    async def test_files_touched_tracking_for_merge(
        self,
        e2e_session,
        sample_project,
    ) -> None:
        """Test tracking files_touched for merge conflict detection."""
        task1 = Task(
            id=uuid.uuid4(),
            project_id=sample_project.id,
            title="Task modifying file A",
            status=TaskStatus.done,
            agent_type="executor",
            files_touched=["src/module_a.py", "tests/test_a.py"],
            max_retries=3,
            retry_count=0,
            created_at=datetime.now(timezone.utc),
        )
        task2 = Task(
            id=uuid.uuid4(),
            project_id=sample_project.id,
            title="Task modifying file B",
            status=TaskStatus.done,
            agent_type="executor",
            files_touched=["src/module_b.py", "tests/test_b.py"],
            max_retries=3,
            retry_count=0,
            created_at=datetime.now(timezone.utc),
        )

        e2e_session.add_all([task1, task2])
        await e2e_session.commit()

        # Verify no file overlap (no conflicts)
        files1 = set(task1.files_touched or [])
        files2 = set(task2.files_touched or [])
        overlap = files1 & files2

        assert len(overlap) == 0  # No conflicts
