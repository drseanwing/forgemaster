"""Parallel group scheduler for Forgemaster orchestrator.

Implements scheduling logic that respects parallel group boundaries
and wave-based execution ordering. The scheduler analyses task
dependencies and ``parallel_group`` assignments to determine which
groups of tasks can be dispatched concurrently, and enforces barrier
synchronisation between groups/waves.

Key concepts:

- **Parallel group**: a set of tasks sharing the same ``parallel_group``
  value.  Tasks without a group are treated as singleton groups.
- **Wave**: groups that share no inter-group dependencies can run in
  the same wave.  Wave ordering is derived from topological sort of
  group-level dependencies.
- **Barrier**: before advancing to the next wave every group in the
  current wave must reach a terminal state (all tasks ``done`` or
  ``failed``).
- **Parallelisation type**: governs how many tasks within a group are
  returned at once (``FULL`` -- all; ``FILE_ISOLATED`` -- non-overlapping
  file sets; ``SEQUENTIAL`` -- one at a time).

The scheduler is consumed by ``MultiWorkerDispatcher`` which calls
``get_schedulable_tasks`` instead of its own ``_select_ready_tasks``.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from enum import Enum
from typing import Any, Callable

import structlog
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from forgemaster.database.models.task import Task, TaskStatus

logger = structlog.get_logger(__name__)

# Type alias matching the pattern used in dispatcher.py
SessionFactory = Callable[[], AsyncSession]


# ---------------------------------------------------------------------------
# Enums and Models
# ---------------------------------------------------------------------------


class GroupStatus(str, Enum):
    """Lifecycle states for a scheduled parallel group.

    Attributes:
        PENDING: Group has not yet been activated.
        ACTIVE: Group's tasks are eligible for dispatch.
        COMPLETING: Some tasks complete, waiting on remainder.
        COMPLETED: All tasks reached terminal state.
        FAILED: Group failed (all remaining tasks failed).
    """

    PENDING = "pending"
    ACTIVE = "active"
    COMPLETING = "completing"
    COMPLETED = "completed"
    FAILED = "failed"


class ScheduledGroup(BaseModel):
    """A parallel group enriched with scheduling metadata.

    Attributes:
        group_id: Unique identifier for the group.
        task_ids: UUIDs (as strings) of tasks belonging to this group.
        parallelization_type: Strategy governing intra-group concurrency.
        shared_files: File paths shared across group tasks.
        wave: Execution wave number (1-based).
        status: Current lifecycle state.
        tasks_completed: Number of tasks that finished successfully.
        tasks_failed: Number of tasks that failed.
        tasks_total: Total number of tasks in the group.
    """

    group_id: str
    task_ids: list[str] = Field(default_factory=list)
    parallelization_type: str = "FULL"
    shared_files: list[str] = Field(default_factory=list)
    wave: int = 1
    status: GroupStatus = GroupStatus.PENDING
    tasks_completed: int = 0
    tasks_failed: int = 0
    tasks_total: int = 0


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


class ParallelGroupScheduler:
    """Schedules tasks respecting parallel group boundaries.

    This scheduler analyses task dependencies and ``parallel_group``
    assignments to determine which groups of tasks can be dispatched
    concurrently, and implements barrier synchronisation between
    groups/waves.

    Attributes:
        session_factory: Callable returning an ``AsyncSession``.
    """

    def __init__(self, session_factory: SessionFactory) -> None:
        """Initialise the parallel group scheduler.

        Args:
            session_factory: Callable returning a new ``AsyncSession``.
        """
        self.session_factory = session_factory
        self._groups: dict[str, ScheduledGroup] = {}
        self._project_groups: dict[str, list[str]] = defaultdict(list)
        self._logger = logger.bind(component="ParallelGroupScheduler")

    # ------------------------------------------------------------------
    # Group Detection  (P3-021)
    # ------------------------------------------------------------------

    async def detect_parallel_groups(
        self,
        project_id: str,
    ) -> list[ScheduledGroup]:
        """Detect parallel groups from task data.

        Analyses tasks in the project and groups them by their
        ``parallel_group`` field.  Tasks without a ``parallel_group`` are
        treated as singleton groups.  Groups are ordered by dependency
        wave.

        Args:
            project_id: UUID string of the project.

        Returns:
            List of ``ScheduledGroup`` instances ordered by wave.
        """
        async with self.session_factory() as session:
            tasks = await self._fetch_project_tasks(session, project_id)

        # Bucket tasks by parallel_group
        group_buckets: dict[str, list[Task]] = defaultdict(list)
        singleton_counter = 0

        for task in tasks:
            if task.parallel_group:
                group_buckets[task.parallel_group].append(task)
            else:
                # Singleton group for ungrouped tasks
                singleton_counter += 1
                key = f"__singleton_{singleton_counter}"
                group_buckets[key].append(task)

        # Build ScheduledGroup instances
        groups: list[ScheduledGroup] = []
        for group_id, group_tasks in group_buckets.items():
            par_type = self._infer_parallelization_type(group_tasks)
            shared = self._detect_shared_files(group_tasks)

            sg = ScheduledGroup(
                group_id=group_id,
                task_ids=[str(t.id) for t in group_tasks],
                parallelization_type=par_type,
                shared_files=shared,
                wave=1,  # will be recomputed below
                status=GroupStatus.PENDING,
                tasks_completed=0,
                tasks_failed=0,
                tasks_total=len(group_tasks),
            )
            groups.append(sg)

        # Compute waves based on inter-group dependencies
        waves = self.compute_waves(groups, tasks)
        for sg in groups:
            # wave was set inside compute_waves
            pass

        # Sort by wave then group_id for determinism
        groups.sort(key=lambda g: (g.wave, g.group_id))

        # Cache
        self._groups = {g.group_id: g for g in groups}
        self._project_groups[project_id] = [g.group_id for g in groups]

        self._logger.info(
            "parallel_groups_detected",
            project_id=project_id,
            groups_count=len(groups),
            waves=len({g.wave for g in groups}),
        )

        return groups

    # ------------------------------------------------------------------
    # Group-Aware Task Selection  (P3-022)
    # ------------------------------------------------------------------

    async def get_next_schedulable_group(
        self,
        project_id: str,
    ) -> ScheduledGroup | None:
        """Get the next group ready for scheduling.

        A group is ready when:

        - All previous waves are completed.
        - The group's own status is ``PENDING``.

        Args:
            project_id: UUID string of the project.

        Returns:
            The next ``ScheduledGroup`` to activate, or ``None``.
        """
        group_ids = self._project_groups.get(project_id, [])
        groups = [self._groups[gid] for gid in group_ids if gid in self._groups]

        if not groups:
            return None

        # Determine the current wave: the smallest wave that has at
        # least one non-completed group.
        active_wave: int | None = None
        for g in groups:
            if g.status not in (GroupStatus.COMPLETED, GroupStatus.FAILED):
                if active_wave is None or g.wave < active_wave:
                    active_wave = g.wave

        if active_wave is None:
            return None  # everything done

        # Check if the preceding wave is fully completed
        preceding_waves_complete = all(
            g.status in (GroupStatus.COMPLETED, GroupStatus.FAILED)
            for g in groups
            if g.wave < active_wave
        )

        if not preceding_waves_complete:
            return None  # barrier not satisfied

        # Return the first PENDING group in the active wave
        for g in groups:
            if g.wave == active_wave and g.status == GroupStatus.PENDING:
                return g

        return None

    async def get_schedulable_tasks(
        self,
        project_id: str,
        max_tasks: int | None = None,
    ) -> list[Task]:
        """Get tasks that can be dispatched now.

        Respects group boundaries -- only returns tasks from the current
        active group(s).  Behaviour depends on parallelisation type:

        - ``FULL``: return all ``READY`` tasks in the group.
        - ``FILE_ISOLATED``: return ``READY`` tasks whose
          ``files_touched`` do not overlap with already-running tasks.
        - ``SEQUENTIAL`` / ``SEQ``: return only the first ``READY`` task
          by priority.

        Args:
            project_id: UUID string of the project.
            max_tasks: Optional cap on the number of tasks returned.

        Returns:
            List of ``Task`` instances eligible for dispatch.
        """
        group_ids = self._project_groups.get(project_id, [])
        active_groups = [
            self._groups[gid]
            for gid in group_ids
            if gid in self._groups
            and self._groups[gid].status == GroupStatus.ACTIVE
        ]

        if not active_groups:
            # Try to activate the next schedulable group first
            next_group = await self.get_next_schedulable_group(project_id)
            if next_group is not None:
                next_group.status = GroupStatus.ACTIVE
                active_groups = [next_group]
            else:
                return []

        schedulable: list[Task] = []

        async with self.session_factory() as session:
            for group in active_groups:
                tasks = await self._fetch_group_ready_tasks(session, group)

                if group.parallelization_type in ("FULL", "PAR-A"):
                    schedulable.extend(tasks)

                elif group.parallelization_type in ("FILE_ISOLATED", "PAR-B", "PAR-C"):
                    running_files = await self._get_running_files(session, group)
                    for task in tasks:
                        touched = set(task.files_touched or [])
                        if not touched & running_files:
                            schedulable.append(task)
                            # Add these files so subsequent tasks in the
                            # same loop iteration don't overlap.
                            running_files |= touched

                elif group.parallelization_type in ("SEQUENTIAL", "SEQ"):
                    # Only the highest-priority ready task
                    if tasks:
                        schedulable.append(tasks[0])

                else:
                    # Unknown type -- treat as FULL
                    schedulable.extend(tasks)

        # Apply cap
        if max_tasks is not None:
            schedulable = schedulable[:max_tasks]

        self._logger.info(
            "schedulable_tasks_selected",
            project_id=project_id,
            count=len(schedulable),
            groups=[g.group_id for g in active_groups],
        )

        return schedulable

    # ------------------------------------------------------------------
    # Group Completion Barrier  (P3-023)
    # ------------------------------------------------------------------

    async def check_group_completion(
        self,
        group_id: str,
    ) -> bool:
        """Check if all tasks in a group are complete.

        This is the barrier check.  Returns ``True`` when every task in
        the group has reached a terminal state (``done`` or ``failed``).

        Args:
            group_id: Identifier of the group to check.

        Returns:
            ``True`` if all tasks are in a terminal state.
        """
        group = self._groups.get(group_id)
        if group is None:
            self._logger.warning("check_completion_unknown_group", group_id=group_id)
            return False

        async with self.session_factory() as session:
            task_ids = [uuid.UUID(tid) for tid in group.task_ids]
            if not task_ids:
                return True

            stmt = select(Task).where(Task.id.in_(task_ids))
            result = await session.execute(stmt)
            tasks = list(result.scalars().all())

        terminal = {TaskStatus.done, TaskStatus.failed}
        completed = sum(1 for t in tasks if t.status == TaskStatus.done)
        failed = sum(1 for t in tasks if t.status == TaskStatus.failed)
        all_terminal = all(t.status in terminal for t in tasks)

        # Update cached counters
        group.tasks_completed = completed
        group.tasks_failed = failed

        if all_terminal:
            if failed == len(tasks):
                group.status = GroupStatus.FAILED
            else:
                group.status = GroupStatus.COMPLETED

            self._logger.info(
                "group_completed",
                group_id=group_id,
                completed=completed,
                failed=failed,
                status=group.status.value,
            )
        elif completed + failed > 0:
            group.status = GroupStatus.COMPLETING

        return all_terminal

    async def advance_to_next_wave(
        self,
        project_id: str,
    ) -> ScheduledGroup | None:
        """Advance to the next wave after current wave completes.

        Checks whether all groups in the current wave have reached a
        terminal state.  If so, activates the next wave's first pending
        group by transitioning its tasks from ``PENDING`` to ``READY``
        in the database.

        Args:
            project_id: UUID string of the project.

        Returns:
            The first ``ScheduledGroup`` in the new wave, or ``None``
            if no advancement is possible.
        """
        group_ids = self._project_groups.get(project_id, [])
        groups = [self._groups[gid] for gid in group_ids if gid in self._groups]

        if not groups:
            return None

        # Refresh completion state only for groups that have been
        # activated (ACTIVE or COMPLETING).  PENDING groups haven't
        # started yet and should not be checked.
        for g in groups:
            if g.status in (GroupStatus.ACTIVE, GroupStatus.COMPLETING):
                await self.check_group_completion(g.group_id)

        # Find the highest completed wave -- this is the wave whose
        # barrier has been satisfied.
        all_waves = sorted({g.wave for g in groups})
        highest_completed_wave: int | None = None

        for wave in all_waves:
            wave_groups = [g for g in groups if g.wave == wave]
            wave_done = all(
                g.status in (GroupStatus.COMPLETED, GroupStatus.FAILED)
                for g in wave_groups
            )
            if wave_done:
                highest_completed_wave = wave
            else:
                # If a wave has ACTIVE/COMPLETING groups, the barrier
                # for this wave hasn't been passed yet.  Stop scanning.
                has_active = any(
                    g.status in (GroupStatus.ACTIVE, GroupStatus.COMPLETING)
                    for g in wave_groups
                )
                if has_active:
                    return None  # barrier not satisfied
                # If the wave is entirely PENDING, we need to check if
                # the preceding wave was completed.  If so, this is the
                # next wave to activate.
                break

        if highest_completed_wave is None:
            return None  # no wave completed yet

        # Find the next wave after the highest completed one that has
        # at least one PENDING group.
        next_wave: int | None = None
        for wave in all_waves:
            if wave > highest_completed_wave:
                has_pending = any(
                    g.wave == wave and g.status == GroupStatus.PENDING
                    for g in groups
                )
                if has_pending:
                    next_wave = wave
                    break

        if next_wave is None:
            return None

        # Activate the first pending group in the next wave and
        # transition its tasks from PENDING to READY.
        next_group: ScheduledGroup | None = None
        for g in groups:
            if g.wave == next_wave and g.status == GroupStatus.PENDING:
                g.status = GroupStatus.ACTIVE
                if next_group is None:
                    next_group = g

        if next_group is not None:
            await self._promote_group_tasks(next_group)

            self._logger.info(
                "advanced_to_next_wave",
                project_id=project_id,
                from_wave=highest_completed_wave,
                to_wave=next_wave,
                group_id=next_group.group_id,
            )

        return next_group

    # ------------------------------------------------------------------
    # Scheduling Status
    # ------------------------------------------------------------------

    async def get_scheduling_status(
        self,
        project_id: str,
    ) -> dict[str, Any]:
        """Get overview of scheduling state: waves, groups, progress.

        Args:
            project_id: UUID string of the project.

        Returns:
            Dictionary containing wave count, per-group progress, and
            overall completion percentage.
        """
        group_ids = self._project_groups.get(project_id, [])
        groups = [self._groups[gid] for gid in group_ids if gid in self._groups]

        waves: dict[int, list[dict[str, Any]]] = defaultdict(list)
        total_tasks = 0
        total_completed = 0
        total_failed = 0

        for g in groups:
            total_tasks += g.tasks_total
            total_completed += g.tasks_completed
            total_failed += g.tasks_failed
            waves[g.wave].append({
                "group_id": g.group_id,
                "status": g.status.value,
                "parallelization_type": g.parallelization_type,
                "tasks_total": g.tasks_total,
                "tasks_completed": g.tasks_completed,
                "tasks_failed": g.tasks_failed,
                "shared_files": g.shared_files,
            })

        completion_pct = (
            ((total_completed + total_failed) / total_tasks * 100.0)
            if total_tasks > 0
            else 0.0
        )

        return {
            "project_id": project_id,
            "total_groups": len(groups),
            "total_waves": len(waves),
            "total_tasks": total_tasks,
            "total_completed": total_completed,
            "total_failed": total_failed,
            "completion_percentage": round(completion_pct, 1),
            "waves": {
                wave_num: wave_groups
                for wave_num, wave_groups in sorted(waves.items())
            },
        }

    # ------------------------------------------------------------------
    # Wave Computation
    # ------------------------------------------------------------------

    def compute_waves(
        self,
        groups: list[ScheduledGroup],
        tasks: list[Task] | None = None,
    ) -> dict[int, list[ScheduledGroup]]:
        """Organise groups into execution waves based on dependencies.

        Wave 1 contains groups whose tasks have no dependencies on tasks
        in other groups.  Wave 2 contains groups that depend on Wave 1
        groups, and so on.  This is a topological sort at the group
        level.

        Args:
            groups: List of ``ScheduledGroup`` instances to organise.
            tasks: Optional pre-fetched task list.  If ``None``, waves
                   default to 1 for all groups.

        Returns:
            Mapping from wave number to groups in that wave.
        """
        if not groups:
            return {}

        if tasks is None:
            # Without task data we cannot determine inter-group deps.
            for g in groups:
                g.wave = 1
            return {1: list(groups)}

        # Build task_id -> group_id mapping
        task_to_group: dict[str, str] = {}
        for g in groups:
            for tid in g.task_ids:
                task_to_group[tid] = g.group_id

        # Build a lookup from task UUID to task
        task_lookup: dict[str, Task] = {str(t.id): t for t in tasks}

        # Build group-level adjacency (group A -> group B if any task
        # in B depends on a task in A).
        group_deps: dict[str, set[str]] = defaultdict(set)
        group_ids_set = {g.group_id for g in groups}

        for g in groups:
            for tid in g.task_ids:
                task = task_lookup.get(tid)
                if task is None or not task.dependencies:
                    continue
                for dep_uuid in task.dependencies:
                    dep_id = str(dep_uuid)
                    dep_group = task_to_group.get(dep_id)
                    if dep_group and dep_group != g.group_id and dep_group in group_ids_set:
                        group_deps[g.group_id].add(dep_group)

        # Kahn's algorithm (topological sort)
        in_degree: dict[str, int] = {g.group_id: 0 for g in groups}
        adjacency: dict[str, list[str]] = defaultdict(list)

        for gid, deps in group_deps.items():
            in_degree[gid] = len(deps)
            for dep_gid in deps:
                adjacency[dep_gid].append(gid)

        wave_map: dict[int, list[ScheduledGroup]] = {}
        group_lookup = {g.group_id: g for g in groups}
        current_wave_ids = [gid for gid, deg in in_degree.items() if deg == 0]
        wave_num = 1

        while current_wave_ids:
            wave_groups: list[ScheduledGroup] = []
            next_wave_ids: list[str] = []

            for gid in current_wave_ids:
                g = group_lookup[gid]
                g.wave = wave_num
                wave_groups.append(g)

                for neighbour in adjacency.get(gid, []):
                    in_degree[neighbour] -= 1
                    if in_degree[neighbour] == 0:
                        next_wave_ids.append(neighbour)

            wave_map[wave_num] = wave_groups
            current_wave_ids = next_wave_ids
            wave_num += 1

        # Any remaining groups with unresolved in-degrees (cycles or
        # missing deps) get assigned to the final wave.
        remaining = [
            group_lookup[gid]
            for gid, deg in in_degree.items()
            if deg > 0
        ]
        if remaining:
            for g in remaining:
                g.wave = wave_num
            wave_map[wave_num] = remaining

        self._logger.info(
            "waves_computed",
            total_waves=len(wave_map),
            groups_per_wave={w: len(gs) for w, gs in wave_map.items()},
        )

        return wave_map

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _fetch_project_tasks(
        self,
        session: AsyncSession,
        project_id: str,
    ) -> list[Task]:
        """Fetch all tasks for a project.

        Args:
            session: Active database session.
            project_id: UUID string of the project.

        Returns:
            List of ``Task`` instances.
        """
        stmt = (
            select(Task)
            .where(Task.project_id == uuid.UUID(project_id))
            .order_by(Task.priority.asc(), Task.created_at.asc())
        )
        result = await session.execute(stmt)
        return list(result.scalars().all())

    async def _fetch_group_ready_tasks(
        self,
        session: AsyncSession,
        group: ScheduledGroup,
    ) -> list[Task]:
        """Fetch ``READY`` tasks belonging to a scheduled group.

        Args:
            session: Active database session.
            group: The group whose tasks to fetch.

        Returns:
            List of ``Task`` instances in ``READY`` status, sorted by
            priority.
        """
        task_ids = [uuid.UUID(tid) for tid in group.task_ids]
        if not task_ids:
            return []

        stmt = (
            select(Task)
            .where(
                Task.id.in_(task_ids),
                Task.status == TaskStatus.ready,
            )
            .order_by(Task.priority.asc(), Task.created_at.asc())
        )
        result = await session.execute(stmt)
        return list(result.scalars().all())

    async def _get_running_files(
        self,
        session: AsyncSession,
        group: ScheduledGroup,
    ) -> set[str]:
        """Get files touched by currently running tasks in a group.

        Args:
            session: Active database session.
            group: The group to inspect.

        Returns:
            Set of file path strings.
        """
        task_ids = [uuid.UUID(tid) for tid in group.task_ids]
        if not task_ids:
            return set()

        stmt = select(Task).where(
            Task.id.in_(task_ids),
            Task.status.in_([TaskStatus.running, TaskStatus.assigned]),
        )
        result = await session.execute(stmt)
        running = result.scalars().all()

        files: set[str] = set()
        for t in running:
            if t.files_touched:
                files.update(t.files_touched)
        return files

    async def _promote_group_tasks(
        self,
        group: ScheduledGroup,
    ) -> None:
        """Transition PENDING tasks in a group to READY.

        Args:
            group: Group whose tasks should be promoted.
        """
        task_ids = [uuid.UUID(tid) for tid in group.task_ids]
        if not task_ids:
            return

        async with self.session_factory() as session:
            stmt = select(Task).where(
                Task.id.in_(task_ids),
                Task.status == TaskStatus.pending,
            )
            result = await session.execute(stmt)
            pending_tasks = list(result.scalars().all())

            for task in pending_tasks:
                task.status = TaskStatus.ready

            await session.commit()

            self._logger.info(
                "group_tasks_promoted",
                group_id=group.group_id,
                promoted_count=len(pending_tasks),
            )

    # ------------------------------------------------------------------
    # Heuristic helpers
    # ------------------------------------------------------------------

    def _infer_parallelization_type(self, tasks: list[Task]) -> str:
        """Infer the parallelisation type for a group of tasks.

        Uses the ``parallel_group`` naming convention and file overlap
        analysis to decide whether the group is fully parallel, file
        isolated, or sequential.

        Args:
            tasks: Tasks in the group.

        Returns:
            Parallelisation type string.
        """
        if len(tasks) <= 1:
            return "SEQUENTIAL"

        # Check for file overlaps
        file_sets: list[set[str]] = []
        for t in tasks:
            file_sets.append(set(t.files_touched or []))

        has_overlap = False
        for i in range(len(file_sets)):
            for j in range(i + 1, len(file_sets)):
                if file_sets[i] & file_sets[j]:
                    has_overlap = True
                    break
            if has_overlap:
                break

        if not has_overlap:
            return "FULL"

        # Has overlap but separate file domains exist
        return "FILE_ISOLATED"

    def _detect_shared_files(self, tasks: list[Task]) -> list[str]:
        """Detect files modified by multiple tasks in a group.

        Args:
            tasks: Tasks in the group.

        Returns:
            Sorted list of shared file paths.
        """
        file_counts: dict[str, int] = defaultdict(int)
        for t in tasks:
            for fp in t.files_touched or []:
                file_counts[fp] += 1

        return sorted(fp for fp, count in file_counts.items() if count > 1)
