# FORGEMASTER Implementation Task List

**Version:** 1.0.0  
**Generated:** 2025-02-05  
**Source:** FORGEMASTER-Planning-Document-v3.md

---

## Task Notation Guide

| Symbol | Meaning |
|--------|---------|
| `[P#]` | Phase number |
| `[SEQ]` | Must be completed sequentially |
| `[PAR-X]` | Can be parallelised within group X |
| `🔴` | Critical path - blocks other tasks |
| `🟡` | High priority |
| `🟢` | Standard priority |
| `⚪` | Can be deferred within phase |

---

## Phase 1: Core Orchestrator (MVP)

### 1.1 Project Scaffolding [SEQ] 🔴

| ID | Task | Agent Type | Files Touched | Dependencies |
|----|------|------------|---------------|--------------|
| P1-001 | Create repository with standard Python project structure | executor | `pyproject.toml`, `src/forgemaster/__init__.py`, `.gitignore`, `README.md` | None |
| P1-002 | Configure pyproject.toml with all dependencies | executor | `pyproject.toml` | P1-001 |
| P1-003 | Create CLAUDE.md with project context | executor | `CLAUDE.md` | P1-001 |
| P1-004 | Set up GitHub Actions CI workflow skeleton | executor | `.github/workflows/ci.yml` | P1-001 |
| P1-005 | Create Docker directory structure | executor | `docker/Dockerfile`, `docker/docker-compose.yml`, `docker/docker-compose.dev.yml` | P1-001 |

### 1.2 Configuration System [SEQ] 🔴

| ID | Task | Agent Type | Files Touched | Dependencies |
|----|------|------------|---------------|--------------|
| P1-006 | Define configuration schema dataclasses | executor | `src/forgemaster/config.py` | P1-002 |
| P1-007 | Implement TOML configuration loader | executor | `src/forgemaster/config.py` | P1-006 |
| P1-008 | Add environment variable override support | executor | `src/forgemaster/config.py` | P1-007 |
| P1-009 | Create default configuration template | executor | `forgemaster.toml.example` | P1-007 |
| P1-010 | Write unit tests for configuration loading | tester | `tests/unit/test_config.py` | P1-008 |

### 1.3 Logging Infrastructure [SEQ] 🟡

| ID | Task | Agent Type | Files Touched | Dependencies |
|----|------|------------|---------------|--------------|
| P1-011 | Configure structlog with JSON output | executor | `src/forgemaster/logging.py` | P1-006 |
| P1-012 | Implement log file rotation handler | executor | `src/forgemaster/logging.py` | P1-011 |
| P1-013 | Add correlation ID middleware | executor | `src/forgemaster/logging.py` | P1-011 |
| P1-014 | Write unit tests for logging configuration | tester | `tests/unit/test_logging.py` | P1-013 |

### 1.4 Database Foundation [SEQ] 🔴

| ID | Task | Agent Type | Files Touched | Dependencies |
|----|------|------------|---------------|--------------|
| P1-015 | Create database connection manager | executor | `src/forgemaster/database/__init__.py`, `src/forgemaster/database/connection.py` | P1-011 |
| P1-016 | Define SQLAlchemy base model class | executor | `src/forgemaster/database/models/__init__.py`, `src/forgemaster/database/models/base.py` | P1-015 |
| P1-017 | Configure Alembic migration environment | executor | `alembic.ini`, `alembic/env.py`, `alembic/versions/` | P1-016 |
| P1-018 | Create projects table model | executor | `src/forgemaster/database/models/project.py` | P1-016 |
| P1-019 | Create tasks table model with state machine enum | executor | `src/forgemaster/database/models/task.py` | P1-016 |
| P1-020 | Create agent_sessions table model | executor | `src/forgemaster/database/models/session.py` | P1-016 |
| P1-021 | Create lessons_learned table model | executor | `src/forgemaster/database/models/lesson.py` | P1-016 |
| P1-022 | Create embedding_queue table model | executor | `src/forgemaster/database/models/embedding.py` | P1-016 |
| P1-023 | Generate initial Alembic migration | executor | `alembic/versions/001_initial_schema.py` | P1-018, P1-019, P1-020, P1-021, P1-022 |
| P1-024 | Enable pgvector extension in migration | executor | `alembic/versions/001_initial_schema.py` | P1-023 |
| P1-025 | Create database indexes migration | executor | `alembic/versions/002_indexes.py` | P1-024 |

### 1.5 Database Query Layer [PAR-A] 🟡

| ID | Task | Agent Type | Files Touched | Dependencies |
|----|------|------------|---------------|--------------|
| P1-026 | Implement project CRUD queries | executor | `src/forgemaster/database/queries/project.py` | P1-025 |
| P1-027 | Implement task CRUD queries | executor | `src/forgemaster/database/queries/task.py` | P1-025 |
| P1-028 | Implement session CRUD queries | executor | `src/forgemaster/database/queries/session.py` | P1-025 |
| P1-029 | Implement lesson CRUD queries | executor | `src/forgemaster/database/queries/lesson.py` | P1-025 |
| P1-030 | Implement embedding queue queries | executor | `src/forgemaster/database/queries/embedding.py` | P1-025 |

### 1.6 Database Tests [PAR-B] 🟢

| ID | Task | Agent Type | Files Touched | Dependencies |
|----|------|------------|---------------|--------------|
| P1-031 | Write integration tests for project queries | tester | `tests/integration/test_project_queries.py` | P1-026 |
| P1-032 | Write integration tests for task queries | tester | `tests/integration/test_task_queries.py` | P1-027 |
| P1-033 | Write integration tests for session queries | tester | `tests/integration/test_session_queries.py` | P1-028 |
| P1-034 | Write integration tests for lesson queries | tester | `tests/integration/test_lesson_queries.py` | P1-029 |

### 1.7 Task State Machine [SEQ] 🔴

| ID | Task | Agent Type | Files Touched | Dependencies |
|----|------|------------|---------------|--------------|
| P1-035 | Define task state enum with valid transitions | executor | `src/forgemaster/orchestrator/state_machine.py` | P1-019 |
| P1-036 | Implement state transition validator | executor | `src/forgemaster/orchestrator/state_machine.py` | P1-035 |
| P1-037 | Create state transition handler | executor | `src/forgemaster/orchestrator/state_machine.py` | P1-036 |
| P1-038 | Add dependency resolution logic | executor | `src/forgemaster/orchestrator/state_machine.py` | P1-037 |
| P1-039 | Write unit tests for state machine | tester | `tests/unit/test_state_machine.py` | P1-038 |

### 1.8 Agent Session Wrapper [SEQ] 🔴

| ID | Task | Agent Type | Files Touched | Dependencies |
|----|------|------------|---------------|--------------|
| P1-040 | Create Claude Agent SDK integration module | executor | `src/forgemaster/agents/__init__.py`, `src/forgemaster/agents/sdk_wrapper.py` | P1-011 |
| P1-041 | Implement agent session lifecycle manager | executor | `src/forgemaster/agents/session.py` | P1-040 |
| P1-042 | Add session health monitoring | executor | `src/forgemaster/agents/session.py` | P1-041 |
| P1-043 | Implement token counting tracker | executor | `src/forgemaster/agents/session.py` | P1-041 |
| P1-044 | Create agent result schema validator | executor | `src/forgemaster/agents/result_schema.py` | P1-040 |
| P1-045 | Implement result parsing logic | executor | `src/forgemaster/agents/result_schema.py` | P1-044 |
| P1-046 | Write unit tests for session wrapper | tester | `tests/unit/test_agent_session.py` | P1-045 |

### 1.9 Single Worker Dispatcher [SEQ] 🔴

| ID | Task | Agent Type | Files Touched | Dependencies |
|----|------|------------|---------------|--------------|
| P1-047 | Create dispatcher base class | executor | `src/forgemaster/orchestrator/__init__.py`, `src/forgemaster/orchestrator/dispatcher.py` | P1-038, P1-041 |
| P1-048 | Implement task queue polling logic | executor | `src/forgemaster/orchestrator/dispatcher.py` | P1-047 |
| P1-049 | Add priority-based task selection | executor | `src/forgemaster/orchestrator/dispatcher.py` | P1-048 |
| P1-050 | Implement task assignment logic | executor | `src/forgemaster/orchestrator/dispatcher.py` | P1-049 |
| P1-051 | Create result handler callback | executor | `src/forgemaster/orchestrator/result_handler.py` | P1-045 |
| P1-052 | Implement lesson extraction from results | executor | `src/forgemaster/orchestrator/result_handler.py` | P1-051 |
| P1-053 | Write unit tests for dispatcher | tester | `tests/unit/test_dispatcher.py` | P1-052 |

### 1.10 Session Health Monitor [SEQ] 🟡

| ID | Task | Agent Type | Files Touched | Dependencies |
|----|------|------------|---------------|--------------|
| P1-054 | Create health monitor service | executor | `src/forgemaster/orchestrator/health_monitor.py` | P1-042 |
| P1-055 | Implement idle timeout detection | executor | `src/forgemaster/orchestrator/health_monitor.py` | P1-054 |
| P1-056 | Add session kill logic | executor | `src/forgemaster/orchestrator/health_monitor.py` | P1-055 |
| P1-057 | Implement retry scheduling | executor | `src/forgemaster/orchestrator/health_monitor.py` | P1-056 |
| P1-058 | Write unit tests for health monitor | tester | `tests/unit/test_health_monitor.py` | P1-057 |

### 1.11 Context Generation [SEQ] 🟡

| ID | Task | Agent Type | Files Touched | Dependencies |
|----|------|------------|---------------|--------------|
| P1-059 | Create Jinja2 template loader | executor | `src/forgemaster/context/__init__.py`, `src/forgemaster/context/loader.py` | P1-006 |
| P1-060 | Define base system prompt template | executor | `agent-templates/base.j2` | P1-059 |
| P1-061 | Create architecture context template | executor | `context-templates/architecture.j2` | P1-059 |
| P1-062 | Create standards context template | executor | `context-templates/standards.j2` | P1-059 |
| P1-063 | Implement context file generator | executor | `src/forgemaster/context/generator.py` | P1-061, P1-062 |
| P1-064 | Add task-specific context injection | executor | `src/forgemaster/context/generator.py` | P1-063 |
| P1-065 | Write unit tests for context generation | tester | `tests/unit/test_context_generator.py` | P1-064 |

### 1.12 Embedding System [SEQ] 🟢

| ID | Task | Agent Type | Files Touched | Dependencies |
|----|------|------------|---------------|--------------|
| P1-066 | Create Ollama client wrapper | executor | `src/forgemaster/intelligence/__init__.py`, `src/forgemaster/intelligence/ollama_client.py` | P1-011 |
| P1-067 | Implement embedding generation function | executor | `src/forgemaster/intelligence/embeddings.py` | P1-066 |
| P1-068 | Create embedding queue processor | executor | `src/forgemaster/intelligence/embedding_worker.py` | P1-067, P1-030 |
| P1-069 | Add fallback to OpenAI embeddings | executor | `src/forgemaster/intelligence/embeddings.py` | P1-067 |
| P1-070 | Write unit tests for embedding generation | tester | `tests/unit/test_embeddings.py` | P1-069 |

### 1.13 Git Operations [SEQ] 🟡

| ID | Task | Agent Type | Files Touched | Dependencies |
|----|------|------------|---------------|--------------|
| P1-071 | Create GitPython wrapper module | executor | `src/forgemaster/pipeline/__init__.py`, `src/forgemaster/pipeline/git_ops.py` | P1-011 |
| P1-072 | Implement branch creation function | executor | `src/forgemaster/pipeline/git_ops.py` | P1-071 |
| P1-073 | Implement commit function | executor | `src/forgemaster/pipeline/git_ops.py` | P1-072 |
| P1-074 | Implement merge function | executor | `src/forgemaster/pipeline/git_ops.py` | P1-073 |
| P1-075 | Add merge conflict detection | executor | `src/forgemaster/pipeline/git_ops.py` | P1-074 |
| P1-076 | Write integration tests for git operations | tester | `tests/integration/test_git_ops.py` | P1-075 |

### 1.14 CLI Interface [SEQ] 🟡

| ID | Task | Agent Type | Files Touched | Dependencies |
|----|------|------------|---------------|--------------|
| P1-077 | Create CLI entry point | executor | `src/forgemaster/main.py`, `src/forgemaster/cli/__init__.py` | P1-047 |
| P1-078 | Implement project create command | executor | `src/forgemaster/cli/project.py` | P1-077 |
| P1-079 | Implement project list command | executor | `src/forgemaster/cli/project.py` | P1-078 |
| P1-080 | Implement task create command | executor | `src/forgemaster/cli/task.py` | P1-077 |
| P1-081 | Implement task list command | executor | `src/forgemaster/cli/task.py` | P1-080 |
| P1-082 | Implement orchestrator start command | executor | `src/forgemaster/cli/orchestrator.py` | P1-077 |
| P1-083 | Write CLI integration tests | tester | `tests/integration/test_cli.py` | P1-082 |

### 1.15 systemd Integration [SEQ] 🟢

| ID | Task | Agent Type | Files Touched | Dependencies |
|----|------|------------|---------------|--------------|
| P1-084 | Create systemd service unit file | executor | `systemd/forgemaster.service` | P1-082 |
| P1-085 | Implement health check endpoint | executor | `src/forgemaster/health.py` | P1-082 |
| P1-086 | Add watchdog notification support | executor | `src/forgemaster/health.py` | P1-085 |

### 1.16 Docker Deployment [SEQ] 🟡

| ID | Task | Agent Type | Files Touched | Dependencies |
|----|------|------------|---------------|--------------|
| P1-087 | Write orchestrator Dockerfile | executor | `docker/Dockerfile` | P1-082 |
| P1-088 | Configure docker-compose.yml for production | executor | `docker/docker-compose.yml` | P1-087 |
| P1-089 | Configure docker-compose.dev.yml for development | executor | `docker/docker-compose.dev.yml` | P1-088 |
| P1-090 | Add PostgreSQL service to compose | executor | `docker/docker-compose.yml` | P1-088 |
| P1-091 | Add Ollama service to compose | executor | `docker/docker-compose.yml` | P1-090 |
| P1-092 | Configure rootless Docker compatibility | executor | `docker/docker-compose.yml` | P1-091 |
| P1-093 | Write Docker deployment tests | tester | `tests/e2e/test_docker_deployment.py` | P1-092 |

### 1.17 Secrets Injection Hook [SEQ] 🟡

| ID | Task | Agent Type | Files Touched | Dependencies |
|----|------|------------|---------------|--------------|
| P1-094 | Create inject-secrets.sh script | executor | `scripts/hooks/inject-secrets.sh` | P1-006 |
| P1-095 | Document hook installation procedure | executor | `docs/secrets-injection.md` | P1-094 |

---

## Phase 2: Architecture Pipeline

### 2.1 Specification Ingestion [SEQ] 🔴

| ID | Task | Agent Type | Files Touched | Dependencies |
|----|------|------------|---------------|--------------|
| P2-001 | Create specification parser module | executor | `src/forgemaster/architecture/__init__.py`, `src/forgemaster/architecture/spec_parser.py` | P1-063 |
| P2-002 | Implement markdown spec ingestion | executor | `src/forgemaster/architecture/spec_parser.py` | P2-001 |
| P2-003 | Implement JSON spec ingestion | executor | `src/forgemaster/architecture/spec_parser.py` | P2-002 |
| P2-004 | Add spec validation logic | executor | `src/forgemaster/architecture/spec_parser.py` | P2-003 |
| P2-005 | Write unit tests for spec parser | tester | `tests/unit/test_spec_parser.py` | P2-004 |

### 2.2 Interview Agent [SEQ] 🔴

| ID | Task | Agent Type | Files Touched | Dependencies |
|----|------|------------|---------------|--------------|
| P2-006 | Create interviewer agent definition | architect | `src/forgemaster/agents/definitions/interviewer.py` | P1-044 |
| P2-007 | Write interviewer system prompt template | architect | `agent-templates/interviewer.j2` | P2-006 |
| P2-008 | Implement question generation logic | executor | `src/forgemaster/architecture/interviewer.py` | P2-007 |
| P2-009 | Create spec clarification workflow | executor | `src/forgemaster/architecture/interviewer.py` | P2-008 |
| P2-010 | Write integration tests for interviewer | tester | `tests/integration/test_interviewer.py` | P2-009 |

### 2.3 Architect Agent [SEQ] 🔴

| ID | Task | Agent Type | Files Touched | Dependencies |
|----|------|------------|---------------|--------------|
| P2-011 | Create architect agent definition | architect | `src/forgemaster/agents/definitions/architect.py` | P1-044 |
| P2-012 | Write architect system prompt template | architect | `agent-templates/architect.j2` | P2-011 |
| P2-013 | Implement architecture document generator | executor | `src/forgemaster/architecture/architect.py` | P2-012 |
| P2-014 | Add technology decision framework | executor | `src/forgemaster/architecture/architect.py` | P2-013 |
| P2-015 | Write integration tests for architect | tester | `tests/integration/test_architect.py` | P2-014 |

### 2.4 Task Decomposition [SEQ] 🔴

| ID | Task | Agent Type | Files Touched | Dependencies |
|----|------|------------|---------------|--------------|
| P2-016 | Create planner agent definition | architect | `src/forgemaster/agents/definitions/planner.py` | P1-044 |
| P2-017 | Write planner system prompt template | architect | `agent-templates/planner.j2` | P2-016 |
| P2-018 | Implement task breakdown algorithm | executor | `src/forgemaster/architecture/planner.py` | P2-017 |
| P2-019 | Add dependency graph generator | executor | `src/forgemaster/architecture/planner.py` | P2-018 |
| P2-020 | Implement parallel group assignment | executor | `src/forgemaster/architecture/planner.py` | P2-019 |
| P2-021 | Write integration tests for planner | tester | `tests/integration/test_planner.py` | P2-020 |

### 2.5 Repository Scaffolding [SEQ] 🟡

| ID | Task | Agent Type | Files Touched | Dependencies |
|----|------|------------|---------------|--------------|
| P2-022 | Create repository template system | executor | `src/forgemaster/architecture/scaffolder.py` | P2-014 |
| P2-023 | Implement Python project scaffolding | executor | `src/forgemaster/architecture/templates/python/` | P2-022 |
| P2-024 | Implement TypeScript project scaffolding | executor | `src/forgemaster/architecture/templates/typescript/` | P2-022 |
| P2-025 | Add CLAUDE.md generator | executor | `src/forgemaster/architecture/scaffolder.py` | P2-023 |
| P2-026 | Write tests for scaffolding | tester | `tests/integration/test_scaffolder.py` | P2-025 |

### 2.6 Nginx Integration [SEQ] 🟢

| ID | Task | Agent Type | Files Touched | Dependencies |
|----|------|------------|---------------|--------------|
| P2-027 | Create nginx-proxy-add.sh script | executor | `scripts/nginx-proxy-add.sh` | P1-006 |
| P2-028 | Create nginx-proxy-remove.sh script | executor | `scripts/nginx-proxy-remove.sh` | P2-027 |
| P2-029 | Create nginx-proxy-modify.sh script | executor | `scripts/nginx-proxy-modify.sh` | P2-028 |
| P2-030 | Implement Hostinger DNS API integration | executor | `scripts/dns-hostinger.sh` | P2-027 |
| P2-031 | Document nginx automation usage | executor | `docs/nginx-automation.md` | P2-030 |

---

## Phase 3: Parallelisation

### 3.1 Git Worktree Management [SEQ] 🔴

| ID | Task | Agent Type | Files Touched | Dependencies |
|----|------|------------|---------------|--------------|
| P3-001 | Implement worktree creation function | executor | `src/forgemaster/pipeline/worktree.py` | P1-075 |
| P3-002 | Implement worktree cleanup function | executor | `src/forgemaster/pipeline/worktree.py` | P3-001 |
| P3-003 | Add worktree pool manager | executor | `src/forgemaster/pipeline/worktree.py` | P3-002 |
| P3-004 | Implement worktree-to-branch mapping | executor | `src/forgemaster/pipeline/worktree.py` | P3-003 |
| P3-005 | Write integration tests for worktree management | tester | `tests/integration/test_worktree.py` | P3-004 |

### 3.2 Multi-Worker Dispatcher [SEQ] 🔴

| ID | Task | Agent Type | Files Touched | Dependencies |
|----|------|------------|---------------|--------------|
| P3-006 | Extend dispatcher for multiple workers | executor | `src/forgemaster/orchestrator/dispatcher.py` | P3-004 |
| P3-007 | Implement worker slot allocation | executor | `src/forgemaster/orchestrator/dispatcher.py` | P3-006 |
| P3-008 | Add concurrent task limit enforcement | executor | `src/forgemaster/orchestrator/dispatcher.py` | P3-007 |
| P3-009 | Implement worker health tracking | executor | `src/forgemaster/orchestrator/dispatcher.py` | P3-008 |
| P3-010 | Write unit tests for multi-worker dispatcher | tester | `tests/unit/test_multi_worker_dispatcher.py` | P3-009 |

### 3.3 File Conflict Detection [SEQ] 🔴

| ID | Task | Agent Type | Files Touched | Dependencies |
|----|------|------------|---------------|--------------|
| P3-011 | Create file lock tracking table | executor | `src/forgemaster/database/models/file_lock.py`, `alembic/versions/003_file_locks.py` | P3-006 |
| P3-012 | Implement file lock acquisition | executor | `src/forgemaster/orchestrator/file_locker.py` | P3-011 |
| P3-013 | Implement file lock release | executor | `src/forgemaster/orchestrator/file_locker.py` | P3-012 |
| P3-014 | Add conflict detection before dispatch | executor | `src/forgemaster/orchestrator/file_locker.py` | P3-013 |
| P3-015 | Write unit tests for file locking | tester | `tests/unit/test_file_locker.py` | P3-014 |

### 3.4 Merge Coordinator [SEQ] 🟡

| ID | Task | Agent Type | Files Touched | Dependencies |
|----|------|------------|---------------|--------------|
| P3-016 | Create merge coordinator service | executor | `src/forgemaster/orchestrator/merge_coordinator.py` | P3-014 |
| P3-017 | Implement merge queue logic | executor | `src/forgemaster/orchestrator/merge_coordinator.py` | P3-016 |
| P3-018 | Add automatic merge attempt | executor | `src/forgemaster/orchestrator/merge_coordinator.py` | P3-017 |
| P3-019 | Implement conflict escalation to architect | executor | `src/forgemaster/orchestrator/merge_coordinator.py` | P3-018 |
| P3-020 | Write integration tests for merge coordinator | tester | `tests/integration/test_merge_coordinator.py` | P3-019 |

### 3.5 Parallel Group Scheduling [SEQ] 🟡

| ID | Task | Agent Type | Files Touched | Dependencies |
|----|------|------------|---------------|--------------|
| P3-021 | Implement parallel group detection | executor | `src/forgemaster/orchestrator/scheduler.py` | P3-006 |
| P3-022 | Add group-aware task selection | executor | `src/forgemaster/orchestrator/scheduler.py` | P3-021 |
| P3-023 | Implement group completion barrier | executor | `src/forgemaster/orchestrator/scheduler.py` | P3-022 |
| P3-024 | Write unit tests for parallel scheduling | tester | `tests/unit/test_parallel_scheduler.py` | P3-023 |

---

## Phase 4: Review Cycles + Intelligence

### 4.1 Review Cycle Orchestration [SEQ] 🔴

| ID | Task | Agent Type | Files Touched | Dependencies |
|----|------|------------|---------------|--------------|
| P4-001 | Create review cycle state machine | executor | `src/forgemaster/review/__init__.py`, `src/forgemaster/review/cycle.py` | P3-019 |
| P4-002 | Implement review trigger logic | executor | `src/forgemaster/review/cycle.py` | P4-001 |
| P4-003 | Add review task generation | executor | `src/forgemaster/review/cycle.py` | P4-002 |
| P4-004 | Implement review result aggregation | executor | `src/forgemaster/review/cycle.py` | P4-003 |
| P4-005 | Write unit tests for review cycle | tester | `tests/unit/test_review_cycle.py` | P4-004 |

### 4.2 Specialist Reviewer Agents [PAR-C] 🟡

| ID | Task | Agent Type | Files Touched | Dependencies |
|----|------|------------|---------------|--------------|
| P4-006 | Create frontend reviewer agent definition | architect | `src/forgemaster/agents/definitions/reviewer_frontend.py` | P4-001 |
| P4-007 | Create backend reviewer agent definition | architect | `src/forgemaster/agents/definitions/reviewer_backend.py` | P4-001 |
| P4-008 | Create database reviewer agent definition | architect | `src/forgemaster/agents/definitions/reviewer_database.py` | P4-001 |
| P4-009 | Create spec compliance reviewer agent definition | architect | `src/forgemaster/agents/definitions/reviewer_spec.py` | P4-001 |
| P4-010 | Create security reviewer agent definition | architect | `src/forgemaster/agents/definitions/reviewer_security.py` | P4-001 |
| P4-011 | Create accessibility reviewer agent definition | architect | `src/forgemaster/agents/definitions/reviewer_accessibility.py` | P4-001 |
| P4-012 | Create integration reviewer agent definition | architect | `src/forgemaster/agents/definitions/reviewer_integration.py` | P4-001 |
| P4-013 | Create dependency reviewer agent definition | architect | `src/forgemaster/agents/definitions/reviewer_dependency.py` | P4-001 |
| P4-014 | Create Docker/infra reviewer agent definition | architect | `src/forgemaster/agents/definitions/reviewer_docker.py` | P4-001 |
| P4-015 | Create SCM/CI reviewer agent definition | architect | `src/forgemaster/agents/definitions/reviewer_scm.py` | P4-001 |
| P4-016 | Create error handling reviewer agent definition | architect | `src/forgemaster/agents/definitions/reviewer_errors.py` | P4-001 |
| P4-017 | Create documentation reviewer agent definition | architect | `src/forgemaster/agents/definitions/reviewer_docs.py` | P4-001 |

### 4.3 Reviewer Prompt Templates [PAR-D] 🟢

| ID | Task | Agent Type | Files Touched | Dependencies |
|----|------|------------|---------------|--------------|
| P4-018 | Write frontend reviewer prompt template | executor | `agent-templates/reviewer-frontend.j2` | P4-006 |
| P4-019 | Write backend reviewer prompt template | executor | `agent-templates/reviewer-backend.j2` | P4-007 |
| P4-020 | Write database reviewer prompt template | executor | `agent-templates/reviewer-database.j2` | P4-008 |
| P4-021 | Write spec compliance reviewer prompt template | executor | `agent-templates/reviewer-spec.j2` | P4-009 |
| P4-022 | Write security reviewer prompt template | executor | `agent-templates/reviewer-security.j2` | P4-010 |
| P4-023 | Write accessibility reviewer prompt template | executor | `agent-templates/reviewer-accessibility.j2` | P4-011 |
| P4-024 | Write integration reviewer prompt template | executor | `agent-templates/reviewer-integration.j2` | P4-012 |
| P4-025 | Write dependency reviewer prompt template | executor | `agent-templates/reviewer-dependency.j2` | P4-013 |
| P4-026 | Write Docker/infra reviewer prompt template | executor | `agent-templates/reviewer-docker.j2` | P4-014 |
| P4-027 | Write SCM/CI reviewer prompt template | executor | `agent-templates/reviewer-scm.j2` | P4-015 |
| P4-028 | Write error handling reviewer prompt template | executor | `agent-templates/reviewer-errors.j2` | P4-016 |
| P4-029 | Write documentation reviewer prompt template | executor | `agent-templates/reviewer-docs.j2` | P4-017 |

### 4.4 Finding Consolidation [SEQ] 🟡

| ID | Task | Agent Type | Files Touched | Dependencies |
|----|------|------------|---------------|--------------|
| P4-030 | Create finding deduplication logic | executor | `src/forgemaster/review/consolidator.py` | P4-004 |
| P4-031 | Implement finding severity ranking | executor | `src/forgemaster/review/consolidator.py` | P4-030 |
| P4-032 | Add fix task generation from findings | executor | `src/forgemaster/review/consolidator.py` | P4-031 |
| P4-033 | Write unit tests for finding consolidation | tester | `tests/unit/test_finding_consolidator.py` | P4-032 |

### 4.5 Lesson Verification Protocol [SEQ] 🟡

| ID | Task | Agent Type | Files Touched | Dependencies |
|----|------|------------|---------------|--------------|
| P4-034 | Implement lesson test discovery | executor | `src/forgemaster/intelligence/lesson_verifier.py` | P1-068 |
| P4-035 | Add pre-fix test execution | executor | `src/forgemaster/intelligence/lesson_verifier.py` | P4-034 |
| P4-036 | Add post-fix test execution | executor | `src/forgemaster/intelligence/lesson_verifier.py` | P4-035 |
| P4-037 | Implement verification status update | executor | `src/forgemaster/intelligence/lesson_verifier.py` | P4-036 |
| P4-038 | Write integration tests for lesson verification | tester | `tests/integration/test_lesson_verifier.py` | P4-037 |

### 4.6 Semantic Context Pre-selection [SEQ] 🟢

| ID | Task | Agent Type | Files Touched | Dependencies |
|----|------|------------|---------------|--------------|
| P4-039 | Implement dual search strategy | executor | `src/forgemaster/intelligence/context_search.py` | P1-068 |
| P4-040 | Add semantic similarity search | executor | `src/forgemaster/intelligence/context_search.py` | P4-039 |
| P4-041 | Add full-text keyword search | executor | `src/forgemaster/intelligence/context_search.py` | P4-040 |
| P4-042 | Add file overlap search | executor | `src/forgemaster/intelligence/context_search.py` | P4-041 |
| P4-043 | Implement result merging algorithm | executor | `src/forgemaster/intelligence/context_search.py` | P4-042 |
| P4-044 | Write unit tests for context search | tester | `tests/unit/test_context_search.py` | P4-043 |

---

## Phase 5: Build/Deploy Pipeline

### 5.1 Docker Build System [SEQ] 🔴

| ID | Task | Agent Type | Files Touched | Dependencies |
|----|------|------------|---------------|--------------|
| P5-001 | Create docker-py wrapper module | executor | `src/forgemaster/pipeline/docker_ops.py` | P3-019 |
| P5-002 | Implement image build function | executor | `src/forgemaster/pipeline/docker_ops.py` | P5-001 |
| P5-003 | Add rootless Docker compatibility checks | executor | `src/forgemaster/pipeline/docker_ops.py` | P5-002 |
| P5-004 | Implement build log streaming | executor | `src/forgemaster/pipeline/docker_ops.py` | P5-003 |
| P5-005 | Write integration tests for Docker build | tester | `tests/integration/test_docker_build.py` | P5-004 |

### 5.2 Image Tagging [SEQ] 🟡

| ID | Task | Agent Type | Files Touched | Dependencies |
|----|------|------------|---------------|--------------|
| P5-006 | Implement git SHA tagging | executor | `src/forgemaster/pipeline/docker_ops.py` | P5-004 |
| P5-007 | Add semantic version tagging | executor | `src/forgemaster/pipeline/docker_ops.py` | P5-006 |
| P5-008 | Implement latest tag management | executor | `src/forgemaster/pipeline/docker_ops.py` | P5-007 |

### 5.3 Registry Operations [SEQ] 🟡

| ID | Task | Agent Type | Files Touched | Dependencies |
|----|------|------------|---------------|--------------|
| P5-009 | Implement registry authentication | executor | `src/forgemaster/pipeline/registry.py` | P5-008 |
| P5-010 | Implement image push function | executor | `src/forgemaster/pipeline/registry.py` | P5-009 |
| P5-011 | Add push retry logic | executor | `src/forgemaster/pipeline/registry.py` | P5-010 |
| P5-012 | Write integration tests for registry operations | tester | `tests/integration/test_registry.py` | P5-011 |

### 5.4 Container Management [SEQ] 🟡

| ID | Task | Agent Type | Files Touched | Dependencies |
|----|------|------------|---------------|--------------|
| P5-013 | Implement container stop function | executor | `src/forgemaster/pipeline/container.py` | P5-011 |
| P5-014 | Implement container start function | executor | `src/forgemaster/pipeline/container.py` | P5-013 |
| P5-015 | Add compose service restart | executor | `src/forgemaster/pipeline/container.py` | P5-014 |
| P5-016 | Write integration tests for container management | tester | `tests/integration/test_container.py` | P5-015 |

### 5.5 Health Check System [SEQ] 🟡

| ID | Task | Agent Type | Files Touched | Dependencies |
|----|------|------------|---------------|--------------|
| P5-017 | Implement health endpoint poller | executor | `src/forgemaster/pipeline/health.py` | P5-015 |
| P5-018 | Add health check timeout handling | executor | `src/forgemaster/pipeline/health.py` | P5-017 |
| P5-019 | Implement rollback trigger logic | executor | `src/forgemaster/pipeline/health.py` | P5-018 |
| P5-020 | Add rollback execution function | executor | `src/forgemaster/pipeline/health.py` | P5-019 |
| P5-021 | Write integration tests for health check system | tester | `tests/integration/test_health_check.py` | P5-020 |

---

## Phase 6: Resilience Hardening

### 6.1 Session Handover Protocol [SEQ] 🔴

| ID | Task | Agent Type | Files Touched | Dependencies |
|----|------|------------|---------------|--------------|
| P6-001 | Implement context exhaustion detection | executor | `src/forgemaster/orchestrator/handover.py` | P5-020 |
| P6-002 | Create save-and-exit prompt injection | executor | `src/forgemaster/orchestrator/handover.py` | P6-001 |
| P6-003 | Implement handover context persistence | executor | `src/forgemaster/orchestrator/handover.py` | P6-002 |
| P6-004 | Add continuation session spawning | executor | `src/forgemaster/orchestrator/handover.py` | P6-003 |
| P6-005 | Write integration tests for session handover | tester | `tests/integration/test_handover.py` | P6-004 |

### 6.2 Crash Recovery [SEQ] 🔴

| ID | Task | Agent Type | Files Touched | Dependencies |
|----|------|------------|---------------|--------------|
| P6-006 | Implement orphan session detection | executor | `src/forgemaster/orchestrator/recovery.py` | P6-004 |
| P6-007 | Add session cleanup logic | executor | `src/forgemaster/orchestrator/recovery.py` | P6-006 |
| P6-008 | Implement task retry scheduling | executor | `src/forgemaster/orchestrator/recovery.py` | P6-007 |
| P6-009 | Add startup recovery routine | executor | `src/forgemaster/orchestrator/recovery.py` | P6-008 |
| P6-010 | Write integration tests for crash recovery | tester | `tests/integration/test_crash_recovery.py` | P6-009 |

### 6.3 Idle Watchdog [SEQ] 🟡

| ID | Task | Agent Type | Files Touched | Dependencies |
|----|------|------------|---------------|--------------|
| P6-011 | Implement activity timestamp tracking | executor | `src/forgemaster/orchestrator/watchdog.py` | P6-009 |
| P6-012 | Add idle detection logic | executor | `src/forgemaster/orchestrator/watchdog.py` | P6-011 |
| P6-013 | Implement watchdog kill action | executor | `src/forgemaster/orchestrator/watchdog.py` | P6-012 |
| P6-014 | Write unit tests for idle watchdog | tester | `tests/unit/test_watchdog.py` | P6-013 |

### 6.4 API Rate Limit Handling [SEQ] 🟡

| ID | Task | Agent Type | Files Touched | Dependencies |
|----|------|------------|---------------|--------------|
| P6-015 | Implement token bucket rate limiter | executor | `src/forgemaster/orchestrator/rate_limiter.py` | P6-009 |
| P6-016 | Add HTTP 429 response handler | executor | `src/forgemaster/orchestrator/rate_limiter.py` | P6-015 |
| P6-017 | Implement exponential backoff | executor | `src/forgemaster/orchestrator/rate_limiter.py` | P6-016 |
| P6-018 | Add adaptive parallelism reduction | executor | `src/forgemaster/orchestrator/rate_limiter.py` | P6-017 |
| P6-019 | Write unit tests for rate limiter | tester | `tests/unit/test_rate_limiter.py` | P6-018 |

### 6.5 E2E Test Suite [SEQ] 🟡

| ID | Task | Agent Type | Files Touched | Dependencies |
|----|------|------------|---------------|--------------|
| P6-020 | Create E2E test fixtures | tester | `tests/e2e/conftest.py` | P6-018 |
| P6-021 | Write full task lifecycle E2E test | tester | `tests/e2e/test_task_lifecycle.py` | P6-020 |
| P6-022 | Write parallel execution E2E test | tester | `tests/e2e/test_parallel_execution.py` | P6-021 |
| P6-023 | Write review cycle E2E test | tester | `tests/e2e/test_review_cycle.py` | P6-022 |
| P6-024 | Write resilience E2E test | tester | `tests/e2e/test_resilience.py` | P6-023 |

---

## Phase 7: Dashboard & API

### 7.1 FastAPI Foundation [SEQ] 🔴

| ID | Task | Agent Type | Files Touched | Dependencies |
|----|------|------------|---------------|--------------|
| P7-001 | Create FastAPI application factory | executor | `src/forgemaster/web/__init__.py`, `src/forgemaster/web/app.py` | P6-024 |
| P7-002 | Configure CORS middleware | executor | `src/forgemaster/web/app.py` | P7-001 |
| P7-003 | Add request logging middleware | executor | `src/forgemaster/web/middleware.py` | P7-002 |
| P7-004 | Implement health endpoint | executor | `src/forgemaster/web/routes/health.py` | P7-003 |
| P7-005 | Write unit tests for FastAPI setup | tester | `tests/unit/test_fastapi_app.py` | P7-004 |

### 7.2 REST API Endpoints [PAR-E] 🟡

| ID | Task | Agent Type | Files Touched | Dependencies |
|----|------|------------|---------------|--------------|
| P7-006 | Implement project CRUD endpoints | executor | `src/forgemaster/web/routes/projects.py` | P7-004 |
| P7-007 | Implement task CRUD endpoints | executor | `src/forgemaster/web/routes/tasks.py` | P7-004 |
| P7-008 | Implement session query endpoints | executor | `src/forgemaster/web/routes/sessions.py` | P7-004 |
| P7-009 | Implement lesson query endpoints | executor | `src/forgemaster/web/routes/lessons.py` | P7-004 |

### 7.3 REST API Tests [PAR-F] 🟢

| ID | Task | Agent Type | Files Touched | Dependencies |
|----|------|------------|---------------|--------------|
| P7-010 | Write tests for project endpoints | tester | `tests/integration/test_api_projects.py` | P7-006 |
| P7-011 | Write tests for task endpoints | tester | `tests/integration/test_api_tasks.py` | P7-007 |
| P7-012 | Write tests for session endpoints | tester | `tests/integration/test_api_sessions.py` | P7-008 |
| P7-013 | Write tests for lesson endpoints | tester | `tests/integration/test_api_lessons.py` | P7-009 |

### 7.4 Webhook System [SEQ] 🟡

| ID | Task | Agent Type | Files Touched | Dependencies |
|----|------|------------|---------------|--------------|
| P7-014 | Create webhook dispatcher module | executor | `src/forgemaster/web/webhooks.py` | P7-009 |
| P7-015 | Implement task completion webhook | executor | `src/forgemaster/web/webhooks.py` | P7-014 |
| P7-016 | Implement review cycle webhook | executor | `src/forgemaster/web/webhooks.py` | P7-015 |
| P7-017 | Implement build failure webhook | executor | `src/forgemaster/web/webhooks.py` | P7-016 |
| P7-018 | Implement deploy success webhook | executor | `src/forgemaster/web/webhooks.py` | P7-017 |
| P7-019 | Write integration tests for webhooks | tester | `tests/integration/test_webhooks.py` | P7-018 |

### 7.5 Server-Sent Events [SEQ] 🟡

| ID | Task | Agent Type | Files Touched | Dependencies |
|----|------|------------|---------------|--------------|
| P7-020 | Create SSE endpoint | executor | `src/forgemaster/web/routes/events.py` | P7-018 |
| P7-021 | Implement task status event | executor | `src/forgemaster/web/routes/events.py` | P7-020 |
| P7-022 | Implement session activity event | executor | `src/forgemaster/web/routes/events.py` | P7-021 |
| P7-023 | Write integration tests for SSE | tester | `tests/integration/test_sse.py` | P7-022 |

### 7.6 Dashboard UI [SEQ] 🟢

| ID | Task | Agent Type | Files Touched | Dependencies |
|----|------|------------|---------------|--------------|
| P7-024 | Create base HTML template with htmx | executor | `src/forgemaster/web/templates/base.html` | P7-022 |
| P7-025 | Implement task board view | executor | `src/forgemaster/web/templates/tasks.html` | P7-024 |
| P7-026 | Implement session logs viewer | executor | `src/forgemaster/web/templates/sessions.html` | P7-025 |
| P7-027 | Add real-time task updates | executor | `src/forgemaster/web/templates/tasks.html` | P7-026 |
| P7-028 | Write Playwright tests for dashboard | tester | `tests/e2e/test_dashboard.py` | P7-027 |

### 7.7 n8n Integration [SEQ] 🟢

| ID | Task | Agent Type | Files Touched | Dependencies |
|----|------|------------|---------------|--------------|
| P7-029 | Create n8n webhook client | executor | `src/forgemaster/integrations/n8n.py` | P7-018 |
| P7-030 | Implement notification payload formatter | executor | `src/forgemaster/integrations/n8n.py` | P7-029 |
| P7-031 | Add n8n integration tests | tester | `tests/integration/test_n8n.py` | P7-030 |

---

## Parallel Execution Groups Summary

| Group | Tasks | Safe to Run Together | Reason |
|-------|-------|---------------------|--------|
| PAR-A | P1-026 to P1-030 | Yes | Separate query files, no shared code |
| PAR-B | P1-031 to P1-034 | Yes | Independent test files |
| PAR-C | P4-006 to P4-017 | Yes | Independent agent definition files |
| PAR-D | P4-018 to P4-029 | Yes | Independent template files |
| PAR-E | P7-006 to P7-009 | Yes | Independent route files |
| PAR-F | P7-010 to P7-013 | Yes | Independent test files |

---

## Critical Path Summary

The critical path through the project is:

```
P1-001 → P1-006 → P1-015 → P1-016 → P1-023 → P1-035 → P1-040 → P1-047 →
P2-001 → P2-006 → P2-011 → P2-016 →
P3-001 → P3-006 → P3-011 → P3-016 →
P4-001 →
P5-001 →
P6-001 → P6-006 →
P7-001
```

Any delay on critical path tasks directly impacts project completion.

---

*Generated from FORGEMASTER-Planning-Document-v3.md*
