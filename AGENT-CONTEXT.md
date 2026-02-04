# FORGEMASTER Agent Context

**Version:** 1.0.0  
**Last Updated:** 2025-02-05T00:00:00Z  
**Current Phase:** 1 - Core Orchestrator (MVP)

---

## ⚠️ AGENT OPERATING REQUIREMENTS

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  ALL AGENTS MUST:                                                            │
│                                                                              │
│  1. READ this document before starting any task                             │
│  2. FOLLOW the coding standards specified here                              │
│  3. UPDATE TASK-STATUS.md on task completion                                │
│  4. COMMIT and PUSH changes before marking task complete                    │
│  5. REQUEST code review for all completed work                              │
│  6. FIX all issues identified in review (no deferrals)                      │
│  7. DOCUMENT fixes in LESSONS-LEARNED.md                                    │
│  8. INCLUDE relevant lessons from LESSONS-LEARNED.md in your approach       │
│                                                                              │
│  DO NOT proceed to next task until current task is DONE                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Project Overview

### What is FORGEMASTER?

FORGEMASTER is an autonomous development orchestration system that manages Claude Code agents for software development. It provides:

- Persistent task state across sessions
- Automatic error recovery
- Multi-agent parallel execution
- Periodic review cycles
- Lessons-learned intelligence

### Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Runtime | Python 3.12+ / asyncio | Core orchestration |
| Agent SDK | claude-agent-sdk | Claude API integration |
| Database | PostgreSQL 16 + pgvector | State persistence |
| Embeddings | Ollama (nomic-embed-text) | Vector generation |
| Git | GitPython | Source control |
| Docker | docker-py (rootless) | Container operations |
| Web | FastAPI + htmx + SSE | Dashboard/API |
| Logging | structlog (JSON) | Structured logging |
| Config | TOML | Configuration |

### Repository Structure

```
forgemaster/
├── .github/workflows/         # CI/CD
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── docker-compose.dev.yml
├── src/forgemaster/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── database/              # Models, migrations, queries
│   ├── orchestrator/          # Dispatcher, monitor, merger
│   ├── agents/                # Definitions, prompts, session
│   ├── intelligence/          # Lessons, embeddings, search
│   ├── pipeline/              # Git, Docker, deploy
│   ├── context/               # File generation
│   └── web/                   # FastAPI app
├── agent-templates/           # Jinja2 prompt templates
├── context-templates/         # Context file templates
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── scripts/
├── docs/
├── CLAUDE.md
├── pyproject.toml
└── README.md
```

---

## Agent Types

### Executor Agent

**Model:** claude-sonnet-4-5-20250929  
**Tools:** Read, Write, Edit, Bash, Grep, Glob  
**Purpose:** Feature implementation

**Use for:**
- Writing new code
- Implementing features
- Creating tests
- Documentation

**Do not use for:**
- Architecture decisions
- Complex debugging
- Security reviews

### Architect Agent

**Model:** claude-opus-4-5-20251101  
**Tools:** Read, Write, Bash, Grep, Glob  
**Purpose:** Architecture, complex debugging, merge review

**Use for:**
- System design
- Complex refactoring
- Merge conflict resolution
- Cross-cutting concerns

### Tester Agent

**Model:** claude-sonnet-4-5-20250929  
**Tools:** Read, Write, Bash  
**Purpose:** Test creation and execution

**Use for:**
- Unit test creation
- Integration test creation
- E2E test creation
- Test debugging

### Fixer Agent

**Model:** claude-sonnet-4-5-20250929  
**Tools:** Read, Write, Edit, Bash, Grep  
**Purpose:** Bug fixing

**Use for:**
- Fixing review findings
- Bug resolution
- Test fixes

### Reviewer Agents

**Model:** claude-sonnet-4-5-20250929 (opus for security/integration)  
**Tools:** Read, Grep, Glob, Bash(test)  
**Purpose:** Domain-specific code review

| Reviewer | Focus |
|----------|-------|
| Frontend | Component structure, React patterns, styling |
| Backend | API design, auth, data validation |
| Database | Schema, indexing, queries, migrations |
| Spec Compliance | Feature completeness |
| Security | Auth/authz, OWASP, secrets |
| Accessibility | WCAG, ARIA, keyboard nav |
| Integration | API contracts, naming |
| Dependency | Versions, licenses |
| Docker/Infra | Rootless compat, volumes |
| SCM/CI | Branch hygiene, workflows |
| Error Handling | Exceptions, retry, degradation |
| Documentation | README, API docs, comments |

---

## Coding Standards

### Python Standards

**Tooling:**
- Python 3.12+
- Formatter: Black (line length 100)
- Linter: Ruff
- Type Checker: mypy (strict mode)
- Type hints required on all function signatures

**Naming Conventions:**

| Element | Convention | Example |
|---------|------------|---------|
| Files | snake_case | `task_dispatcher.py` |
| Classes | PascalCase | `TaskDispatcher` |
| Functions | snake_case | `dispatch_task()` |
| Constants | UPPER_SNAKE | `MAX_RETRIES` |
| Private | underscore prefix | `_internal_method()` |

**Required Patterns:**

```python
# Module docstring - REQUIRED at top of every file
"""
Module description explaining purpose.

This module handles X, Y, Z.
"""

# Function docstring - REQUIRED for all public functions
async def dispatch_task(task: Task, worker: Worker) -> DispatchResult:
    """
    Dispatch a task to an available worker.
    
    Args:
        task: The task to dispatch
        worker: The worker to assign
        
    Returns:
        DispatchResult with status and session info
        
    Raises:
        WorkerBusyError: If worker is not available
    """
```

### Error Handling Pattern

```python
# REQUIRED for all external operations
try:
    result = await agent_session.query(prompt)
except AgentTimeoutError as e:
    logger.error("agent_timeout", 
                 session_id=session.id, 
                 task_id=task.id, 
                 elapsed_seconds=e.elapsed)
    await handle_timeout(session, task)
except AgentContextFullError as e:
    logger.warning("context_exhausted", session_id=session.id)
    await handle_context_exhaustion(session, task)
except Exception as e:
    logger.exception("unexpected_agent_error", session_id=session.id)
    await handle_unexpected_error(session, task, e)
```

### Logging Pattern

```python
import structlog
logger = structlog.get_logger(__name__)

# Always include identifiers for traceability
logger.info("task_dispatched", 
            task_id=task.id, 
            agent_type=task.agent_type,
            model=model, 
            worker_slot=slot)
```

### Commit Message Format

```
type(scope): brief description

Detailed explanation if needed.

Task: FM-{task_id}
```

**Types:** feat, fix, refactor, docs, test, chore, style, perf

---

## Phase 1 Context

### Phase Objective

Build the minimum viable orchestrator:
- Database with task queue
- Single-worker dispatcher
- Agent session management
- Basic context generation
- CLI for project/task management

### Key Decisions Made

1. **PostgreSQL + pgvector** for all storage (no Redis at this scale)
2. **Structlog with JSON** for machine-readable logs
3. **TOML configuration** (pyproject.toml compatible)
4. **Alembic** for database migrations
5. **GitPython** for git operations
6. **Rootless Docker** compatibility required

### Dependencies (pyproject.toml)

```toml
[project]
dependencies = [
    "sqlalchemy[asyncio]>=2.0",
    "asyncpg>=0.29",
    "alembic>=1.13",
    "pgvector>=0.2",
    "structlog>=24.1",
    "pydantic>=2.5",
    "pydantic-settings>=2.1",
    "toml>=0.10",
    "jinja2>=3.1",
    "gitpython>=3.1",
    "docker>=7.0",
    "httpx>=0.26",
    "typer>=0.9",
    "rich>=13.7",
]

[project.optional-dependencies]
test = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "pytest-cov>=4.1",
    "httpx>=0.26",
]
```

### Environment Variables

```bash
# Database
DATABASE_URL=postgresql+asyncpg://forgemaster:password@localhost:5432/forgemaster
PGVECTOR_ENABLED=true

# Ollama
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=nomic-embed-text

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
LOG_FILE=/data/forgemaster/logs/forgemaster.log
```

### Before Committing Checklist

```bash
# 1. Format code
black src/ tests/

# 2. Lint
ruff check src/ --fix

# 3. Type check
mypy src/ --strict

# 4. Run unit tests
pytest tests/unit/ -v

# 5. Run integration tests (if applicable)
pytest tests/integration/ -v

# 6. Commit with proper message
git add .
git commit -m "type(scope): description

Task: FM-{task_id}"
```

---

## Current Lessons Learned

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  No lessons recorded yet for Phase 1.                                        │
│                                                                              │
│  As issues are discovered and fixed, they will appear here automatically.    │
│  Agents should incorporate these lessons into their approach.                │
└─────────────────────────────────────────────────────────────────────────────┘
```

*When lessons are added to LESSONS-LEARNED.md with verification status VERIFIED, relevant ones will be included here for agent context.*

---

## Task Workflow

### Starting a Task

1. Check TASK-STATUS.md for your assigned task
2. Verify all dependencies are DONE
3. Read this AGENT-CONTEXT.md for current standards
4. Check LESSONS-LEARNED.md for relevant lessons
5. Create feature branch: `feat/{task_id}-{short-description}`
6. Update TASK-STATUS.md: status → RUNNING
7. Commit and push status update

### During Task Execution

1. Follow coding standards strictly
2. Write tests alongside code
3. Use structured logging
4. Handle errors gracefully
5. Document complex logic

### Completing a Task

1. Run all checks (format, lint, type, test)
2. Update TASK-STATUS.md: status → REVIEW
3. Commit and push all changes
4. Request code review

### After Code Review

**If issues found:**
1. Fix each issue
2. Document in LESSONS-LEARNED.md
3. Re-run all checks
4. Request re-review

**If approved:**
1. Update TASK-STATUS.md: status → DONE
2. Merge to integration branch
3. Delete feature branch

---

## Phase Completion Review Protocol

At the end of each phase, a comprehensive review is conducted:

### Review Order

1. **Backend Review** - API design, business logic
2. **Database Review** - Schema, queries, migrations
3. **Security Review** - Auth, secrets, OWASP
4. **Docker/Infra Review** - Containers, rootless compat
5. **Error Handling Review** - Exceptions, degradation
6. **Documentation Review** - Comments, README, API docs
7. **Frontend Review** - (Phase 7 only)
8. **Accessibility Review** - (Phase 7 only)

### Review Requirements

- Each reviewer examines all phase code
- Findings recorded with severity
- ALL findings must be fixed
- No "deferred" or "known issues"
- Each fix documented in LESSONS-LEARNED.md
- Re-review after fixes

### Sign-off Criteria

- [ ] All specialist reviews completed
- [ ] All findings resolved
- [ ] All fixes documented as lessons
- [ ] All tests passing
- [ ] TASK-STATUS.md updated
- [ ] Integration branch merged to main

---

## Communication Protocol

### Structured Output

All agents return results in standard JSON schema:

```json
{
  "task_id": "uuid",
  "status": "completed | partial | failed",
  "summary": "What was accomplished",
  "files_modified": ["path/to/file.ts"],
  "files_created": ["path/to/new.ts"],
  "files_deleted": [],
  "tests_run": {
    "total": 5,
    "passed": 4,
    "failed": 1,
    "details": "test output summary"
  },
  "issues_discovered": [
    {
      "severity": "medium",
      "description": "Found issue",
      "location": "src/file.py:42",
      "suggested_fix": "Do this instead"
    }
  ],
  "lessons_learned": [
    {
      "symptom": "Error observed",
      "root_cause": "Why it happened",
      "fix": "What fixed it"
    }
  ],
  "handover_notes": "If continuation needed..."
}
```

### Issue Severity Levels

| Level | Response | Example |
|-------|----------|---------|
| CRITICAL | Immediate halt | DB connection lost |
| HIGH | Fix before continue | Test failure |
| MEDIUM | Fix in current task | Style violation |
| LOW | Fix if time permits | Minor refactor suggestion |

---

## Quick Reference

### Essential Commands

```bash
# Development environment
docker compose -f docker/docker-compose.dev.yml up -d

# Run tests
pytest tests/unit/ -v
pytest tests/integration/ -v
pytest tests/e2e/ -v

# Database migrations
alembic upgrade head
alembic revision --autogenerate -m "description"

# Lint and format
black src/ tests/
ruff check src/ --fix
mypy src/ --strict
```

### Essential Files

| File | Purpose |
|------|---------|
| `docs/TASK-STATUS.md` | Current task states |
| `docs/LESSONS-LEARNED.md` | Institutional knowledge |
| `docs/AGENT-CONTEXT.md` | This file |
| `CLAUDE.md` | Project context for Claude Code |
| `pyproject.toml` | Dependencies and config |

---

## Updates

When phase changes or significant lessons are learned, update this document:

| Date | Change | Author |
|------|--------|--------|
| 2025-02-05 | Initial creation | Orchestrator |

---

*This context document is regenerated at phase transitions and when high-value lessons are verified.*
