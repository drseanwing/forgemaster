# FORGEMASTER Lessons Learned

**Version:** 1.0.0  
**Last Updated:** 2025-02-05T00:00:00Z  
**Total Lessons:** 0  
**Verified Lessons:** 0

---

## ⚠️ MANDATORY AGENT INSTRUCTIONS

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  WHEN FIXING ANY ISSUE DURING CODE REVIEW:                                   │
│                                                                              │
│  1. DOCUMENT the issue, root cause, and fix in this file                    │
│  2. USE the template provided below                                          │
│  3. COMMIT with message: "lesson: [LESSON_ID] - [SHORT_DESCRIPTION]"        │
│  4. UPDATE verification status after tests pass                              │
│                                                                              │
│  ALL ISSUES FOUND IN REVIEW MUST BE DOCUMENTED HERE                         │
│  NO EXCEPTIONS - THIS IS HOW THE SYSTEM LEARNS                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Lesson Entry Template

```markdown
### L-{NNNN}: {Short Title}

**Task ID:** {TASK_ID}  
**Phase:** {PHASE_NUMBER}  
**Discovered:** {ISO_TIMESTAMP}  
**Severity:** {CRITICAL | HIGH | MEDIUM | LOW}  
**Category:** {CATEGORY_TAG}  
**Verification Status:** {UNVERIFIED | VERIFIED | DISPUTED}  
**Confidence Score:** {0.0 - 1.0}

#### Symptom
{What was observed - the error message, behaviour, or failure}

#### Root Cause
{Why it happened - the underlying reason}

#### Fix Applied
{What was done to resolve it - specific code changes or configuration}

#### Files Affected
- `{path/to/file1}`
- `{path/to/file2}`

#### Prevention
{How to prevent this in future - patterns to follow or avoid}

#### Related Lessons
- {L-XXXX if related}

#### Verification
- Pre-fix test: {PASS | FAIL | N/A}
- Post-fix test: {PASS | FAIL | N/A}
- Test file: `{path/to/test}`
```

---

## Category Tags

| Tag | Description |
|-----|-------------|
| `python` | Python language patterns |
| `asyncio` | Async/await patterns |
| `sqlalchemy` | Database ORM issues |
| `postgresql` | PostgreSQL-specific |
| `pgvector` | Vector extension |
| `alembic` | Migration issues |
| `docker` | Container configuration |
| `rootless-docker` | Rootless Docker specifics |
| `git` | Git operations |
| `worktree` | Git worktree management |
| `claude-sdk` | Claude Agent SDK |
| `fastapi` | FastAPI framework |
| `structlog` | Logging configuration |
| `testing` | Test patterns |
| `ci-cd` | CI/CD pipeline |
| `security` | Security patterns |
| `performance` | Performance issues |
| `configuration` | Config management |
| `error-handling` | Error handling patterns |
| `type-hints` | Type annotation issues |

---

## Lessons by Phase

### Phase 1: Core Orchestrator (MVP)

*No lessons recorded yet.*

---

### Phase 2: Architecture Pipeline

*No lessons recorded yet.*

---

### Phase 3: Parallelisation

*No lessons recorded yet.*

---

### Phase 4: Review Cycles + Intelligence

*No lessons recorded yet.*

---

### Phase 5: Build/Deploy Pipeline

*No lessons recorded yet.*

---

### Phase 6: Resilience Hardening

*No lessons recorded yet.*

---

### Phase 7: Dashboard & API

*No lessons recorded yet.*

---

## Lessons by Category

### Python Patterns

*No lessons in this category yet.*

### Database (PostgreSQL / SQLAlchemy)

*No lessons in this category yet.*

### Docker / Infrastructure

*No lessons in this category yet.*

### Git Operations

*No lessons in this category yet.*

### Claude Agent SDK

*No lessons in this category yet.*

### Testing

*No lessons in this category yet.*

### Security

*No lessons in this category yet.*

### Error Handling

*No lessons in this category yet.*

---

## High-Value Lessons (Verified, High Confidence)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  These lessons have been verified through testing and have high confidence   │
│  scores. They should be included in agent context for relevant tasks.        │
└─────────────────────────────────────────────────────────────────────────────┘
```

*No verified high-value lessons yet.*

---

## Disputed Lessons

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  These lessons failed verification - symptom not reproducible or fix         │
│  ineffective. They should be reviewed and either corrected or archived.      │
└─────────────────────────────────────────────────────────────────────────────┘
```

*No disputed lessons.*

---

## Lesson Statistics

| Metric | Value |
|--------|-------|
| Total Lessons | 0 |
| Verified | 0 |
| Unverified | 0 |
| Disputed | 0 |
| Avg Confidence | N/A |

### By Severity

| Severity | Count |
|----------|-------|
| CRITICAL | 0 |
| HIGH | 0 |
| MEDIUM | 0 |
| LOW | 0 |

### By Category

| Category | Count |
|----------|-------|
| - | 0 |

---

## Review Finding → Lesson Workflow

When a specialist reviewer finds an issue:

```
1. REVIEWER identifies issue during code review
           │
           ▼
2. FIXER agent assigned to resolve issue
           │
           ▼
3. FIXER creates lesson entry (UNVERIFIED, confidence 0.5)
           │
           ▼
4. FIXER implements fix in code
           │
           ▼
5. FIXER runs relevant tests
           │
           ├── Tests pass → Update to VERIFIED, confidence 0.9
           │
           └── Tests fail → Fix again OR mark DISPUTED with reason
           │
           ▼
6. FIXER commits both code fix AND lesson entry
           │
           ▼
7. FIXER updates TASK-STATUS.md
```

---

## Example Lessons (Reference)

### L-0000: Example - Rootless Docker Volume Permissions

**Task ID:** P1-092  
**Phase:** 1  
**Discovered:** 2025-02-05T10:00:00Z  
**Severity:** HIGH  
**Category:** `rootless-docker`  
**Verification Status:** VERIFIED  
**Confidence Score:** 0.9

#### Symptom
Container fails to write to mounted volume with "Permission denied" error despite correct file permissions on host.

#### Root Cause
Rootless Docker runs containers with user namespace remapping. The container's root user (UID 0) is mapped to the host user's UID, but volume mounts don't automatically adjust ownership.

#### Fix Applied
Added `userns_mode: keep-id` to docker-compose.yml service definition:

```yaml
services:
  orchestrator:
    userns_mode: keep-id
    volumes:
      - ./data:/data
```

#### Files Affected
- `docker/docker-compose.yml`
- `docker/docker-compose.dev.yml`

#### Prevention
Always include `userns_mode: keep-id` in docker-compose services when using rootless Docker with volume mounts. Add this to project scaffolding templates.

#### Related Lessons
- None

#### Verification
- Pre-fix test: FAIL
- Post-fix test: PASS
- Test file: `tests/e2e/test_docker_deployment.py::test_volume_write`

---

### L-0001: Example - SQLAlchemy Async Session Scope

**Task ID:** P1-015  
**Phase:** 1  
**Discovered:** 2025-02-05T11:00:00Z  
**Severity:** MEDIUM  
**Category:** `sqlalchemy`, `asyncio`  
**Verification Status:** VERIFIED  
**Confidence Score:** 0.9

#### Symptom
"Instance is not bound to a Session" error when accessing lazy-loaded relationships after the async context manager exits.

#### Root Cause
SQLAlchemy async sessions don't support lazy loading outside the session context. Accessing relationships after `async with session:` block closes triggers detached instance errors.

#### Fix Applied
Used `selectinload()` to eagerly load relationships within the session:

```python
# Before (broken)
async with session.begin():
    project = await session.get(Project, project_id)
# project.tasks fails here

# After (fixed)
async with session.begin():
    stmt = select(Project).options(selectinload(Project.tasks)).where(Project.id == project_id)
    project = (await session.execute(stmt)).scalar_one()
# project.tasks works here
```

#### Files Affected
- `src/forgemaster/database/queries/project.py`

#### Prevention
Always use eager loading (`selectinload`, `joinedload`) for relationships that will be accessed outside the session context. Document this pattern in coding standards.

#### Related Lessons
- None

#### Verification
- Pre-fix test: FAIL
- Post-fix test: PASS
- Test file: `tests/integration/test_project_queries.py::test_get_project_with_tasks`

---

## Change Log

| Timestamp | Lesson ID | Action | Agent |
|-----------|-----------|--------|-------|
| 2025-02-05T00:00:00Z | - | Document created | orchestrator |

---

*This document captures institutional knowledge. Every fix teaches the system.*
