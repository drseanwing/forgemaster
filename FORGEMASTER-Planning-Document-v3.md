# FORGEMASTER: Autonomous Development Orchestration System

**Version:** 3.0.0  
**Date:** 2025-02-05  
**Status:** ARCHITECTURE FINALISED — Python-native implementation with infrastructure integration

---

## Quick Reference

| Component | Technology | Purpose |
|-----------|------------|---------|
| Orchestrator | Python 3.12+ / asyncio | Task dispatch, session management |
| Agent SDK | claude-agent-sdk | Claude API session management |
| Database | PostgreSQL 16 + pgvector | Task queue, state, semantic search |
| Embeddings | Ollama (nomic-embed-text) | Local vector generation |
| Git | GitPython | Worktree management |
| Docker | docker-py (rootless) | Container operations |
| Web/API | FastAPI + htmx + SSE | Dashboard, webhooks, events |
| Secrets | Claude Code hook injection | Runtime credential injection |
| Reverse Proxy | nginx-proxy-manager scripts | Automated subdomain/SSL setup |

---

## 1. Problem Statement

### Core Failures Addressed

| Failure | Description | Solution |
|---------|-------------|----------|
| Session Fragility | Context exhaustion causes hangs, auto-compaction loses state | External orchestrator with DB-persisted state |
| State Amnesia | New sessions start cold, must re-discover environment | Configuration stored in DB, projected to agents |
| Workflow Inconsistency | Ad-hoc reimplementation of git/docker operations | Deterministic pipeline operations (non-agent) |
| Task Drift | Volatile todo lists, granularity varies, completed tasks vanish | External task queue with enforced state machine |
| Silent Deferral | Issues filed mentally, no persistent record | Structured issue capture in agent result schema |
| Parallelisation Ceiling | Single-session subagent nesting limit | Multi-worktree parallel agent execution |

### Requirements Matrix

| Requirement | Priority | Implementation |
|-------------|----------|----------------|
| Persistent state across sessions | CRITICAL | PostgreSQL task/state storage |
| Automatic error recovery | CRITICAL | Session health monitor + retry logic |
| Standardised build/deploy | HIGH | Deterministic pipeline scripts |
| Granular task tracking | HIGH | DB task queue with state machine |
| Multi-agent parallel execution | HIGH | Git worktrees + file conflict detection |
| Spec → Architecture → Implementation | HIGH | Interview/Architect agent pipeline |
| Periodic review cycles | HIGH | 14 specialist reviewer agents |
| Model selection per task | MEDIUM | Automatic routing by complexity |
| Lessons-learned persistence | MEDIUM | pgvector semantic search |
| Remote VPS operation | HIGH | VPS-MCP + nginx automation |

---

## 2. Architecture Decisions

### Selected: Hybrid Python Orchestrator

Custom Python orchestrator using Claude Agent SDK with OMC-proven patterns (specialisation, ralph methodology, skill system) translated to versioned configuration.

### Technology Stack

```
Runtime:        Python 3.12+ / asyncio
Agent SDK:      claude-agent-sdk (Python binding)
Database:       PostgreSQL 16 + pgvector
Embedding:      Ollama (nomic-embed-text), fallback OpenAI
Git:            GitPython + subprocess (worktree management)
Docker:         docker-py (rootless compatible)
Scheduling:     apscheduler or asyncio timers
Events:         PostgreSQL LISTEN/NOTIFY
Dashboard:      FastAPI + htmx + SSE
Config:         TOML (pyproject.toml compatible)
Logging:        structlog (JSON to file)
Secrets:        Claude Code hook injection
Proxy:          nginx-proxy-manager automation
```

### Rejected Alternatives

| Approach | Reason for Rejection |
|----------|---------------------|
| n8n orchestration | Impedance mismatch (request/response vs 5-30min sessions), lost visibility on session health |
| Redis caching | Unnecessary at 3-5 worker scale; PostgreSQL + in-process state sufficient |
| External vector DB | pgvector extension is zero-overhead, co-located with relational data |
| Speculative code generation | Anchoring bias, misprediction waste, optimises wrong bottleneck |

### Conditional Future Additions

| Technology | Trigger Condition |
|------------|-------------------|
| Redis | 20+ parallel workers OR multi-instance orchestrator |
| External vector DB | >100K embeddings with latency requirements |
| Multi-model routing | Phase 8 after core stability proven |

---

## 3. System Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FORGEMASTER                                  │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌──────────────┐    ┌────────────────────┐     │
│  │  Spec Input  │───▶│  Architect   │───▶│  Task Generator    │     │
│  │  (API/CLI)   │    │  Agent       │    │  & Prioritiser     │     │
│  └─────────────┘    └──────────────┘    └────────┬───────────┘     │
│                                                   │                 │
│                                                   ▼                 │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │                    TASK QUEUE (PostgreSQL)                  │    │
│  │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐    │    │
│  │  │READY │ │READY │ │BLOCK │ │ RUN  │ │ RUN  │ │DONE  │    │    │
│  │  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘ └──────┘    │    │
│  └────────────────────────────────────────────────────────────┘    │
│          │                         ▲                                │
│          ▼                         │ results                       │
│  ┌───────────────────┐    ┌───────┴──────────┐                    │
│  │   DISPATCHER      │    │  RESULT HANDLER   │                    │
│  │   (parallel)      │    │  (lessons, merge) │                    │
│  └───────┬───────────┘    └───────────────────┘                    │
│          │                                                          │
│          ▼                                                          │
│  ┌───────────────────────────────────────────────────────────┐     │
│  │              AGENT WORKERS (Claude Agent SDK)              │     │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐      │     │
│  │  │Worker 1 │  │Worker 2 │  │Worker 3 │  │Worker N │      │     │
│  │  │Worktree │  │Worktree │  │Worktree │  │Worktree │      │     │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘      │     │
│  └───────────────────────────────────────────────────────────┘     │
│                            │                                        │
│                            ▼                                        │
│  ┌───────────────────────────────────────────────────────────┐     │
│  │                 DETERMINISTIC PIPELINE                     │     │
│  │  Git Ops → Docker Build → Registry Push → Deploy → Health │     │
│  └───────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
```

### Deployment Architecture

```
VPS (Azure)
├── Docker Compose (rootless)
│   ├── forgemaster-orchestrator    (Python, long-running)
│   ├── forgemaster-postgres        (PostgreSQL 16 + pgvector)
│   ├── forgemaster-ollama          (embedding generation)
│   └── forgemaster-dashboard       (FastAPI)
├── Git worktrees
│   ├── /workspace/main             (integration branch)
│   └── /workspace/worker-N         (feature branches)
├── Shared volumes
│   ├── /data/forgemaster/state     (DB data)
│   ├── /data/forgemaster/logs      (structured logs)
│   └── /data/forgemaster/context   (generated context files)
├── Nginx reverse proxies           (managed via nginx-proxy-manager)
│   └── forgemaster.vps.resuseducation.com
└── systemd
    └── forgemaster.service         (watchdog, auto-restart)
```

---

## 4. Infrastructure Automation

### 4.1 Secrets Injection System

Forgemaster uses a Claude Code PreToolUse hook to inject API keys at runtime without storing credentials in repositories.

**Location:** `~/.claude/hooks/inject-secrets.sh` (chmod 600)

**Hook Flow:**
```
Claude Code executes: docker compose up
           │
           ▼
PreToolUse hook fires (matcher: Bash)
inject-secrets.sh receives JSON via stdin
           │
           ▼
Command matches docker compose pattern?
           │
    ┌──────┴──────┐
    ▼             ▼
 No match      Match: prepend env vars
 (pass through) env VAR1=val1 VAR2=val2 docker compose up
           │
           ▼
Return modified tool_input via JSON
Docker Compose resolves ${VAR} placeholders
```

**Secret Registry (in inject-secrets.sh):**
```bash
declare -A SECRETS=(
    ["GITHUB_TOKEN"]="ghp_..."
    ["DOCKERHUB_USERNAME"]="..."
    ["DOCKERHUB_TOKEN"]="dckr_pat_..."
    ["ANTHROPIC_API_KEY"]="sk-ant-..."
    ["OPENAI_API_KEY"]="sk-..."
    ["N8N_PROD_API_KEY"]="n8n_api_..."
    ["N8N_DEV_API_KEY"]="n8n_api_..."
    ["HOSTINGER_API_TOKEN"]="..."
)
```

**Repository Pattern for Compose Files:**
```yaml
services:
  app:
    environment:
      - GITHUB_TOKEN=${GITHUB_TOKEN:?GITHUB_TOKEN must be set}
      - API_KEY=${ANTHROPIC_API_KEY:?ANTHROPIC_API_KEY must be set}
```

**Security Model:**
- File permissions: chmod 600 (owner read/write only)
- Location: Outside any git repo (~/.claude/hooks/)
- Equivalent security to ~/.ssh/id_rsa, ~/.docker/config.json
- Logs record secret names only, never values

### 4.2 Nginx Reverse Proxy Automation

Automated subdomain creation with DNS, SSL, and security hardening.

**Scripts:**
| Script | Purpose |
|--------|---------|
| `nginx-proxy-manager.sh` | Interactive menu for all operations |
| `nginx-proxy-add.sh` | Create new proxy with DNS + SSL |
| `nginx-proxy-remove.sh` | Remove proxy, DNS record, and SSL cert |
| `nginx-proxy-modify.sh` | Change internal/external ports |

**Add Proxy Flow:**
```
nginx-proxy-add.sh -s forgemaster -i 8080
           │
           ▼
[1] Add DNS A record via Hostinger API
    forgemaster.vps.resuseducation.com → VPS IP
           │
           ▼
[2] Wait for DNS propagation (poll every 10s)
           │
           ▼
[3] Create basic nginx config + enable site
           │
           ▼
[4] Obtain SSL certificate via Certbot
           │
           ▼
[5] Apply security hardening
    - TLS 1.2/1.3 only
    - Modern cipher suite
    - Security headers (HSTS, CSP, X-Frame-Options)
    - Rate limiting
    - WebSocket support
```

**Usage Examples:**
```bash
# Add proxy on standard HTTPS port
nginx-proxy-add -s forgemaster -i 8080

# Add proxy on custom port
nginx-proxy-add -s api -i 3000 -e 8443

# Modify existing proxy
nginx-proxy-modify -s forgemaster -i 9000

# Remove proxy completely
nginx-proxy-remove -s oldservice
```

**Environment Requirements:**
```bash
export HOSTINGER_API_TOKEN="your_token_here"
```

---

## 5. Database Schema

### Core Tables

```sql
-- Projects
CREATE TABLE projects (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    status project_status NOT NULL DEFAULT 'draft',
    spec_document JSONB,
    architecture_document JSONB,
    config JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Tasks with state machine
CREATE TABLE tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES projects(id),
    title TEXT NOT NULL,
    description TEXT,
    status task_status NOT NULL DEFAULT 'pending',
    agent_type TEXT NOT NULL,
    model_tier TEXT DEFAULT 'auto',
    priority INTEGER NOT NULL DEFAULT 100,
    estimated_minutes INTEGER,
    files_touched TEXT[],
    dependencies UUID[],
    parallel_group TEXT,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    description_embedding vector(768),
    created_at TIMESTAMPTZ DEFAULT now(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    
    CONSTRAINT valid_status CHECK (status IN (
        'pending', 'ready', 'assigned', 'running', 
        'review', 'done', 'failed', 'blocked'
    ))
);

-- Agent sessions
CREATE TABLE agent_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id UUID REFERENCES tasks(id),
    status session_status NOT NULL DEFAULT 'initialising',
    model TEXT NOT NULL,
    worktree_path TEXT,
    started_at TIMESTAMPTZ DEFAULT now(),
    last_activity_at TIMESTAMPTZ DEFAULT now(),
    ended_at TIMESTAMPTZ,
    token_count INTEGER DEFAULT 0,
    result JSONB,
    handover_context JSONB,
    error_message TEXT
);

-- Lessons learned with embeddings
CREATE TABLE lessons_learned (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES projects(id),
    task_id UUID REFERENCES tasks(id),
    symptom TEXT NOT NULL,
    root_cause TEXT NOT NULL,
    fix_applied TEXT NOT NULL,
    files_affected TEXT[],
    pattern_tags TEXT[],
    verification_status TEXT DEFAULT 'unverified',
    confidence_score FLOAT DEFAULT 0.5,
    symptom_embedding vector(768),
    content_embedding vector(768),
    content_tsv tsvector GENERATED ALWAYS AS (
        to_tsvector('english', symptom || ' ' || root_cause || ' ' || fix_applied)
    ) STORED,
    created_at TIMESTAMPTZ DEFAULT now(),
    archived_at TIMESTAMPTZ
);

-- Embedding queue for async processing
CREATE TABLE embedding_queue (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    target_table TEXT NOT NULL,
    target_id UUID NOT NULL,
    target_column TEXT NOT NULL,
    source_text TEXT NOT NULL,
    status TEXT DEFAULT 'pending',
    created_at TIMESTAMPTZ DEFAULT now(),
    processed_at TIMESTAMPTZ,
    error_message TEXT
);

-- Create indexes
CREATE INDEX idx_tasks_project_status ON tasks(project_id, status);
CREATE INDEX idx_tasks_priority ON tasks(priority DESC) WHERE status = 'ready';
CREATE INDEX idx_lessons_project ON lessons_learned(project_id) WHERE archived_at IS NULL;
CREATE INDEX idx_lessons_embedding ON lessons_learned 
    USING hnsw (content_embedding vector_cosine_ops) WHERE content_embedding IS NOT NULL;
CREATE INDEX idx_lessons_tsv ON lessons_learned USING gin(content_tsv);
```

### Task State Machine

```
                    ┌─────────────────┐
                    │    PENDING      │
                    │ (deps not met)  │
                    └────────┬────────┘
                             │ dependencies resolved
                             ▼
                    ┌─────────────────┐
                    │     READY       │
                    │ (awaiting slot) │
                    └────────┬────────┘
                             │ dispatched
                             ▼
                    ┌─────────────────┐
                    │    ASSIGNED     │
                    │ (worktree prep) │
                    └────────┬────────┘
                             │ agent spawned
                             ▼
                    ┌─────────────────┐
        timeout ───▶│    RUNNING      │◀─── retry
                    │ (agent active)  │
                    └────────┬────────┘
                             │ result received
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
     ┌─────────────┐  ┌───────────┐  ┌───────────┐
     │   REVIEW    │  │  FAILED   │  │  BLOCKED  │
     │(merge prep) │  │(max retry)│  │(conflict) │
     └──────┬──────┘  └───────────┘  └───────────┘
            │ merged
            ▼
     ┌─────────────┐
     │    DONE     │
     │ (complete)  │
     └─────────────┘
```

---

## 6. Agent Definitions

### Agent Type Registry

| Agent Type | Model | Tools | Purpose |
|------------|-------|-------|---------|
| `architect` | opus | Read, Write, Bash, Grep, Glob | Architecture, complex debugging, merge review |
| `executor` | sonnet | Read, Write, Edit, Bash, Grep, Glob | Feature implementation |
| `reviewer-*` | sonnet/opus | Read, Grep, Glob, Bash(test) | Domain-specific review |
| `interviewer` | opus | Read | Specification clarification |
| `planner` | opus | Read, Grep, Glob | Task decomposition |
| `researcher` | sonnet-1m | Read, Grep, Glob, WebSearch | Documentation research |
| `tester` | sonnet | Read, Write, Bash | Test creation/execution |
| `fixer` | sonnet | Read, Write, Edit, Bash, Grep | Bug fixing |

### Reviewer Specialists

| Reviewer | Focus Areas |
|----------|-------------|
| Frontend | Component structure, React patterns, styling, responsive design |
| Backend | API design, auth patterns, data validation, business logic |
| Database | Schema design, indexing, query optimisation, migrations |
| Spec Compliance | Feature completeness vs requirements |
| Security | Auth/authz, input validation, secrets handling, OWASP |
| Accessibility | WCAG compliance, ARIA, keyboard nav, screen reader |
| Integration | API contract matching, naming consistency |
| Dependency | Version conflicts, unused deps, license compliance |
| Docker/Infra | Rootless compatibility, volumes, networking |
| SCM/CI | Branch hygiene, CI config, workflow correctness |
| Error Handling | Uncaught errors, retry logic, graceful degradation |
| Documentation | README accuracy, API docs, inline comments |

### System Prompt Structure

```
┌─────────────────────────────────────────┐
│ LAYER 1: Base Identity                  │
│ "You are a {agent_type} agent..."       │
├─────────────────────────────────────────┤
│ LAYER 2: Role-Specific Instructions     │
├─────────────────────────────────────────┤
│ LAYER 3: Project Context (ARCHITECTURE) │
├─────────────────────────────────────────┤
│ LAYER 4: Coding Standards (STANDARDS)   │
├─────────────────────────────────────────┤
│ LAYER 5: Environment Reference          │
├─────────────────────────────────────────┤
│ LAYER 6: Lessons Learned                │
├─────────────────────────────────────────┤
│ LAYER 7: Task Instructions              │
├─────────────────────────────────────────┤
│ LAYER 8: Output Contract (JSON schema)  │
└─────────────────────────────────────────┘
```

### Agent Result Schema

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
  "acceptance_criteria_met": {
    "criterion_1": true,
    "criterion_2": false
  },
  "issues_discovered": [
    {
      "severity": "medium",
      "description": "Found inconsistent naming",
      "location": "src/utils.ts:42",
      "suggested_fix": "Rename to match convention"
    }
  ],
  "lessons_learned": [
    {
      "symptom": "Docker build failed with permission error",
      "root_cause": "Rootless Docker needs --userns-keep-id",
      "fix": "Added userns_mode to compose file"
    }
  ],
  "handover_notes": "If continuation needed: auth middleware 80% done..."
}
```

---

## 7. Coding Standards

### 7.1 Python Standards (Orchestrator)

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

**Error Handling Pattern:**
```python
try:
    result = await agent_session.query(prompt)
except AgentTimeoutError as e:
    logger.error("agent_timeout", session_id=session.id, 
                 task_id=task.id, elapsed_seconds=e.elapsed)
    await handle_timeout(session, task)
except AgentContextFullError as e:
    logger.warning("context_exhausted", session_id=session.id)
    await handle_context_exhaustion(session, task)
except Exception as e:
    logger.exception("unexpected_agent_error", session_id=session.id)
    await handle_unexpected_error(session, task, e)
```

**Logging Pattern:**
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

### 7.2 Agent-Generated Code Standards

**General Principles:**
- Every file: module-level docstring explaining purpose
- Every function: docstring or JSDoc comment
- Error handling: explicit, no unhandled promise rejections, no bare excepts
- External API calls: timeout and retry logic required
- Logging: structured format with correlation IDs
- Configuration: environment variables or config files, never hardcoded
- Tests: required for business logic (unit) and API endpoints (integration)

**Commit Message Format:**
```
type(scope): brief description

Detailed explanation if needed.

Task: FM-{task_uuid_short}
```
Types: feat, fix, refactor, docs, test, chore, style, perf

### 7.3 Documentation Standards

| Type | Requirements |
|------|--------------|
| In-Code | Module, class, public function docstrings; inline for complex algorithms |
| Project Docs | Architecture decisions with rationale; API contracts with examples |
| Agent Context | Generated from DB, never manually edited |
| Deployment | Exact commands, environment requirements |

### 7.4 Error Handling Standards

**Severity Levels:**
| Level | Response | Example |
|-------|----------|---------|
| CRITICAL | Immediate halt + alert | DB connection lost, API auth failed |
| HIGH | Retry with backoff + alert | Agent timeout, merge conflict |
| MEDIUM | Retry, log, continue | Embedding generation failed |
| LOW | Log, continue | Non-critical validation warning |

**Required Patterns:**
```python
# All external operations must have:
# 1. Timeout
# 2. Retry logic with exponential backoff
# 3. Structured error logging
# 4. Graceful degradation path

async def external_operation_with_resilience():
    max_retries = 3
    base_delay = 1.0
    
    for attempt in range(max_retries):
        try:
            async with asyncio.timeout(30):
                return await external_call()
        except asyncio.TimeoutError:
            logger.warning("operation_timeout", attempt=attempt)
        except ExternalServiceError as e:
            logger.warning("service_error", attempt=attempt, error=str(e))
        
        if attempt < max_retries - 1:
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            await asyncio.sleep(delay)
    
    logger.error("operation_failed_all_retries")
    return fallback_value()
```

---

## 8. CI/CD Pipeline

### 8.1 GitHub Actions CI

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  lint-and-type-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - run: pip install ruff mypy
      - run: ruff check src/
      - run: mypy src/ --strict

  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: pgvector/pgvector:pg16
        env:
          POSTGRES_PASSWORD: test
          POSTGRES_DB: forgemaster_test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - run: pip install -e ".[test]"
      - run: pytest tests/unit/ -v --cov=forgemaster
      - run: pytest tests/integration/ -v
        env:
          DATABASE_URL: postgresql://postgres:test@localhost:5432/forgemaster_test

  e2e:
    runs-on: ubuntu-latest
    needs: [lint-and-type-check, test]
    steps:
      - uses: actions/checkout@v4
      - name: Start services
        run: docker compose -f docker/docker-compose.test.yml up -d
      - name: Wait for services
        run: |
          timeout 60 bash -c 'until curl -s http://localhost:8080/health; do sleep 2; done'
      - name: Run E2E tests
        run: pytest tests/e2e/ -v
      - name: Collect logs on failure
        if: failure()
        run: docker compose -f docker/docker-compose.test.yml logs
```

### 8.2 End-to-End Testing Strategy

**E2E Test Categories:**

| Category | Tests | Tools |
|----------|-------|-------|
| API Integration | Health endpoints, task CRUD, project lifecycle | pytest + httpx |
| Agent Lifecycle | Task dispatch → execution → result capture | Mock agent SDK |
| Pipeline Operations | Git operations, Docker builds | Test repositories |
| Dashboard | UI interactions, real-time updates | Playwright |
| Resilience | Failure recovery, timeout handling | Chaos testing |

**E2E Test Structure:**
```python
# tests/e2e/test_task_lifecycle.py
import pytest
from httpx import AsyncClient

@pytest.mark.e2e
async def test_full_task_lifecycle(api_client: AsyncClient, test_project):
    """Test complete task flow: create → dispatch → execute → complete"""
    
    # Create task
    task = await api_client.post("/api/tasks", json={
        "project_id": str(test_project.id),
        "title": "Test implementation task",
        "agent_type": "executor",
        "description": "Implement test feature"
    })
    assert task.status_code == 201
    task_id = task.json()["id"]
    
    # Wait for dispatch
    await wait_for_task_status(api_client, task_id, "running", timeout=60)
    
    # Wait for completion
    await wait_for_task_status(api_client, task_id, "done", timeout=300)
    
    # Verify result captured
    result = await api_client.get(f"/api/tasks/{task_id}")
    assert result.json()["status"] == "done"
    assert result.json()["result"] is not None
```

### 8.3 Deployment Pipeline

```yaml
# .github/workflows/deploy.yml
name: Deploy

on:
  push:
    branches: [main]
    paths-ignore:
      - 'docs/**'
      - '*.md'

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build and push
        run: |
          docker build -t ghcr.io/${{ github.repository }}:${{ github.sha }} .
          docker push ghcr.io/${{ github.repository }}:${{ github.sha }}

  deploy:
    needs: build-and-push
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to VPS
        uses: appleboy/ssh-action@v1
        with:
          host: ${{ secrets.VPS_HOST }}
          username: ${{ secrets.VPS_USER }}
          key: ${{ secrets.VPS_SSH_KEY }}
          script: |
            cd /opt/forgemaster
            docker compose pull
            docker compose up -d
            docker compose exec orchestrator python -m forgemaster.health_check
```

---

## 9. Claude Code Integration

### 9.1 MCP Servers

**Required MCPs for Forgemaster Development:**

| MCP | Purpose | Configuration |
|-----|---------|---------------|
| `vps-mcp` | VPS terminal access for deployment testing | Local install connecting to Azure VPS |
| `github-mcp` | Repository operations, issue management | OAuth with repo scope |
| `n8n-mcp` | Workflow automation integration | API key per instance |

**MCP Configuration (~/.claude/mcp_servers.json):**
```json
{
  "servers": {
    "vps": {
      "command": "node",
      "args": ["/path/to/vps-mcp/index.js"],
      "env": {
        "VPS_HOST": "vps.resuseducation.com",
        "VPS_USER": "claude",
        "VPS_KEY_PATH": "~/.ssh/vps_deploy_key"
      }
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@anthropic/github-mcp"]
    },
    "n8n": {
      "command": "node",
      "args": ["/path/to/n8n-mcp/index.js"],
      "env": {
        "N8N_PROD_URL": "https://n8n.resuseducation.com",
        "N8N_API_KEY": "${N8N_PROD_API_KEY}"
      }
    }
  }
}
```

### 9.2 Claude Code Skills

**Relevant Skills for Forgemaster:**

| Skill | Location | Use Case |
|-------|----------|----------|
| n8n workflows | `/mnt/skills/n8n/` | Creating/modifying automation workflows |
| Docker operations | Built-in | Container management |
| Git operations | Built-in | Branch/worktree management |
| PostgreSQL | Built-in | Database queries and migrations |

**Project CLAUDE.md Template:**
```markdown
# Forgemaster Development Context

## Quick Reference
- Python 3.12+ with type hints
- PostgreSQL 16 + pgvector for database
- Rootless Docker (use userns_mode: keep-id)
- Structured logging with structlog

## Environment
- VPS: vps.resuseducation.com (use vps-mcp for deployment)
- Registry: ghcr.io/seanellul/forgemaster
- Database: PostgreSQL 16 with pgvector extension

## Key Commands
```bash
# Run tests
pytest tests/unit/ -v

# Run with coverage
pytest --cov=forgemaster --cov-report=html

# Type check
mypy src/ --strict

# Lint
ruff check src/ --fix
```

## Before Committing
1. Run `ruff check src/`
2. Run `mypy src/ --strict`
3. Run `pytest tests/unit/`
4. Use commit format: `type(scope): description`
```

### 9.3 Hooks Configuration

**~/.claude/settings.json:**
```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "command": "~/.claude/hooks/inject-secrets.sh"
      }
    ]
  },
  "permissions": {
    "allow": [
      "Bash(docker compose *)",
      "Bash(git *)",
      "Bash(pytest *)",
      "Bash(python -m forgemaster*)"
    ],
    "deny": [
      "Bash(rm -rf /)",
      "Bash(sudo *)"
    ]
  }
}
```

### 9.4 n8n Integration

**Forgemaster-n8n Webhook Patterns:**

| Event | n8n Webhook | Payload |
|-------|-------------|---------|
| Task completed | `/webhook/forgemaster/task-complete` | `{task_id, status, summary}` |
| Review cycle triggered | `/webhook/forgemaster/review-start` | `{project_id, cycle_id, tasks_reviewed}` |
| Build failed | `/webhook/forgemaster/build-failed` | `{project_id, error, logs}` |
| Deploy successful | `/webhook/forgemaster/deploy-success` | `{project_id, version, url}` |

**n8n Workflow: Slack Notifications:**
```
Webhook Trigger → Format Message → Slack Send
```

---

## 10. Lesson Intelligence System

### Lesson Capture Sources

| Source | Verification | Confidence |
|--------|--------------|------------|
| Agent self-report | Queued for test verification | 0.5 (unverified) |
| Review cycle findings | Auto-verified via fix task | 0.9 (verified) |
| Manual entry | Human verified | 0.8 (manual) |

### Lesson Verification Protocol

```python
async def verify_lesson(lesson: Lesson) -> VerificationResult:
    # Only verified lessons get high confidence injection
    
    # 1. Find related test
    test_path = find_related_test(lesson.task_id, lesson.files_affected)
    if not test_path:
        return VerificationResult(status='unverified', reason='no_test')
    
    # 2. Test was failing before fix?
    pre_fix = run_test_at_commit(test_path, lesson.pre_fix_sha)
    if pre_fix.passed:
        return VerificationResult(status='disputed', reason='symptom_not_reproducible')
    
    # 3. Test passes after fix?
    post_fix = run_test_at_commit(test_path, lesson.post_fix_sha)
    if not post_fix.passed:
        return VerificationResult(status='disputed', reason='fix_ineffective')
    
    # Verified: symptom real, fix effective
    lesson.verification_status = 'verified'
    lesson.confidence_score = 0.9
    return VerificationResult(status='verified')
```

### Dual Search Strategy

```python
async def find_relevant_lessons(project_id, task, limit=10):
    results = []
    
    # Signal 1: Semantic similarity (pgvector)
    if task.description_embedding:
        semantic = await db.execute("""
            SELECT *, 1 - (content_embedding <=> $1) AS similarity
            FROM lessons_learned
            WHERE project_id = $2 AND archived_at IS NULL
              AND verification_status != 'disputed'
              AND content_embedding IS NOT NULL
            ORDER BY content_embedding <=> $1
            LIMIT $3
        """, task.description_embedding, project_id, limit)
        results.extend(semantic)
    
    # Signal 2: Full-text keyword search
    keyword = await db.execute("""
        SELECT *, ts_rank(content_tsv, query) AS rank
        FROM lessons_learned, to_tsquery($1) query
        WHERE content_tsv @@ query AND project_id = $2
        ORDER BY rank DESC LIMIT $3
    """, build_tsquery(task.description), project_id, limit)
    results.extend(keyword)
    
    # Signal 3: File overlap
    if task.files_touched:
        file_match = await db.execute("""
            SELECT * FROM lessons_learned
            WHERE project_id = $1 AND files_affected && $2
        """, project_id, task.files_touched)
        results.extend(file_match)
    
    return merge_and_rank(results, limit)
```

---

## 11. Model Selection

### Automatic Routing

```python
def select_model(task: Task) -> str:
    # Explicit override
    if task.model_tier != 'auto':
        return MODEL_MAP[task.model_tier]
    
    # Agent type defaults
    if task.agent_type in ('architect', 'interviewer', 'planner'):
        return OPUS  # Always Opus for architecture
    
    if task.agent_type.startswith('reviewer-'):
        if task.agent_type in ('reviewer-security', 'reviewer-integration'):
            return OPUS  # High-stakes reviews
        return SONNET
    
    if task.agent_type in ('researcher', 'documenter'):
        return SONNET_1M  # Long context
    
    # Complexity heuristics
    if task.estimated_minutes > 20:
        return OPUS
    if len(task.files_touched or []) > 5:
        return OPUS
    if 'refactor' in task.title.lower() or 'debug' in task.title.lower():
        return OPUS
    
    return SONNET  # Default
```

### Model Map

| Tier | Model String | Use Case |
|------|--------------|----------|
| opus | claude-opus-4-5-20251101 | Architecture, security review, complex debugging |
| sonnet | claude-sonnet-4-5-20250929 | Standard implementation, routine reviews |
| sonnet-1m | claude-sonnet-4-5-20250929 (1M context) | Documentation, research |
| haiku | claude-haiku-4-5-20251001 | Simple lookups, formatting |

---

## 12. Resilience & Recovery

### Failure Mitigation Matrix

| Failure | Detection | Response |
|---------|-----------|----------|
| Agent hang | Idle timer > 5 min | Kill, retry with fresh context |
| Context exhaustion | Token count > 80% | Request save-and-exit, spawn continuation |
| Invalid output | JSON validation | Retry with format reminder |
| Orchestrator crash | systemd watchdog | Auto-restart, resume from DB |
| Merge conflict | Git return code | Spawn architect to resolve |
| Docker build failure | Exit code | Log, create fix task |
| API rate limit | HTTP 429 | Exponential backoff, reduce parallelism |

### Session Handover Protocol

```
1. DETECT: Token count > 80% OR idle timeout approaching

2. INJECT: Save-and-exit prompt
   "URGENT: Context nearly full. Output handover JSON:
    {progress_percentage, completed_steps, remaining_steps,
     current_state, files_modified, key_decisions, blockers}"

3. PERSIST: Save to agent_sessions.handover_context

4. SPAWN: New session with:
   - Original task prompt
   - Handover context
   - Diff of changes made

5. RESUME: New agent continues from handover state
```

### Orchestrator Health Check

```python
# systemd service: /etc/systemd/system/forgemaster.service
[Service]
Type=notify
WatchdogSec=120
Restart=always
RestartSec=10

# On restart, orchestrator:
# 1. Connect to PostgreSQL
# 2. Find sessions marked 'running'
# 3. Check if still alive
# 4. Kill orphans, mark for retry
# 5. Resume dispatch loop
```

---

## 13. Implementation Roadmap

### Phase 1: Core Orchestrator (MVP) — 2-3 weeks

- [ ] Database schema + pgvector + Alembic migrations
- [ ] Configuration system (TOML)
- [ ] Task queue with state machine
- [ ] Agent session wrapper (Claude Agent SDK)
- [ ] Single-worker dispatcher
- [ ] Session health monitor
- [ ] Basic context file generation
- [ ] Lesson capture (unverified)
- [ ] Embedding queue + Ollama worker
- [ ] Git operations (branch, commit, merge)
- [ ] CLI for project/task management
- [ ] Structured logging (structlog)
- [ ] systemd service + watchdog
- [ ] Docker Compose deployment
- [ ] Secrets injection hook integration

### Phase 2: Architecture Pipeline — 1-2 weeks

- [ ] Specification ingestion
- [ ] Interview agent
- [ ] Architect agent
- [ ] Task decomposition
- [ ] Repository scaffolding
- [ ] nginx-proxy-manager integration for project URLs

### Phase 3: Parallelisation — 1-2 weeks

- [ ] Git worktree management
- [ ] Multi-worker dispatcher
- [ ] File conflict detection
- [ ] Merge coordinator
- [ ] Parallel group scheduling

### Phase 4: Review Cycles + Intelligence — 1-2 weeks

- [ ] Review cycle orchestration
- [ ] 14 specialist reviewers
- [ ] Finding consolidation
- [ ] Lesson verification protocol
- [ ] Semantic context pre-selection

### Phase 5: Build/Deploy Pipeline — 1 week

- [ ] Docker build (rootless)
- [ ] Image tagging (git SHA)
- [ ] Registry push
- [ ] Container restart
- [ ] Health check + rollback

### Phase 6: Resilience Hardening — 1 week

- [ ] Session handover protocol
- [ ] Crash recovery
- [ ] Idle watchdog
- [ ] API rate limit handling
- [ ] E2E test suite

### Phase 7: Dashboard & API — 1-2 weeks

- [ ] Full REST API
- [ ] Webhook endpoints
- [ ] Real-time task board
- [ ] Session logs viewer
- [ ] n8n notification integration

---

## 14. Risk Register

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Claude Agent SDK breaking changes | Medium | High | Pin version, integration tests |
| Context exhaustion | High | Medium | Handover protocol, task size limits |
| Git merge conflicts | Medium | Medium | File conflict detection, architect resolution |
| API rate limits | Medium | Medium | Token bucket, adaptive parallelism |
| Bad lessons propagation | Medium | Medium-High | Test-verified lessons only |
| Integration failures | High | High | Contract tests, CI on every merge |
| Orchestrator complexity | Medium | High | Ruthless scope limits, library-first |
| Model regression | Medium | Medium-High | Pinned versions, regression tests |
| Secrets exposure | Low | Critical | Hook injection, never in repos |
| VPS access failure | Low | High | Health monitoring, fallback procedures |

---

## Appendix A: File Structure

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
│   ├── nginx-proxy-add.sh
│   ├── nginx-proxy-remove.sh
│   └── nginx-proxy-modify.sh
├── docs/
├── CLAUDE.md
├── pyproject.toml
└── README.md
```

---

## Appendix B: Environment Variables

```bash
# Database
DATABASE_URL=postgresql://forgemaster:password@localhost:5432/forgemaster
PGVECTOR_ENABLED=true

# Anthropic API
ANTHROPIC_API_KEY=sk-ant-...

# Ollama
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=nomic-embed-text

# Git/Docker
GITHUB_TOKEN=ghp_...
DOCKERHUB_USERNAME=...
DOCKERHUB_TOKEN=dckr_pat_...
DOCKER_REGISTRY=ghcr.io/seanellul

# VPS/Nginx
VPS_HOST=vps.resuseducation.com
HOSTINGER_API_TOKEN=...

# n8n Integration
N8N_PROD_URL=https://n8n.resuseducation.com
N8N_PROD_API_KEY=n8n_api_...

# Notifications
SLACK_WEBHOOK_URL=https://hooks.slack.com/...
```

---

*Document Version: 3.0.0 | Optimised for LLM ingestion | Last updated: 2025-02-05*
