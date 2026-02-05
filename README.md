# Forgemaster

Autonomous development orchestration system that manages Claude AI agent sessions,
task queues, and deterministic build/deploy pipelines.

## Overview

Forgemaster is a Python-based orchestrator that coordinates multiple Claude AI agents
working in parallel across git worktrees. It provides:

- **Task Queue**: PostgreSQL-backed task management with state machine enforcement
- **Agent Orchestration**: Parallel Claude agent sessions with health monitoring
- **Deterministic Pipelines**: Git, Docker, and deployment operations outside agent control
- **Knowledge Persistence**: Semantic search over lessons learned via pgvector
- **Context Generation**: Jinja2-based agent context projection from database state
- **Dashboard**: FastAPI + htmx real-time monitoring interface

## Requirements

- Python 3.12+
- PostgreSQL 16 with pgvector extension
- Docker (rootless mode supported)
- Ollama with nomic-embed-text model

## Quick Start

```bash
# Install in development mode
pip install -e ".[dev,test]"

# Run tests
pytest tests/unit/ -v

# Start the CLI
forgemaster --help
```

## Project Structure

```
src/forgemaster/
    database/       # SQLAlchemy models, migrations, queries
    orchestrator/   # Task dispatcher, state machine, health monitor
    agents/         # Agent definitions, session management
    intelligence/   # Embeddings, semantic search
    pipeline/       # Git, Docker, deployment operations
    context/        # Jinja2 context generation
    web/            # FastAPI dashboard and API
    cli/            # Typer CLI commands
```

## License

Proprietary
