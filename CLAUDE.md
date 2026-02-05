# Forgemaster Development Context

## Quick Reference
- Python 3.12+ with type hints on all functions
- PostgreSQL 16 + pgvector for database
- Rootless Docker (use userns_mode: keep-id)
- Structured logging with structlog (JSON format)
- TOML configuration

## Environment
- Database: PostgreSQL 16 with pgvector extension
- Embeddings: Ollama with nomic-embed-text
- Registry: ghcr.io

## Key Commands
```bash
# Install dev dependencies
pip install -e ".[dev,test]"

# Run tests
pytest tests/unit/ -v
pytest tests/integration/ -v

# Type check
mypy src/forgemaster/ --strict

# Lint and format
ruff check src/ --fix
black src/ tests/

# Database migrations
alembic upgrade head
alembic revision --autogenerate -m "description"
```

## Before Committing
1. Run `black src/ tests/`
2. Run `ruff check src/ --fix`
3. Run `mypy src/forgemaster/ --strict`
4. Run `pytest tests/unit/ -v`
5. Use commit format: `type(scope): description`

## Project Structure
```
src/forgemaster/
    config.py          # Configuration system
    main.py            # CLI entry point
    database/          # Models, migrations, queries
    orchestrator/      # Dispatcher, state machine, monitor
    agents/            # Agent definitions, session wrapper
    intelligence/      # Embeddings, lesson search
    pipeline/          # Git, Docker operations
    context/           # Jinja2 context generation
    web/               # FastAPI dashboard/API
    cli/               # CLI commands
```

## Coding Standards
- Black formatter (line length 100)
- Ruff linter
- mypy strict mode
- structlog for all logging
- Type hints required on ALL functions
- Module docstrings required on ALL files
- Async/await for all I/O operations
