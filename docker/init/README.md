# PostgreSQL Initialization Scripts

This directory contains SQL scripts that run automatically when the PostgreSQL container is first started.

## Scripts

### 01-extensions.sql
Installs required PostgreSQL extensions:
- `vector` - pgvector extension for embedding storage
- `uuid-ossp` - UUID generation functions

## Execution Order

Scripts in this directory are executed in alphabetical order by PostgreSQL's Docker entrypoint.

## Notes

- Scripts only run on **first container startup** (when the data directory is empty)
- To re-run scripts, delete the `forgemaster-db` Docker volume:
  ```bash
  docker compose down -v
  docker compose up
  ```
- Scripts run as the `POSTGRES_USER` (forgemaster) with full database privileges
