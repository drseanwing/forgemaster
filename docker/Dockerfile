# syntax=docker/dockerfile:1

# --- Build stage ---
FROM python:3.12-slim AS builder

WORKDIR /build

COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir build \
    && python -m build --wheel --outdir /build/dist

# --- Runtime stage ---
FROM python:3.12-slim AS runtime

# Create non-root user
RUN groupadd --gid 1000 forgemaster \
    && useradd --uid 1000 --gid forgemaster --shell /bin/bash --create-home forgemaster

WORKDIR /app

# Install the wheel from build stage
COPY --from=builder /build/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl \
    && rm -rf /tmp/*.whl

# Copy any additional runtime files
COPY agent-templates/ /app/agent-templates/
COPY context-templates/ /app/context-templates/

# Switch to non-root user
USER forgemaster

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()" || exit 1

EXPOSE 8000

ENTRYPOINT ["forgemaster"]
