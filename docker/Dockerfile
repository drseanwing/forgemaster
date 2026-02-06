# syntax=docker/dockerfile:1

# --- Build stage ---
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY src/ src/

# Build wheel with production dependencies only
RUN pip install --no-cache-dir build \
    && python -m build --wheel --outdir /build/dist

# --- Runtime stage ---
FROM python:3.12-slim AS runtime

# Install runtime dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd --gid 1000 forgemaster \
    && useradd --uid 1000 --gid forgemaster --shell /bin/bash --create-home forgemaster

WORKDIR /app

# Install the wheel from build stage (production dependencies only)
COPY --from=builder /build/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl \
    && rm -rf /tmp/*.whl /root/.cache

# Copy application source code
COPY --chown=forgemaster:forgemaster src/forgemaster/ /app/forgemaster/

# Copy additional runtime files
COPY --chown=forgemaster:forgemaster agent-templates/ /app/agent-templates/
COPY --chown=forgemaster:forgemaster context-templates/ /app/context-templates/

# Create directories for logs and context
RUN mkdir -p /app/logs /app/context \
    && chown -R forgemaster:forgemaster /app

# Switch to non-root user
USER forgemaster

# Health check - verify the FastAPI /health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

ENTRYPOINT ["forgemaster"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
