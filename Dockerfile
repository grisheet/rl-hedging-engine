# =============================================================================
# rl-hedging-engine — Production-grade Dockerfile
# Multi-stage build: keeps the final image lean (no build tools / dev deps).
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: builder — install all dependencies into a venv
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build essentials (needed to compile some torch wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create isolated virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip first
RUN pip install --no-cache-dir --upgrade pip

# Copy dependency manifests before source code (better layer caching)
COPY requirements.txt pyproject.toml ./

# Install runtime dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install the package itself (no extras, editable for src layout)
COPY . .
RUN pip install --no-cache-dir -e . --no-deps

# ---------------------------------------------------------------------------
# Stage 2: runtime — minimal image with only what is needed to run
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

LABEL maintainer="Grisheet"
LABEL description="RL-based Derivatives Hedging Engine"
LABEL version="0.1.0"

WORKDIR /app

# Copy venv from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy source code and configs
COPY --from=builder /app/src ./src
COPY --from=builder /app/scripts ./scripts
COPY --from=builder /app/configs ./configs
COPY --from=builder /app/pyproject.toml ./pyproject.toml

# Create directories for runtime artefacts
RUN mkdir -p /app/checkpoints /app/logs /app/results

# Non-root user for security
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Default command: run training with the default config
# Override at `docker run` time with e.g.:
#   docker run rl-hedging-engine python scripts/evaluate.py --checkpoint ...
CMD ["python", "scripts/train.py", "--config", "configs/default.yaml"]
