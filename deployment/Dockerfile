# ---------------------------------------------------------------------------
# Payroll Mapping Engine — Production Dockerfile
# ---------------------------------------------------------------------------
# Multi-stage build:
#   Stage 1 (builder) – install dependencies into a venv.
#   Stage 2 (runtime) – copy only the venv and app source; no build tools.
#
# Security practices:
#   - Non-root user (appuser) in the runtime stage.
#   - No secrets in the image; all secrets are injected at runtime via
#     environment variables or Azure Key Vault.
#   - Minimal runtime image (python:3.12-slim).
# ---------------------------------------------------------------------------

# ---- Stage 1: dependency builder ----------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /build

# System deps needed to compile some Python packages (e.g. cryptography)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libffi-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt


# ---- Stage 2: runtime image ----------------------------------------------
FROM python:3.12-slim AS runtime

LABEL maintainer="payroll-platform-team"
LABEL description="Enterprise Payroll Mapping Engine"

# Non-root user
RUN groupadd --gid 1001 appgroup \
    && useradd --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser

WORKDIR /app

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application source
COPY app/ ./app/
COPY data/ ./data/

# Ensure the venv is used for all Python commands
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Application defaults (override via env vars or Key Vault at runtime)
    ENVIRONMENT=production \
    LOG_JSON=true \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    API_WORKERS=2

USER appuser

EXPOSE 8000

# Health check — used by Azure Container Apps and Docker orchestrators
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')"

CMD ["sh", "-c", \
     "uvicorn app.main:app \
        --host $API_HOST \
        --port $API_PORT \
        --workers $API_WORKERS \
        --log-level warning \
        --no-access-log"]
