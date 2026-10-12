# ==============================================================================
# Production Dockerfile for Heat-Risk FastAPI Environmental Intelligence API
# Target Architecture: AWS ECS Fargate / ECR (Linux amd64 / arm64)
# ==============================================================================

FROM python:3.13-slim AS production

# Set essential runtime environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/backend \
    PORT=8000 \
    HOST=0.0.0.0 \
    ENVIRONMENT=production \
    HEAT_MODEL_TYPE=ml \
    HEAT_MODEL_ARTIFACT_DIR=/app/ml/artifacts/heat_model_v1

WORKDIR /app

# Install native runtime dependencies:
# - libgomp1: Required OpenMP runtime for LightGBM
# - curl: Used by Docker HEALTHCHECK probe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies from the authoritative requirements manifest
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy backend application source code and verified ML model artifacts
COPY backend /app/backend
COPY ml/artifacts/heat_model_v1 /app/ml/artifacts/heat_model_v1

# Create a non-root system user for least privilege security
RUN groupadd -r appgroup && useradd -r -g appgroup -d /app -s /sbin/nologin appuser \
    && chown -R appuser:appgroup /app

USER appuser

# Expose the API port
EXPOSE 8000

# Docker Healthcheck targeting the application liveness endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Launch Uvicorn with graceful signal handling
CMD ["uvicorn", "app.main:app", "--app-dir", "/app/backend", "--host", "0.0.0.0", "--port", "8000"]
