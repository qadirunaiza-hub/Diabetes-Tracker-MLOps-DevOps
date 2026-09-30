# ── Base image ────────────────────────────────────────────────────────────── #
# We use a slim Python image to keep the container small.
# The "slim" variant omits many system packages that aren't needed at runtime.
FROM python:3.11-slim

# ── System dependencies ───────────────────────────────────────────────────── #
# libgomp1: required by scikit-learn's OpenMP-accelerated code (Random Forest, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────── #
WORKDIR /app

# ── Install Python dependencies first (layer-caching optimisation) ─────────── #
# Copying requirements.txt before the rest of the source means Docker re-uses
# the dependency layer on rebuilds unless requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application source ───────────────────────────────────────────────── #
COPY . .

# ── Environment ───────────────────────────────────────────────────────────── #
ENV PYTHONPATH=/app
ENV FLASK_PORT=5001
ENV FLASK_ENV=production
ENV MLFLOW_TRACKING_URI=/app/mlruns
ENV MODEL_NAME=DiabetesPredictor
ENV MODEL_STAGE=Production
ENV LOG_LEVEL=INFO

# Ensure log and model directories exist inside the container
RUN mkdir -p /app/logs /app/models /app/docs/plots

# ── Expose port ───────────────────────────────────────────────────────────── #
EXPOSE 5001

# ── Health check ──────────────────────────────────────────────────────────── #
# Docker will call GET /health every 30 seconds. If it fails 3 times the
# container is marked "unhealthy" and orchestrators (e.g., Kubernetes) restart it.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5001/health')"

# ── Entrypoint ────────────────────────────────────────────────────────────── #
# gunicorn is a production-grade WSGI server (Flask's built-in server is
# single-threaded and not suitable for concurrent requests in production).
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "--workers", "2", "--timeout", "60", "api.app:create_app()"]
