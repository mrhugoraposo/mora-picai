# MoRA Research Console — single robust FastAPI service (serves UI + API).
# Lean CPU image: the live inference uses scikit-learn logistic models + the persisted
# MoRA reliability references — NO torch/GPU at runtime. SimpleITK is included only for
# optional UPLOAD-mode radiomics extraction.
#
# Build context = repo root:  docker build -f infrastructure/docker/Dockerfile -t mora-console .
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# libgomp1 is needed by scikit-learn / SimpleITK
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- python deps (web service only; copied first for layer caching) ---
COPY services/api/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# --- application code ---
COPY services/api /app/services/api

# --- pipeline source (the app reuses califusion.data.* for clinical schema/marksheet) ---
COPY services/ai/califusion-cnn/src /app/services/ai/califusion-cnn/src

# models/ and data/processed/ are mounted READ-ONLY at runtime (see docker-compose.yml).
# We point the engine at the mounted locations via env.
ENV CALIFUSION_PIPELINE_ROOT=/app/services/ai/califusion-cnn \
    MODELS_DIR=/app/models \
    PYTHONPATH=/app/services/api

WORKDIR /app/services/api

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# Gunicorn-less: uvicorn with a couple of workers is plenty for a home-server console.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
