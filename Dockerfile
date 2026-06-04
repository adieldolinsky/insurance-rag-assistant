# Insurance RAG — production image for AWS EC2
# Build:  docker build -t insurance-rag .
# Run:    docker run -d -p 5000:5000 --env-file .env insurance-rag
# Prefer EC2 IAM instance profile for credentials (do not bake keys into the image).

FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.cache/huggingface \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    HF_HUB_DISABLE_SYMLINKS=1 \
    REQUIRE_AWS_ENV=1 \
    PORT=5000

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app.py config.py ./
COPY services/ ./services/
COPY templates/ ./templates/

RUN mkdir -p tmp_uploads tmp_outputs .cache/huggingface \
    && useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 5000

# Single Gunicorn worker: upload job status (JOB_STATUS) is in-process memory.
# Mount host volumes at /app/tmp_uploads and /app/tmp_outputs for large PDF scratch space.
CMD ["gunicorn", \
     "--bind", "0.0.0.0:5000", \
     "--workers", "1", \
     "--threads", "8", \
     "--timeout", "300", \
     "--graceful-timeout", "30", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "app:app"]

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://127.0.0.1:5000/health || exit 1
