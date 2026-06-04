# Insurance RAG — Flask + Docling + Amazon Bedrock
# Build:  docker build -t insurance-rag .
# Run:    # Run:    docker run -d -p 5000:5000 --name my-insurance-app -e AWS_REGION="us-east-1" -e BEDROCK_MODEL_ID="us.anthropic.claude-sonnet-4-5-20250929-v1:0" -e S3_BUCKET_NAME="insurance-private-mvp" -e KNOWLEDGE_BASE_ID="NXYJDUMTAJ" username/insurance-rag:latest

FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    HF_HUB_DISABLE_SYMLINKS=1

WORKDIR /app

# System libraries for Docling / PDF / ONNX runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install gunicorn

COPY app.py config.py ./
COPY services/ ./services/
COPY templates/ ./templates/

RUN mkdir -p tmp_uploads tmp_outputs

EXPOSE 5000

# Single worker: JOB_STATUS lives in process memory (upload polling).
# Long timeout: Docling conversion + Bedrock converse on large contexts.
CMD ["gunicorn", \
     "--bind", "0.0.0.0:5000", \
     "--workers", "1", \
     "--threads", "8", \
     "--timeout", "300", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "app:app"]

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://127.0.0.1:5000/ || exit 1
