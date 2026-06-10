# Insurance RAG API — python:3.10-slim with Docling OCR system deps
FROM python:3.10-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
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
    libsm6 \
    libxext6 \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Pre-download Docling + RapidOCR *pytorch* weights (runtime engine).
# DocumentConverter() alone only fetches onnx models; a real convert pulls .pth files.
RUN mkdir -p .cache/huggingface \
    && python -c "\
from pypdf import PdfWriter; \
w = PdfWriter(); \
w.add_blank_page(width=612, height=792); \
f = open('/tmp/warmup.pdf', 'wb'); \
w.write(f); \
f.close(); \
from docling.document_converter import DocumentConverter; \
DocumentConverter().convert('/tmp/warmup.pdf'); \
print('Docling OCR warmup complete')"

COPY config.py .
COPY backend/app.py .
COPY backend/services/ ./services/
COPY backend/templates/ ./templates/
COPY backend/static/ ./static/

RUN mkdir -p tmp_uploads tmp_outputs .cache/huggingface \
    && useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app \
    && chown -R appuser:appuser /usr/local/lib/python3.10/site-packages/rapidocr

USER appuser

EXPOSE 5000

CMD ["gunicorn", \
     "--bind", "0.0.0.0:5000", \
     "--workers", "1", \
     "--threads", "4", \
     "--timeout", "300", \
     "--graceful-timeout", "30", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "app:app"]

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://127.0.0.1:5000/health || exit 1
