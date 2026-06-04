# Insurance RAG — Flask Web Application

A modular Flask web app for analyzing and comparing Israeli health-insurance
policies using **Amazon Bedrock Knowledge Bases**, **S3**, and **Docling** for
PDF parsing.

## Features

- **Hebrew RTL UI** (Tailwind CDN) with a compact upload control and a chat interface.
- **Background ingestion**: uploads return `202 Accepted` immediately while a worker
  thread converts the PDF, classifies the insurer, tags metadata, and triggers a KB
  ingestion job (with conflict-aware exponential backoff).
- **Memory-safe Docling parsing**: PDFs are split into small page-chunks, converted,
  merged, and the temporary chunks deleted. Inline company markers are injected so every
  Knowledge Base chunk keeps its source-policy identity.
- **Automated AI metadata extraction**: Bedrock identifies the insurer and writes a
  `.metadata.json` sidecar used for retrieval filtering.
- **Precision RAG** designed against three failure modes:
  - *Scope-aware Top-K* and *per-company retrieval fan-out* so broad/comparison queries
    get balanced, sufficient context (no "amnesia").
  - *Sealed, per-company context sections* (grouped, never interleaved) to prevent
    cross-contamination of figures between insurers.
  - *Anti-mirroring* prompt rules + explicit empty-section placeholders so missing data
    is reported as "לא מצוין במסמכים" instead of being copied across columns.

## Project structure

```
app.py                       # Flask entry point and routing
config.py                    # AWS IDs, retrieval tuning, company aliases, logging
services/
  bedrock_service.py         # RAG: query planning, retrieval fan-out, generation
  upload_service.py          # Background ingestion (Docling split/merge, tagging, S3)
  policy_registry.py         # Source of truth for policies currently in the KB
templates/
  index.html                 # Hebrew RTL frontend
requirements.txt
```

`tmp_uploads/` and `tmp_outputs/` are runtime scratch directories (git-ignored,
auto-created on startup).

## Configuration

Settings live in `config.py` and can be overridden via environment variables:

| Variable | Default |
| --- | --- |
| `AWS_REGION` | `us-east-1` |
| `BEDROCK_MODEL_ID` | `us.anthropic.claude-haiku-4-5-20251001-v1:0` |
| `S3_BUCKET_NAME` | `insurance-private-mvp` |
| `KNOWLEDGE_BASE_ID` | `NXYJDUMTAJ` |
| `KNOWLEDGE_BASE_DATA_SOURCE_ID` | *(auto-resolved if empty)* |
| `RAG_NARROW_RESULTS` / `RAG_BROAD_RESULTS` | `40` / `80` |
| `RAG_RESULTS_PER_COMPANY` / `RAG_RESULTS_PER_COMPANY_BROAD` | `40` / `50` |

> Metadata filtering requires `insurance_company` to be configured as a filterable
> metadata field on the Knowledge Base data source (Bedrock console / `update_data_source`).

## Running locally

```bash
# (Windows PowerShell) activate the virtual environment
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt

# Ensure AWS credentials are configured (e.g. via `aws configure` or env vars)
python app.py
```

Then open http://127.0.0.1:5000
