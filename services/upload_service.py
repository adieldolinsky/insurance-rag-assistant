"""
Background processing for document ingestion.

Pipeline (executed inside a worker thread):
  a. Convert PDF -> Markdown using a memory-safe "Split & Merge" Docling approach.
  b. Classify the insurance company from the document text via Bedrock.
  c. Prepend a policy header to the Markdown so each chunk keeps its identity.
  d. Write a Bedrock-compatible ``.metadata.json`` sidecar tagging the company.
  e. Upload the ``.md`` + ``.metadata.json`` to S3 and trigger a KB ingestion job
     (with conflict-aware exponential backoff).
"""
from __future__ import annotations

import os

# These flags MUST be set before importing docling on Windows to avoid the
# HuggingFace symlink warning/error.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from botocore.exceptions import ClientError
from PyPDF2 import PdfReader, PdfWriter
from docling.document_converter import DocumentConverter

from config import (
    MODEL_ID,
    BUCKET_NAME,
    KNOWLEDGE_BASE_ID,
    DATA_SOURCE_ID,
    PAGES_PER_CHUNK,
    OUTPUT_FOLDER,
    METADATA_KEY,
    INGESTION_MAX_RETRIES,
    INGESTION_RETRY_DELAY_SECONDS,
    INGESTION_RETRY_MAX_DELAY_SECONDS,
    normalize_company,
    UNKNOWN_COMPANY,
)
from services import policy_registry
from services.aws_clients import (
    bedrock_agent_client,
    bedrock_runtime_client,
    s3_client as get_s3_client,
)

logger = logging.getLogger(__name__)

# In-memory job status registry so the frontend can poll progress.
JOB_STATUS: dict[str, dict[str, Any]] = {}

UNKNOWN_TIER = "Unknown"


@dataclass(frozen=True)
class PolicyMetadata:
    """Structured policy metadata extracted from the uploaded document."""

    company: str
    tier: str | None
    source_filename: str


def _set_status(
    job_id: str,
    state: str,
    message: str,
    filename: str | None = None,
    company: str | None = None,
) -> None:
    payload: dict[str, Any] = {"state": state, "message": message}
    if filename is not None:
        payload["filename"] = filename
    if company is not None:
        payload["company"] = company
    JOB_STATUS[job_id] = payload
    logger.info("[%s] %s: %s", job_id, state, message)


# ---------------------------------------------------------------------------
# Step (a): Split & Merge PDF -> Markdown
# ---------------------------------------------------------------------------
def convert_pdf_to_markdown(
    input_pdf_path: str, pages_per_chunk: int = PAGES_PER_CHUNK
) -> str:
    """
    Split a PDF into small page-chunks, convert each with Docling, and merge the
    Markdown. Chunking bounds Docling's peak memory and isolates per-chunk
    failures so one bad page range cannot abort the whole document.
    """
    if not os.path.exists(input_pdf_path):
        raise FileNotFoundError(f"PDF not found: {input_pdf_path}")
    if pages_per_chunk < 1:
        pages_per_chunk = 1

    reader = PdfReader(input_pdf_path)
    total_pages = len(reader.pages)
    if total_pages == 0:
        raise ValueError("PDF has no pages.")

    logger.info("Converting PDF (%d pages, %d per chunk).", total_pages, pages_per_chunk)
    converter = DocumentConverter()
    parts: list[str] = []
    failed_ranges: list[str] = []

    for start_page in range(0, total_pages, pages_per_chunk):
        end_page = min(start_page + pages_per_chunk, total_pages)
        temp_pdf_path = os.path.join(
            OUTPUT_FOLDER, f"temp_chunk_{start_page}_{end_page}.pdf"
        )
        try:
            writer = PdfWriter()
            for i in range(start_page, end_page):
                writer.add_page(reader.pages[i])
            with open(temp_pdf_path, "wb") as fh:
                writer.write(fh)

            result = converter.convert(temp_pdf_path)
            markdown = result.document.export_to_markdown()
            if markdown.strip():
                parts.append(markdown.strip())
            logger.info("Converted pages %d-%d.", start_page + 1, end_page)
        except Exception as exc:  # noqa: BLE001 - keep processing remaining chunks
            failed_ranges.append(f"{start_page + 1}-{end_page}")
            logger.warning("Failed pages %d-%d: %s", start_page + 1, end_page, exc)
        finally:
            if os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)

    if failed_ranges:
        logger.warning("Chunks that failed conversion: %s", ", ".join(failed_ranges))

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Step (b): identify the insurance company and optional tier via Bedrock
# ---------------------------------------------------------------------------
def extract_policy_metadata(markdown_text: str, source_filename: str) -> PolicyMetadata:
    """Classify insurer + optional tier using a Hybrid Approach (LLM + Heuristics) with linguistic trap prevention."""
    snippet = markdown_text[:3000]
    bedrock_client = bedrock_runtime_client()

    system_prompt = (
        "You are an expert data extraction API for Israeli insurance documents. "
        "Your ONLY job is to return a valid JSON object. Do not use markdown blocks (no ```json).\n\n"
        "Task:\n"
        "1. Identify the issuing insurance company. Look closely at the VERY FIRST line of the document.\n"
        "   Known companies: 'מכבי' (Maccabi), 'כללית' (Clalit), 'מגדל' (Migdal), 'הראל' (Harel), 'הפניקס' (Phoenix), 'מנורה' (Menora), 'כלל' (Clal), 'איילון' (Ayalon).\n"
        "   CRITICAL HEBREW WARNING: Do NOT misclassify a document as 'Clal' (כלל) just because it contains standard phrases like 'התנאים הכלליים' (General Conditions) or 'בכלל'. If the document starts with 'מגדל', it is Migdal.\n"
        "2. Identify the specific policy tier (e.g., 'Sheli', 'Zahav', 'Platinum', 'Mushlam'). If none is explicitly named, set tier to 'Standard'.\n\n"
        "Output EXACTLY in this format:\n"
        "{\"company\": \"Migdal\", \"tier\": \"Standard\"}\n"
    )

    company = UNKNOWN_COMPANY
    tier = None
    raw = ""

    try:
        user_text = f"Filename: {source_filename}\n\nDocument Text:\n{snippet}"
        response = bedrock_client.converse(
            modelId=MODEL_ID,
            system=[{"text": system_prompt}],
            messages=[{"role": "user", "content": [{"text": user_text}]}],
            inferenceConfig={"maxTokens": 100, "temperature": 0.0},
        )
        raw = response["output"]["message"]["content"][0]["text"].strip()
        
        if raw.startswith("```"):
            raw = raw.strip("`").removeprefix("json").strip()

        parsed = json.loads(raw)
        company = normalize_company(parsed.get("company"))
        
        tier_raw = parsed.get("tier")
        tier = str(tier_raw).strip() if tier_raw and str(tier_raw).strip() not in ["Unknown", "Standard", ""] else None

    except Exception as exc:
        logger.error("LLM metadata extraction failed: %s", exc)

    # --- HYBRID FALLBACK (Bulletproof Heuristics) ---
    # אם המודל נכשל או החזיר Unknown, אנחנו תופסים פיקוד עם פייתון רגיל:
    if company == UNKNOWN_COMPANY:
        logger.info("LLM returned Unknown. Triggering heuristic fallback scan.")
        search_text = (source_filename + " " + markdown_text[:500]).lower()
        
        if "מגדל" in search_text or "migdal" in search_text: company = "Migdal"
        elif "מכבי" in search_text or "maccabi" in search_text: company = "Maccabi"
        elif "כללית" in search_text or "clalit" in search_text: company = "Clalit"
        elif "הראל" in search_text or "harel" in search_text: company = "Harel"
        elif "פניקס" in search_text or "phoenix" in search_text: company = "Phoenix"
        elif "כלל" in search_text or "clal" in search_text: company = "Clal"
        elif "מנורה" in search_text or "menora" in search_text: company = "Menora"
        elif "איילון" in search_text or "ayalon" in search_text: company = "Ayalon"
        
        company = normalize_company(company)

    logger.info("Classified metadata: company='%s', tier='%s' (raw='%s')", company, tier, raw)
    return PolicyMetadata(company=company, tier=tier, source_filename=source_filename)


# ---------------------------------------------------------------------------
# Step (c): Clean Markdown formatting (Semantic Preservation)
# ---------------------------------------------------------------------------
# We prepend a single clear header identifying the policy and its tier.
# Unlike previous brute-force methods, we DO NOT inject markers into the body. 
# This preserves the semantic structure of the Markdown (especially complex 
# financial tables) so Bedrock's internal chunking algorithm can process them 
# accurately. Cross-contamination defense is now strictly handled via Metadata.

def format_policy_markdown(company: str, tier: str | None, filename: str, body: str) -> str:
    """
    Prepend a clean single header to the document without injecting disruptive
    markers into the body. This preserves Markdown tables for proper embedding.
    """
    tier_display = tier if tier else "Standard"
    header = f"# פוליסת ביטוח: {company} (סוג: {tier_display})\n"
    header += f"**קובץ מקור:** {filename}\n"
    header += "---\n\n"
    
    return header + body.strip()

def write_clean_markdown(
    output_md_path: str, company: str, tier: str | None, filename: str, body: str
) -> None:
    """Write the semantically clean Markdown to disk."""
    with open(output_md_path, "w", encoding="utf-8") as fh:
        fh.write(format_policy_markdown(company, tier, filename, body))


# ---------------------------------------------------------------------------
# Step (d): write a Bedrock-compatible metadata sidecar
# ---------------------------------------------------------------------------
def write_metadata_file(md_path: str, metadata: PolicyMetadata) -> str:
    """
    Create '<file>.md.metadata.json' for Bedrock metadata filtering.

    Includes:
    - company (existing key used by retrieval filters)
    - tier (optional; stored as Unknown when not identifiable)
    - source_filename (useful for tracing + selective filtering)
    """
    metadata_path = f"{md_path}.metadata.json"
    metadata_payload = {
        "metadataAttributes": {
            METADATA_KEY: metadata.company,
            "tier": metadata.tier or UNKNOWN_TIER,
            "source_filename": metadata.source_filename,
        }
    }
    with open(metadata_path, "w", encoding="utf-8") as fh:
        json.dump(metadata_payload, fh, ensure_ascii=False, indent=2)
    return metadata_path


# ---------------------------------------------------------------------------
# Step (e): upload to S3 + trigger Knowledge Base ingestion
# ---------------------------------------------------------------------------
def _resolve_data_source_id(agent_client: Any) -> str:
    if DATA_SOURCE_ID:
        return DATA_SOURCE_ID
    response = agent_client.list_data_sources(knowledgeBaseId=KNOWLEDGE_BASE_ID)
    summaries = response.get("dataSourceSummaries", [])
    if not summaries:
        raise RuntimeError("No data sources found for the Knowledge Base.")
    return summaries[0]["dataSourceId"]


def _start_ingestion_job_with_retry(agent_client: Any, data_source_id: str) -> str:
    """
    Bedrock allows only one ingestion job per Knowledge Base at a time. Retry on
    ConflictException with exponential backoff (base delay, capped).
    """
    for attempt in range(INGESTION_MAX_RETRIES):
        try:
            ingestion = agent_client.start_ingestion_job(
                knowledgeBaseId=KNOWLEDGE_BASE_ID,
                dataSourceId=data_source_id,
            )
            return ingestion["ingestionJob"]["ingestionJobId"]
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code != "ConflictException" or attempt >= INGESTION_MAX_RETRIES - 1:
                raise
            delay = min(
                INGESTION_RETRY_DELAY_SECONDS * (2**attempt),
                INGESTION_RETRY_MAX_DELAY_SECONDS,
            )
            logger.info(
                "Ingestion conflict (attempt %d/%d); retrying in %ds.",
                attempt + 1,
                INGESTION_MAX_RETRIES,
                delay,
            )
            time.sleep(delay)

    raise RuntimeError("Failed to start ingestion job after maximum retries.")


def upload_and_ingest(md_path: str, metadata_path: str) -> str:
    """Upload the markdown + metadata to S3 and start a KB ingestion job."""
    s3 = get_s3_client()
    agent_client = bedrock_agent_client()

    md_key = os.path.basename(md_path)
    metadata_key = os.path.basename(metadata_path)

    s3.upload_file(md_path, BUCKET_NAME, md_key)
    s3.upload_file(metadata_path, BUCKET_NAME, metadata_key)
    logger.info("Uploaded to s3://%s/%s", BUCKET_NAME, md_key)
    # Local copies are intentionally kept for manual inspection (debugging).
    logger.info("Local debug artifacts retained: %s | %s", md_path, metadata_path)

    data_source_id = _resolve_data_source_id(agent_client)
    ingestion_job_id = _start_ingestion_job_with_retry(agent_client, data_source_id)
    logger.info("Started ingestion job: %s", ingestion_job_id)
    return ingestion_job_id


# ---------------------------------------------------------------------------
# Orchestrator (entry point for the background thread)
# ---------------------------------------------------------------------------
def process_document(job_id: str, input_pdf_path: str, original_filename: str) -> None:
    """Run the full ingestion pipeline. Designed to run inside a worker thread."""
    base_name = os.path.splitext(os.path.basename(original_filename))[0]
    output_md_path = os.path.join(OUTPUT_FOLDER, f"{base_name}.md")

    try:
        _set_status(job_id, "processing", "מעלה ומעבד מסמך...", filename=original_filename)

        markdown_body = convert_pdf_to_markdown(input_pdf_path)
        if not markdown_body.strip():
            _set_status(
                job_id, "error", "לא ניתן לחלץ טקסט מהמסמך.", filename=original_filename
            )
            return

        policy_meta = extract_policy_metadata(markdown_body, source_filename=base_name)
        
        # הוספנו את ה-tier החסר לקריאה של הפונקציה
        write_clean_markdown(
            output_md_path,
            policy_meta.company,
            policy_meta.tier,
            base_name,
            markdown_body,
        )
        metadata_path = write_metadata_file(output_md_path, policy_meta)
        upload_and_ingest(output_md_path, metadata_path)

        logger.info(
            "Ingestion complete. Local files kept for inspection: %s, %s",
            output_md_path,
            metadata_path,
        )

        # Reflect the new policy in the registry immediately for the UI.
        policy_registry.register_local(base_name, policy_meta.company)

        _set_status(
            job_id,
            "done",
            "המסמך הועלה בהצלחה",
            filename=original_filename,
            company=policy_meta.company,
        )
    except Exception as exc:  # noqa: BLE001 - report any failure to the UI
        _set_status(
            job_id, "error", "שגיאה בעיבוד המסמך. נסו שוב.", filename=original_filename
        )
        logger.exception("[%s] processing failed: %s", job_id, exc)
    finally:
        # Only the temporary uploaded PDF is removed; .md and .metadata.json stay
        # in OUTPUT_FOLDER for semantic/table debugging.
        if os.path.exists(input_pdf_path):
            os.remove(input_pdf_path)
