"""
Background ingestion: Docling PDF->MD, S3 upload (markdown only), Lambda ETL, KB sync.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from botocore.exceptions import ClientError
from docling.document_converter import DocumentConverter
from pypdf import PdfReader, PdfWriter

from config import (
    ADMIN_LAMBDA_NAME,
    BUCKET_NAME,
    DATA_SOURCE_ID,
    INGESTION_MAX_RETRIES,
    INGESTION_RETRY_DELAY_SECONDS,
    INGESTION_RETRY_MAX_DELAY_SECONDS,
    KNOWLEDGE_BASE_ID,
    MODEL_ID,
    OUTPUT_FOLDER,
    PAGES_PER_CHUNK,
    USER_LAMBDA_NAME,
)
from services.aws_clients import (
    bedrock_agent_client,
    bedrock_runtime_client,
    lambda_client,
    s3_client,
)

logger = logging.getLogger(__name__)

JOB_STATUS: dict[str, dict[str, Any]] = {}


def _set_status(job_id: str, state: str, message: str, **extra: Any) -> None:
    payload: dict[str, Any] = {"state": state, "message": message, **extra}
    JOB_STATUS[job_id] = payload
    logger.info("[%s] %s: %s", job_id, state, message)


def extract_policy_metadata(pdf_path: str) -> dict[str, Any]:
    """Read first 2 pages and extract metadata via Bedrock."""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for i in range(min(2, len(reader.pages))):
            text += reader.pages[i].extract_text() + "\n"

        prompt = f"""
        You are a strict data extraction AI for Israeli insurance policies.
        Analyze the following text.
        Extract:
        1. "company"
        2. "policy_name"
        3. "shaban_tier" (or null)

        Return ONLY a raw JSON object.
        Schema: {{"company": "string", "policy_name": "string", "shaban_tier": "string" | null}}

        Text:
        {text}
        """

        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 150,
            "messages": [{"role": "user", "content": prompt}],
        })

        response = bedrock_runtime_client().invoke_model(
            modelId=MODEL_ID,
            body=body,
            contentType="application/json",
            accept="application/json",
        )

        response_body = json.loads(response.get("body").read())
        extracted_text = response_body["content"][0]["text"]

        if extracted_text.startswith("```json"):
            extracted_text = extracted_text.replace("```json\n", "").replace("```", "")

        return json.loads(extracted_text)

    except Exception as exc:  # noqa: BLE001
        logger.warning("Metadata extraction failed: %s", exc)
        return {"company": "לא ידוע", "policy_name": "לא ידוע", "shaban_tier": None}


def format_policy_markdown(
    company: str,
    policy_name: str,
    shaban_tier: str | None,
    filename: str,
    body: str,
) -> str:
    tier_display = f" (Supplemental Tier: {shaban_tier})" if shaban_tier else ""
    header = f"# Insurance Policy: {company} - {policy_name}{tier_display}\n"
    header += f"**Source File:** {filename}\n---\n\n"
    return header + body


def convert_pdf_with_docling_chunked(pdf_path: str, chunk_size: int | None = None) -> str:
    """Split & merge pattern to prevent OOM during Docling conversion."""
    chunk_size = chunk_size or PAGES_PER_CHUNK
    converter = DocumentConverter()
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    full_md = ""

    for i in range(0, total_pages, chunk_size):
        writer = PdfWriter()
        chunk_path = os.path.join(OUTPUT_FOLDER, f"temp_chunk_{i}.pdf")

        for j in range(i, min(i + chunk_size, total_pages)):
            writer.add_page(reader.pages[j])

        with open(chunk_path, "wb") as fh:
            writer.write(fh)

        try:
            result = converter.convert(chunk_path)
            full_md += result.document.export_to_markdown() + "\n\n"
        finally:
            if os.path.exists(chunk_path):
                os.remove(chunk_path)

    return full_md


def _resolve_data_source_id(agent: Any) -> str:
    if DATA_SOURCE_ID:
        return DATA_SOURCE_ID
    response = agent.list_data_sources(knowledgeBaseId=KNOWLEDGE_BASE_ID)
    summaries = response.get("dataSourceSummaries", [])
    if not summaries:
        raise RuntimeError("No data sources found for the Knowledge Base.")
    return summaries[0]["dataSourceId"]


def trigger_bedrock_ingestion() -> None:
    """Start a Knowledge Base sync job with conflict-aware retry."""
    agent = bedrock_agent_client()
    data_source_id = _resolve_data_source_id(agent)

    for attempt in range(INGESTION_MAX_RETRIES):
        try:
            agent.start_ingestion_job(
                knowledgeBaseId=KNOWLEDGE_BASE_ID,
                dataSourceId=data_source_id,
            )
            logger.info("Bedrock ingestion job started.")
            return
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code != "ConflictException" or attempt >= INGESTION_MAX_RETRIES - 1:
                raise
            delay = min(
                INGESTION_RETRY_DELAY_SECONDS * (2**attempt),
                INGESTION_RETRY_MAX_DELAY_SECONDS,
            )
            logger.info("Ingestion conflict; retrying in %ds.", delay)
            time.sleep(delay)


def _invoke_etl_lambda(
    function_name: str,
    payload: dict[str, Any],
) -> None:
    lambda_client().invoke(
        FunctionName=function_name,
        InvocationType="Event",
        Payload=json.dumps(payload),
    )


def process_document(
    job_id: str,
    pdf_path: str,
    filename: str,
    upload_type: str = "admin",
    session_id: str | None = None,
) -> None:
    """
    Background worker: Docling -> S3 (markdown only) -> Lambda ETL + KB ingestion.
    Deletes the local PDF when finished.
    """
    try:
        _set_status(job_id, "processing", "מעלה ומעבד מסמך...", filename=filename)

        metadata = extract_policy_metadata(pdf_path)
        company = metadata.get("company", "unknown")
        policy_name = metadata.get("policy_name", "unknown")
        shaban_tier = metadata.get("shaban_tier")

        JOB_STATUS[job_id]["metadata"] = metadata

        raw_md = convert_pdf_with_docling_chunked(pdf_path)
        if not raw_md.strip():
            _set_status(job_id, "error", "לא ניתן לחלץ טקסט מהמסמך.", filename=filename)
            return

        final_md = format_policy_markdown(
            company, policy_name, shaban_tier, filename, raw_md
        )

        md_basename = filename.replace(".pdf", ".md")
        if upload_type == "user" and session_id:
            session_prefix = f"user_sessions/{session_id}"
            md_key = f"{session_prefix}/{md_basename}"
            output_csv_key = f"{session_prefix}/personal_rates.csv"
            lambda_name = USER_LAMBDA_NAME
            lambda_payload = {
                "session_id": session_id,
                "s3_bucket": BUCKET_NAME,
                "s3_md_key": md_key,
                "output_csv_key": output_csv_key,
                "company": company,
                "policy_name": policy_name,
                "shaban_tier": shaban_tier,
            }
        else:
            md_key = f"global_policies/kb/{job_id}_{md_basename}"
            lambda_name = ADMIN_LAMBDA_NAME
            lambda_payload = {
                "s3_bucket": BUCKET_NAME,
                "s3_md_key": md_key,
                "company": company,
                "policy_name": policy_name,
                "shaban_tier": shaban_tier,
            }

        s3_client().put_object(
            Bucket=BUCKET_NAME,
            Key=md_key,
            Body=final_md.encode("utf-8"),
        )
        logger.info("Uploaded markdown to s3://%s/%s", BUCKET_NAME, md_key)

        trigger_bedrock_ingestion()
        _invoke_etl_lambda(lambda_name, lambda_payload)

        _set_status(
            job_id,
            "done",
            "המסמך הועלה בהצלחה",
            filename=filename,
            company=company,
            md_key=md_key,
        )

    except Exception as exc:  # noqa: BLE001
        logger.exception("[%s] processing failed: %s", job_id, exc)
        _set_status(
            job_id,
            "error",
            "שגיאה בעיבוד המסמך. נסו שוב.",
            filename=filename,
        )
    finally:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
