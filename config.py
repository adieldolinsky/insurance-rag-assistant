"""
Central configuration for the Insurance RAG application.

NOTE: The AWS identifiers below were extracted from the original scripts and
must NOT be changed unless the underlying AWS resources change. They can be
overridden via environment variables for staging/production deployments.
"""
from __future__ import annotations

import logging
import os
import re

# --- AWS / Bedrock configuration (preserved from the original scripts) ---
REGION: str = os.environ.get("AWS_REGION", "us-east-1")
MODEL_ID: str = os.environ.get(
    "BEDROCK_MODEL_ID", "us.anthropic.claude-opus-4-6-v1"
)
BUCKET_NAME: str = os.environ.get("S3_BUCKET_NAME", "insurance-private-mvp")
KNOWLEDGE_BASE_ID: str = os.environ.get("KNOWLEDGE_BASE_ID", "NXYJDUMTAJ")

# Data source id for the Knowledge Base ingestion job.
# Optional: if empty, the app resolves the first data source automatically.
DATA_SOURCE_ID: str = os.environ.get("KNOWLEDGE_BASE_DATA_SOURCE_ID", "")

# --- AWS HTTP timeouts (botocore defaults read_timeout=60, too low for large converse) ---
BOTO_CONNECT_TIMEOUT: int = int(os.environ.get("BOTO_CONNECT_TIMEOUT", "10"))
BOTO_READ_TIMEOUT: int = int(os.environ.get("BOTO_READ_TIMEOUT", "300"))

# --- Retrieval tuning (scope-aware) ---
# Bedrock retrieve() allows at most 100 results per call.
MAX_RESULTS_PER_CALL: int = 110
SEARCH_TYPE: str = os.environ.get("RAG_SEARCH_TYPE", "HYBRID")

# Single-scope queries (one or no specific company).
NARROW_NUMBER_OF_RESULTS: int = int(os.environ.get("RAG_NARROW_RESULTS", "50"))
BROAD_NUMBER_OF_RESULTS: int = int(os.environ.get("RAG_BROAD_RESULTS", "100"))

# Comparison queries fan out one retrieval per company; these are per-company K.
RESULTS_PER_COMPANY: int = int(os.environ.get("RAG_RESULTS_PER_COMPANY", "50"))
RESULTS_PER_COMPANY_BROAD: int = int(os.environ.get("RAG_RESULTS_PER_COMPANY_BROAD", "60"))

# Total chunks kept across a fan-out (keeps the prompt within budget).
MAX_TOTAL_RESULTS: int = int(os.environ.get("RAG_MAX_TOTAL_RESULTS", "130"))
# Minimum per-company K so no policy is starved in a wide comparison.
MIN_RESULTS_PER_COMPANY: int = 15
# Cap the number of companies compared in a single unnamed "all policies" ask.
MAX_COMPARE_COMPANIES: int = int(os.environ.get("RAG_MAX_COMPARE_COMPANIES", "4"))

# Backwards-compatible default used by any external callers.
NUMBER_OF_RESULTS: int = NARROW_NUMBER_OF_RESULTS

# Metadata attribute used to tag and filter documents by insurer.
METADATA_KEY: str = "insurance_company"

# --- PDF processing ---
PAGES_PER_CHUNK: int = int(os.environ.get("PAGES_PER_CHUNK", "5"))

# --- Ingestion retry (Bedrock allows one ingestion job per KB at a time) ---
INGESTION_MAX_RETRIES: int = int(os.environ.get("INGESTION_MAX_RETRIES", "15"))
INGESTION_RETRY_DELAY_SECONDS: int = int(
    os.environ.get("INGESTION_RETRY_DELAY_SECONDS", "10")
)
INGESTION_RETRY_MAX_DELAY_SECONDS: int = 120

# --- Policy registry caching ---
POLICY_REGISTRY_TTL_SECONDS: int = int(os.environ.get("POLICY_REGISTRY_TTL_SECONDS", "30"))

# --- Local working directories ---
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER: str = os.path.join(BASE_DIR, "tmp_uploads")
OUTPUT_FOLDER: str = os.path.join(BASE_DIR, "tmp_outputs")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- Company normalization ---
# Canonical English insurer name -> aliases (English + Hebrew) used to detect
# which policy a user is asking about and to normalize the classifier output.
COMPANY_ALIASES: dict[str, list[str]] = {
    "Maccabi": ["maccabi", "מכבי", "מכבי שלי", "מכבי זהב"],
    "Migdal": ["migdal", "מגדל"],
    "Harel": ["harel", "הראל"],
    "Clal": ["clal", "כלל"],
    "Clalit": ["clalit", "כללית", "מושלם", "כללית מושלם"],
    "Phoenix": ["phoenix", "הפניקס", "פניקס"],
    "Menora": ["menora", "menora mivtachim", "מנורה", "מנורה מבטחים"],
    "Ayalon": ["ayalon", "איילון", "אילון"],
    "Meuhedet": ["meuhedet", "מאוחדת"],
    "Leumit": ["leumit", "לאומית"],
}

UNKNOWN_COMPANY: str = "Unknown"

# Logging is configured once via setup_logging() from the app entry point.
LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO").upper()


def setup_logging() -> None:
    """Configure root logging once for the whole application."""
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


_HEBREW_RANGE = "\u0590-\u05ff"
# Single-letter Hebrew prefixes that attach directly to a noun
# (ו=and, ה=the, ב=in, כ=as, ל=to, מ=from, ש=that).
_HEBREW_PREFIXES = "והבכלמש"
_hebrew_re = re.compile(f"[{_HEBREW_RANGE}]")


def alias_in_text(text: str, alias: str) -> bool:
    """
    Whole-word alias match.

    - Latin aliases use ``\\b`` boundaries, so "clal" does not match inside
      "clalit".
    - Hebrew aliases allow a single attached prefix letter (e.g. "בכללית" ->
      "כללית") while a trailing-letter boundary still stops "כלל" from matching
      "כללית".
    """
    if _hebrew_re.search(alias):
        pattern = (
            rf"(?<![{_HEBREW_RANGE}])[{_HEBREW_PREFIXES}]?"
            rf"{re.escape(alias)}(?![{_HEBREW_RANGE}])"
        )
    else:
        pattern = rf"\b{re.escape(alias)}\b"
    return re.search(pattern, text) is not None


def normalize_company(raw: str | None) -> str:
    """
    Map any raw/alias company string to its canonical English name, choosing the
    longest matching alias so e.g. "Clalit" wins over "Clal".
    """
    if not raw or not raw.strip():
        return UNKNOWN_COMPANY

    text = raw.strip().lower()
    best_company: str | None = None
    best_len = 0
    for canonical, aliases in COMPANY_ALIASES.items():
        for candidate in (canonical.lower(), *(a.lower() for a in aliases)):
            if len(candidate) > best_len and alias_in_text(text, candidate):
                best_company, best_len = canonical, len(candidate)

    if best_company:
        return best_company
    # Title-case unknown single tokens so they at least look like a name.
    return raw.strip().split()[0].title()
