"""
Central configuration for the Insurance RAG application.

AWS resource IDs below match the original MVP and can be overridden via environment
variables. On EC2, attach an IAM instance profile (no static keys in the image).
"""
from __future__ import annotations

import logging
import os
import re
import sys

# --- AWS / Bedrock (preserved defaults; override in production) ---
REGION: str = os.environ.get(
    "AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
)
MODEL_ID: str = os.environ.get(
    "BEDROCK_MODEL_ID", "us.anthropic.claude-opus-4-6-v1"
)
BUCKET_NAME: str = os.environ.get("S3_BUCKET_NAME", "insurance-private-mvp")
KNOWLEDGE_BASE_ID: str = os.environ.get("KNOWLEDGE_BASE_ID", "NXYJDUMTAJ")

# Optional: if empty, upload_service resolves the first data source automatically.
DATA_SOURCE_ID: str = os.environ.get("KNOWLEDGE_BASE_DATA_SOURCE_ID", "")

# When set, startup fails if required env vars are not explicitly provided.
REQUIRE_AWS_ENV: bool = os.environ.get("REQUIRE_AWS_ENV", "").lower() in (
    "1",
    "true",
    "yes",
)

# --- HTTP server ---
APP_PORT: int = int(os.environ.get("PORT", "5000"))
APP_BIND_HOST: str = os.environ.get("BIND_HOST", "0.0.0.0")

# --- AWS HTTP timeouts (botocore default read_timeout=60 is too low for RAG) ---
BOTO_CONNECT_TIMEOUT: int = int(os.environ.get("BOTO_CONNECT_TIMEOUT", "10"))
BOTO_READ_TIMEOUT: int = int(os.environ.get("BOTO_READ_TIMEOUT", "300"))
BOTO_MAX_RETRY_ATTEMPTS: int = int(os.environ.get("BOTO_MAX_RETRY_ATTEMPTS", "10"))

# --- Retrieval tuning (scope-aware) ---
# Bedrock retrieve() allows at most 100 results per call.
MAX_RESULTS_PER_CALL: int = 100
SEARCH_TYPE: str = os.environ.get("RAG_SEARCH_TYPE", "HYBRID")

NARROW_NUMBER_OF_RESULTS: int = int(os.environ.get("RAG_NARROW_RESULTS", "50"))
BROAD_NUMBER_OF_RESULTS: int = int(os.environ.get("RAG_BROAD_RESULTS", "100"))

RESULTS_PER_COMPANY: int = int(os.environ.get("RAG_RESULTS_PER_COMPANY", "50"))
RESULTS_PER_COMPANY_BROAD: int = int(os.environ.get("RAG_RESULTS_PER_COMPANY_BROAD", "60"))

MAX_TOTAL_RESULTS: int = int(os.environ.get("RAG_MAX_TOTAL_RESULTS", "130"))
MIN_RESULTS_PER_COMPANY: int = 15
MAX_COMPARE_COMPANIES: int = int(os.environ.get("RAG_MAX_COMPARE_COMPANIES", "4"))

NUMBER_OF_RESULTS: int = NARROW_NUMBER_OF_RESULTS
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
POLICY_REGISTRY_TTL_SECONDS: int = int(
    os.environ.get("POLICY_REGISTRY_TTL_SECONDS", "30")
)

# --- Local working directories (mount EBS volumes here on EC2) ---
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER: str = os.environ.get(
    "UPLOAD_FOLDER", os.path.join(BASE_DIR, "tmp_uploads")
)
OUTPUT_FOLDER: str = os.environ.get(
    "OUTPUT_FOLDER", os.path.join(BASE_DIR, "tmp_outputs")
)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- Company normalization ---
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
LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO").upper()


def setup_logging() -> None:
    """Configure root logging once for the whole application."""
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def validate_aws_config() -> list[str]:
    """Return configuration problems. Empty list means OK."""
    problems: list[str] = []
    if REQUIRE_AWS_ENV:
        if not (os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")):
            problems.append("Missing required environment variable: AWS_REGION")
        for name in ("S3_BUCKET_NAME", "KNOWLEDGE_BASE_ID", "BEDROCK_MODEL_ID"):
            if not os.environ.get(name):
                problems.append(f"Missing required environment variable: {name}")

    for label, value in (
        ("RAG_NARROW_RESULTS", NARROW_NUMBER_OF_RESULTS),
        ("RAG_BROAD_RESULTS", BROAD_NUMBER_OF_RESULTS),
        ("RAG_RESULTS_PER_COMPANY", RESULTS_PER_COMPANY),
        ("RAG_RESULTS_PER_COMPANY_BROAD", RESULTS_PER_COMPANY_BROAD),
    ):
        if value > MAX_RESULTS_PER_CALL:
            problems.append(
                f"{label} ({value}) exceeds Bedrock per-call maximum ({MAX_RESULTS_PER_CALL})"
            )
    return problems


def validate_aws_config_or_exit() -> None:
    problems = validate_aws_config()
    for msg in problems:
        logging.error("Configuration error: %s", msg)
    if problems and REQUIRE_AWS_ENV:
        sys.exit(1)


_HEBREW_RANGE = "\u0590-\u05ff"
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
    return raw.strip().split()[0].title()
