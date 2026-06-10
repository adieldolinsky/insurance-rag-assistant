"""
Central configuration — every key below maps 1:1 to a variable in .env / .env.example.
"""
from __future__ import annotations

import logging
import os
import sys

# --- AWS region & credentials ---
REGION: str = os.environ.get(
    "AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
)
AWS_PROFILE: str = os.environ.get("AWS_PROFILE", "")
# Local Docker: require keys in .env. EC2: set AWS_USE_ENV_CREDENTIALS=0 (IAM role).
AWS_USE_ENV_CREDENTIALS: bool = os.environ.get(
    "AWS_USE_ENV_CREDENTIALS", "1"
).lower() in ("1", "true", "yes")

# --- AWS resources ---
S3_BUCKET_NAME: str = os.environ.get("S3_BUCKET_NAME", "insurance-private-mvp")
KNOWLEDGE_BASE_ID: str = os.environ.get("KNOWLEDGE_BASE_ID", "NXYJDUMTAJ")
KNOWLEDGE_BASE_DATA_SOURCE_ID: str = os.environ.get("KNOWLEDGE_BASE_DATA_SOURCE_ID", "")

# Alias used by services (same value as S3_BUCKET_NAME / KNOWLEDGE_BASE_DATA_SOURCE_ID)
BUCKET_NAME: str = S3_BUCKET_NAME
DATA_SOURCE_ID: str = KNOWLEDGE_BASE_DATA_SOURCE_ID

# --- Bedrock models & agent ---
BEDROCK_MODEL_ID: str = os.environ.get(
    "BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
)
MODEL_ID: str = BEDROCK_MODEL_ID

BEDROCK_AGENT_ID: str = os.environ.get("BEDROCK_AGENT_ID", "827DXDYFRW")
BEDROCK_AGENT_ALIAS_ID: str = os.environ.get("BEDROCK_AGENT_ALIAS_ID", "2DQU6OOOXR")

# --- Lambda ETL function names ---
ADMIN_LAMBDA_NAME: str = os.environ.get("ADMIN_LAMBDA_NAME", "Admin_Policy_ETL_Lambda")
USER_LAMBDA_NAME: str = os.environ.get("USER_LAMBDA_NAME", "Lambda3_User_ETL")

# --- Startup validation ---
REQUIRE_AWS_ENV: bool = os.environ.get("REQUIRE_AWS_ENV", "").lower() in (
    "1",
    "true",
    "yes",
)

# --- HTTP server ---
PORT: int = int(os.environ.get("PORT", "5000"))
APP_PORT: int = PORT
BIND_HOST: str = os.environ.get("BIND_HOST", "0.0.0.0")
APP_BIND_HOST: str = BIND_HOST

# --- AWS HTTP client tuning ---
BOTO_CONNECT_TIMEOUT: int = int(os.environ.get("BOTO_CONNECT_TIMEOUT", "10"))
BOTO_READ_TIMEOUT: int = int(os.environ.get("BOTO_READ_TIMEOUT", "300"))
BOTO_MAX_RETRY_ATTEMPTS: int = int(os.environ.get("BOTO_MAX_RETRY_ATTEMPTS", "10"))

# --- PDF / Docling ---
PAGES_PER_CHUNK: int = int(os.environ.get("PAGES_PER_CHUNK", "3"))

# --- Bedrock KB ingestion retry ---
INGESTION_MAX_RETRIES: int = int(os.environ.get("INGESTION_MAX_RETRIES", "15"))
INGESTION_RETRY_DELAY_SECONDS: int = int(
    os.environ.get("INGESTION_RETRY_DELAY_SECONDS", "10")
)
INGESTION_RETRY_MAX_DELAY_SECONDS: int = int(
    os.environ.get("INGESTION_RETRY_MAX_DELAY_SECONDS", "120")
)

# --- Local scratch directories ---
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER: str = os.environ.get(
    "UPLOAD_FOLDER", os.path.join(BASE_DIR, "tmp_uploads")
)
OUTPUT_FOLDER: str = os.environ.get(
    "OUTPUT_FOLDER", os.path.join(BASE_DIR, "tmp_outputs")
)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO").upper()


def setup_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def validate_aws_config() -> list[str]:
    """Return config problems. Empty list means OK."""
    problems: list[str] = []
    if not REQUIRE_AWS_ENV:
        return problems

    if not (os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")):
        problems.append("Missing required environment variable: AWS_REGION")

    for name in (
        "S3_BUCKET_NAME",
        "KNOWLEDGE_BASE_ID",
        "BEDROCK_MODEL_ID",
        "BEDROCK_AGENT_ID",
        "BEDROCK_AGENT_ALIAS_ID",
    ):
        if not os.environ.get(name):
            problems.append(f"Missing required environment variable: {name}")

    key_id = (os.environ.get("AWS_ACCESS_KEY_ID") or "").strip()
    key_secret = (os.environ.get("AWS_SECRET_ACCESS_KEY") or "").strip()
    if key_id and not key_secret:
        problems.append(
            "AWS_SECRET_ACCESS_KEY is required when AWS_ACCESS_KEY_ID is set"
        )
    elif key_secret and not key_id:
        problems.append(
            "AWS_ACCESS_KEY_ID is required when AWS_SECRET_ACCESS_KEY is set"
        )
    elif AWS_USE_ENV_CREDENTIALS and not (key_id and key_secret):
        problems.append(
            "Missing AWS credentials: set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY "
            "in .env. On EC2, set AWS_USE_ENV_CREDENTIALS=0 and use an IAM instance profile."
        )

    return problems


def validate_aws_config_or_exit() -> None:
    problems = validate_aws_config()
    for msg in problems:
        logging.error("Configuration error: %s", msg)
    if problems and REQUIRE_AWS_ENV:
        sys.exit(1)
