"""
Shared boto3 session and clients for EC2 / Docker deployment.

Uses the default AWS credential chain (recommended on EC2):
  1. Environment variables (local dev only — avoid in production images)
  2. Shared credentials file (~/.aws/credentials)
  3. ECS task role / EC2 instance profile (production)

Never embed access keys in application code or Docker images.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError

from config import (
    BOTO_CONNECT_TIMEOUT,
    BOTO_MAX_RETRY_ATTEMPTS,
    BOTO_READ_TIMEOUT,
    REGION,
)

logger = logging.getLogger(__name__)


def get_botocore_config() -> Config:
    """Adaptive retries help with Bedrock throttling under load."""
    return Config(
        connect_timeout=BOTO_CONNECT_TIMEOUT,
        read_timeout=BOTO_READ_TIMEOUT,
        retries={"max_attempts": BOTO_MAX_RETRY_ATTEMPTS, "mode": "adaptive"},
        user_agent_extra="insurance-rag",
    )


@lru_cache(maxsize=1)
def _session() -> boto3.Session:
    return boto3.Session(region_name=REGION)


def _client(service_name: str) -> Any:
    return _session().client(service_name, config=get_botocore_config())


@lru_cache(maxsize=1)
def bedrock_runtime_client() -> Any:
    return _client("bedrock-runtime")


@lru_cache(maxsize=1)
def bedrock_agent_runtime_client() -> Any:
    return _client("bedrock-agent-runtime")


@lru_cache(maxsize=1)
def bedrock_agent_client() -> Any:
    return _client("bedrock-agent")


@lru_cache(maxsize=1)
def s3_client() -> Any:
    return _client("s3")


def check_aws_connectivity() -> tuple[bool, str]:
    """
    Readiness probe: verify credentials via STS (no Bedrock call).
    """
    try:
        sts = _session().client("sts", config=get_botocore_config())
        identity = sts.get_caller_identity()
        return True, identity.get("Arn", "unknown")
    except NoCredentialsError:
        return (
            False,
            "No AWS credentials found. Attach an IAM instance profile to the EC2 host.",
        )
    except (ClientError, BotoCoreError) as exc:
        return False, f"AWS STS check failed: {exc}"


def check_s3_bucket(bucket_name: str) -> tuple[bool, str]:
    try:
        s3_client().head_bucket(Bucket=bucket_name)
        return True, "ok"
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "Unknown")
        message = exc.response.get("Error", {}).get("Message", str(exc))
        return False, f"S3 head_bucket failed ({code}): {message}"
    except (BotoCoreError, NoCredentialsError) as exc:
        return False, str(exc)
