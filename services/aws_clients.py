"""
Shared boto3 clients with extended timeouts for Bedrock.

Default botocore read_timeout is 60s, which is too low for converse() calls
with large retrieved context (40+ chunks) and maxTokens=4096.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import boto3
from botocore.config import Config

from config import REGION, BOTO_CONNECT_TIMEOUT, BOTO_READ_TIMEOUT


def get_botocore_config() -> Config:
    return Config(
        connect_timeout=BOTO_CONNECT_TIMEOUT,
        read_timeout=BOTO_READ_TIMEOUT,
        retries={"max_attempts": 3, "mode": "adaptive"},
    )


@lru_cache(maxsize=1)
def bedrock_runtime_client() -> Any:
    return boto3.client(
        "bedrock-runtime",
        region_name=REGION,
        config=get_botocore_config(),
    )


@lru_cache(maxsize=1)
def bedrock_agent_runtime_client() -> Any:
    return boto3.client(
        "bedrock-agent-runtime",
        region_name=REGION,
        config=get_botocore_config(),
    )


@lru_cache(maxsize=1)
def bedrock_agent_client() -> Any:
    return boto3.client(
        "bedrock-agent",
        region_name=REGION,
        config=get_botocore_config(),
    )


@lru_cache(maxsize=1)
def s3_client() -> Any:
    return boto3.client(
        "s3",
        region_name=REGION,
        config=get_botocore_config(),
    )
