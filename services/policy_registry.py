"""
Policy registry: the single source of truth for which policies currently live
in the Knowledge Base.

Policies are discovered by listing the ``*.md.metadata.json`` sidecar files in
the S3 bucket and reading the ``insurance_company`` attribute from each. Results
are cached briefly to avoid hammering S3 on every chat request. Newly uploaded
documents are also registered locally so the UI reflects them immediately,
before the next S3 refresh.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, asdict
from typing import Any

from botocore.exceptions import ClientError

from config import (
    BUCKET_NAME,
    METADATA_KEY,
    POLICY_REGISTRY_TTL_SECONDS,
    normalize_company,
)
from services.aws_clients import s3_client as get_s3_client

logger = logging.getLogger(__name__)

_METADATA_SUFFIX = ".md.metadata.json"


@dataclass(frozen=True)
class Policy:
    """A single policy document known to the Knowledge Base."""

    filename: str  # human-friendly name, e.g. "maccabi_sheli_regulations"
    company: str  # canonical insurer name, e.g. "Maccabi"
    tier: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _PolicyCache:
    """Thread-safe, TTL-based cache of the policy registry."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._policies: dict[str, Policy] = {}
        self._fetched_at: float = 0.0

    def is_fresh(self) -> bool:
        return bool(self._policies) and (
            time.time() - self._fetched_at < POLICY_REGISTRY_TTL_SECONDS
        )

    def replace(self, policies: list[Policy]) -> None:
        with self._lock:
            self._policies = {p.filename: p for p in policies}
            self._fetched_at = time.time()

    def add(self, policy: Policy) -> None:
        with self._lock:
            self._policies[policy.filename] = policy

    def values(self) -> list[Policy]:
        with self._lock:
            return list(self._policies.values())


_cache = _PolicyCache()


def _s3_client() -> Any:
    return get_s3_client()


def _read_metadata_attributes(s3_client: Any, key: str) -> tuple[str, str | None]:
    """Read a single metadata sidecar object and return its canonical company and tier."""
    obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=key)
    raw = obj["Body"].read().decode("utf-8")
    data = json.loads(raw)
    
    # חילוץ הנתונים מתוך בלוק ה-metadataAttributes
    attrs = data.get("metadataAttributes", {})
    company = normalize_company(attrs.get(METADATA_KEY))
    tier = attrs.get("tier")
    
    # ניקוי ה-Tier אם הוא הוגדר כלא ידוע
    if tier == "Unknown" or not tier:
        tier = None
        
    return company, tier


def _fetch_from_s3() -> list[Policy]:
    """List metadata sidecars in S3 and build the policy registry."""
    s3_client = _s3_client()
    policies: list[Policy] = []

    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET_NAME):
        for entry in page.get("Contents", []):
            key = entry["Key"]
            if not key.endswith(_METADATA_SUFFIX):
                continue
            filename = key[: -len(_METADATA_SUFFIX)]
            try:
                # שימוש בפונקציה החדשה שמחזירה שני ערכים
                company, tier = _read_metadata_attributes(s3_client, key)
            except (ClientError, ValueError, KeyError) as exc:
                logger.warning("Could not read metadata for %s: %s", key, exc)
                company = normalize_company(None)
                tier = None
                
            # יצירת אובייקט ה-Policy עם 3 הפרמטרים
            policies.append(Policy(filename=filename, company=company, tier=tier))

    logger.info("Policy registry refreshed: %d policies found.", len(policies))
    return policies


def list_policies(force_refresh: bool = False) -> list[Policy]:
    """Return the current registry, refreshing from S3 when the cache is stale."""
    if not force_refresh and _cache.is_fresh():
        return _cache.values()

    try:
        policies = _fetch_from_s3()
        _cache.replace(policies)
    except ClientError as exc:
        # On S3 failure, fall back to whatever we have cached rather than crash.
        logger.error("Failed to list policies from S3: %s", exc)

    return _cache.values()


def register_local(filename: str, company: str, tier: str | None = None) -> None:
    """Immediately add a freshly uploaded policy to the cache for instant UI feedback."""
    # הוספת ה-tier לאובייקט שנשמר במטמון המקומי
    _cache.add(Policy(filename=filename, company=normalize_company(company), tier=tier))


def get_known_companies(force_refresh: bool = False) -> list[str]:
    """Return the sorted unique set of insurer names currently in the KB."""
    companies = {p.company for p in list_policies(force_refresh=force_refresh)}
    return sorted(companies)
