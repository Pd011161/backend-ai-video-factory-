"""S3 storage service — upload/download media bytes.

AWS credentials are read from env vars (set in .env):
  AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_REGION, AWS_BUCKET, AWS_URL
"""

from __future__ import annotations

import os
from functools import lru_cache
from urllib.parse import quote, unquote


def is_configured() -> bool:
    """Return True when all required AWS env vars are present."""
    return bool(
        os.getenv("AWS_BUCKET")
        and os.getenv("AWS_ACCESS_KEY")
        and os.getenv("AWS_SECRET_KEY")
        and os.getenv("AWS_REGION")
    )


@lru_cache(maxsize=1)
def _client():
    import boto3  # imported lazily so the app starts without boto3 when S3 is disabled

    return boto3.client(
        "s3",
        region_name=os.getenv("AWS_REGION"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("AWS_SECRET_KEY"),
    )


def upload_bytes(key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
    """Upload *data* to S3 at *key* and return the public URL."""
    bucket = os.getenv("AWS_BUCKET", "")
    _client().put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)
    base = os.getenv("AWS_URL", "").rstrip("/")
    return f"{base}/{quote(key, safe='/')}"


def download_bytes(key: str) -> bytes:
    """Download *key* from S3 and return raw bytes."""
    bucket = os.getenv("AWS_BUCKET", "")
    response = _client().get_object(Bucket=bucket, Key=key)
    return response["Body"].read()


def delete_bytes(key: str) -> None:
    """Delete *key* from S3 (no-op if it doesn't exist)."""
    bucket = os.getenv("AWS_BUCKET", "")
    _client().delete_object(Bucket=bucket, Key=key)


def key_from_url(url: str) -> str | None:
    """The object key for a URL this module returned from upload_bytes(); None if it
    doesn't match AWS_URL (e.g. a local /api/media/ URL — nothing to delete on S3)."""
    base = os.getenv("AWS_URL", "").rstrip("/")
    if not base or not (url or "").startswith(base + "/"):
        return None
    return unquote(url[len(base) + 1:])
