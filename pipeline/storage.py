"""Video storage: local disk or any S3-compatible bucket.

Returns a URL the browser can fetch. With the local backend that is the
/videos mount; with S3 it is either the CDN base URL or a presigned link.
"""

from __future__ import annotations

from pathlib import Path

from config import (
    S3_BUCKET,
    S3_ENDPOINT_URL,
    S3_PREFIX,
    S3_PUBLIC_BASE_URL,
    S3_REGION,
    STORAGE_BACKEND,
)


def _s3_client():
    import boto3

    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT_URL,
        region_name=S3_REGION,
        aws_access_key_id=__import__("os").environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=__import__("os").environ.get("AWS_SECRET_ACCESS_KEY"),
    )


def upload(local_path: Path, key: str) -> str:
    """Store a finished video and return a fetchable URL."""
    if STORAGE_BACKEND != "s3" or not S3_BUCKET:
        return f"/videos/{key}"

    object_key = f"{S3_PREFIX}/{key}" if S3_PREFIX else key
    client = _s3_client()
    client.upload_file(
        str(local_path),
        S3_BUCKET,
        object_key,
        ExtraArgs={"ContentType": "video/mp4"},
    )
    if S3_PUBLIC_BASE_URL:
        return f"{S3_PUBLIC_BASE_URL.rstrip('/')}/{object_key}"
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": S3_BUCKET, "Key": object_key},
        ExpiresIn=604800,
    )
