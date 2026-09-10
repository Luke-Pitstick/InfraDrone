"""R2 storage for browser-compatible result videos under videos/."""

import os
from functools import lru_cache
from pathlib import Path, PurePosixPath
from uuid import uuid4

import boto3
from botocore.config import Config

CONTENT_TYPES = {".mp4": "video/mp4", ".webm": "video/webm"}
URL_TTL = 3600


@lru_cache
def client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT_URL"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
        config=Config(signature_version="s3v4", connect_timeout=5, read_timeout=10,
                      retries={"mode": "standard", "total_max_attempts": 2}),
    )


def bucket():
    return os.environ["R2_BUCKET_NAME"]


def validate_key(key: str):
    if (not key.startswith("videos/") or ".." in key.split("/")
            or PurePosixPath(key).suffix.lower() not in CONTENT_TYPES):
        raise ValueError("Expected an MP4 or WebM key under videos/")
    return key


def list_videos(limit: int, cursor: str | None, prefix: str):
    params = {"Bucket": bucket(), "Prefix": "videos/" + prefix, "MaxKeys": limit}
    if cursor:
        params["ContinuationToken"] = cursor
    page = client().list_objects_v2(**params)
    return {
        "videos": [
            {"key": obj["Key"], "size": obj["Size"],
             "last_modified": obj["LastModified"].isoformat()}
            for obj in page.get("Contents", [])
            if PurePosixPath(obj["Key"]).suffix.lower() in CONTENT_TYPES
        ],
        "next_cursor": page.get("NextContinuationToken"),
    }


def playback(key: str):
    validate_key(key)
    metadata = client().head_object(Bucket=bucket(), Key=key)
    return {
        "key": key,
        "url": client().generate_presigned_url(
            "get_object", Params={"Bucket": bucket(), "Key": key}, ExpiresIn=URL_TTL,
        ),
        "expires_in": URL_TTL,
        "content_type": metadata["ContentType"],
        "size": metadata["ContentLength"],
    }


def upload_video(path: str, job_id: str):
    """Trusted producer helper; multipart upload keeps large files out of memory."""
    from uuid import UUID

    source = Path(path)
    extension = source.suffix.lower()
    if extension not in CONTENT_TYPES or not source.is_file():
        raise ValueError("Expected an existing MP4 or WebM video")
    key = f"videos/{UUID(job_id)}/{uuid4()}{extension}"
    client().upload_file(str(source), bucket(), key,
                         ExtraArgs={"ContentType": CONTENT_TYPES[extension]})
    return key


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Upload a result video to R2 using producer credentials")
    parser.add_argument("path")
    parser.add_argument("--job-id", required=True, help="Inference job UUID")
    args = parser.parse_args()
    print(upload_video(args.path, args.job_id))
