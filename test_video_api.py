from datetime import datetime, timezone
from unittest.mock import Mock

import boto3
import pytest
from botocore.stub import Stubber
from fastapi.testclient import TestClient

import modal_api
import video_storage


@pytest.fixture
def storage(monkeypatch):
    s3 = boto3.client("s3", region_name="auto", endpoint_url="https://example.r2.cloudflarestorage.com",
                      aws_access_key_id="test", aws_secret_access_key="test")
    monkeypatch.setenv("R2_BUCKET_NAME", "test-videos")
    monkeypatch.setattr(video_storage, "client", lambda: s3)
    with Stubber(s3) as stub:
        yield stub
        stub.assert_no_pending_responses()


@pytest.fixture
def api():
    return TestClient(modal_api.api.get_raw_f()())


def test_listing_preserves_cursor_and_filters_non_videos(api, storage):
    storage.add_response("list_objects_v2", {
        "Contents": [{"Key": "videos/job/result.mp4", "Size": 42,
                      "LastModified": datetime(2026, 9, 10, tzinfo=timezone.utc)},
                     {"Key": "videos/job/manifest.json", "Size": 1}],
        "NextContinuationToken": "next", "IsTruncated": True,
    }, {"Bucket": "test-videos", "Prefix": "videos/job/", "MaxKeys": 2,
        "ContinuationToken": "previous"})
    r = api.get("/videos", params={"limit": 2, "cursor": "previous", "prefix": "job/"},
                headers={"Origin": "http://localhost:5173"})
    assert r.status_code == 200
    assert [v["key"] for v in r.json()["videos"]] == ["videos/job/result.mp4"]
    assert r.json()["next_cursor"] == "next"
    assert r.headers["access-control-allow-origin"] == "*"


def test_empty_listing(api, storage):
    storage.add_response("list_objects_v2", {},
                         {"Bucket": "test-videos", "Prefix": "videos/", "MaxKeys": 50})
    assert api.get("/videos").json() == {"videos": [], "next_cursor": None}


def test_playback_is_expiring_signed_url(api, storage):
    storage.add_response("head_object", {"ContentLength": 42, "ContentType": "video/mp4"},
                         {"Bucket": "test-videos", "Key": "videos/job/result.mp4"})
    r = api.get("/videos/playback", params={"key": "videos/job/result.mp4"})
    assert r.status_code == 200
    assert "X-Amz-Signature=" in r.json()["url"]
    assert "X-Amz-Expires=3600" in r.json()["url"]
    assert r.headers["cache-control"] == "no-store"


@pytest.mark.parametrize("key", ["private/a.mp4", "videos/../a.mp4", "videos/a.json"])
def test_invalid_keys_rejected(api, key):
    assert api.get("/videos/playback", params={"key": key}).status_code == 400


@pytest.mark.parametrize("code,status,expected", [("404", 404, 404), ("AccessDenied", 403, 503)])
def test_storage_errors(api, storage, code, status, expected):
    storage.add_client_error("head_object", service_error_code=code, http_status_code=status,
                              expected_params={"Bucket": "test-videos", "Key": "videos/a.mp4"})
    assert api.get("/videos/playback", params={"key": "videos/a.mp4"}).status_code == expected


def test_limit_bounds_and_no_public_upload(api):
    assert api.get("/videos?limit=101").status_code == 422
    assert api.post("/videos").status_code == 405
    assert api.get("/health").json() == {"status": "ok"}


def test_upload_uses_video_mime_and_unique_key(monkeypatch, tmp_path):
    s3 = Mock()
    monkeypatch.setattr(video_storage, "client", lambda: s3)
    monkeypatch.setenv("R2_BUCKET_NAME", "test-videos")
    path = tmp_path / "result.mp4"
    path.write_bytes(b"test")
    key = video_storage.upload_video(str(path), "1dcda5fd-374e-483a-905e-9ebc7fae2717")
    assert key.startswith("videos/1dcda5fd-374e-483a-905e-9ebc7fae2717/")
    s3.upload_file.assert_called_once_with(str(path), "test-videos", key,
                                          ExtraArgs={"ContentType": "video/mp4"})
