"""The AWS code path (S3Store, presigned POST, async self-invoke, presigned GET)
with S3 faked by moto, plus the dead-worker rule."""

from datetime import datetime, timedelta, timezone

import boto3
import pytest
from fastapi.testclient import TestClient
from moto import mock_aws

import lumen.api as api
from lumen import synth
from lumen.jobs import STALE_AFTER_S, visible_status

BUCKET = "lumen-test"


@pytest.fixture()
def aws(monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("LUMEN_STORAGE", "s3")
    monkeypatch.setenv("LUMEN_BUCKET", BUCKET)
    monkeypatch.setenv("AWS_LAMBDA_FUNCTION_NAME", "lumen-test-fn")
    with mock_aws():
        s3 = boto3.client("s3")
        s3.create_bucket(Bucket=BUCKET)
        api.get_store.cache_clear()
        invoked = []
        monkeypatch.setattr(api, "_invoke_async", invoked.append)
        yield TestClient(api.app), s3, invoked
        api.get_store.cache_clear()


def test_s3_flow_end_to_end(aws, tmp_path):
    client, s3, invoked = aws
    truth = synth.generate(tmp_path, sets=1, seed=5)
    names = [f["file"] for f in truth["sets"][0]["frames"]]
    created = client.post("/api/jobs", json={"files": names}).json()
    job_id = created["job"]["id"]
    for name in names:
        target = created["uploads"][name]
        assert target["method"] == "POST" and target["fields"]["key"] == f"jobs/{job_id}/in/{name}"
        assert "policy" in target["fields"]  # size limit travels in the signed policy
        s3.put_object(Bucket=BUCKET, Key=target["fields"]["key"], Body=(tmp_path / name).read_bytes())

    assert client.put(f"/api/jobs/{job_id}/files/{names[0]}", content=b"x").status_code == 400  # S3 mode: no PUT
    started = client.post(f"/api/jobs/{job_id}/start").json()
    assert started["status"] == "queued" and invoked == [job_id]

    assert api.handler({"lumen_job": job_id}, None) == "done"  # what the async invocation runs
    job = client.get(f"/api/jobs/{job_id}").json()
    assert job["status"] == "done"
    urls = job["sets"][0]["urls"]
    assert all("X-Amz-Signature" in urls[k] for k in ("before", "after", "preview"))
    res = client.get(f"/api/jobs/{job_id}/outputs/set01/after.jpg", follow_redirects=False)
    assert res.status_code == 307 and "X-Amz-Signature" in res.headers["location"]


def test_dead_worker_is_reported_failed():
    now = datetime(2026, 10, 9, 12, 0, tzinfo=timezone.utc)
    started = (now - timedelta(seconds=STALE_AFTER_S + 5)).isoformat()
    job = {"status": "running", "started_at": started}
    shown = visible_status(job, now)
    assert shown["status"] == "failed" and "time or memory" in shown["error"]
    fresh = {"status": "running", "started_at": (now - timedelta(seconds=30)).isoformat()}
    assert visible_status(fresh, now)["status"] == "running"
    assert visible_status({"status": "done"}, now)["status"] == "done"


def test_presigned_upload_uses_the_regional_endpoint(monkeypatch):
    """Deployed in ap-southeast-2, the global s3.amazonaws.com host answered uploads with a 307."""
    from lumen.store import S3Store

    monkeypatch.setenv("AWS_REGION", "ap-southeast-2")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    url = S3Store("photos").s3.generate_presigned_post("photos", "jobs/x/in/a.jpg")["url"]
    assert url == "https://photos.s3.ap-southeast-2.amazonaws.com/"
