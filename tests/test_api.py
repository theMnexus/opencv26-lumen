import pytest
from fastapi.testclient import TestClient

import lumen.api as api
from lumen import synth


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("LUMEN_STORAGE", "local")
    monkeypatch.setenv("LUMEN_DATA_DIR", str(tmp_path / "store"))
    monkeypatch.setenv("LUMEN_SYNC", "1")
    monkeypatch.delenv("AWS_LAMBDA_FUNCTION_NAME", raising=False)
    api.get_store.cache_clear()
    yield TestClient(api.app)
    api.get_store.cache_clear()


@pytest.fixture(scope="module")
def photos(tmp_path_factory):
    out = tmp_path_factory.mktemp("photos")
    truth = synth.generate(out, sets=2, seed=5)
    return [(f["file"], (out / f["file"]).read_bytes()) for s in truth["sets"] for f in s["frames"]]


def _create_and_upload(client, photos):
    created = client.post("/api/jobs", json={"files": [name for name, _ in photos]}).json()
    job_id = created["job"]["id"]
    for name, data in photos:
        target = created["uploads"][name]
        assert target["method"] == "PUT"
        assert client.put(target["url"], content=data).status_code == 200
    return job_id


def test_health_reports_opencv_5(client):
    body = client.get("/api/health").json()
    assert body["ok"] and body["opencv"].startswith("5.")


def test_index_and_security_headers(client):
    res = client.get("/")
    assert res.status_code == 200 and "Lumen" in res.text
    assert "default-src 'self'" in res.headers["content-security-policy"]
    assert res.headers["x-content-type-options"] == "nosniff"
    assert client.get("/static/app.js").status_code == 200


def test_full_flow(client, photos):
    job_id = _create_and_upload(client, photos)
    started = client.post(f"/api/jobs/{job_id}/start")
    assert started.status_code == 200
    job = client.get(f"/api/jobs/{job_id}").json()
    assert job["status"] == "done" and len(job["sets"]) == 2
    for s in job["sets"]:
        assert s["status"] == "ok" and s["steps"][-1]["decision"] == "done"
        for kind in ("before", "after", "preview"):
            img = client.get(s["urls"][kind])
            assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"
            assert img.content[:2] == b"\xff\xd8"


def test_lambda_worker_event_runs_the_job(client, photos):
    job_id = _create_and_upload(client, photos[:3])
    assert api.handler({"lumen_job": job_id}, None) == "done"


@pytest.mark.parametrize("names", [["../evil.jpg"], ["a.exe"], [".hidden.jpg"], ["a.jpg", "a.jpg"], []])
def test_bad_file_lists_are_rejected(client, names):
    res = client.post("/api/jobs", json={"files": names})
    assert res.status_code in (400, 422)


def test_unannounced_upload_is_rejected(client, photos):
    job_id = client.post("/api/jobs", json={"files": ["a.jpg"]}).json()["job"]["id"]
    res = client.put(f"/api/jobs/{job_id}/files/b.jpg", content=b"x")
    assert res.status_code == 400 and "announced" in res.json()["error"]


def test_start_needs_every_file(client):
    job_id = client.post("/api/jobs", json={"files": ["a.jpg", "b.jpg"]}).json()["job"]["id"]
    client.put(f"/api/jobs/{job_id}/files/a.jpg", content=b"x")
    res = client.post(f"/api/jobs/{job_id}/start")
    assert res.status_code == 400 and "not uploaded" in res.json()["error"]


def test_start_twice_is_rejected(client, photos):
    job_id = _create_and_upload(client, photos[:3])
    assert client.post(f"/api/jobs/{job_id}/start").status_code == 200
    assert client.post(f"/api/jobs/{job_id}/start").status_code == 400


@pytest.mark.parametrize("path", [
    "/api/jobs/not-a-uuid",
    "/api/jobs/" + "0" * 32,
    "/api/jobs/" + "0" * 32 + "/outputs/set01/secret.txt",
    "/api/jobs/" + "0" * 32 + "/outputs/..%2F..%2Fjob/after.jpg",
])
def test_bad_ids_and_paths(client, path):
    assert client.get(path).status_code in (400, 404)


def test_unreadable_image_fails_cleanly(client):
    job_id = client.post("/api/jobs", json={"files": ["a.jpg", "b.jpg"]}).json()["job"]["id"]
    for name in ("a.jpg", "b.jpg"):
        client.put(f"/api/jobs/{job_id}/files/{name}", content=b"not a jpeg")
    job = client.post(f"/api/jobs/{job_id}/start").json()
    assert job["status"] == "failed" and job["error"]


def test_one_bracket_skips_grouping(client, photos):
    """The person can say the upload is one room (staged brackets with minutes between shots)."""
    room = photos[:3]
    created = client.post("/api/jobs", json={"files": [n for n, _ in room], "one_bracket": True}).json()
    job_id = created["job"]["id"]
    assert created["job"]["one_bracket"] is True
    for name, data in room:
        client.put(created["uploads"][name]["url"], content=data)
    job = client.post(f"/api/jobs/{job_id}/start").json()
    assert job["status"] == "done" and len(job["sets"]) == 1 and job["grouping"] == []
    assert job["sets"][0]["status"] == "ok"


def test_one_bracket_must_be_a_boolean(client, photos):
    res = client.post("/api/jobs", json={"files": ["a.jpg"], "one_bracket": "maybe"})
    assert res.status_code == 422
