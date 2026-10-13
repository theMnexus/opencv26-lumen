"""A person can accept, reject, or drop a frame and re-run a set; the decision lands in the trace."""

import io

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

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
def brackets(tmp_path_factory):
    """Two sets: set A as shot, set B pitched 25 degrees so the agent asks a person."""
    out = tmp_path_factory.mktemp("review")
    truth = synth.generate(out, sets=2, seed=5)
    f = synth.FOCAL_PX
    k = np.array([[f, 0, synth.WIDTH / 2], [0, f, synth.HEIGHT / 2], [0, 0, 1]])
    a = np.radians(25)
    pitch = k @ np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]]) @ np.linalg.inv(k)
    photos = []
    for i, s in enumerate(truth["sets"]):
        for fr in s["frames"]:
            path = out / fr["file"]
            if i == 1:  # keep the EXIF, as a real camera file would
                with Image.open(path) as src:
                    exif = src.info["exif"]
                img = cv2.warpPerspective(cv2.imread(str(path)), pitch, (synth.WIDTH, synth.HEIGHT),
                                          borderMode=cv2.BORDER_REPLICATE)
                buf = io.BytesIO()
                Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).save(buf, "JPEG", quality=92, exif=exif)
                data = buf.getvalue()
            else:
                data = path.read_bytes()
            photos.append((fr["file"], data))
    return photos


def _done_job(client, photos):
    created = client.post("/api/jobs", json={"files": [n for n, _ in photos]}).json()
    job_id = created["job"]["id"]
    for name, data in photos:
        client.put(created["uploads"][name]["url"], content=data)
    job = client.post(f"/api/jobs/{job_id}/start").json()
    assert job["status"] == "done"
    return job_id, job


def _set(job, status):
    return next(s for s in job["sets"] if s["status"] == status)


def test_accept_turns_needs_review_into_ok(client, brackets):
    job_id, job = _done_job(client, brackets)
    pending = _set(job, "needs_review")
    assert pending["after"] is not None  # the agent made an image before stopping
    res = client.post(f"/api/jobs/{job_id}/sets/{pending['id']}/review", json={"action": "accept"})
    assert res.status_code == 200
    reviewed = next(s for s in res.json()["sets"] if s["id"] == pending["id"])
    assert reviewed["status"] == "ok" and reviewed["review"]["status_before"] == "needs_review"
    assert reviewed["steps"][0]["rule"] == "person_review" and reviewed["steps"][0]["decision"] == "accepted"
    assert reviewed["steps"][1]["rule"] == "need_alignment"  # the agent's own trace is kept


def test_reject(client, brackets):
    job_id, job = _done_job(client, brackets)
    pending = _set(job, "needs_review")
    res = client.post(f"/api/jobs/{job_id}/sets/{pending['id']}/review", json={"action": "reject"})
    assert next(s for s in res.json()["sets"] if s["id"] == pending["id"])["status"] == "rejected"


def test_drop_frame_reruns_the_agent(client, brackets):
    job_id, job = _done_job(client, brackets)
    target = _set(job, "ok")
    dropped = target["frames"][-1]
    res = client.post(f"/api/jobs/{job_id}/sets/{target['id']}/review",
                      json={"action": "drop_frame", "frame": dropped})
    assert res.status_code == 200
    rerun = next(s for s in res.json()["sets"] if s["id"] == target["id"])
    assert dropped not in rerun["frames"] and len(rerun["frames"]) == len(target["frames"]) - 1
    assert rerun["steps"][0]["tool"] == "drop_frame" and rerun["steps"][-1]["decision"] == "done"
    assert client.get(rerun["urls"]["after"]).status_code == 200


@pytest.mark.parametrize("body, path_set, status", [
    ({"action": "delete_everything"}, "set01", 422),
    ({"action": "accept"}, "../x", 404),
    ({"action": "accept"}, "setXY", 400),
    ({"action": "accept"}, "set99", 400),
    ({"action": "drop_frame", "frame": "not-in-set.jpg"}, "set01", 400),
])
def test_bad_review_requests(client, brackets, body, path_set, status):
    job_id, _ = _done_job(client, brackets)
    assert client.post(f"/api/jobs/{job_id}/sets/{path_set}/review", json=body).status_code == status


def test_review_needs_a_finished_job(client):
    job_id = client.post("/api/jobs", json={"files": ["a.jpg"]}).json()["job"]["id"]
    res = client.post(f"/api/jobs/{job_id}/sets/set01/review", json={"action": "accept"})
    assert res.status_code == 400 and "finished" in res.json()["error"]


def test_cannot_accept_when_no_image_was_made(client, brackets):
    job_id, job = _done_job(client, brackets[:1] + brackets[3:4])  # two single frames from two rooms
    lonely = _set(job, "needs_review")
    assert lonely["after"] is None
    res = client.post(f"/api/jobs/{job_id}/sets/{lonely['id']}/review", json={"action": "accept"})
    assert res.status_code == 400 and "nothing to accept" in res.json()["error"]
