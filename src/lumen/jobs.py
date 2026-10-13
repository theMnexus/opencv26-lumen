"""A job = one uploaded folder of photos -> grouped, processed, traced.

Lifecycle: uploading -> queued -> running -> done | failed.
State lives in jobs/<id>/job.json in the store, so the API and the worker
(same process locally, a second Lambda invocation on AWS) share nothing else.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import time
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from lumen.agent import link_dicts, run_set, trace_dict
from lumen.group import group_frames, one_bracket
from lumen.io import load_frame
from lumen.store import MAX_UPLOAD_BYTES, LocalStore, S3Store

log = logging.getLogger(__name__)

JOB_ID = re.compile(r"^[0-9a-f]{32}$")
FILE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,119}\.(jpe?g|png|tiff?)$", re.IGNORECASE)
MAX_FILES = 60
PREVIEW_LONG_SIDE = 1600
JPEG_QUALITY = 92
GROUP_REDUCE = 4  # grouping only needs thumbnails (64 px DCT, 256 px edges)
# Lambda timeout is 900 s. A worker killed for time or memory never reaches the
# except below, so a job still queued/running after this long is reported failed.
STALE_AFTER_S = 960
# Peak memory per megapixel per frame: 24 MP x 5 frames peaked at 7.5 GB (62 MB), rounded up.
MB_PER_FRAME_MP = 70
RUNTIME_MB = 400  # Python, OpenCV and numpy before any image is loaded

Store = LocalStore | S3Store


class JobError(ValueError):
    """Bad request from the client: the message is safe to show."""


def _key(job_id: str, *parts: str) -> str:
    return "/".join(["jobs", job_id, *parts])


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def check_job_id(job_id: str) -> str:
    if not JOB_ID.match(job_id):
        raise JobError("bad job id")
    return job_id


def load_job(store: Store, job_id: str) -> dict:
    key = _key(check_job_id(job_id), "job.json")
    if not store.exists(key):
        raise JobError("job not found")
    return json.loads(store.get(key))


def _save_job(store: Store, job: dict) -> dict:
    store.put(_key(job["id"], "job.json"), json.dumps(job).encode(), "application/json")
    return job


def create_job(store: Store, files: list[str], one_set: bool = False) -> tuple[dict, dict]:
    """Returns the job and an upload target per file. one_set: the person says it is one bracket."""
    if not 1 <= len(files) <= MAX_FILES:
        raise JobError(f"send 1 to {MAX_FILES} files")
    bad = [f for f in files if not FILE_NAME.match(f)]
    if bad:
        raise JobError(f"unsupported file names: {bad[:5]}")
    if len(set(files)) != len(files):
        raise JobError("file names must be unique")
    job = {"id": uuid.uuid4().hex, "status": "uploading", "created_at": _now(), "files": files,
           "one_bracket": bool(one_set), "sets": [], "grouping": [], "error": None, "seconds": None}
    _save_job(store, job)
    return job, {name: store.upload_target(job["id"], name) for name in files}


def save_upload(store: Store, job_id: str, name: str, data: bytes) -> None:
    job = load_job(store, job_id)
    if job["status"] != "uploading":
        raise JobError("job is not accepting files")
    if name not in job["files"]:
        raise JobError("file was not announced when the job was created")
    if not 0 < len(data) <= MAX_UPLOAD_BYTES:
        raise JobError(f"file must be 1 byte to {MAX_UPLOAD_BYTES // 2**20} MB")
    store.put(_key(job_id, "in", name), data)


def start_job(store: Store, job_id: str) -> dict:
    job = load_job(store, job_id)
    if job["status"] != "uploading":
        raise JobError(f"job is already {job['status']}")
    missing = [f for f in job["files"] if not store.exists(_key(job_id, "in", f))]
    if missing:
        raise JobError(f"{len(missing)} files not uploaded yet, e.g. {missing[0]}")
    return _save_job(store, {**job, "status": "queued", "queued_at": _now()})


REVIEW_ACTIONS = ("accept", "reject", "drop_frame")


def _person_step(action: str, frame: str | None, status_before: str, decision: str) -> dict:
    return {"step": 0, "rule": "person_review", "tool": action, "observe": {"status_before": status_before},
            "params": {"frame": frame}, "result": {}, "decision": decision}


def set_megapixel_budget() -> float | None:
    """On Lambda, how many megapixels x frames one set may use; None = no limit (local)."""
    memory = os.environ.get("AWS_LAMBDA_FUNCTION_MEMORY_SIZE")
    if not memory:
        return None
    return max(1.0, (int(memory) - RUNTIME_MB) / MB_PER_FRAME_MP)


def load_set(paths: list[Path]) -> list:
    """Full-resolution frames, shrunk evenly when the set would not fit the memory budget.
    A 3 GB Lambda ran out of memory on 7 frames of 16 MP; the output is then smaller, not lost."""
    frames = [load_frame(p) for p in paths]
    budget = set_megapixel_budget()
    if budget is None:
        return frames
    per_frame = budget / len(frames)
    out = []
    for f in frames:
        h, w = f.image.shape[:2]
        scale = (per_frame * 1e6 / (h * w)) ** 0.5
        if scale >= 1:
            out.append(f)
            continue
        size = (max(1, round(w * scale)), max(1, round(h * scale)))
        out.append(replace(f, image=cv2.resize(f.image, size, interpolation=cv2.INTER_AREA)))
    return out


def _rerun_without(store: Store, job_id: str, set_entry: dict, frame: str) -> dict:
    keep = [f for f in set_entry["frames"] if f != frame]
    if len(keep) < 2:
        raise JobError("a bracket needs at least 2 frames after dropping one")
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for name in keep:
            path = Path(tmp) / name
            path.write_bytes(store.get(_key(job_id, "in", name)))
            paths.append(path)
        state = run_set(load_set(paths))
    return _save_set(store, job_id, set_entry["id"], state)


def review_set(store: Store, job_id: str, set_id: str, action: str, frame: str | None = None) -> dict:
    """A person decides on one set. The decision is stored in the set and as the first trace step."""
    job = load_job(store, job_id)
    if job["status"] != "done":
        raise JobError("only finished jobs can be reviewed")
    if action not in REVIEW_ACTIONS:
        raise JobError(f"action must be one of {', '.join(REVIEW_ACTIONS)}")
    index = next((i for i, s in enumerate(job["sets"]) if s["id"] == set_id), None)
    if index is None:
        raise JobError("set not found")
    current = job["sets"][index]
    if action == "accept":
        if current.get("after") is None:
            raise JobError("nothing to accept: the agent produced no image for this set")
        updated, decision = {**current, "status": "ok", "reason": "accepted by a person"}, "accepted"
    elif action == "reject":
        updated, decision = {**current, "status": "rejected", "reason": "rejected by a person"}, "rejected"
    else:
        if frame not in current["frames"]:
            raise JobError("frame is not part of this set")
        updated, decision = _rerun_without(store, job_id, current, frame), "dropped frame, re-ran the agent"
    # A re-run brings its own trace; accept/reject keep the agent's trace. Older person steps are replaced.
    agent_steps = [st for st in updated["steps"] if st["rule"] != "person_review"]
    person = _person_step(action, frame, current["status"], decision)
    review = {"action": action, "frame": frame, "at": _now(), "status_before": current["status"]}
    updated = {**updated, "review": review, "steps": [person, *agent_steps]}
    return _save_job(store, {**job, "sets": [*job["sets"][:index], updated, *job["sets"][index + 1:]]})


def visible_status(job: dict, now: datetime | None = None) -> dict:
    """The job as the UI should see it: a worker that died silently becomes 'failed'."""
    if job["status"] not in ("queued", "running"):
        return job
    since = job.get("started_at") or job.get("queued_at")
    if since is None:
        return job
    age = ((now or datetime.now(timezone.utc)) - datetime.fromisoformat(since)).total_seconds()
    if age <= STALE_AFTER_S:
        return job
    return {**job, "status": "failed",
            "error": f"worker stopped after {int(age)}s (time or memory limit); try fewer photos per upload"}


def _jpeg(image: np.ndarray, long_side: int | None = None) -> bytes:
    if long_side and max(image.shape[:2]) > long_side:
        scale = long_side / max(image.shape[:2])
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return buf.tobytes()


def _save_set(store: Store, job_id: str, set_id: str, state) -> dict:
    out = {"id": set_id, **trace_dict(state), "before": None, "after": None, "preview": None}
    ref = state.alignment.reference_index if state.alignment is not None else len(state.frames) // 2
    before = _key(job_id, "out", set_id, "before.jpg")
    store.put(before, _jpeg(state.frames[ref].image, PREVIEW_LONG_SIDE), "image/jpeg")
    out["before"] = before
    if state.image is not None:
        out["after"] = _key(job_id, "out", set_id, "after.jpg")
        out["preview"] = _key(job_id, "out", set_id, "preview.jpg")
        store.put(out["after"], _jpeg(state.image), "image/jpeg")
        store.put(out["preview"], _jpeg(state.image, PREVIEW_LONG_SIDE), "image/jpeg")
    return out


def run_job(store: Store, job_id: str) -> dict:
    """Group the folder on 1/4-size decodes, then process one set at a time at full
    resolution and drop it before the next: memory stays at one set, not one folder."""
    job = _save_job(store, {**load_job(store, job_id), "status": "running", "started_at": _now()})
    started = time.time()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for name in job["files"]:
                path = Path(tmp) / name
                path.write_bytes(store.get(_key(job_id, "in", name)))
                paths.append(path)
            thumbs = [load_frame(p, GROUP_REDUCE) for p in paths]
            grouping = one_bracket(thumbs) if job.get("one_bracket") else group_frames(thumbs)
            sets = []
            for i, ids in enumerate(grouping.sets):
                state = run_set(load_set([paths[k] for k in ids]))
                sets.append(_save_set(store, job_id, f"set{i + 1:02d}", state))
                del state
        done = {**job, "status": "done", "sets": sets, "grouping": link_dicts(grouping),
                "seconds": round(time.time() - started, 2)}
        return _save_job(store, done)
    except Exception as exc:  # the job must end in a state the UI can show
        log.exception("job %s failed", job_id)
        message = str(exc) if isinstance(exc, (JobError, ValueError)) else "processing error, see server log"
        return _save_job(store, {**job, "status": "failed", "error": message,
                                 "seconds": round(time.time() - started, 2)})
