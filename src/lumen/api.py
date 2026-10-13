"""HTTP API + web UI. Local: uvicorn lumen.api:app. AWS Lambda: lumen.api.handler.

POST /api/jobs                       {"files": [...]} -> job + upload target per file
PUT  /api/jobs/{id}/files/{name}     raw bytes (local storage only; S3 uses presigned POST)
POST /api/jobs/{id}/start            queue processing (Lambda: async self-invoke)
GET  /api/jobs/{id}                  status, sets, traces, output URLs
GET  /api/jobs/{id}/outputs/{set}/{file}
POST /api/jobs/{id}/sets/{set}/review {"action": "accept"|"reject"|"drop_frame", "frame": name}
"""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from lumen.jobs import (FILE_NAME, MAX_FILES, JobError, check_job_id, create_job, load_job, run_job,
                        review_set, save_upload, start_job, visible_status)
from lumen.store import MAX_UPLOAD_BYTES, LocalStore, store_from_env

log = logging.getLogger(__name__)
WEB = Path(__file__).parent / "web"
OUTPUT_FILES = {"before.jpg", "after.jpg", "preview.jpg"}
CSP = ("default-src 'self'; img-src 'self' data: blob: https://*.amazonaws.com; "
       "connect-src 'self' https://*.amazonaws.com; style-src 'self' https://fonts.googleapis.com; "
       "font-src https://fonts.gstatic.com; frame-ancestors 'none'; base-uri 'self'; object-src 'none'")

app = FastAPI(title="theMnexus Lumen", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=WEB), name="static")


@lru_cache(maxsize=1)
def get_store():
    return store_from_env()


class CreateJob(BaseModel):
    files: list[str] = Field(min_length=1, max_length=MAX_FILES)
    one_bracket: bool = False  # the person says the upload is one room: skip grouping


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Content-Security-Policy"] = CSP
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response


@app.exception_handler(JobError)
async def job_error(_: Request, exc: JobError):
    return JSONResponse({"error": str(exc)}, status_code=400)


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


@app.get("/api/health")
def health():
    import cv2

    return {"ok": True, "opencv": cv2.__version__}


@app.post("/api/jobs")
def new_job(body: CreateJob):
    job, uploads = create_job(get_store(), body.files, body.one_bracket)
    return {"job": job, "uploads": uploads}


@app.put("/api/jobs/{job_id}/files/{name}")
async def upload(job_id: str, name: str, request: Request):
    store = get_store()
    if not isinstance(store, LocalStore):
        raise JobError("upload with the presigned POST target")
    if not FILE_NAME.match(name):
        raise JobError("bad file name")
    declared = int(request.headers.get("content-length") or 0)
    if declared > MAX_UPLOAD_BYTES:
        raise JobError("file too large")
    save_upload(store, check_job_id(job_id), name, await request.body())
    return {"ok": True}


def _invoke_async(job_id: str) -> None:
    import boto3

    boto3.client("lambda").invoke(FunctionName=os.environ["AWS_LAMBDA_FUNCTION_NAME"],
                                  InvocationType="Event", Payload=json.dumps({"lumen_job": job_id}).encode())


@app.post("/api/jobs/{job_id}/start")
def start(job_id: str, tasks: BackgroundTasks):
    store = get_store()
    job = start_job(store, job_id)
    if os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
        _invoke_async(job_id)  # the HTTP invocation returns now; a second one does the work
    elif os.environ.get("LUMEN_SYNC") == "1":
        job = run_job(store, job_id)
    else:
        tasks.add_task(run_job, store, job_id)
    return job


def _with_urls(job: dict) -> dict:
    store = get_store()

    def url(key: str | None) -> str | None:
        if key is None:
            return None
        return store.download_url(key) or "/api/" + key.replace("/out/", "/outputs/", 1)

    sets = [{**s, "urls": {k: url(s.get(k)) for k in ("before", "after", "preview")}} for s in job["sets"]]
    return {**job, "sets": sets}


@app.get("/api/jobs/{job_id}")
def status(job_id: str):
    return _with_urls(visible_status(load_job(get_store(), job_id)))


class ReviewRequest(BaseModel):
    action: Literal["accept", "reject", "drop_frame"]
    frame: str | None = Field(default=None, max_length=125)


@app.post("/api/jobs/{job_id}/sets/{set_id}/review")
def review(job_id: str, set_id: str, body: ReviewRequest):
    """A person accepts, rejects, or drops one frame and re-runs a set; stored in the trace."""
    if not set_id.startswith("set") or not set_id[3:].isdigit():
        raise JobError("bad set id")
    job = review_set(get_store(), check_job_id(job_id), set_id, body.action, body.frame)
    return _with_urls(job)


@app.get("/api/jobs/{job_id}/outputs/{set_id}/{file}")
def output(job_id: str, set_id: str, file: str):
    if file not in OUTPUT_FILES or not set_id.startswith("set") or not set_id[3:].isdigit():
        raise JobError("bad output path")
    key = f"jobs/{check_job_id(job_id)}/out/{set_id}/{file}"
    store = get_store()
    if not store.exists(key):
        return JSONResponse({"error": "not found"}, status_code=404)
    direct = store.download_url(key)
    if direct:
        return RedirectResponse(direct)
    return Response(store.get(key), media_type="image/jpeg")


def handler(event: dict, context):
    """AWS Lambda entry: async worker events carry lumen_job, everything else is HTTP."""
    if "lumen_job" in event:
        return run_job(get_store(), check_job_id(event["lumen_job"]))["status"]
    from mangum import Mangum

    return Mangum(app, lifespan="off")(event, context)
