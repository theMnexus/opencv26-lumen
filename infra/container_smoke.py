"""Drive the Lambda container through the Runtime Interface Emulator, like a Function URL would.

    docker run --rm -p 9000:8080 -e LUMEN_STORAGE=local -e LUMEN_DATA_DIR=/tmp/store lumen:arm64
    uv run python infra/container_smoke.py out/demo-photos

Steps: health -> web page + static files -> create job -> upload every photo -> worker event -> job status.
"""

from __future__ import annotations

import base64
import json
import sys
import time
import urllib.request
from pathlib import Path

RIE = "http://localhost:9000/2015-03-31/functions/function/invocations"


def invoke(event: dict) -> dict:
    req = urllib.request.Request(RIE, data=json.dumps(event).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as res:
        return json.loads(res.read())


def http(method: str, path: str, body: bytes | None = None, content_type: str = "application/json") -> dict:
    event = {
        "version": "2.0", "rawPath": path, "rawQueryString": "", "routeKey": "$default",
        "headers": {"content-type": content_type, "host": "lambda-url.local"},
        "requestContext": {"http": {"method": method, "path": path, "sourceIp": "127.0.0.1"},
                           "domainName": "lambda-url.local", "stage": "$default"},
        "isBase64Encoded": body is not None,
        "body": base64.b64encode(body).decode() if body is not None else None,
    }
    res = invoke(event)
    payload = base64.b64decode(res["body"]) if res.get("isBase64Encoded") else res["body"].encode()
    if res["statusCode"] >= 400:
        raise SystemExit(f"{method} {path} -> {res['statusCode']}: {payload[:200]!r}")
    return json.loads(payload) if res["headers"].get("content-type", "").startswith("application/json") else {}


def main(folder: str) -> None:
    photos = sorted(Path(folder).glob("*.jpg"))
    print("health", http("GET", "/api/health"))
    for path in ("/", "/static/app.js", "/static/app.css"):
        res = invoke({"version": "2.0", "rawPath": path, "rawQueryString": "", "routeKey": "$default",
                      "headers": {"host": "lambda-url.local"}, "isBase64Encoded": False, "body": None,
                      "requestContext": {"http": {"method": "GET", "path": path, "sourceIp": "127.0.0.1"},
                                         "domainName": "lambda-url.local", "stage": "$default"}})
        print("static", path, res["statusCode"], res["headers"].get("content-type"))
        if res["statusCode"] != 200:
            raise SystemExit(f"{path} -> {res['statusCode']}")
    created = http("POST", "/api/jobs", json.dumps({"files": [p.name for p in photos]}).encode())
    job_id = created["job"]["id"]
    for p in photos:
        http("PUT", f"/api/jobs/{job_id}/files/{p.name}", p.read_bytes(), "image/jpeg")
    t0 = time.time()
    print("worker", invoke({"lumen_job": job_id}), f"{time.time() - t0:.1f}s")
    job = http("GET", f"/api/jobs/{job_id}")
    print("job", job["status"], [(s["id"], s["status"]) for s in job["sets"]], f"{job['seconds']}s inside Lambda")
    if job["status"] != "done":
        raise SystemExit(1)


if __name__ == "__main__":
    main(sys.argv[1])
