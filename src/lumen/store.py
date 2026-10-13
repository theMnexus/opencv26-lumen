"""Where job files live: a local folder (dev) or an S3 bucket (AWS).

Selected by LUMEN_STORAGE=local|s3. Keys look like jobs/<id>/in/<name>.
"""

from __future__ import annotations

import os
from pathlib import Path

MAX_UPLOAD_BYTES = 40 * 1024 * 1024
URL_TTL_S = 3600


class LocalStore:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()

    def _path(self, key: str) -> Path:
        path = (self.root / key).resolve()
        if self.root not in path.parents:
            raise ValueError(f"key escapes store root: {key}")
        return path

    def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)  # readers never see a half-written job.json

    def get(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def upload_target(self, job_id: str, name: str) -> dict:
        return {"method": "PUT", "url": f"/api/jobs/{job_id}/files/{name}"}

    def download_url(self, key: str) -> str | None:
        return None  # served by the API itself


class S3Store:
    def __init__(self, bucket: str):
        import boto3  # only needed on AWS
        from botocore.config import Config

        self.bucket = bucket
        # SigV4 explicitly: the default can still presign with SigV2, which newer regions reject.
        # Regional endpoint: presigned URLs on the global host get a 307 for buckets outside
        # us-east-1 until DNS catches up, and a browser upload does not follow it.
        region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
        self.s3 = boto3.client("s3", region_name=region,
                               endpoint_url=f"https://s3.{region}.amazonaws.com" if region else None,
                               config=Config(signature_version="s3v4", s3={"addressing_style": "virtual"}))

    def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=data, ContentType=content_type)

    def get(self, key: str) -> bytes:
        return self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self.s3.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError:
            return False

    def upload_target(self, job_id: str, name: str) -> dict:
        post = self.s3.generate_presigned_post(
            self.bucket, f"jobs/{job_id}/in/{name}",
            Conditions=[["content-length-range", 1, MAX_UPLOAD_BYTES]], ExpiresIn=URL_TTL_S)
        return {"method": "POST", "url": post["url"], "fields": post["fields"]}

    def download_url(self, key: str) -> str | None:
        return self.s3.generate_presigned_url("get_object", Params={"Bucket": self.bucket, "Key": key},
                                              ExpiresIn=URL_TTL_S)


def store_from_env() -> LocalStore | S3Store:
    kind = os.environ.get("LUMEN_STORAGE", "local")
    if kind == "s3":
        return S3Store(os.environ["LUMEN_BUCKET"])
    if kind == "local":
        return LocalStore(os.environ.get("LUMEN_DATA_DIR", "out/jobs-store"))
    raise ValueError(f"LUMEN_STORAGE must be local or s3, got {kind!r}")
