"""Fetch real bracket scenes from the Empa HDR Image Database (CC BY 4.0) and convert them to JPEG.

    uv run python bench/fetch_empa.py FORTH3 FORTH4 Museum1 FORTH1

Each scene: download the zip from Zenodo (resumable), check its MD5 against the
record, extract the RAW frames (macOS "._" files skipped), convert to JPEG with
EXIF (bench/raw_to_jpeg.py). Output in data/empa/ (gitignored) plus
data/empa/SOURCES.md with the required attribution.

Source: P. Zolliker, I. Sprow, Empa HDR Image Database, Zenodo,
doi:10.5281/zenodo.7861544, licensed CC BY 4.0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "empa"
RECORD = "https://zenodo.org/api/records/7861544"
FILE_URL = "https://zenodo.org/records/7861544/files/{key}?download=1"
ATTRIBUTION = ("Empa HDR Image Database, P. Zolliker & I. Sprow, Empa, CC BY 4.0, "
               "doi:10.5281/zenodo.7861544 (https://zenodo.org/records/7861544)")
TRIES = 6


def _record() -> dict:
    """Zenodo's API answers 504 under load: retry with a growing pause."""
    for attempt in range(TRIES):
        try:
            with urllib.request.urlopen(RECORD, timeout=60) as res:
                return json.load(res)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            print(f"zenodo record: {exc}; retry {attempt + 1}/{TRIES}", flush=True)
            time.sleep(5 * (attempt + 1))
    raise SystemExit("Zenodo record unavailable, try again later")


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(key: str, size: int, md5: str) -> Path:
    path = DATA / "zips" / key
    path.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(TRIES):
        if path.exists() and path.stat().st_size == size:
            break
        subprocess.run(["curl", "-fsSL", "-C", "-", "--retry", "3", "-o", str(path), FILE_URL.format(key=key)])
    if not path.exists() or path.stat().st_size != size:
        raise SystemExit(f"{key}: download incomplete after {TRIES} tries")
    if _md5(path) != md5:
        raise SystemExit(f"{key}: MD5 mismatch, delete {path} and retry")
    return path


def extract(zip_path: Path, scene: str) -> Path:
    out = DATA / "raw" / scene
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            name = Path(member).name
            if name.startswith("._") or member.endswith("/") or name != member.split("/")[-1]:
                continue
            target = out / name  # flat: never trust paths inside the archive
            if not target.exists():
                target.write_bytes(zf.read(member))
    return out


def write_sources() -> None:
    """List every converted scene, not only this run's (runs can be split when Zenodo times out)."""
    scenes = sorted(d.name for d in (DATA / "jpg").iterdir() if d.is_dir())
    lines = ["# Data sources", "", f"Scenes: {', '.join(scenes)}", "", f"- {ATTRIBUTION}",
             "- Downloaded and converted by `bench/fetch_empa.py`; RAW decoded with rawpy (LibRaw).", ""]
    (DATA / "SOURCES.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("scenes", nargs="+")
    args = parser.parse_args()
    files = {f["key"]: f for f in _record()["files"]}
    failed = []
    for scene in args.scenes:
        meta = files.get(f"{scene}.zip")
        if meta is None:
            raise SystemExit(f"unknown scene {scene}")
        try:
            zip_path = download(meta["key"], meta["size"], meta["checksum"].removeprefix("md5:"))
        except SystemExit as exc:  # keep going: one slow file should not cost the others
            print(exc, flush=True)
            failed.append(scene)
            continue
        raw_dir = extract(zip_path, scene)
        subprocess.run([sys.executable, str(ROOT / "bench" / "raw_to_jpeg.py"), str(raw_dir),
                        str(DATA / "jpg" / scene)], check=True, stdout=subprocess.DEVNULL)
        print(f"{scene}: ok -> {DATA / 'jpg' / scene}", flush=True)
    write_sources()
    if failed:
        raise SystemExit(f"not fetched: {', '.join(failed)}; run again later")


if __name__ == "__main__":
    main()
