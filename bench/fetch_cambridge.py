"""Fetch the Cambridge HDR deghosting dataset (CC BY 4.0) and keep only the camera JPEGs.

    uv run python bench/fetch_cambridge.py

Downloads exposure_stacks_part1..3.zip (about 3.9 GB, resumable, MD5 checked
against the repository record) into data/cambridge/zips/, then extracts the
JPEG stacks to data/cambridge/jpg/<category>-<n>/{ghosted,ground_truth}/
(RAW files skipped) and writes data/cambridge/SOURCES.md. data/ is gitignored.

Source: K. Karaduzovic-Hadziabdic, J. Hasic Telalovic, R. K. Mantiuk,
"Multi-exposure image stacks for testing HDR deghosting methods",
University of Cambridge, doi:10.17863/CAM.6881, licensed CC BY 4.0.
Paper: Computers & Graphics 63 (2017) 1-17, doi:10.1016/j.cag.2017.01.002.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "cambridge"
URL = "https://api.repository.cam.ac.uk/server/api/core/bitstreams/{uuid}/content"
FILES = {  # name: (bitstream uuid, md5) from the repository record
    "exposure_stacks_part1.zip": ("e52f491b-ea00-4102-ab15-1aa0f088b737", "2788bb6dd6ff0a857d86240c2ac8726d"),
    "exposure_stacks_part2.zip": ("62522fb8-356c-440d-bac5-10780f75c8ba", "14527d8b3882dc6ffaac868d23aaf9ac"),
    "exposure_stacks_part3.zip": ("4391f37e-5005-44b6-8c99-c9adb0c3b945", "21a534f26bcc3673b1dc38a615816a28"),
}
ATTRIBUTION = ("Multi-exposure image stacks for testing HDR deghosting methods, K. Karaduzovic-Hadziabdic, "
               "J. Hasic Telalovic, R. K. Mantiuk, University of Cambridge, CC BY 4.0, doi:10.17863/CAM.6881 "
               "(https://www.repository.cam.ac.uk/handle/1810/261766)")
KINDS = ("ghosted", "ground_truth")
SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
TRIES = 5


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(name: str) -> Path:
    uuid, md5 = FILES[name]
    path = DATA / "zips" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(TRIES):
        if path.exists() and _md5(path) == md5:
            return path
        subprocess.run(["curl", "-fsSL", "-C", "-", "--retry", "5", "-A", "Mozilla/5.0", "-o", str(path),
                        URL.format(uuid=uuid)])
    raise SystemExit(f"{name}: download incomplete or MD5 mismatch after {TRIES} tries; delete it and retry")


def scene_path(member: str) -> tuple[str, str, str] | None:
    """'exposure_stacks/complex/image_set1/ghosted/jpg/x.jpg' -> ('complex-1', 'ghosted', 'x.jpg')."""
    parts = member.split("/")
    if len(parts) < 5 or parts[-2].lower() != "jpg" or not parts[-1].lower().endswith(".jpg"):
        return None
    kind, set_dir, category, name = parts[-3], parts[-4], parts[-5], parts[-1]
    number = re.sub(r"\D", "", set_dir)
    if kind not in KINDS or not number or name.startswith("._") or not all(SAFE.match(x) for x in (category, name)):
        return None
    return f"{category}-{number}", kind, name


def extract(zip_path: Path) -> int:
    count = 0
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            found = scene_path(member)
            if found is None:
                continue
            scene, kind, name = found
            target = DATA / "jpg" / scene / kind / name  # built from checked names only
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_bytes(zf.read(member))
            count += 1
    return count


def main() -> None:
    for name in FILES:
        zip_path = download(name)
        print(f"{name}: MD5 ok, {extract(zip_path)} JPEGs", flush=True)
    scenes = sorted(d.name for d in (DATA / "jpg").iterdir() if d.is_dir())
    (DATA / "SOURCES.md").write_text("\n".join([
        "# Data sources", "", f"Scenes ({len(scenes)}): {', '.join(scenes)}", "", f"- {ATTRIBUTION}",
        "- Paper: Karaduzovic-Hadziabdic, Hasic Telalovic, Mantiuk, Assessment of multi-exposure HDR image "
        "deghosting methods, Computers & Graphics 63 (2017) 1-17, doi:10.1016/j.cag.2017.01.002",
        "- Camera JPEGs extracted unchanged by `bench/fetch_cambridge.py` (RAW files not used).", ""]))
    print(f"{len(scenes)} scenes -> {DATA / 'jpg'}")


if __name__ == "__main__":
    main()
