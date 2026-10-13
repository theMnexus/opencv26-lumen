"""Build the public sample pack: real CC BY brackets, resized, EXIF kept, with attribution.

    uv run python bench/fetch_empa.py FORTH3 FORTH4 Museum1
    uv run python bench/fetch_cambridge.py
    uv run python bench/make_samples.py          # -> out/lumen-samples/ and out/lumen-samples.zip

Frames are resized to a 2400 px long side (about 3.8 MP) so a judge can upload a
folder to the demo in seconds; EXIF (time, exposure, bias) is copied unchanged.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "lumen-samples"
LONG_SIDE = 2400
EMPA = ["FORTH3", "FORTH4", "Museum1"]
# One or two per motion type, plus a multiview scene the agent should refuse to merge.
CAMBRIDGE = ["complex-1", "complex-4", "lolm-1", "lolm-4", "occlusion-1", "nrm-1", "handheld-3", "multiview-1"]
README = """# theMnexus Lumen — sample brackets

Real exposure brackets for trying the demo and re-running the benchmarks. Upload one
folder (or several at once: Lumen groups them) to the live demo or run them locally.

| Folder | What it tests |
|---|---|
| `empa/FORTH3`, `empa/FORTH4` | rooms with a bright window, tilted camera (7 exposures, EV −3…+3) |
| `empa/Museum1` | dark room, exposures up to 8 s |
| `cambridge/<scene>/ghosted` | people or objects moving between exposures (5 exposures) |
| `cambridge/<scene>/ground_truth` | the same bracket without motion: the answer for ghost removal |
| `cambridge/multiview-1` | camera moved between shots: the agent should ask a person |

The Cambridge brackets were staged (objects moved by hand, minutes between exposures), so the
time rule splits them: tick **These photos are one bracket** before uploading one `ghosted/` folder.

## Sources and licences (both CC BY 4.0)

- Empa HDR Image Database, P. Zolliker & I. Sprow, Empa, CC BY 4.0, doi:10.5281/zenodo.7861544
  (https://zenodo.org/records/7861544). RAW converted to JPEG with rawpy (camera white balance).
- Multi-exposure image stacks for testing HDR deghosting methods, K. Karaduzovic-Hadziabdic,
  J. Hasic Telalovic, R. K. Mantiuk, University of Cambridge, CC BY 4.0, doi:10.17863/CAM.6881
  (https://www.repository.cam.ac.uk/handle/1810/261766). Paper: Computers & Graphics 63 (2017) 1–17.
  Camera JPEGs used.

Changes made by theMnexus: images resized to a {long_side} px long side and re-encoded as JPEG
(quality 92); EXIF copied unchanged. Licence: https://creativecommons.org/licenses/by/4.0/
"""


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        exif = im.info.get("exif", b"")
        im = im.convert("RGB")
        scale = LONG_SIDE / max(im.size)
        if scale < 1:
            im = im.resize((round(im.width * scale), round(im.height * scale)), Image.Resampling.LANCZOS)
        im.save(dst, "JPEG", quality=92, exif=exif)


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)  # out/ is generated and gitignored
    count = 0
    for scene in EMPA:
        for src in sorted((ROOT / "data" / "empa" / "jpg" / scene).glob("*.jpg")):
            _copy(src, OUT / "empa" / scene / src.name)
            count += 1
    for scene in CAMBRIDGE:
        for kind in ("ghosted", "ground_truth"):
            for src in sorted((ROOT / "data" / "cambridge" / "jpg" / scene / kind).glob("*.jpg")):
                _copy(src, OUT / "cambridge" / scene / kind / src.name)
                count += 1
    (OUT / "README.md").write_text(README.format(long_side=LONG_SIDE))
    archive = shutil.make_archive(str(OUT), "zip", OUT.parent, OUT.name)
    print(f"{count} images -> {OUT} and {archive} ({Path(archive).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
