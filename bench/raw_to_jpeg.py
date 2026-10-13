"""Convert camera RAW brackets to JPEG with the EXIF the pipeline needs.

    uv run python bench/raw_to_jpeg.py data/empa/raw/FORTH3 data/empa/jpg/FORTH3

Decode: rawpy (LibRaw) with the camera white balance and no auto-brightening,
so the exposure steps between frames stay as shot. EXIF: DateTimeOriginal,
SubSecTimeOriginal, ExposureTime, ExposureBiasValue, FocalLength, Make, Model
are read with exifread and written into the JPEG. Dev tool only: the Lambda
image takes JPEG and does not ship rawpy.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
from pathlib import Path

import exifread
import rawpy
from PIL import Image
from PIL.TiffImagePlugin import IFDRational

RAW_SUFFIXES = {".cr2", ".cr3", ".nef", ".arw", ".dng", ".raf", ".orf", ".rw2"}
JPEG_QUALITY = 95


def _ratio(tag) -> IFDRational | None:
    if tag is None:
        return None
    value = tag.values[0]
    return IFDRational(Fraction(value.num, value.den) if hasattr(value, "num") else Fraction(str(value)))


def _exif(path: Path) -> Image.Exif:
    with path.open("rb") as fh:
        tags = exifread.process_file(fh, details=False)
    exif = Image.Exif()
    for tag_id, key in ((271, "Image Make"), (272, "Image Model")):
        if key in tags:
            exif[tag_id] = str(tags[key]).strip()
    sub = exif.get_ifd(0x8769)
    if "EXIF DateTimeOriginal" in tags:
        sub[36867] = str(tags["EXIF DateTimeOriginal"])
    if "EXIF SubSecTimeOriginal" in tags:
        sub[37521] = str(tags["EXIF SubSecTimeOriginal"])
    for tag_id, key in ((33434, "EXIF ExposureTime"), (37380, "EXIF ExposureBiasValue"), (37386, "EXIF FocalLength")):
        value = _ratio(tags.get(key))
        if value is not None:
            sub[tag_id] = value
    return exif


def convert(src: Path, dst: Path) -> None:
    with rawpy.imread(str(src)) as raw:
        rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=8)
    Image.fromarray(rgb).save(dst, "JPEG", quality=JPEG_QUALITY, exif=_exif(src))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("src_dir", type=Path)
    parser.add_argument("dst_dir", type=Path)
    args = parser.parse_args()
    args.dst_dir.mkdir(parents=True, exist_ok=True)
    raws = sorted(p for p in args.src_dir.iterdir() if p.suffix.lower() in RAW_SUFFIXES)
    if not raws:
        raise SystemExit(f"no RAW files in {args.src_dir}")
    for src in raws:
        dst = args.dst_dir / f"{src.stem}.jpg"
        convert(src, dst)
        print(f"{src.name} -> {dst}")


if __name__ == "__main__":
    main()
