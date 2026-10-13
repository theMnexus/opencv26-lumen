"""Read bracket photos and the EXIF fields the pipeline needs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# EXIF tag ids (CIPA DC-008)
_EXIF_IFD = 0x8769
_MAKE = 271
_MODEL = 272
_EXPOSURE_TIME = 33434
_DATETIME_ORIGINAL = 36867
_EXPOSURE_BIAS = 37380
_FOCAL_LENGTH = 37386
_SUBSEC_ORIGINAL = 37521

_EXIF_DATETIME_FORMAT = "%Y:%m:%d %H:%M:%S"


@dataclass(frozen=True)
class Exif:
    taken_at: datetime | None = None
    exposure_time_s: float | None = None
    exposure_bias_ev: float | None = None
    focal_length_mm: float | None = None
    camera: str | None = None


@dataclass(frozen=True)
class Frame:
    path: Path
    image: np.ndarray  # BGR uint8, EXIF orientation applied
    exif: Exif


def _to_float(value: object) -> float | None:
    """EXIF rationals can be 0/0; treat those and junk as missing."""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return number if math.isfinite(number) else None


def _parse_taken_at(raw: object, subsec: object) -> datetime | None:
    if not isinstance(raw, str):
        return None
    try:
        taken = datetime.strptime(raw.strip("\x00 "), _EXIF_DATETIME_FORMAT)
    except ValueError:
        return None
    digits = str(subsec).strip("\x00 ") if subsec is not None else ""
    if digits.isdigit():
        taken = taken.replace(microsecond=int(digits[:6].ljust(6, "0")))
    return taken


def _camera_name(make: object, model: object) -> str | None:
    parts = [str(p).strip("\x00 ") for p in (make, model) if p]
    name = " ".join(p for p in parts if p)
    return name or None


def read_exif(path: str | Path) -> Exif:
    """Read EXIF; missing or broken tags become None, never an exception."""
    with Image.open(path) as img:
        base = img.getexif()
        sub = base.get_ifd(_EXIF_IFD)
    return Exif(
        taken_at=_parse_taken_at(sub.get(_DATETIME_ORIGINAL), sub.get(_SUBSEC_ORIGINAL)),
        exposure_time_s=_to_float(sub.get(_EXPOSURE_TIME)),
        exposure_bias_ev=_to_float(sub.get(_EXPOSURE_BIAS)),
        focal_length_mm=_to_float(sub.get(_FOCAL_LENGTH)),
        camera=_camera_name(base.get(_MAKE), base.get(_MODEL)),
    )


_REDUCED = {1: cv2.IMREAD_COLOR, 2: cv2.IMREAD_REDUCED_COLOR_2, 4: cv2.IMREAD_REDUCED_COLOR_4,
            8: cv2.IMREAD_REDUCED_COLOR_8}


def load_image(path: str | Path, reduce: int = 1) -> np.ndarray:
    """BGR uint8. cv2.imread applies EXIF orientation for colour reads.

    reduce 2/4/8 decodes JPEGs at that fraction directly (DCT scaling): used for
    grouping a whole folder without holding every 24 MP frame in memory.
    """
    image = cv2.imread(str(path), _REDUCED[reduce])
    if image is None:
        raise ValueError(f"cannot decode image: {Path(path).name}")
    return image


def load_frame(path: str | Path, reduce: int = 1) -> Frame:
    p = Path(path)
    return Frame(path=p, image=load_image(p, reduce), exif=read_exif(p))
