from datetime import datetime
from fractions import Fraction

import numpy as np
import pytest
from PIL import Image
from PIL.TiffImagePlugin import IFDRational

from lumen.io import load_frame, load_image, read_exif


def _write_jpeg(path, exif_ifd=None, make=None, model=None):
    pixels = np.full((40, 60, 3), 128, np.uint8)
    img = Image.fromarray(pixels)
    exif = Image.Exif()
    if make:
        exif[271] = make
    if model:
        exif[272] = model
    if exif_ifd:
        sub = exif.get_ifd(0x8769)
        sub.update(exif_ifd)
    img.save(path, "JPEG", exif=exif)
    return path


def test_reads_all_fields(tmp_path):
    path = _write_jpeg(
        tmp_path / "a.jpg",
        exif_ifd={
            36867: "2026:10:09 10:00:05",
            37521: "25",
            33434: IFDRational(Fraction(1, 60)),
            37380: IFDRational(Fraction(-2, 1)),
            37386: IFDRational(Fraction(24, 1)),
        },
        make="Sony",
        model="A7 IV",
    )
    exif = read_exif(path)
    assert exif.taken_at == datetime(2026, 10, 9, 10, 0, 5, 250000)
    assert exif.exposure_time_s == pytest.approx(1 / 60)
    assert exif.exposure_bias_ev == pytest.approx(-2.0)
    assert exif.focal_length_mm == pytest.approx(24.0)
    assert exif.camera == "Sony A7 IV"


def test_missing_exif_is_none(tmp_path):
    exif = read_exif(_write_jpeg(tmp_path / "b.jpg"))
    assert exif.taken_at is None
    assert exif.exposure_time_s is None
    assert exif.exposure_bias_ev is None
    assert exif.focal_length_mm is None
    assert exif.camera is None


def test_broken_date_is_none(tmp_path):
    path = _write_jpeg(tmp_path / "c.jpg", exif_ifd={36867: "not a date"})
    assert read_exif(path).taken_at is None


def test_load_frame_returns_bgr(tmp_path):
    frame = load_frame(_write_jpeg(tmp_path / "d.jpg"))
    assert frame.image.shape == (40, 60, 3)
    assert frame.image.dtype == np.uint8


def test_undecodable_file_raises(tmp_path):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    with pytest.raises(ValueError):
        load_image(bad)
