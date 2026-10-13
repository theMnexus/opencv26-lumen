"""Merge aligned brackets into one image, with ghost detection and removal.

Engines:
  mertens  createMergeMertens: exposure fusion, needs no exposure times.
  debevec  createMergeDebevec with the sRGB response (JPEG input) +
           createTonemapDrago: radiance map, needs EXIF ExposureTime for every
           frame. Keeps window detail that Mertens washes out.
           createCalibrateDebevec was tried first: with 3 frames its per-channel
           curves disagree and walls turn purple/green (09/10), so the fixed sRGB
           curve is used instead.
Ghosts: each frame is mapped to the reference exposure with a per-channel
look-up table learned from well-exposed pixels; where the mapped frame still
differs from the reference (absdiff, threshold, morphology, connected
components), the frame takes the reference content instead. The scene then
shows the moving object once, where the reference frame saw it.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from lumen.qc import clip_ratio, mef_ssim

WELL_LOW, WELL_HIGH = 12, 243  # uint8 range trusted for exposure matching
GHOST_DIFF = 28  # levels, largest channel difference after exposure matching
# Extra levels allowed per unit of LUT slope above 1 (noise grows with the mapping).
# Measured 2026-10-09: k=28 (limit x slope) cut synthetic recall to 0.72; k=4/8/12 kept
# 0.84/0.83/0.82 with 0 % false ghost on real Empa brackets and a private interior test set (not published); k=8 also had
# 0 % false alarms on synthetic still sets.
GHOST_NOISE_PER_SLOPE = 8.0
DRAGO_SATURATION, DRAGO_BIAS = 1.0, 0.85
SRGB_GAMMA = 2.2
GHOST_MIN_AREA = 0.0005  # share of image; smaller blobs are noise
GHOST_FEATHER = 15  # px, Gaussian sigma for the replacement mask edge
GHOST_GROW = 25  # px dilation before feathering; 9 px left a faint outline (09/10)
OPEN_K, CLOSE_K = 5, 15


@dataclass(frozen=True)
class MergeResult:
    image: np.ndarray  # BGR uint8
    engine: str
    ghost_ratio: float  # share of image flagged as moving, before removal
    deghosted: bool
    clip_ratio: float
    mef_ssim: float
    note: str


# ---------- exposure matching ----------

QUANTILES = np.linspace(0.0, 1.0, 257)
MIN_MATCH_PIXELS = 1000


def exposure_lut(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """(3, 256) uint8 LUT per channel so that LUT[src] ~ dst (quantile matching).

    For each channel, the quantiles of src are mapped onto the quantiles of
    dst over pixels where that channel is well exposed in both images. This
    follows any monotonic camera tone curve, and a small moving object barely
    moves the quantiles. History (2026-10-09):
      - free 256-bin LUT: sparse bins filled by the moving object, up to 62 %
        false ghost on synthetic sets;
      - one sRGB-linear exposure ratio per channel: exact on synthetic gamma
        images, but 11.5 % false ghost on a real static tripod bracket
        (Empa FORTH3, EV -3..+3), where the tone curve is not a pure gamma;
      - quantile matching: 0 % on FORTH3, synthetic recall 0.86, false alarms
        at most 0.24 % of the image.
    """
    ok = (src > WELL_LOW) & (src < WELL_HIGH) & (dst > WELL_LOW) & (dst < WELL_HIGH)
    levels = np.arange(256, dtype=np.float64)
    luts = []
    for c in range(3):
        s, d = src[..., c][ok[..., c]], dst[..., c][ok[..., c]]
        if s.size < MIN_MATCH_PIXELS:
            luts.append(levels)
            continue
        src_q, keep = np.unique(np.quantile(s, QUANTILES), return_index=True)
        dst_q = np.quantile(d, QUANTILES)[keep]
        luts.append(np.maximum.accumulate(np.interp(levels, src_q, dst_q)))
    return np.clip(np.stack(luts), 0, 255).round().astype(np.uint8)


def apply_lut(image: np.ndarray, lut: np.ndarray) -> np.ndarray:
    return np.stack([cv2.LUT(image[..., c], lut[c]) for c in range(3)], axis=2)


# ---------- ghosts ----------

def ghost_mask(frame: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """uint8 mask (255 = moving) of where `frame` disagrees with `reference`."""
    lut = exposure_lut(frame, reference)
    mapped = apply_lut(frame, lut)
    # Per channel, compared only where that channel is well exposed in both
    # frames; then the largest channel difference. Measured 09/10:
    #   grey difference misses a red object on a grey wall (IoU 0.04-0.08);
    #   any-channel validity flags a half-clipped ceiling (4 % false ghost);
    #   all-channel validity misses an object whose red channel clips.
    # The limit grows with the LUT slope: mapping a frame 4 stops up turns one
    # level of its noise into many levels. A fixed limit flagged large parts of
    # static rooms as moving (the darkest frame of wide brackets in a private interior test set (not published)).
    ok = (frame > WELL_LOW) & (frame < WELL_HIGH) & (reference > WELL_LOW) & (reference < WELL_HIGH)
    slope = np.maximum(1.0, np.gradient(lut.astype(np.float32), axis=1))  # (3, 256)
    limit = GHOST_DIFF + GHOST_NOISE_PER_SLOPE * (np.stack([slope[c][frame[..., c]] for c in range(3)], axis=2) - 1)
    over = ok & (cv2.absdiff(mapped, reference).astype(np.float32) > limit)
    mask = over.any(axis=2).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((OPEN_K, OPEN_K), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((CLOSE_K, CLOSE_K), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    min_area = GHOST_MIN_AREA * mask.size
    keep = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    return np.isin(labels, keep).astype(np.uint8) * 255


def _replacement_alpha(mask: np.ndarray) -> np.ndarray:
    """1.0 over the moving object plus a margin, soft only outside that core.

    Blurring the core itself left ~22 % of the object at its edge (09/10).
    """
    disk = lambda size: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))  # noqa: E731
    core = cv2.dilate(mask, disk(GHOST_GROW)).astype(np.float32) / 255
    halo = cv2.GaussianBlur(cv2.dilate(mask, disk(GHOST_GROW + 4 * GHOST_FEATHER)).astype(np.float32) / 255,
                            (0, 0), GHOST_FEATHER)
    return np.maximum(core, halo)[..., None]


def remove_ghosts(frames: list[np.ndarray], reference_index: int) -> tuple[list[np.ndarray], float]:
    """Replace moving regions in every non-reference frame with reference content."""
    reference = frames[reference_index]
    union = np.zeros(reference.shape[:2], np.uint8)
    out = []
    for i, frame in enumerate(frames):
        if i == reference_index:
            out.append(frame)
            continue
        mask = ghost_mask(frame, reference)
        union |= mask
        if not mask.any():
            out.append(frame)
            continue
        alpha = _replacement_alpha(mask)
        from_ref = apply_lut(reference, exposure_lut(reference, frame)).astype(np.float32)
        out.append((alpha * from_ref + (1 - alpha) * frame.astype(np.float32)).round().astype(np.uint8))
    return out, float((union > 0).mean())


# ---------- engines ----------

def _to_uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(np.nan_to_num(image) * 255, 0, 255).round().astype(np.uint8)


def fuse_mertens(frames: list[np.ndarray]) -> np.ndarray:
    return _to_uint8(cv2.createMergeMertens().process(frames))


def _srgb_response() -> np.ndarray:
    levels = (np.arange(256, dtype=np.float32) + 0.5) / 256
    return np.repeat(levels ** SRGB_GAMMA, 3).reshape(256, 1, 3)


def fuse_debevec(frames: list[np.ndarray], exposure_times_s: list[float]) -> np.ndarray:
    times = np.asarray(exposure_times_s, np.float32)
    radiance = cv2.createMergeDebevec().process(frames, times, _srgb_response())
    tonemap = cv2.createTonemapDrago(SRGB_GAMMA, DRAGO_SATURATION, DRAGO_BIAS)
    return _to_uint8(tonemap.process(radiance))


def merge_set(frames: list[np.ndarray], reference_index: int, engine: str = "mertens",
              exposure_times_s: list[float | None] | None = None, deghost: bool = True) -> MergeResult:
    """frames: aligned BGR uint8 (output of align.align_set)."""
    notes = []
    if engine == "debevec" and (exposure_times_s is None or any(t is None for t in exposure_times_s)):
        engine = "mertens"
        notes.append("no ExposureTime for every frame: debevec -> mertens")
    inputs, ghost_ratio = remove_ghosts(frames, reference_index) if deghost else (frames, 0.0)
    if engine == "debevec":
        fused = fuse_debevec(inputs, exposure_times_s)  # type: ignore[arg-type]
    elif engine == "mertens":
        fused = fuse_mertens(inputs)
    else:
        raise ValueError(f"unknown merge engine: {engine}")
    return MergeResult(fused, engine, ghost_ratio, deghost, clip_ratio(fused), mef_ssim(frames, fused), "; ".join(notes))
