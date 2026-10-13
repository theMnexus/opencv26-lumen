"""Align every frame of a bracket set to the reference (middle) frame.

Per frame:
  1. AlignMTB: exposure-invariant integer shift (coarse start).
  2. findTransformECCMultiScale (OpenCV 5): homography on gradient maps,
     pyramid levels, masks that drop clipped pixels in both images.
  3. If ECC throws or the tile residual stays > MAX_RESIDUAL_PX:
     SIFT + ratio test + findHomography(RANSAC); keep the better result.
Estimation runs on a ~1.5 MP copy; the homography is scaled to full size.
Homography convention matches synth: reference pixel -> frame pixel.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from lumen.qc import grey_gradient, tile_residual_px

ESTIMATION_PIXELS = 1.5e6
# Pixels of the ~1.5 MP working copy. The full-resolution limit is this / scale:
# a 24 MP frame measured at scale 0.25 turns 0.5 px of tile noise into 2 px,
# and a fixed 2 px full-res limit dropped good frames (09/10).
MAX_RESIDUAL_PX = 2.0
CLIP_LOW, CLIP_HIGH = 6, 249  # grey levels treated as clipped
ECC_LEVELS = 4
ECC_ITERS = 100
ECC_EPS = 1e-6
# Measured 09/10 (seed 5-6, clean + phone profiles, gradient pre-blur 2 px):
# same room cc >= 0.921, different rooms <= 0.652. The first value, 0.7, was
# set on clean images only and rejected good noisy phone frames (cc 0.645).
MIN_ECC_CC = 0.8
SIFT_RATIO = 0.75
SIFT_RANSAC_PX = 3.0
SIFT_MIN_INLIERS = 30


@dataclass(frozen=True)
class FrameAlignment:
    index: int
    homography: np.ndarray  # 3x3, full-res, reference pixel -> frame pixel
    method: str  # "reference" | "ecc" | "sift" | "mtb"
    residual_px: float  # full-res pixels, from qc.tile_residual_px
    ecc_cc: float | None
    note: str


@dataclass(frozen=True)
class SetAlignment:
    reference_index: int
    frames: list[FrameAlignment]
    aligned: list[np.ndarray]  # BGR uint8 frames warped onto the reference
    residual_limit_px: float  # full-res pixels; above this a frame is badly aligned

    @property
    def worst_residual_px(self) -> float:
        return max(f.residual_px for f in self.frames)


def _scale_for(image: np.ndarray) -> float:
    h, w = image.shape[:2]
    return min(1.0, float(np.sqrt(ESTIMATION_PIXELS / (h * w))))


def _small_grey(image: np.ndarray, scale: float) -> np.ndarray:
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if scale == 1.0:
        return grey
    return cv2.resize(grey, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def _valid_mask(grey: np.ndarray) -> np.ndarray:
    mask = cv2.inRange(grey, CLIP_LOW, CLIP_HIGH)
    return cv2.erode(mask, np.ones((5, 5), np.uint8))


def _rescale(h_small: np.ndarray, scale: float) -> np.ndarray:
    s = np.diag([scale, scale, 1.0])
    out = np.linalg.inv(s) @ h_small @ s
    return out / out[2, 2]


def _warp_to_reference(image: np.ndarray, homography: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return cv2.warpPerspective(image, homography, size, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_REPLICATE)


def _mtb_start(ref: np.ndarray, img: np.ndarray) -> np.ndarray:
    dx, dy = cv2.createAlignMTB().calculateShift(ref, img)
    # calculateShift returns the shift to apply to img; the warp goes the other way
    return np.array([[1, 0, -dx], [0, 1, -dy], [0, 0, 1]], np.float32)


def _ecc(ref: np.ndarray, img: np.ndarray, start: np.ndarray) -> tuple[np.ndarray, float]:
    params = cv2.ECCParameters()
    params.motionType = cv2.MOTION_HOMOGRAPHY
    params.nlevels = ECC_LEVELS
    params.criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, ECC_ITERS, ECC_EPS)
    cc, warp = cv2.findTransformECCMultiScale(
        grey_gradient(ref), grey_gradient(img), start.copy(), params, _valid_mask(ref), _valid_mask(img))
    return warp.astype(np.float64), float(cc)


def _sift(ref: np.ndarray, img: np.ndarray) -> np.ndarray | None:
    sift = cv2.SIFT_create()
    kp_r, des_r = sift.detectAndCompute(cv2.equalizeHist(ref), _valid_mask(ref))
    kp_i, des_i = sift.detectAndCompute(cv2.equalizeHist(img), _valid_mask(img))
    if des_r is None or des_i is None or len(kp_r) < SIFT_MIN_INLIERS or len(kp_i) < SIFT_MIN_INLIERS:
        return None
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des_r, des_i, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < SIFT_RATIO * n.distance]
    if len(good) < SIFT_MIN_INLIERS:
        return None
    src = np.float32([kp_r[m.queryIdx].pt for m in good])
    dst = np.float32([kp_i[m.trainIdx].pt for m in good])
    homography, inliers = cv2.findHomography(src, dst, cv2.RANSAC, SIFT_RANSAC_PX)
    if homography is None or int(inliers.sum()) < SIFT_MIN_INLIERS:
        return None
    return homography


def _residual_full_px(ref_small: np.ndarray, img_small: np.ndarray, h_small: np.ndarray, scale: float) -> float:
    size = ref_small.shape[::-1]
    aligned = _warp_to_reference(img_small, h_small, size)
    valid = _warp_to_reference(np.full_like(img_small, 255), h_small, size)
    valid = cv2.bitwise_and(valid, _valid_mask(ref_small))
    return tile_residual_px(ref_small, aligned, valid) / scale


def align_pair(ref_small: np.ndarray, img_small: np.ndarray, scale: float, index: int) -> FrameAlignment:
    """ECC is trusted when it converges under the residual limit; otherwise the
    lowest-residual candidate among MTB, ECC and SIFT wins."""
    start = _mtb_start(ref_small, img_small)

    def residual(h: np.ndarray) -> float:
        return _residual_full_px(ref_small, img_small, h, scale)

    def done(method: str, h: np.ndarray, res: float, cc: float | None, note: str) -> FrameAlignment:
        return FrameAlignment(index, _rescale(h, scale), method, res, cc, note)

    limit = MAX_RESIDUAL_PX / scale

    candidates = [("mtb", start.astype(np.float64), residual(start.astype(np.float64)), None)]
    note = ""
    try:
        warp, cc = _ecc(ref_small, img_small, start)
        if cc < MIN_ECC_CC:
            candidates.append(("ecc", warp, float("inf"), cc))
            note = f"ecc cc {cc:.2f} < {MIN_ECC_CC}: not the same scene"
        else:
            ecc_res = residual(warp)
            if ecc_res <= limit:
                return done("ecc", warp, ecc_res, cc, "")
            candidates.append(("ecc", warp, ecc_res, cc))
            note = f"ecc residual {ecc_res:.2f}px > {limit:.2f}px"
    except cv2.error as exc:
        note = f"ecc failed: {str(exc).strip().splitlines()[-1][:80]}"
    h_sift = _sift(ref_small, img_small)
    if h_sift is not None:
        candidates.append(("sift", h_sift, residual(h_sift), None))
    else:
        note += "; sift found too few matches"
    method, h, res, cc = min(candidates, key=lambda c: c[2])
    return done(method, h, res, cc, f"{note}; picked {method}")


def align_set(images: list[np.ndarray], reference_index: int | None = None) -> SetAlignment:
    """images: BGR uint8 frames of one bracket, any order; reference defaults to the middle."""
    ref_i = len(images) // 2 if reference_index is None else reference_index
    scale = _scale_for(images[ref_i])
    smalls = [_small_grey(img, scale) for img in images]
    h, w = images[ref_i].shape[:2]
    frames, aligned = [], []
    for i, img in enumerate(images):
        if i == ref_i:
            fa = FrameAlignment(i, np.eye(3), "reference", 0.0, None, "")
        else:
            fa = align_pair(smalls[ref_i], smalls[i], scale, i)
        frames.append(fa)
        aligned.append(img if i == ref_i else _warp_to_reference(img, fa.homography, (w, h)))
    return SetAlignment(ref_i, frames, aligned, MAX_RESIDUAL_PX / scale)
