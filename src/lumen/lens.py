"""Self-calibrate radial lens distortion from lines that should be straight.

1. Canny + findContours give edge chains; chains are cut where they turn
   sharply (corners), long pieces are kept as "should be straight" candidates.
2. k1 is searched so the pieces are straightest after undistortPoints
   (plumb-line cost: RMS distance of every point to its piece's fitted line).
   Pieces already curved at k1 = 0 (plants, round objects) are dropped first.
   A trimmed mean was tried first (09/10): it dropped exactly the long border
   lines that carry the distortion signal and missed k1 on 6 of 16 sets.
3. getOptimalNewCameraMatrix + initUndistortRectifyMap + remap undo it.

The focal length in pixels is unknown from the image alone. k1 is reported
for the assumed focal (default: image width); a different guess gives a
different k1 but nearly the same corrected image.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

WORK_LONG_SIDE = 1000  # px, analysis size
CANNY_LOW, CANNY_HIGH = 40, 120
MIN_PIECE = 0.08  # piece length as share of the image diagonal
TURN_WINDOW = 12  # px along the chain used to measure direction
MAX_TURN_DEG = 25.0  # sharper turn = corner, cut the chain there
MAX_START_CURVE = 0.03  # RMS / length at k1 = 0; above = really curved, ignore
# Real Empa Museum1 (display cases, 6 pieces) produced k1 = +0.069 from glass
# edges; rooms with walls and frames give 86-88 pieces, synthetic rooms 26-66.
MIN_PIECES = 12
K1_RANGE = (-0.6, 0.3)
K1_COARSE_STEP = 0.01
# Measured 09/10 (seed 3-4, 16 sets): undistorted sets gain 0 %, distorted >= 11 %.
MIN_GAIN = 0.05


@dataclass(frozen=True)
class LensResult:
    image: np.ndarray
    k1: float  # for focal_px below; 0.0 = not corrected
    focal_px: float
    curvature_before_px: float  # plumb-line RMS distance to fitted lines, full-res px
    curvature_after_px: float
    pieces: int
    note: str


def _work_copy(image: np.ndarray) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    scale = min(1.0, WORK_LONG_SIDE / max(h, w))
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    if scale < 1.0:
        grey = cv2.resize(grey, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return grey, scale


def _split_on_turns(chain: np.ndarray) -> list[np.ndarray]:
    if len(chain) < 2 * TURN_WINDOW + 2:
        return [chain]
    ahead = chain[2 * TURN_WINDOW:] - chain[TURN_WINDOW:-TURN_WINDOW]
    behind = chain[TURN_WINDOW:-TURN_WINDOW] - chain[:-2 * TURN_WINDOW]
    a1, a2 = np.arctan2(ahead[:, 1], ahead[:, 0]), np.arctan2(behind[:, 1], behind[:, 0])
    turn = np.degrees(np.abs(np.angle(np.exp(1j * (a1 - a2)))))
    cuts = np.flatnonzero(turn > MAX_TURN_DEG) + TURN_WINDOW
    return [p for p in np.split(chain, cuts) if len(p)]


def line_pieces(grey: np.ndarray) -> list[np.ndarray]:
    """Long, nearly straight edge pieces as float64 (N, 2) point arrays."""
    edges = cv2.Canny(cv2.GaussianBlur(grey, (5, 5), 0), CANNY_LOW, CANNY_HIGH)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    min_len = MIN_PIECE * float(np.hypot(*grey.shape))
    pieces = []
    for contour in contours:
        for piece in _split_on_turns(contour.reshape(-1, 2).astype(np.float64)):
            span = float(np.linalg.norm(piece[-1] - piece[0]))
            if span >= min_len and _rms_to_line(piece) <= MAX_START_CURVE * span:
                pieces.append(piece)
    return pieces


def _sq_dist_to_line(points: np.ndarray) -> float:
    """Sum of squared distances to the best-fit line (smallest singular value squared)."""
    centred = points - points.mean(axis=0)
    return float(np.linalg.svd(centred, compute_uv=False)[-1] ** 2)


def _rms_to_line(points: np.ndarray) -> float:
    return float(np.sqrt(_sq_dist_to_line(points) / len(points)))


def _camera(shape: tuple[int, int], focal_px: float) -> np.ndarray:
    h, w = shape
    return np.array([[focal_px, 0, w / 2], [0, focal_px, h / 2], [0, 0, 1]], np.float64)


def straightness(pieces: list[np.ndarray], k1: float, camera: np.ndarray) -> float:
    """Plumb-line cost: RMS distance (px) of all piece points to their fitted lines after undistortion."""
    dist = np.array([k1, 0, 0, 0, 0], np.float64)
    total, count = 0.0, 0
    for piece in pieces:
        pts = piece.reshape(-1, 1, 2)
        if k1:
            pts = cv2.undistortPoints(pts, camera, dist, P=camera)
        total += _sq_dist_to_line(pts.reshape(-1, 2))
        count += len(piece)
    return float(np.sqrt(total / count))


def estimate_k1(pieces: list[np.ndarray], camera: np.ndarray) -> float:
    coarse = np.arange(K1_RANGE[0], K1_RANGE[1] + 1e-9, K1_COARSE_STEP)
    costs = [straightness(pieces, float(k), camera) for k in coarse]
    best = float(coarse[int(np.argmin(costs))])
    fine = np.linspace(best - K1_COARSE_STEP, best + K1_COARSE_STEP, 21)
    fine_costs = [straightness(pieces, float(k), camera) for k in fine]
    return float(fine[int(np.argmin(fine_costs))])


def undistort(image: np.ndarray, k1: float, focal_px: float) -> np.ndarray:
    h, w = image.shape[:2]
    camera = _camera((h, w), focal_px)
    dist = np.array([k1, 0, 0, 0, 0], np.float64)
    new_camera, _ = cv2.getOptimalNewCameraMatrix(camera, dist, (w, h), 0, (w, h))
    map_x, map_y = cv2.initUndistortRectifyMap(camera, dist, None, new_camera, (w, h), cv2.CV_32FC1)
    return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def correct_lens(image: np.ndarray, focal_px: float | None = None) -> LensResult:
    """focal_px: full-resolution focal length in pixels if known (EXIF + sensor); else image width."""
    h, w = image.shape[:2]
    focal = float(focal_px or w)
    grey, scale = _work_copy(image)
    pieces = line_pieces(grey)
    if len(pieces) < MIN_PIECES:
        return LensResult(image, 0.0, focal, float("nan"), float("nan"), len(pieces),
                          f"only {len(pieces)} straight-line pieces (< {MIN_PIECES}): not corrected")
    camera = _camera(grey.shape, focal * scale)
    before = straightness(pieces, 0.0, camera)
    k1 = estimate_k1(pieces, camera)
    if not K1_RANGE[0] + K1_COARSE_STEP < k1 < K1_RANGE[1] - K1_COARSE_STEP:
        # The best value sits on the search limit: no real minimum. Seen on a
        # wide-angle room in a private interior test set (not published), where a positive k1 made no physical sense.
        return LensResult(image, 0.0, focal, before / scale, before / scale, len(pieces),
                          f"best k1 {k1:+.3f} is at the search limit: not corrected")
    after = straightness(pieces, k1, camera)
    if after > before * (1 - MIN_GAIN):
        return LensResult(image, 0.0, focal, before / scale, before / scale, len(pieces),
                          f"best k1 {k1:+.3f} improves straightness < {MIN_GAIN:.0%}: not corrected")
    return LensResult(undistort(image, k1, focal), k1, focal, before / scale, after / scale, len(pieces), "")
