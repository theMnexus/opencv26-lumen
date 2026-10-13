"""Make vertical lines vertical (camera roll and keystone from pitch).

1. createLineSegmentDetector finds segments; near-vertical ones are kept.
2. Vertical vanishing point = smallest eigenvector of sum(w * l l^T) over the
   segment lines l (homogeneous, normalised coordinates), refit once without
   segments that miss it by more than MAX_VP_ERR_DEG.
3. Homography: rotate so the vanishing point is straight below/above the
   centre, then send it to infinity, so verticals become parallel and upright.
4. Zoom about the centre until no empty border is left (pointPolygonTest).
Run after lens correction: barrel distortion bends the verticals.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

WORK_LONG_SIDE = 1000  # px, analysis size
MIN_SEGMENT = 0.04  # share of the image diagonal
MAX_FROM_VERTICAL_DEG = 20.0
MIN_VERTICALS = 4
MAX_VP_ERR_DEG = 1.5
MIN_CORRECTION_DEG = 0.2  # smaller tilt is left alone
MAX_KEYSTONE = 0.35  # |w / y| of the vanishing point (normalised); stronger = ask a person
MAX_ZOOM = 1.35  # crop limit; beyond it the correction eats the photo


@dataclass(frozen=True)
class TiltResult:
    image: np.ndarray
    roll_deg: float  # estimated roll present in the input, getRotationMatrix2D sign
    keystone: float  # |w / y| of the vertical vanishing point; 0 = verticals parallel
    tilt_before_deg: float  # length-weighted mean |angle from vertical| of vertical segments
    tilt_after_deg: float
    zoom: float
    verticals: int
    note: str


def _work_grey(image: np.ndarray) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    scale = min(1.0, WORK_LONG_SIDE / max(h, w))
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    if scale < 1.0:
        grey = cv2.resize(grey, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return grey, scale


def vertical_segments(grey: np.ndarray) -> np.ndarray:
    """(N, 4) x1 y1 x2 y2 of long, near-vertical LSD segments."""
    lines = cv2.createLineSegmentDetector().detect(grey)[0]
    if lines is None:
        return np.zeros((0, 4))
    seg = lines.reshape(-1, 4).astype(np.float64)
    d = seg[:, 2:] - seg[:, :2]
    length = np.hypot(d[:, 0], d[:, 1])
    from_vertical = np.degrees(np.arctan(np.abs(d[:, 0]) / np.maximum(np.abs(d[:, 1]), 1e-9)))
    keep = (length >= MIN_SEGMENT * np.hypot(*grey.shape)) & (from_vertical <= MAX_FROM_VERTICAL_DEG)
    return seg[keep]


def tilt_deg(segments: np.ndarray) -> float:
    """Length-weighted median |angle from vertical| of near-vertical segments.

    Median, not mean: on the real Empa FORTH3 room every window frame and wall
    corner was within 0.5 deg after straightening, but a chair and a fire
    extinguisher (not vertical in the world) pulled the mean to 1.7 deg.
    """
    if len(segments) == 0:
        return float("nan")
    d = segments[:, 2:] - segments[:, :2]
    length = np.hypot(d[:, 0], d[:, 1])
    angle = np.degrees(np.arctan(np.abs(d[:, 0]) / np.maximum(np.abs(d[:, 1]), 1e-9)))
    order = np.argsort(angle)
    cumulative = np.cumsum(length[order])
    return float(angle[order][np.searchsorted(cumulative, cumulative[-1] / 2)])


def _lines(segments: np.ndarray, centre: np.ndarray, norm: float) -> tuple[np.ndarray, np.ndarray]:
    p1 = np.c_[(segments[:, :2] - centre) / norm, np.ones(len(segments))]
    p2 = np.c_[(segments[:, 2:] - centre) / norm, np.ones(len(segments))]
    lines = np.cross(p1, p2)
    lines /= np.linalg.norm(lines[:, :2], axis=1, keepdims=True)
    return lines, np.linalg.norm(p2[:, :2] - p1[:, :2], axis=1)


def _vp_error_deg(segments: np.ndarray, vp: np.ndarray, centre: np.ndarray, norm: float) -> np.ndarray:
    mid = (segments[:, :2] + segments[:, 2:]) / 2
    direction = segments[:, 2:] - segments[:, :2]
    mid_n = np.c_[(mid - centre) / norm, np.ones(len(mid))]
    to_vp = vp[:2] - mid_n[:, :2] * vp[2]  # works for finite and infinite vp
    cos = np.abs((direction * to_vp).sum(1)) / (np.linalg.norm(direction, axis=1) * np.linalg.norm(to_vp, axis=1) + 1e-12)
    return np.degrees(np.arccos(np.clip(cos, 0, 1)))


def vanishing_point(segments: np.ndarray, centre: np.ndarray, norm: float) -> np.ndarray:
    """Unit 3-vector in centred, normalised coordinates, sign chosen so y >= 0."""
    keep = np.ones(len(segments), bool)
    vp = np.array([0.0, 1.0, 0.0])
    for _ in range(2):
        lines, weight = _lines(segments[keep], centre, norm)
        moment = (lines * weight[:, None]).T @ lines
        vp = np.linalg.eigh(moment)[1][:, 0]
        keep = _vp_error_deg(segments, vp, centre, norm) <= MAX_VP_ERR_DEG
        if keep.sum() < MIN_VERTICALS:
            break
    return vp if vp[1] >= 0 else -vp


def _upright_homography(vp: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Centred, normalised homography that makes verticals upright; also roll (deg) and keystone."""
    theta = np.arctan2(vp[0], vp[1])
    c, s = np.cos(theta), np.sin(theta)
    rotate = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    v = rotate @ vp
    keystone = -v[2] / v[1]
    unkey = np.array([[1, 0, 0], [0, 1, 0], [0, keystone, 1]])
    # theta rotates the image clockwise on screen (y down); getRotationMatrix2D(+) is counter-clockwise
    return unkey @ rotate, float(np.degrees(theta)), float(abs(keystone))


def _fit_zoom(homography: np.ndarray, w: int, h: int) -> float:
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64).reshape(-1, 1, 2)
    polygon = cv2.perspectiveTransform(corners, homography).astype(np.float32)
    centre = np.array([w / 2, h / 2])
    lo, hi = 0.5, 3.0
    for _ in range(30):
        zoom = (lo + hi) / 2
        scaled = ((polygon.reshape(-1, 2) - centre) * zoom + centre).astype(np.float32)
        inside = all(cv2.pointPolygonTest(scaled, (float(x), float(y)), False) >= 0
                     for x, y in corners.reshape(-1, 2))
        lo, hi = (lo, zoom) if inside else (zoom, hi)
    return hi


def _pixel_homography(h_norm: np.ndarray, w: int, h: int) -> np.ndarray:
    norm = float(max(w, h))
    to_norm = np.array([[1 / norm, 0, -w / 2 / norm], [0, 1 / norm, -h / 2 / norm], [0, 0, 1]])
    return np.linalg.inv(to_norm) @ h_norm @ to_norm


def straighten(image: np.ndarray) -> TiltResult:
    h, w = image.shape[:2]
    grey, scale = _work_grey(image)
    segments = vertical_segments(grey)
    before = tilt_deg(segments)
    if len(segments) < MIN_VERTICALS:
        return TiltResult(image, 0.0, 0.0, before, before, 1.0, len(segments),
                          f"only {len(segments)} vertical segments (< {MIN_VERTICALS}): not corrected")
    gh, gw = grey.shape
    vp = vanishing_point(segments, np.array([gw / 2, gh / 2]), float(max(gw, gh)))
    h_norm, roll, keystone = _upright_homography(vp)
    if abs(roll) < MIN_CORRECTION_DEG and before < MIN_CORRECTION_DEG:
        return TiltResult(image, roll, keystone, before, before, 1.0, len(segments), "already upright")
    if keystone > MAX_KEYSTONE:
        return TiltResult(image, roll, keystone, before, before, 1.0, len(segments),
                          f"keystone {keystone:.2f} > {MAX_KEYSTONE}: needs review")
    homography = _pixel_homography(h_norm, w, h)
    zoom = _fit_zoom(homography, w, h)
    if zoom > MAX_ZOOM:
        return TiltResult(image, roll, keystone, before, before, 1.0, len(segments),
                          f"zoom {zoom:.2f} > {MAX_ZOOM} to hide borders: needs review")
    centre = np.array([[zoom, 0, w / 2 * (1 - zoom)], [0, zoom, h / 2 * (1 - zoom)], [0, 0, 1]])
    out = cv2.warpPerspective(image, centre @ homography, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
    after = tilt_deg(vertical_segments(_work_grey(out)[0]))
    return TiltResult(out, roll, keystone, before, after, zoom, len(segments), "")
