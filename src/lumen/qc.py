"""Quality numbers the agent reads. Every function returns plain floats."""

from __future__ import annotations

import cv2
import numpy as np

RESIDUAL_TILES = 4  # 4 x 4 grid of local phase correlations
# Measured 09/10 (synthetic seed 3): same room -> 13/16 tiles respond >= 0.15,
# different room after a forced ECC fit -> 4/16. Flat tiles also respond low.
MIN_TILE_RESPONSE = 0.15
MIN_RELIABLE_SHARE = 0.5  # fewer reliable tiles than this -> not the same scene
MIN_TILE_VALID = 0.8  # share of valid pixels a tile needs


PRE_BLUR_SIGMA = 2.0  # px before Sobel; see grey_gradient


def grey_gradient(image: np.ndarray) -> np.ndarray:
    """Exposure-robust float32 map: Sobel magnitude of the equalised grey image.

    The blur before Sobel matters on noisy phone JPEGs: equalising a dark frame
    amplifies its noise and the raw gradient drowns in it. Measured 09/10
    (ECC cc, seed 5-6): without it same-room pairs fell to 0.645 on the phone
    profile; with sigma 2 same-room >= 0.921, different rooms <= 0.652.
    """
    grey = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eq = cv2.GaussianBlur(cv2.equalizeHist(grey).astype(np.float32), (0, 0), PRE_BLUR_SIGMA)
    gx = cv2.Sobel(eq, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(eq, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.GaussianBlur(cv2.magnitude(gx, gy), (5, 5), 0)


CLIP_HIGH = 250  # uint8: all channels at or above = blown highlight
CLIP_LOW = 5  # uint8: all channels at or below = crushed shadow
MEF_SIZE = 600  # px, long side used for MEF-SSIM
MEF_WINDOW = 8
MEF_STEP = 4
MEF_P = 4.0  # structure weight exponent (Ma et al. 2015)
MEF_C2 = (0.03 * 255) ** 2


def clip_ratio(image: np.ndarray) -> float:
    """Share of pixels with every channel blown or crushed."""
    blown = (image.min(axis=2) >= CLIP_HIGH) if image.ndim == 3 else image >= CLIP_HIGH
    crushed = (image.max(axis=2) <= CLIP_LOW) if image.ndim == 3 else image <= CLIP_LOW
    return float((blown | crushed).mean())


def _patches(grey: np.ndarray) -> np.ndarray:
    windows = np.lib.stride_tricks.sliding_window_view(grey, (MEF_WINDOW, MEF_WINDOW))
    return windows[::MEF_STEP, ::MEF_STEP].reshape(-1, MEF_WINDOW * MEF_WINDOW)


def _mef_grey(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return cv2.resize(grey, size, interpolation=cv2.INTER_AREA).astype(np.float64)


def mef_ssim(inputs: list[np.ndarray], fused: np.ndarray) -> float:
    """Single-scale MEF-SSIM (Ma, Zeng, Wang, IEEE TIP 2015), no reference image needed.

    For each patch the ideal fused patch keeps the strongest input contrast and
    a contrast-weighted mix of the input structures; the score is the SSIM
    structure term between that ideal and the fused patch. Range about [0, 1].
    """
    h, w = fused.shape[:2]
    scale = MEF_SIZE / max(h, w)
    size = (max(MEF_WINDOW, round(w * scale)), max(MEF_WINDOW, round(h * scale)))
    x = np.stack([_patches(_mef_grey(img, size)) for img in inputs])  # K, N, n
    y = _patches(_mef_grey(fused, size))
    xc = x - x.mean(axis=2, keepdims=True)
    contrast = np.linalg.norm(xc, axis=2)  # K, N
    structure = xc / np.maximum(contrast[..., None], 1e-12)
    weights = contrast ** MEF_P
    mixed = (weights[..., None] * structure).sum(axis=0) / np.maximum(weights.sum(axis=0)[:, None], 1e-12)
    mixed /= np.maximum(np.linalg.norm(mixed, axis=1, keepdims=True), 1e-12)
    ideal = contrast.max(axis=0)[:, None] * mixed
    yc = y - y.mean(axis=1, keepdims=True)
    n = x.shape[2] - 1
    cov = (ideal * yc).sum(axis=1) / n
    var_i, var_y = (ideal ** 2).sum(axis=1) / n, (yc ** 2).sum(axis=1) / n
    return float(np.mean((2 * cov + MEF_C2) / (var_i + var_y + MEF_C2)))


def homography_error_px(estimated: np.ndarray, truth: np.ndarray, width: int, height: int) -> float:
    """Mean distance between where the two homographies send a 5 x 5 grid."""
    xs, ys = np.meshgrid(np.linspace(0, width - 1, 5), np.linspace(0, height - 1, 5))
    pts = np.stack([xs.ravel(), ys.ravel()], axis=1).reshape(-1, 1, 2).astype(np.float64)
    a = cv2.perspectiveTransform(pts, np.asarray(estimated, np.float64))
    b = cv2.perspectiveTransform(pts, np.asarray(truth, np.float64))
    return float(np.linalg.norm(a - b, axis=2).mean())


def tile_residual_px(reference: np.ndarray, aligned: np.ndarray, valid: np.ndarray) -> float:
    """Median of local shifts left after alignment, in pixels of the given images.

    Works without ground truth: each tile of the gradient maps is phase-correlated;
    a perfect alignment leaves every tile shift near zero. Median, not p90: a
    moving person (ghost) shifts one or two tiles by tens of pixels while the
    camera alignment is fine (measured 09/10: p90 17.4 px vs truth error 0.15 px).
    A real misalignment moves every tile, so the median still catches it.
    Returns inf when too few tiles correlate: the images do not show the same scene.
    """
    ref_g, ali_g = grey_gradient(reference), grey_gradient(aligned)
    h, w = ref_g.shape
    th, tw = h // RESIDUAL_TILES, w // RESIDUAL_TILES
    window = cv2.createHanningWindow((tw, th), cv2.CV_32F)
    shifts, checked = [], 0
    for r in range(RESIDUAL_TILES):
        for c in range(RESIDUAL_TILES):
            sl = (slice(r * th, (r + 1) * th), slice(c * tw, (c + 1) * tw))
            if (valid[sl] > 0).mean() < MIN_TILE_VALID:
                continue
            checked += 1
            (dx, dy), response = cv2.phaseCorrelate(ref_g[sl], ali_g[sl], window)
            if response >= MIN_TILE_RESPONSE:
                shifts.append(float(np.hypot(dx, dy)))
    if not checked or len(shifts) < MIN_RELIABLE_SHARE * checked:
        return float("inf")
    return float(np.median(shifts))


# Foreign edges: edges in an edited region that no input frame has. Seams between two
# sources show up here; real texture does not, because some frame recorded it.
# Measured 13/10 inside the window region (own room, -4..+4 EV, at 6000, 3000 and 1500 px):
# the per-pixel blend that left tree-shaped blotches scored 0.103-0.112, the one-source blend
# 0.000-0.003; synthetic sets (seeds 5-6) <= 0.002; Empa FORTH4 0.012.
EDGE_BLUR = 1.5  # px, after equalising, so every exposure gives comparable edges
RESULT_CANNY = (50, 150)
SOURCE_CANNY = (15, 45)  # lower: a fusion sharpens texture a single frame shows faintly
EDGE_REACH = 7  # px, a result edge this close to a source edge counts as recorded
EDGE_WORK_SIDE = 800  # px, long side of the box the check runs on: Canny finds grain at 24 MP


def _edges(image: np.ndarray, thresholds: tuple[int, int]) -> np.ndarray:
    grey = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(cv2.GaussianBlur(cv2.equalizeHist(grey), (0, 0), EDGE_BLUR), *thresholds) > 0


def foreign_edge_share(result: np.ndarray, sources: list[np.ndarray], mask: np.ndarray) -> float:
    """Share of the result's edges inside mask (uint8) that no source frame has nearby."""
    if not mask.any():
        return 0.0
    x, y, w, h = cv2.boundingRect(mask)
    box = (slice(y, y + h), slice(x, x + w))
    scale = min(1.0, EDGE_WORK_SIDE / max(w, h))

    def small(image: np.ndarray) -> np.ndarray:
        return cv2.resize(image[box], None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else image[box]

    recorded = np.logical_or.reduce([_edges(small(f), SOURCE_CANNY) for f in sources]).astype(np.uint8)
    recorded = cv2.dilate(recorded, np.ones((EDGE_REACH, EDGE_REACH), np.uint8)) > 0
    found = _edges(small(result), RESULT_CANNY) & (small(mask) > 0)
    return float((found & ~recorded).sum() / max(int(found.sum()), 1))
