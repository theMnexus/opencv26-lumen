"""Bring window detail back from the darker aligned frames.

1. Window regions: large blown blobs in the merged image (all channels high),
   closed so the mullions inside count as window, kept only where the dark
   frame still has unclipped detail.
2. Fill: an exposure fusion (merge_set, with ghost removal) of the frames at
   or below the reference, cut to the windows' box; with only one dark frame,
   that frame brightened in sRGB-linear space so the window's p99 lands just
   under clipping.
3. Blend: the whole region comes from the fill, with a 3 px feathered edge.
   A per-pixel gate on merged luminance was used until 13/10: on a real room
   the blown sky came from the -4 EV frame (mean 72 where the merged image was
   225) while the unblown branches kept the merged pixels, which gave dark
   blotches shaped like the trees and branches that looked like a negative.
Only pixels the camera actually recorded are used.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from lumen.merge import merge_set
from lumen.qc import foreign_edge_share

BLOWN = 245  # all channels at or above = blown in the merged image
DARK_OK = 245  # dark frame must stay below this to hold detail
MIN_WINDOW_AREA = 0.002  # share of image
CLOSE_K = 25  # px, joins window panes across mullions
TARGET_P99 = 240  # uint8 level for the window's brightest 1 % after gain
EDGE_SIGMA = 3  # px, feather of the region edge
SRGB_GAMMA = 2.2


@dataclass(frozen=True)
class WindowResult:
    image: np.ndarray
    window_ratio: float  # share of image detected as window
    clip_before: float  # share of window pixels blown, before
    clip_after: float
    gain: float
    note: str
    foreign_edges: float = 0.0  # qc.foreign_edge_share inside the window region


def window_mask(merged: np.ndarray, dark: np.ndarray) -> np.ndarray:
    blown = (merged.min(axis=2) >= BLOWN).astype(np.uint8) * 255
    region = cv2.morphologyEx(blown, cv2.MORPH_CLOSE, np.ones((CLOSE_K, CLOSE_K), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(region)
    dark_detail = dark.max(axis=2) < DARK_OK
    keep = []
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] < MIN_WINDOW_AREA * region.size:
            continue
        inside = labels == i
        if dark_detail[inside].mean() > 0.5:  # the dark frame can fill it
            keep.append(i)
    return np.isin(labels, keep).astype(np.uint8) * 255


def _gain_for(dark: np.ndarray, mask: np.ndarray) -> float:
    p99 = (float(np.percentile(dark[mask > 0].max(axis=1), 99)) / 255) ** SRGB_GAMMA
    target = (TARGET_P99 / 255) ** SRGB_GAMMA
    return max(1.0, target / max(p99, 1e-6))


def _blend(merged: np.ndarray, fill: np.ndarray, mask: np.ndarray) -> np.ndarray:
    alpha = cv2.GaussianBlur(mask.astype(np.float32) / 255, (0, 0), EDGE_SIGMA)[..., None]
    return (alpha * fill + (1 - alpha) * merged).round().clip(0, 255).astype(np.uint8)


def _gain_lut(gain: float) -> np.ndarray:
    levels = (np.arange(256) / 255.0) ** SRGB_GAMMA
    return (255 * np.clip(levels * gain, 0, 1) ** (1 / SRGB_GAMMA)).round().astype(np.uint8)


def _fill(dark: np.ndarray, darker: list[np.ndarray], mask: np.ndarray, box: tuple) -> tuple[np.ndarray, float]:
    if len(darker) >= 2:
        return merge_set([f[box] for f in darker], len(darker) - 1, "mertens").image, 1.0
    gain = _gain_for(dark, mask)
    return cv2.LUT(dark[box], _gain_lut(gain)), gain


def pull_windows(merged: np.ndarray, dark: np.ndarray, darker: list[np.ndarray] = (),
                 recorded: list[np.ndarray] = ()) -> WindowResult:
    """merged: fused BGR uint8; dark: darkest frame aligned to the same reference;
    darker: aligned frames from the darkest up to the reference (agent passes them);
    recorded: every aligned frame, for the foreign-edge check (default: darker, else dark).
    The merged image is not a source for that check: Mertens itself adds edges where the
    contrast is low, which hid the seams (0.061 vs 0.018 instead of 0.256 vs 0.019).

    Work happens inside the windows' bounding box (+ a blur margin): at 24 MP
    the full-frame float64 version took 3.8 s and several GB (09/10).
    """
    mask = window_mask(merged, dark)
    if not mask.any():
        return WindowResult(merged, 0.0, 0.0, 0.0, 1.0, "no blown window with dark-frame detail")
    inside = mask > 0
    clip_before = float((merged.min(axis=2) >= BLOWN)[inside].mean())
    x, y, w, h = cv2.boundingRect(mask)
    pad = 16  # > 3 sigma of the edge blur
    box = (slice(max(0, y - pad), y + h + pad), slice(max(0, x - pad), x + w + pad))
    out = merged.copy()
    fill_box, gain = _fill(dark, list(darker), mask, box)
    out[box] = _blend(merged[box], fill_box, mask[box])
    clip_after = float((out.min(axis=2) >= BLOWN)[inside].mean())
    sources = list(recorded) or list(darker) or [dark]
    foreign = foreign_edge_share(out[box], [f[box] for f in sources], mask[box])
    return WindowResult(out, float(inside.mean()), clip_before, clip_after, gain, "", foreign)
