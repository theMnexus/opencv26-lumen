"""Split a folder of photos into bracket sets.

Frames are ordered by capture time. Two neighbours join the same set when
all of these hold:
  1. the time gap is small (or a timestamp is missing),
  2. the exposure (bias and shutter time) does not repeat inside the current set,
  3. the scene matches: phase correlation of edge maps (main signal), or a
     near-identical DCT signature when the timestamps confirm one burst.
Every decision is kept as a Link so the agent trace can show why.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from lumen.io import Frame

MAX_GAP_S = 5.0  # brackets fire within ~2 s; next room takes longer to set up
SAME_SHUTTER = 0.01  # shutter times within 1 % count as the same exposure
THUMB_DCT = 64  # px, square grey thumbnail for the DCT signature
DCT_BAND = 8  # low-frequency block kept (DC dropped)
THUMB_EDGE_W = 256  # px, width of the edge map for phase correlation
# Measured 09/10 on 12 synthetic sets (seed 1), time and EV removed:
#   phase response  same set min 0.459 | different set max 0.320
#   dct cosine      same set min 0.985 | different set max 0.984  (does not separate)
MIN_PHASE_RESPONSE = 0.40
MAX_PHASE_SHIFT = 0.10  # fraction of thumbnail width; tripod brackets barely move
MIN_DCT_COS_TIMED = 0.99  # only used when the time gap is known and small
# A slow camera can idle longer before its brightest frame than the photographer
# takes to move to the next room (a private interior test set (not published)), so a long gap alone may not split.
# Same-bracket phase stays well above this; different rooms stay below 0.45.
STRONG_PHASE_RESPONSE = 0.6


@dataclass(frozen=True)
class Link:
    left: int
    right: int
    gap_s: float | None
    dct_cos: float
    phase_response: float
    phase_shift: float
    joined: bool
    reason: str


@dataclass(frozen=True)
class Grouping:
    sets: list[list[int]]  # frame indices, each set ordered dark -> bright
    links: list[Link]


def _equalized_grey(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Histogram-equalised grey thumbnail: removes most of the exposure difference."""
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(grey, (width, height), interpolation=cv2.INTER_AREA)
    return cv2.equalizeHist(small)


def dct_signature(image: np.ndarray) -> np.ndarray:
    thumb = _equalized_grey(image, THUMB_DCT, THUMB_DCT).astype(np.float32)
    coeffs = cv2.dct(thumb)[:DCT_BAND, :DCT_BAND].flatten()[1:]
    norm = float(np.linalg.norm(coeffs))
    return coeffs / norm if norm > 0 else coeffs


def edge_map(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    height = max(1, round(THUMB_EDGE_W * h / w))
    thumb = _equalized_grey(image, THUMB_EDGE_W, height).astype(np.float32)
    gx = cv2.Sobel(thumb, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(thumb, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def _phase(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    if a.shape != b.shape:  # portrait vs landscape: not the same bracket
        return 0.0, 1.0
    window = cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(a, b, window)
    return float(response), float(np.hypot(dx, dy) / a.shape[1])


def _gap_s(a: Frame, b: Frame) -> float | None:
    """Idle time between two shots: from the end of the earlier exposure to the next start.

    Start-to-start time is wrong for dark rooms: the real Empa Museum1 bracket
    has a 4 s frame followed 8.7 s later by the next one, which a plain 5 s gap
    would cut into two sets.
    """
    if a.exif.taken_at is None or b.exif.taken_at is None:
        return None
    first, second = (a, b) if a.exif.taken_at <= b.exif.taken_at else (b, a)
    gap = (second.exif.taken_at - first.exif.taken_at).total_seconds()
    return max(0.0, gap - (first.exif.exposure_time_s or 0.0))


def _decide(gap: float | None, ev_repeats: bool, dct_cos: float, response: float, shift: float) -> tuple[bool, str]:
    if ev_repeats:
        return False, "exposure bias repeats in current set"
    if gap is not None and gap > MAX_GAP_S:
        if response >= STRONG_PHASE_RESPONSE and shift <= MAX_PHASE_SHIFT:
            return True, f"slow camera ({gap:.1f}s idle) but phase response {response:.2f} >= {STRONG_PHASE_RESPONSE}"
        return False, f"time gap {gap:.1f}s > {MAX_GAP_S}s"
    if response >= MIN_PHASE_RESPONSE and shift <= MAX_PHASE_SHIFT:
        return True, f"phase response {response:.2f} >= {MIN_PHASE_RESPONSE}"
    if gap is not None and dct_cos >= MIN_DCT_COS_TIMED:
        return True, f"same burst ({gap:.1f}s) and dct cos {dct_cos:.3f} >= {MIN_DCT_COS_TIMED}"
    return False, f"scene differs (phase {response:.2f}, dct {dct_cos:.3f})"


def _capture_order(frames: list[Frame]) -> list[int]:
    """By capture time; frames without time keep input order at the end."""
    timed = sorted((f.exif.taken_at, i) for i, f in enumerate(frames) if f.exif.taken_at is not None)
    untimed = [i for i, f in enumerate(frames) if f.exif.taken_at is None]
    return [i for _, i in timed] + untimed


def _brightness(frame: Frame) -> float:
    return float(cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY).mean())


def _same_exposure(a: Frame, b: Frame) -> bool:
    """Same bias and same shutter time = the same shot again. Manual-mode brackets keep the
    bias at 0 and change only the shutter, so bias alone split them into single frames."""
    bias_a, bias_b = a.exif.exposure_bias_ev, b.exif.exposure_bias_ev
    if bias_a is None or bias_b is None or bias_a != bias_b:
        return False
    time_a, time_b = a.exif.exposure_time_s, b.exif.exposure_time_s
    return time_a is None or time_b is None or abs(time_a - time_b) <= SAME_SHUTTER * max(time_a, time_b)


def _dark_to_bright(indices: list[int], frames: list[Frame]) -> list[int]:
    exifs = [frames[i].exif for i in indices]
    if all(e.exposure_bias_ev is not None for e in exifs):
        times = [e.exposure_time_s for e in exifs]
        keys = [(e.exposure_bias_ev, t if all(x is not None for x in times) else 0.0) for e, t in zip(exifs, times)]
        if len(set(keys)) == len(keys):
            return [i for _, i in sorted(zip(keys, indices))]
    return sorted(indices, key=lambda i: _brightness(frames[i]))


def one_bracket(frames: list[Frame]) -> Grouping:
    """The person says the whole upload is one room: no grouping, just order dark -> bright.
    Needed for staged brackets (objects moved by hand, minutes between shots) that the
    time and phase rules rightly split."""
    return Grouping([_dark_to_bright(list(range(len(frames))), frames)] if frames else [], [])


def group_frames(frames: list[Frame]) -> Grouping:
    if not frames:
        return Grouping([], [])
    order = _capture_order(frames)
    sigs = {i: dct_signature(frames[i].image) for i in order}
    edges = {i: edge_map(frames[i].image) for i in order}
    sets: list[list[int]] = [[order[0]]]
    links: list[Link] = []
    for left, right in zip(order, order[1:]):
        current = sets[-1]
        ev_repeats = any(_same_exposure(frames[i], frames[right]) for i in current)
        dct_cos = float(np.dot(sigs[left], sigs[right]))
        response, shift = _phase(edges[left], edges[right])
        gap = _gap_s(frames[left], frames[right])
        joined, reason = _decide(gap, ev_repeats, dct_cos, response, shift)
        links.append(Link(left, right, gap, dct_cos, response, shift, joined, reason))
        if joined:
            current.append(right)
        else:
            sets.append([right])
    return Grouping([_dark_to_bright(s, frames) for s in sets], links)


def exact_match_rate(predicted: list[list[int]], truth: list[list[int]]) -> float:
    """Share of truth sets reproduced exactly (order inside a set ignored)."""
    if not truth:
        return 1.0
    found = {frozenset(s) for s in predicted}
    return sum(frozenset(s) in found for s in truth) / len(truth)
