"""See -> decide -> act loop over the OpenCV 5 tools.

Every step: observe QC numbers, take the first rule that fires, run its tool,
measure again, then accept, revert, or stop and ask a person. The rule table
is fixed, so the same photos always give the same trace.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np

from lumen.align import SetAlignment, align_set
from lumen.group import Grouping, group_frames
from lumen.io import Frame
from lumen.lens import correct_lens
from lumen.merge import MergeResult, merge_set
from lumen.qc import clip_ratio
from lumen.tilt import straighten
from lumen.window import pull_windows

# Step budget: the fixed steps (align, reference swap, merge, window, Debevec, lens, tilt, finish)
# plus a drop + realign per frame. A flat 10 cut off a real 5-frame bracket that had to drop
# three frames, one step before the end.
FIXED_STEPS = 10
STEPS_PER_FRAME = 2
GHOST_REVIEW = 0.15  # share of image moving; above it the photo set needs a person
WINDOW_TRIGGER = 0.005  # share of image blown or crushed before trying the window pull
# A Debevec merge may not lose more MEF-SSIM than this. Synthetic sets lose 0.025-0.03 while
# gaining window detail (keep); the real Empa FORTH3 bracket lost 0.081 (reject).
MAX_MEF_SSIM_LOSS = 0.05
# A window pull that adds edges no frame recorded is a seam, not detail (qc.foreign_edge_share):
# the blotchy blend scored 0.10-0.11 on a real room, the accepted one <= 0.012.
MAX_FOREIGN_EDGES = 0.05
TILT_TOLERANCE_DEG = 0.05  # a straightening that measures worse than this is reverted


@dataclass(frozen=True)
class Step:
    step: int
    rule: str
    tool: str
    observe: dict
    params: dict
    result: dict
    decision: str


@dataclass(frozen=True)
class State:
    frames: tuple[Frame, ...]  # one bracket, dark -> bright
    reference: int | None = None  # None = middle frame
    alignment: SetAlignment | None = None
    merged: MergeResult | None = None
    image: np.ndarray | None = None
    tried: frozenset[str] = frozenset()
    status: str = "running"  # running | ok | needs_review
    reason: str = ""
    trace: tuple[Step, ...] = field(default_factory=tuple)
    off: frozenset[str] = frozenset()  # steps the person turned off at upload: OPTIONAL_STEPS


OPTIONAL_STEPS = ("deghost", "window", "lens", "tilt")  # merge itself is the product, so it cannot be turned off
Action = tuple[State, dict, dict, str]  # new state, params, result, decision
Rule = Callable[[State, dict], tuple[str, str, Callable[[], Action]] | None]


def _turned_off(s: State, key: str) -> Action:
    """The step is still written to the trace, so the record shows the person turned it off."""
    return replace(s, tried=s.tried | {key}), {}, {"note": "turned off at upload"}, "off"


# ---------- observe ----------

def observe(state: State) -> dict:
    obs: dict = {"frames": len(state.frames)}
    if state.alignment is not None:
        obs["worst_residual_px"] = state.alignment.worst_residual_px
        obs["residual_limit_px"] = state.alignment.residual_limit_px
    if state.merged is not None:
        obs["ghost_ratio"] = state.merged.ghost_ratio
    if state.image is not None:
        obs["clip_ratio"] = clip_ratio(state.image)
    return obs


def _times(frames: tuple[Frame, ...]) -> list[float | None]:
    return [f.exif.exposure_time_s for f in frames]


# ---------- rules ----------

def need_alignment(s: State, obs: dict):
    if s.alignment is not None:
        return None

    def act() -> Action:
        result = align_set([f.image for f in s.frames], s.reference)
        per_frame = [{"frame": fa.index, "method": fa.method, "residual_px": fa.residual_px, "note": fa.note}
                     for fa in result.frames]
        return replace(s, alignment=result), {"reference": result.reference_index}, \
            {"worst_residual_px": result.worst_residual_px, "frames": per_frame}, "accept"
    return "need_alignment", "align_set", act


def bad_frame(s: State, obs: dict):
    if s.alignment is None or s.merged is not None or obs["worst_residual_px"] <= s.alignment.residual_limit_px:
        return None
    ref = s.alignment.reference_index
    others = [fa for fa in s.alignment.frames if fa.index != ref]
    if all(fa.residual_px > s.alignment.residual_limit_px for fa in others) and len(s.frames) >= 3 \
            and "swap_reference" not in s.tried:
        # Nothing lines up with the reference: the reference itself may be the stranger.
        new_ref = next(i for i in range(len(s.frames)) if i != ref)

        def swap() -> Action:
            return replace(s, reference=new_ref, alignment=None, tried=s.tried | {"swap_reference"}), \
                {"old_reference": ref, "new_reference": new_ref}, {}, "change reference, realign"
        return "bad_reference", "swap_reference", swap
    worst = max(others, key=lambda fa: fa.residual_px)

    def act() -> Action:
        kept = tuple(f for i, f in enumerate(s.frames) if i != worst.index)
        params = {"drop_frame": worst.index, "residual_px": worst.residual_px}
        if len(kept) < 2:
            return replace(s, status="needs_review", reason="no frame lines up with the reference"), \
                params, {"frames_left": len(kept)}, "review"
        return replace(s, frames=kept, reference=None, alignment=None), params, \
            {"frames_left": len(kept)}, "drop frame, realign"
    return "bad_frame", "drop_frame", act


def need_merge(s: State, obs: dict):
    if s.alignment is None or s.merged is not None:
        return None

    deghost = "deghost" not in s.off

    def act() -> Action:
        m = merge_set(s.alignment.aligned, s.alignment.reference_index, "mertens", deghost=deghost)
        return replace(s, merged=m, image=m.image), {"engine": "mertens", "deghost": deghost}, \
            {"ghost_ratio": m.ghost_ratio, "clip_ratio": m.clip_ratio, "mef_ssim": m.mef_ssim}, "accept"
    return "need_merge", "merge_set", act


def too_much_motion(s: State, obs: dict):
    if s.merged is None or obs["ghost_ratio"] <= GHOST_REVIEW:
        return None

    def act() -> Action:
        return replace(s, status="needs_review", reason=f"{obs['ghost_ratio']:.0%} of the frame moves"), \
            {}, {}, "review"
    return "too_much_motion", "ask_person", act


def blown_window(s: State, obs: dict):
    if s.image is None or "window" in s.tried or obs["clip_ratio"] < WINDOW_TRIGGER:
        return None
    if "window" in s.off:
        return "blown_window", "pull_windows", lambda: _turned_off(s, "window")

    def act() -> Action:
        al = s.alignment
        w = pull_windows(s.image, al.aligned[0], al.aligned[:al.reference_index + 1], al.aligned)
        result = {"window_ratio": w.window_ratio, "clip_before": w.clip_before, "clip_after": w.clip_after,
                  "gain": w.gain, "foreign_edges": w.foreign_edges, "note": w.note}
        if w.window_ratio == 0:
            # Nothing to pull (clipping elsewhere, or none): not a failure, so no Debevec retry.
            return replace(s, tried=s.tried | {"window"}), {}, result, "skip"
        if w.clip_after < w.clip_before and w.foreign_edges <= MAX_FOREIGN_EDGES:
            return replace(s, image=w.image, tried=s.tried | {"window"}), {}, result, "accept"
        return replace(s, tried=s.tried | {"window", "window_failed"}), {}, result, "revert"
    return "blown_window", "pull_windows", act


def debevec_fallback(s: State, obs: dict):
    if "window_failed" not in s.tried or "debevec" in s.tried or None in _times(s.frames):
        return None

    def act() -> Action:
        m = merge_set(s.alignment.aligned, s.alignment.reference_index, "debevec", _times(s.frames))
        result = {"clip_before": obs["clip_ratio"], "clip_after": m.clip_ratio,
                  "mef_ssim_before": s.merged.mef_ssim, "mef_ssim_after": m.mef_ssim}
        tried = s.tried | {"debevec"}
        # Less clipping is not enough: a real bracket once traded MEF-SSIM 0.965 -> 0.884 for it.
        if m.clip_ratio < obs["clip_ratio"] and m.mef_ssim >= s.merged.mef_ssim - MAX_MEF_SSIM_LOSS:
            return replace(s, merged=m, image=m.image, tried=tried), {"engine": "debevec"}, result, "accept"
        return replace(s, tried=tried), {"engine": "debevec"}, result, "revert"
    return "debevec_fallback", "merge_set", act


def lens(s: State, obs: dict):
    if s.image is None or "lens" in s.tried:
        return None
    if "lens" in s.off:
        return "lens", "correct_lens", lambda: _turned_off(s, "lens")

    def act() -> Action:
        r = correct_lens(s.image)
        result = {"k1": r.k1, "curvature_before_px": r.curvature_before_px,
                  "curvature_after_px": r.curvature_after_px, "pieces": r.pieces, "note": r.note}
        new = replace(s, image=r.image, tried=s.tried | {"lens"})
        return new, {"focal_px": r.focal_px}, result, "accept" if r.k1 else "skip"
    return "lens", "correct_lens", act


def tilt(s: State, obs: dict):
    if s.image is None or "tilt" in s.tried:
        return None
    if "tilt" in s.off:
        return "tilt", "straighten", lambda: _turned_off(s, "tilt")

    def act() -> Action:
        r = straighten(s.image)
        result = {"roll_deg": r.roll_deg, "keystone": r.keystone, "tilt_before_deg": r.tilt_before_deg,
                  "tilt_after_deg": r.tilt_after_deg, "zoom": r.zoom, "note": r.note}
        tried = s.tried | {"tilt"}
        if "needs review" in r.note:
            return replace(s, tried=tried, status="needs_review", reason=r.note), {}, result, "review"
        if r.image is s.image:
            return replace(s, tried=tried), {}, result, "skip"
        if r.tilt_after_deg > r.tilt_before_deg + TILT_TOLERANCE_DEG:
            # Measured worse after the warp (seen in a private interior test set (not published)): keep the image as it was.
            return replace(s, tried=tried), {}, result, "revert"
        return replace(s, image=r.image, tried=tried), {}, result, "accept"
    return "tilt", "straighten", act


def finished(s: State, obs: dict):
    def act() -> Action:
        return replace(s, status="ok"), {}, {}, "done"
    return "finished", "none", act


RULES: tuple[Rule, ...] = (need_alignment, bad_frame, need_merge, too_much_motion,
                           blown_window, debevec_fallback, lens, tilt, finished)


# ---------- loop ----------

def run_set(frames: list[Frame], off: frozenset[str] | set[str] = frozenset()) -> State:
    """off: optional steps (OPTIONAL_STEPS) the person turned off."""
    unknown = set(off) - set(OPTIONAL_STEPS)
    if unknown:
        raise ValueError(f"cannot turn off {sorted(unknown)}; optional steps are {OPTIONAL_STEPS}")
    state = State(frames=tuple(frames), off=frozenset(off))
    if len(frames) < 2:
        return replace(state, status="needs_review", reason="single frame, not a bracket")
    max_steps = FIXED_STEPS + STEPS_PER_FRAME * len(frames)
    for number in range(1, max_steps + 1):
        obs = observe(state)
        name, tool, act = next(hit for rule in RULES if (hit := rule(state, obs)) is not None)
        new, params, result, decision = act()
        step = Step(number, name, tool, obs, params, result, decision)
        state = replace(new, trace=state.trace + (step,))
        if state.status != "running":
            return state
    return replace(state, status="needs_review", reason=f"no result after {max_steps} steps")


@dataclass(frozen=True)
class FolderResult:
    sets: list[State]
    grouping: list[dict]


def link_dicts(grouping: Grouping) -> list[dict]:
    return [{"left": link.left, "right": link.right, "joined": link.joined, "reason": link.reason}
            for link in grouping.links]


def run_folder(frames: list[Frame]) -> FolderResult:
    """Whole folder in memory: fine for tests and benches. jobs.run_job streams set by set."""
    grouping = group_frames(frames)
    return FolderResult([run_set([frames[i] for i in ids]) for ids in grouping.sets], link_dicts(grouping))


# ---------- trace export ----------

def _clean(value):
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        v = float(value)
        return None if math.isnan(v) else ("inf" if math.isinf(v) else round(v, 4))
    if isinstance(value, np.integer):
        return int(value)
    return value


def trace_dict(state: State) -> dict:
    """JSON-safe summary of one set: status, reason, frames used and every step."""
    return _clean({
        "status": state.status,
        "reason": state.reason,
        "frames": [f.path.name for f in state.frames],
        "steps": [s.__dict__ for s in state.trace],
    })
