"""Build the web UI's static evidence: real before/after images and the measurement report.

    uv run python bench/build_web_assets.py

Writes into src/lumen/web/img/ (all sources CC BY 4.0):
- hero-ev*.webp: five of the seven Empa FORTH3 exposures;
- real-<scene>-{before,after}.webp: Empa rooms, the 0 EV frame and Lumen's result
  (full-resolution run, the same code as the Lambda service), plus a window crop pair
  (reference frame vs the merge, before lens and tilt) where the reference frame is blown;
- ghost-<scene>-{off,on}.webp: Cambridge brackets merged with ghost removal off and on.
and src/lumen/web/report.json with the numbers from the same runs and from the
benchmark files in docs/, so the page never shows a number that was not measured.
"""

from __future__ import annotations

import csv
import datetime
import json
import statistics
from collections import defaultdict
from pathlib import Path

import cv2

import numpy as np

from lumen.agent import run_set
from lumen.group import group_frames
from lumen.align import align_set
from lumen.io import load_frame
from lumen.merge import merge_set
from lumen.tilt import _work_grey, vertical_segments
from lumen.window import window_mask

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "src" / "lumen" / "web"
DOCS = ROOT / "docs"
EMPA = ROOT / "data" / "empa" / "jpg"
CAMBRIDGE = ROOT / "data" / "cambridge" / "jpg"
HERO_SCENE = "FORTH3"
HERO_FRAMES = (0, 2, 3, 4, 6)  # EV -3, -1, 0, +1, +3 of the seven
EMPA_SCENES = ("FORTH3", "FORTH4", "Museum1")
# Large gains, a small object, and one bracket where the agent stops (parallax).
GHOST_SCENES = ("occlusion-1", "lolm-4", "solm-1", "multiview-1")
GHOST_REDUCE = 2  # as bench/ghost_real.py
THUMB_W, RESULT_W, CROP_W, GHOST_W = 360, 1600, 900, 1200
PLUMB_LINES = 16  # longest detected verticals drawn over the cover image

# Measured on the deployed stack (ap-southeast-2, arm64, 3008 MB), 2026-10-10; see docs/REPORT.md.
AWS_RUNS = [
    {"scene": "Empa FORTH3", "frames": 7, "megapixels": 16, "lambda_s": 18.3, "memory_mb": 1362, "status": "done"},
    {"scene": "Cambridge complex-1 (one bracket)", "frames": 5, "megapixels": 3.8, "lambda_s": 12.9,
     "memory_mb": 1009, "status": "needs review: 16 % of the frame moves"},
]
ARM_USD_PER_GB_S = 0.0000133334  # Lambda arm64 price per GB-second
SYNTH_METRICS = {  # metric: (label, unit, better)
    "align_err_px": ("Alignment error vs truth", "px", "low"),
    "ghost_recall": ("Ghost pixels found", "", "high"),
    "ghost_false_alarm": ("Static pixels flagged as ghost", "", "low"),
    "window_clip_after": ("Window still clipped after pull", "", "low"),
    "roll_err_deg": ("Roll error", "°", "low"),
    "tilt_after_deg": ("Vertical tilt after", "°", "low"),
    "agent_seconds_per_set": ("Agent time per set (MacBook)", "s", "low"),
}


def _webp(image, width: int, path: Path) -> None:
    h, w = image.shape[:2]
    small = cv2.resize(image, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
    if not cv2.imwrite(str(path), small, [cv2.IMWRITE_WEBP_QUALITY, 80]):
        raise SystemExit(f"could not write {path}")


def _ev0_index(frames) -> int:
    return min(range(len(frames)), key=lambda i: abs(frames[i].exif.exposure_bias_ev or 0.0))


def _plumb(image) -> list[list[float]]:
    """The longest near-vertical LSD segments of the image, as fractions of the analysis copy."""
    grey, _ = _work_grey(image)
    seg = vertical_segments(grey)
    seg = seg[np.argsort(-np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1]))][:PLUMB_LINES]
    h, w = grey.shape
    return [[round(x1 / w, 4), round(y1 / h, 4), round(x2 / w, 4), round(y2 / h, 4)] for x1, y1, x2, y2 in seg]


def _window_crop(state, out: Path, name: str) -> dict | None:
    """Window box of the reference frame (blown there, detailed in the dark frame): frame vs merge."""
    al = state.alignment
    ref = al.aligned[al.reference_index]
    mask = window_mask(ref, al.aligned[0])
    if not mask.any():
        return None
    x, y, w, h = cv2.boundingRect(mask)
    pad = round(0.15 * max(w, h))
    H, W = mask.shape
    box = (slice(max(0, y - pad), min(H, y + h + pad)), slice(max(0, x - pad), min(W, x + w + pad)))
    _webp(ref[box], CROP_W, out / f"{name}-win-before.webp")
    _webp(state.merged.image[box], CROP_W, out / f"{name}-win-after.webp")
    blown = lambda img: float((img[box].min(axis=2) >= 245).mean())  # noqa: E731  same threshold as window.BLOWN
    return {"before": f"/static/img/{name}-win-before.webp", "after": f"/static/img/{name}-win-after.webp",
            "blown_before": round(blown(ref), 4), "blown_after": round(blown(state.merged.image), 4)}


def empa_gallery(out: Path) -> tuple[dict, list[dict]]:
    hero, rooms = {}, []
    for scene in EMPA_SCENES:
        files = sorted((EMPA / scene).glob("*.jpg"))
        if len(files) != 7:
            raise SystemExit(f"run bench/fetch_empa.py {scene} first")
        # Same frame order as bench/run.py (grouping sorts dark -> bright), so the numbers match docs/traces.
        sets = group_frames([load_frame(p, 4) for p in files]).sets
        if len(sets) != 1:
            raise SystemExit(f"{scene}: grouping found {len(sets)} sets, expected 1")
        frames = [load_frame(files[i]) for i in sets[0]]
        state = run_set(frames)
        before = frames[_ev0_index(frames)].image
        name = f"real-{scene}"
        _webp(before, RESULT_W, out / f"{name}-before.webp")
        _webp(state.image, RESULT_W, out / f"{name}-after.webp")
        h, w = state.image.shape[:2]
        step = {s.tool: s for s in state.trace}
        tilt, merge = step["straighten"].result, step["merge_set"].result
        rooms.append({"scene": scene, "status": state.status, "frames": len(frames),
                      "size": [RESULT_W, round(h * RESULT_W / w)],
                      "before": f"/static/img/{name}-before.webp", "after": f"/static/img/{name}-after.webp",
                      "tilt_before_deg": round(tilt["tilt_before_deg"], 2),
                      "tilt_after_deg": round(tilt["tilt_after_deg"] if step["straighten"].decision == "accept"
                                              else tilt["tilt_before_deg"], 2),
                      "ghost": round(merge["ghost_ratio"], 4), "clip": round(merge["clip_ratio"], 4),
                      "steps": [{"tool": s.tool, "decision": s.decision} for s in state.trace],
                      "window": _window_crop(state, out, name)})
        if scene == HERO_SCENE:
            small = [load_frame(files[i], 4) for i in HERO_FRAMES]
            thumbs = []
            for n, f in enumerate(small):
                _webp(f.image, THUMB_W, out / f"hero-ev{n}.webp")
                thumbs.append({"src": f"/static/img/hero-ev{n}.webp", "ev": f.exif.exposure_bias_ev,
                               "shutter_s": f.exif.exposure_time_s})
            hero = {"frames": thumbs, "scene": scene, "plumb": _plumb(state.image)}
        print(f"{scene}: {state.status}, tilt {rooms[-1]['tilt_before_deg']} -> {rooms[-1]['tilt_after_deg']}", flush=True)
    return hero, rooms


def _ghost_merge(folder: Path, deghost: bool) -> tuple[np.ndarray, float]:
    frames = [load_frame(p, GHOST_REDUCE) for p in sorted(folder.glob("*.jpg"))]
    al = align_set([f.image for f in frames])
    m = merge_set(al.aligned, al.reference_index, deghost=deghost)
    return m.image, m.ghost_ratio


def ghost_gallery(out: Path) -> list[dict]:
    rows = {r["scene"]: r for r in csv.DictReader((DOCS / "RESULTS-ghost.csv").open())}
    items = []
    for scene in GHOST_SCENES:
        off, _ = _ghost_merge(CAMBRIDGE / scene / "ghosted", deghost=False)
        on, flagged = _ghost_merge(CAMBRIDGE / scene / "ghosted", deghost=True)
        _webp(off, GHOST_W, out / f"ghost-{scene}-off.webp")
        _webp(on, GHOST_W, out / f"ghost-{scene}-on.webp")
        r = rows[scene]
        h, w = on.shape[:2]
        items.append({"scene": scene, "category": r["category"], "size": [GHOST_W, round(h * GHOST_W / w)],
                      "off": f"/static/img/ghost-{scene}-off.webp", "on": f"/static/img/ghost-{scene}-on.webp",
                      "flagged": round(flagged, 4), "psnr_off": float(r["psnr_off"]), "psnr_on": float(r["psnr_on"]),
                      "agent": r["agent"], "agent_reason": r["agent_reason"]})
        print(f"{scene}: flagged {flagged:.1%}, agent {r['agent']}", flush=True)
    return items


def ghost_report() -> dict:
    rows = list(csv.DictReader((DOCS / "RESULTS-ghost.csv").open()))
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    mean = lambda rs, k: round(statistics.mean(float(r[k]) for r in rs), 2)  # noqa: E731
    gains = [float(r["psnr_on"]) - float(r["psnr_off"]) for r in rows]
    return {"scenes": len(rows), "psnr_off": mean(rows, "psnr_off"), "psnr_on": mean(rows, "psnr_on"),
            "ssim_off": round(statistics.mean(float(r["ssim_off"]) for r in rows), 3),
            "ssim_on": round(statistics.mean(float(r["ssim_on"]) for r in rows), 3),
            "better": sum(g > 0.05 for g in gains), "same": sum(abs(g) <= 0.05 for g in gains),
            "worse": sum(g < -0.05 for g in gains),
            "agent_ok": sum(r["agent"] == "ok" for r in rows),
            "categories": [{"name": c, "scenes": len(rs), "psnr_off": mean(rs, "psnr_off"),
                            "psnr_on": mean(rs, "psnr_on"),
                            "flagged": round(statistics.mean(float(r["flagged"]) for r in rs), 3)}
                           for c, rs in sorted(by_cat.items())]}


def empa_report() -> list[dict]:
    out = []
    for path in sorted(DOCS.glob("traces/real-*.json")):
        trace = json.loads(path.read_text())
        step = {s["tool"]: s for s in trace["steps"]}
        merge, tilt = step["merge_set"]["result"], step["straighten"]["result"]
        out.append({"scene": path.stem.removeprefix("real-"), "status": trace["status"],
                    "frames": len(trace["frames"]), "ghost": round(merge["ghost_ratio"], 4),
                    "clip": round(merge["clip_ratio"], 4),
                    "tilt_before_deg": round(tilt["tilt_before_deg"], 2),
                    "tilt_after_deg": round(tilt["tilt_after_deg"], 2),
                    "lens": step["correct_lens"]["decision"]})
    return out


def synthetic_report() -> list[dict]:
    values = defaultdict(list)
    for r in csv.DictReader((DOCS / "RESULTS-synthetic.csv").open()):
        values[(r["metric"], r["profile"])].append(float(r["value"]))
    rows = []
    for metric, (label, unit, better) in SYNTH_METRICS.items():
        row = {"label": label, "unit": unit}
        for profile in ("clean", "phone"):
            v = values[(metric, profile)]
            row[profile] = {"mean": round(statistics.mean(v), 4),
                            "worst": round(max(v) if better == "low" else min(v), 4), "n": len(v)}
        rows.append(row)
    return rows


def main() -> None:
    aws = [{**r, "gb_s": round(r["lambda_s"] * 3008 / 1024, 1),
            "usd": round(r["lambda_s"] * 3008 / 1024 * ARM_USD_PER_GB_S, 5)} for r in AWS_RUNS]
    out = WEB / "img"
    out.mkdir(exist_ok=True)
    hero, rooms = empa_gallery(out)
    report = {"generated": datetime.date.today().isoformat(),  # issue date of this sheet set "hero": hero, "gallery": {"empa": rooms, "ghost": ghost_gallery(out)},
              "ghost": ghost_report(), "empa": empa_report(),
              "synthetic": synthetic_report(), "aws": aws,
              "sources": {"empa": "Empa HDR Image Database, P. Zolliker & I. Sprow, CC BY 4.0, doi:10.5281/zenodo.7861544",
                          "cambridge": "Multi-exposure stacks for testing HDR deghosting, Karaduzovic-Hadziabdic, "
                                       "Hasic Telalovic, Mantiuk, University of Cambridge, CC BY 4.0, doi:10.17863/CAM.6881"}}
    (WEB / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(f"wrote {WEB / 'report.json'} and {len(list(out.glob('*.webp')))} images in {out}")


if __name__ == "__main__":
    main()
