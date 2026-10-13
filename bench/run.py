"""Measure every step and write docs/RESULTS.md, docs/RESULTS-synthetic.csv and docs/traces/*.json.

    uv run python bench/run.py                  # synthetic, clean + phone, seeds 3-6
    uv run python bench/run.py --real data/empa/jpg   # also real brackets: one subfolder per set
                                                     # (or a flat folder + labels.csv with file,set_id)

Synthetic sets have known answers (truth.json); real brackets have no ground
truth, so they report grouping, the agent's decisions, QC numbers and timing.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lumen import synth  # noqa: E402
from lumen.agent import run_set, trace_dict  # noqa: E402
from lumen.align import align_set  # noqa: E402
from lumen.group import exact_match_rate, group_frames  # noqa: E402
from lumen.io import Exif, load_frame  # noqa: E402
from lumen.lens import correct_lens  # noqa: E402
from lumen.merge import ghost_mask, merge_set  # noqa: E402
from lumen.qc import homography_error_px, mef_ssim  # noqa: E402
from lumen.tilt import straighten  # noqa: E402
from lumen.window import pull_windows  # noqa: E402

DOCS = ROOT / "docs"


def _ghost_truth(s: dict) -> np.ndarray:
    mask = np.zeros((synth.HEIGHT, synth.WIDTH), np.uint8)
    for f in s["frames"]:
        if f["ghost_bbox"]:
            x, y, w, h = f["ghost_bbox"]
            cv2.ellipse(mask, (x + w // 2, y + h // 2), (w // 2, h // 2), 0, 0, 360, 255, -1)
    return mask > 0


def measure_set(s: dict, frames: list) -> dict:
    ref = s["reference_index"]
    images = [f.image for f in frames]
    m: dict = {}
    al = align_set(images, ref)
    m["align_err_px"] = max(homography_error_px(fa.homography, np.array(f["homography"]), synth.WIDTH, synth.HEIGHT)
                            for fa, f in zip(al.frames, s["frames"]))
    m["align_all_ecc"] = float(all(fa.method in ("ecc", "reference") for fa in al.frames))
    merged = merge_set(al.aligned, ref)
    m["mef_ssim_fused"], m["mef_ssim_middle_only"] = merged.mef_ssim, mef_ssim(al.aligned, al.aligned[ref])
    found = np.logical_or.reduce([ghost_mask(a, al.aligned[ref]) > 0 for i, a in enumerate(al.aligned) if i != ref])
    truth = _ghost_truth(s)
    far = ~(cv2.dilate(truth.astype(np.uint8), np.ones((25, 25), np.uint8)) > 0)
    m["ghost_false_alarm"] = float((found & far).mean())
    if truth.any():
        m["ghost_recall"] = float((found & truth).sum() / truth.sum())
    debevec = merge_set(al.aligned, ref, "debevec", [f["exposure_time_s"] for f in s["frames"]])
    m["clip_mertens"], m["clip_debevec"] = merged.clip_ratio, debevec.clip_ratio
    window = pull_windows(merged.image, al.aligned[0], al.aligned[:ref + 1])
    m["window_clip_before"], m["window_clip_after"] = window.clip_before, window.clip_after
    k1 = s["lens"]["k1"]
    lens = correct_lens(al.aligned[ref], focal_px=s["lens"]["fx"])
    m["lens_k1_ok"] = float(abs(lens.k1 - k1) <= max(0.1 * abs(k1), 0.01))
    tilt = straighten(correct_lens(al.aligned[ref]).image)
    m["roll_err_deg"], m["tilt_after_deg"] = abs(tilt.roll_deg - s["roll_deg"]), tilt.tilt_after_deg
    t0 = time.time()
    state = run_set(frames)
    m["agent_seconds_per_set"] = time.time() - t0
    lens_step = next((st.decision for st in state.trace if st.tool == "correct_lens"), None)
    m["agent_ok"] = float(state.status == "ok" and lens_step == ("accept" if k1 else "skip"))
    return m


def fault_cases(sets: list[list]) -> tuple[dict, dict]:
    """Injected faults: stranger frame at the end, stranger in the reference slot, two rooms."""
    ok = defaultdict(list)
    traces = {}
    n = len(sets)
    for i in range(n):
        j = (i + 2) % n
        st = run_set([sets[i][0], sets[i][1], sets[j][2]])
        ok["stranger_frame_dropped"].append(st.status == "ok" and any(x.tool == "drop_frame" for x in st.trace))
        traces.setdefault("stranger-frame", st)
        st = run_set([sets[i][0], sets[j][1], sets[i][2]])
        ok["stranger_reference_recovered"].append(
            st.status == "ok" and any(x.tool == "swap_reference" for x in st.trace))
        traces.setdefault("stranger-reference", st)
        st = run_set([sets[i][1], sets[j][1]])
        ok["two_rooms_sent_to_person"].append(st.status == "needs_review")
        traces.setdefault("two-rooms", st)
    return {k: (sum(v), len(v)) for k, v in ok.items()}, traces


def run_synthetic(profiles: list[str], seeds: list[int]) -> tuple[list, dict, dict, dict]:
    rows, faults, grouping = [], {}, {}
    traces: dict = {}
    for profile in profiles:
        fault_total: dict = defaultdict(lambda: [0, 0])
        group_scores = defaultdict(list)
        for seed in seeds:
            with tempfile.TemporaryDirectory() as tmp:
                truth = synth.generate(tmp, sets=6, seed=seed, profile=profile)
                sets = [[load_frame(Path(tmp) / f["file"]) for f in s["frames"]] for s in truth["sets"]]
                flat = [f for s in sets for f in s]
                ids, k = [], 0
                for s in sets:
                    ids.append(list(range(k, k + len(s))))
                    k += len(s)
                group_scores["full_exif"].append(exact_match_rate(group_frames(flat).sets, ids))
                no_exif = [replace(f, exif=Exif()) for f in flat]
                group_scores["image_only"].append(exact_match_rate(group_frames(no_exif).sets, ids))
                for s, frames in zip(truth["sets"], sets):
                    for metric, value in measure_set(s, frames).items():
                        rows.append({"profile": profile, "seed": seed, "set_id": s["set_id"],
                                     "metric": metric, "value": value})
                counts, sample = fault_cases(sets)
                for key, (good, total) in counts.items():
                    fault_total[key][0] += good
                    fault_total[key][1] += total
                for name, state in sample.items():
                    traces.setdefault(f"{profile}-{name}", state)
                traces.setdefault(f"{profile}-normal", run_set(sets[1]))
        faults[profile] = dict(fault_total)
        grouping[profile] = {k: statistics.mean(v) for k, v in group_scores.items()}
    return rows, faults, grouping, traces


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def _real_files(folder: Path, labels_path: Path | None = None) -> tuple[list[Path], dict[Path, str]]:
    """Subfolders = one bracket each (label = folder name); else a flat folder with labels.csv."""
    subdirs = sorted(d for d in folder.iterdir() if d.is_dir())
    if subdirs:
        files = [p for d in subdirs for p in sorted(d.iterdir()) if p.suffix.lower() in IMAGE_SUFFIXES]
        return files, {p: p.parent.name for p in files}
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    labels_path = labels_path or folder / "labels.csv"
    if not labels_path.exists():
        return files, {}
    with labels_path.open() as fh:
        by_name = {r["file"]: r["set_id"] for r in csv.DictReader(fh)}
    return files, {p: by_name.get(p.name, f"unlabelled-{p.name}") for p in files}


def _step(state, tool: str) -> dict:
    return next((st.result | {"decision": st.decision} for st in state.trace if st.tool == tool), {})


def run_real(folder: Path, labels_path: Path | None = None) -> tuple[list, float | None, dict]:
    """Group on 1/4-size decodes, then run the agent set by set at full resolution."""
    files, label = _real_files(folder, labels_path)
    grouping = group_frames([load_frame(p, 4) for p in files])
    score = None
    if label:
        by_set = defaultdict(list)
        for i, p in enumerate(files):
            by_set[label[p]].append(i)
        score = exact_match_rate(grouping.sets, list(by_set.values()))
    rows, traces = [], {}
    for ids in grouping.sets:
        name = label.get(files[ids[0]], files[ids[0]].stem)
        t0 = time.time()
        state = run_set([load_frame(files[i]) for i in ids])
        merge, window = _step(state, "merge_set"), _step(state, "pull_windows")
        lens, tilt = _step(state, "correct_lens"), _step(state, "straighten")
        rows.append({"set": name, "status": state.status, "frames": len(state.frames),
                     "ghost": merge.get("ghost_ratio"), "clip": merge.get("clip_ratio"),
                     "window": window.get("decision", "-"), "lens": lens.get("decision", "-"),
                     "k1": lens.get("k1"), "tilt": (tilt.get("tilt_before_deg"), tilt.get("tilt_after_deg")),
                     "keystone": tilt.get("keystone"), "seconds": time.time() - t0, "reason": state.reason})
        traces[f"real-{name}"] = state
    return rows, score, traces


def _summary(rows: list) -> dict:
    table: dict = defaultdict(lambda: defaultdict(list))
    for r in rows:
        table[r["metric"]][r["profile"]].append(float(r["value"]))
    return table


def _real_lines(real_rows: list, real_score: float | None, real_sources: list[str]) -> list[str]:
    if not real_rows:
        return ["", "## Real brackets", "",
                "Not run yet: `uv run python bench/fetch_empa.py FORTH3 ...`, then `--real data/empa/jpg`."]
    lines = ["", "## Real brackets", ""]
    lines += [f"Source: {line.lstrip('- ')}" for line in real_sources]
    lines += ["", f"Grouping exact match: {real_score:.0%}" if real_score is not None
              else "No set labels: grouping not scored.", "",
              "| Set | Status | Frames | Ghost | Clip after merge | Window | Lens (k1) | Tilt before → after | Keystone | Seconds |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    fmt = lambda v, f: "-" if v is None else format(v, f)  # noqa: E731
    for r in real_rows:
        before, after = r["tilt"]
        lines.append(f"| {r['set']} | {r['status']} {r['reason']} | {r['frames']} | {fmt(r['ghost'], '.2%')} | "
                     f"{fmt(r['clip'], '.2%')} | {r['window']} | {r['lens']} ({fmt(r['k1'], '+.3f')}) | "
                     f"{fmt(before, '.2f')}° → {fmt(after, '.2f')}° | {fmt(r['keystone'], '.2f')} | {r['seconds']:.1f} |")
    return lines


def write_internal(out_dir: Path, folder: Path, real_rows, real_score, traces) -> None:
    """Private data: results stay in out_dir (gitignored), never in docs/."""
    (out_dir / "traces").mkdir(parents=True, exist_ok=True)
    for name, state in traces.items():
        (out_dir / "traces" / f"{name}.json").write_text(json.dumps(trace_dict(state), indent=2))
    lines = ["# Internal results (not for publication)", "", f"Input: `{folder}`"]
    (out_dir / "RESULTS-internal.md").write_text("\n".join(lines + _real_lines(real_rows, real_score, [])) + "\n")


def write_outputs(rows, faults, grouping, traces, real_rows, real_score, real_sources, profiles, seeds) -> None:
    (DOCS / "traces").mkdir(parents=True, exist_ok=True)
    with (DOCS / "RESULTS-synthetic.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, ["profile", "seed", "set_id", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)
    for name, state in traces.items():
        (DOCS / "traces" / f"{name}.json").write_text(json.dumps(trace_dict(state), indent=2))
    table = _summary(rows)
    lines = ["# Results — theMnexus Lumen", "",
             f"Generated by `uv run python bench/run.py`. Synthetic seeds {seeds}, 6 sets each, profiles {profiles}.",
             "Raw numbers: `docs/RESULTS-synthetic.csv`. Traces: `docs/traces/`.", "",
             "## Per-step numbers (synthetic, known answers)", "",
             "| Metric | " + " | ".join(f"{p} mean · worst" for p in profiles) + " |",
             "|---|" + "---|" * len(profiles)]
    worst_is_min = {"align_all_ecc", "lens_k1_ok", "agent_ok", "ghost_recall", "mef_ssim_fused",
                    "mef_ssim_middle_only"}
    for metric, by_profile in table.items():
        cells = []
        for p in profiles:
            v = by_profile.get(p, [])
            worst = (min(v) if metric in worst_is_min else max(v)) if v else float("nan")
            cells.append(f"{statistics.mean(v):.4f} · {worst:.4f}" if v else "—")
        lines.append(f"| `{metric}` | " + " | ".join(cells) + " |")
    lines += ["", "## Grouping (exact set match)", "", "| Profile | Full EXIF | Image only |", "|---|---|---|"]
    lines += [f"| {p} | {g['full_exif']:.0%} | {g['image_only']:.0%} |" for p, g in grouping.items()]
    lines += ["", "## Agent on injected faults", "", "| Case | " + " | ".join(profiles) + " |",
              "|---|" + "---|" * len(profiles)]
    for case in next(iter(faults.values())):
        lines.append(f"| {case} | " + " | ".join(f"{faults[p][case][0]}/{faults[p][case][1]}" for p in profiles) + " |")
    lines += _real_lines(real_rows, real_score, real_sources)
    lines += ["", "## Limits", "",
              "- Synthetic rooms are drawn shapes; real rooms have more texture, plants and mirrors.",
              "- The phone profile adds an S tone curve, noise and JPEG 75, but no local tone mapping or phone HDR.",
              "- Ghost recall is about 0.8: the edge of a moving object can survive at low contrast.",
              "- Lens self-calibration fits k1 only; fisheye lenses are not handled yet.", ""]
    (DOCS / "RESULTS.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[3, 4, 5, 6])
    parser.add_argument("--profiles", nargs="+", default=["clean", "phone"], choices=sorted(synth.PROFILES))
    parser.add_argument("--real", type=Path, help="folder with real brackets (subfolder per set, or labels.csv)")
    parser.add_argument("--labels", type=Path, help="labels.csv (file,set_id) kept outside the photo folder")
    parser.add_argument("--internal-out", type=Path,
                        help="private data: run only --real and write results here instead of docs/")
    args = parser.parse_args()
    if args.internal_out:
        if not args.real:
            raise SystemExit("--internal-out needs --real")
        real_rows, real_score, real_traces = run_real(args.real, args.labels)
        write_internal(args.internal_out, args.real, real_rows, real_score, real_traces)
        print(f"wrote {args.internal_out / 'RESULTS-internal.md'} ({len(real_rows)} sets)")
        return
    rows, faults, grouping, traces = run_synthetic(args.profiles, args.seeds)
    real_rows, real_score, real_traces, real_sources = [], None, {}, []
    if args.real:
        real_rows, real_score, real_traces = run_real(args.real, args.labels)
        sources = next((f for f in (args.real / "SOURCES.md", args.real.parent / "SOURCES.md") if f.exists()), None)
        real_sources = [ln for ln in sources.read_text().splitlines() if ln.startswith("- ")][:1] if sources else []
    write_outputs(rows, faults, grouping, {**traces, **real_traces}, real_rows, real_score, real_sources,
                  args.profiles, args.seeds)
    print(f"wrote {DOCS / 'RESULTS.md'} ({len(rows)} measurements, {len(traces)} traces)")


if __name__ == "__main__":
    main()
