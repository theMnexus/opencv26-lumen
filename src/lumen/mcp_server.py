"""MCP server: the same OpenCV 5 tools the Lumen agent uses, for other agents.

A coding agent (Claude Code, Codex, Kiro, ...) can call these tools, read the
QC numbers, and decide its own next step, or hand the whole bracket to
`run_agent` and read the trace.

    uv run --extra mcp lumen-mcp                 # stdio transport

Safety: every path must resolve inside LUMEN_MCP_ROOT (default: the working
directory) and be an image; outputs are written only to <root>/lumen-out/.
"""

from __future__ import annotations

import functools
import os
import re
from pathlib import Path

import cv2
import numpy as np

from lumen.agent import run_set, trace_dict
from lumen.align import align_set
from lumen.group import group_frames
from lumen.io import load_frame, load_image
from lumen.lens import _camera, _work_copy, correct_lens as lens_fix, line_pieces, straightness
from lumen.merge import merge_set
from lumen.qc import clip_ratio
from lumen.tilt import straighten as tilt_fix, tilt_deg, vertical_segments

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
MAX_PATHS = 60
OUT_DIR = "lumen-out"
SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")
INSTRUCTIONS = (
    "OpenCV 5 tools for real-estate bracket photos. Typical flow: group_photos -> align_bracket "
    "(check residual_px) -> merge_bracket -> correct_lens -> straighten, reading qc_image numbers "
    "between steps; or run_agent for the built-in see/decide/act loop with a trace. Paths are "
    "relative to the server root; a folder stands for the images inside it; outputs go to lumen-out/."
)


def _root() -> Path:
    return Path(os.environ.get("LUMEN_MCP_ROOT", os.getcwd())).resolve()


def _inside_root(raw: str) -> Path:
    root = _root()
    path = (root / raw).resolve()
    if root != path and root not in path.parents:
        raise ValueError(f"path is outside the server root: {raw}")
    return path


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_SUFFIXES and path.is_file()


def _image_path(raw: str) -> Path:
    path = _inside_root(raw)
    if not _is_image(path):
        raise ValueError(f"not an image file: {raw}")
    return path


def _image_paths(raw: list[str], minimum: int = 1) -> list[Path]:
    """File paths, or a folder: a folder stands for the images directly inside it (sorted by name)."""
    files: list[Path] = []
    for item in raw:
        path = _inside_root(item)
        if path.is_dir():
            files += sorted(p for p in path.iterdir() if _is_image(p))
        else:
            files.append(_image_path(item))
    if not minimum <= len(files) <= MAX_PATHS:
        raise ValueError(f"send {minimum} to {MAX_PATHS} images (got {len(files)})")
    return files


def _write(image: np.ndarray, name: str) -> str:
    out = _root() / OUT_DIR
    out.mkdir(exist_ok=True)
    target = out / f"{SAFE_NAME.sub('_', name).strip('._') or 'result'}.jpg"
    if not cv2.imwrite(str(target), image, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise RuntimeError("could not write the output image")
    return str(target.relative_to(_root()))


def _rel(path: Path) -> str:
    return str(path.relative_to(_root()))


# ---------- tools (plain functions; registered in build_server) ----------

def group_photos(paths: list[str]) -> dict:
    """Split photos into bracket sets (EXIF time, exposure bias, phase correlation). Paths may be image files or a folder."""
    files = _image_paths(paths)
    result = group_frames([load_frame(p, 4) for p in files])
    return {"sets": [[_rel(files[i]) for i in ids] for ids in result.sets],
            "links": [{"left": _rel(files[link.left]), "right": _rel(files[link.right]),
                       "joined": link.joined, "reason": link.reason} for link in result.links]}


def align_bracket(paths: list[str]) -> dict:
    """Align a bracket to its middle frame; residual_px above residual_limit_px means a frame does not fit."""
    files = _image_paths(paths, minimum=2)
    result = align_set([load_image(p) for p in files])
    return {"reference": _rel(files[result.reference_index]), "residual_limit_px": result.residual_limit_px,
            "frames": [{"path": _rel(files[fa.index]), "method": fa.method,
                        "residual_px": None if not np.isfinite(fa.residual_px) else round(fa.residual_px, 3),
                        "ecc_cc": fa.ecc_cc, "note": fa.note} for fa in result.frames]}


def merge_bracket(paths: list[str], engine: str = "mertens", output_name: str = "merged") -> dict:
    """Align and merge (mertens | debevec) with ghost removal; writes the result to lumen-out/."""
    if engine not in ("mertens", "debevec"):
        raise ValueError("engine must be mertens or debevec")
    files = _image_paths(paths, minimum=2)
    frames = [load_frame(p) for p in files]
    aligned = align_set([f.image for f in frames])
    merged = merge_set(aligned.aligned, aligned.reference_index, engine, [f.exif.exposure_time_s for f in frames])
    return {"output": _write(merged.image, output_name), "engine": merged.engine, "ghost_ratio": merged.ghost_ratio,
            "clip_ratio": merged.clip_ratio, "mef_ssim": merged.mef_ssim, "note": merged.note}


def correct_lens(path: str, output_name: str = "lens") -> dict:
    """Self-calibrate radial distortion from straight lines; only writes an image when it helps."""
    result = lens_fix(load_image(_image_path(path)))
    out = _write(result.image, output_name) if result.k1 else None
    return {"output": out, "k1": result.k1, "curvature_before_px": result.curvature_before_px,
            "curvature_after_px": result.curvature_after_px, "pieces": result.pieces, "note": result.note}


def straighten(path: str, output_name: str = "upright") -> dict:
    """Remove roll and keystone from the vertical vanishing point; 'needs review' notes mean ask a person."""
    image = load_image(_image_path(path))
    result = tilt_fix(image)
    out = _write(result.image, output_name) if result.image is not image else None
    return {"output": out, "roll_deg": result.roll_deg, "keystone": result.keystone,
            "tilt_before_deg": result.tilt_before_deg, "tilt_after_deg": result.tilt_after_deg,
            "zoom": result.zoom, "note": result.note}


def qc_image(path: str) -> dict:
    """Quality numbers for one image: clipping, vertical tilt, line curvature, sharpness."""
    image = load_image(_image_path(path))
    grey, _ = _work_copy(image)
    pieces = line_pieces(grey)
    curvature = straightness(pieces, 0.0, _camera(grey.shape, grey.shape[1])) if pieces else None
    return {"clip_ratio": clip_ratio(image), "tilt_deg": tilt_deg(vertical_segments(grey)),
            "line_pieces": len(pieces), "curvature_px": curvature,
            "sharpness": float(cv2.Laplacian(grey, cv2.CV_64F).var()), "size": [image.shape[1], image.shape[0]]}


def run_agent(paths: list[str], output_name: str = "lumen") -> dict:
    """Run Lumen's own see/decide/act agent on one bracket; returns the full trace."""
    files = _image_paths(paths, minimum=2)
    state = run_set([load_frame(p) for p in files])
    out = _write(state.image, output_name) if state.image is not None else None
    trace = trace_dict(state)
    return {"status": state.status, "reason": state.reason, "output": out, "trace": trace["steps"]}


TOOLS = (group_photos, align_bracket, merge_bracket, correct_lens, straighten, qc_image, run_agent)
READ_ONLY = {"group_photos", "align_bracket", "qc_image"}


def _as_tool(fn):
    """Bad input (ValueError) becomes an MCP tool error the calling agent can read and fix."""
    from mcp.server.mcpserver.exceptions import ToolError

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ValueError as exc:
            raise ToolError(str(exc)) from exc
    return wrapped


def build_server():
    from mcp.server.mcpserver import MCPServer
    from mcp.types import ToolAnnotations

    server = MCPServer("lumen", title="theMnexus Lumen", instructions=INSTRUCTIONS)
    for fn in TOOLS:
        server.tool(annotations=ToolAnnotations(readOnlyHint=fn.__name__ in READ_ONLY))(_as_tool(fn))
    return server


def main() -> None:
    build_server().run("stdio")


if __name__ == "__main__":
    main()
