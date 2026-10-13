"""Synthetic bracket sets with known answers (truth.json).

Each set is one procedurally drawn room (no third-party images). Faults are
applied with known parameters so every pipeline step can be scored:

  scene radiance (+ moving ghost object per frame)
    -> camera roll  (roll_deg, cv2.getRotationMatrix2D sign: + = counter-clockwise)
    -> lens         (k1, k2 on K = [[fx,0,cx],[0,fx,cy],[0,0,1]])
    -> camera shake (homography: reference-frame pixel -> this-frame pixel)
    -> exposure     (ev, gamma 2.2, clip) -> noise -> JPEG with EXIF

Run:  uv run python -m lumen.synth out/synth --sets 6 --seed 0 [--profile phone]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from PIL.TiffImagePlugin import IFDRational

WIDTH, HEIGHT = 1200, 800
FOCAL_PX = 0.8 * WIDTH  # wide-ish interior lens
EVS = (-2.0, 0.0, 2.0)
BASE_EXPOSURE_S = 1 / 60
EXPOSURE_GAIN = 0.7  # wall radiance 0.25 -> mid-grey at ev 0
NOISE_SIGMA = 1.5  # uint8 levels
JPEG_QUALITY = 92
FRAME_GAP_S = 0.3  # within a bracket
SET_GAP_S = 40.0  # between brackets
START_TIME = datetime(2026, 10, 9, 10, 0, 0)
UNDISTORT_CRITERIA = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-9)


@dataclass(frozen=True)
class Lens:
    k1: float
    k2: float
    fx: float
    cx: float
    cy: float

    def camera_matrix(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fx, self.cy], [0, 0, 1]], np.float64)

    def dist_coeffs(self) -> np.ndarray:
        return np.array([self.k1, self.k2, 0, 0, 0], np.float64)


@dataclass(frozen=True)
class FrameTruth:
    file: str
    ev: float
    exposure_time_s: float
    taken_at: str
    homography: list[list[float]]
    ghost_bbox: list[int] | None


@dataclass(frozen=True)
class SetTruth:
    set_id: str
    reference_index: int
    roll_deg: float
    lens: Lens
    frames: list[FrameTruth]


# ---------- scene ----------

def _smooth_noise(rng: np.random.Generator, shape: tuple[int, int], scale: int) -> np.ndarray:
    small = rng.random((shape[0] // scale + 2, shape[1] // scale + 2)).astype(np.float32)
    big = cv2.resize(small, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)
    return big - big.mean()


def _draw_window(rad: np.ndarray, rng: np.random.Generator, x: int, y: int, w: int, h: int) -> None:
    # Outdoor radiance: clips at ev 0 (> 1 / EXPOSURE_GAIN = 1.43) but stays
    # below clip at ev -2 (< 4 / EXPOSURE_GAIN = 5.7), so only the dark frame
    # keeps the sky and tree detail.
    sky = np.linspace(5.0, 3.0, h, dtype=np.float32)[:, None, None] * np.array([1.1, 1.0, 0.85], np.float32)
    rad[y:y + h, x:x + w] = sky
    for _ in range(int(rng.integers(4, 9))):  # trees
        cx, cy = int(rng.integers(x, x + w)), int(rng.integers(y + h // 3, y + h))
        r = int(rng.integers(h // 10, h // 4))
        tree = rad[y:y + h, x:x + w].copy()
        cv2.circle(tree, (cx - x, cy - y), r, (1.6, 2.4, 1.8), -1)
        rad[y:y + h, x:x + w] = tree
    bar = (0.2, 0.2, 0.2)
    cv2.rectangle(rad, (x, y), (x + w, y + h), bar, 6)
    cv2.line(rad, (x + w // 2, y), (x + w // 2, y + h), bar, 5)
    cv2.line(rad, (x, y + h // 2), (x + w, y + h // 2), bar, 5)


def _draw_room(rng: np.random.Generator) -> np.ndarray:
    """Float32 BGR radiance; random layout per seed so sets look different."""
    rad = np.zeros((HEIGHT, WIDTH, 3), np.float32)
    left, right = int(rng.integers(150, 300)), int(rng.integers(900, 1050))
    top, floor = int(rng.integers(120, 200)), int(rng.integers(560, 660))
    wall = rng.uniform(0.18, 0.32, 3).astype(np.float32)
    rad[:] = wall
    rad[:top] = wall * 1.4  # ceiling
    rad[floor:] = rng.uniform(0.08, 0.15, 3)  # floor
    rad[:, :left] *= 0.75
    rad[:, right:] *= 0.85
    for x in (left, right):  # room corners: long verticals for tilt/lens
        cv2.line(rad, (x, 0), (x, HEIGHT), (0.1, 0.1, 0.1), 3)
    cv2.line(rad, (0, top), (WIDTH, top), (0.12, 0.12, 0.12), 3)
    cv2.line(rad, (0, floor), (WIDTH, floor), (0.06, 0.06, 0.06), 3)
    wx = int(rng.integers(left + 40, right - 380))
    _draw_window(rad, rng, wx, top + 60, int(rng.integers(220, 340)), int(rng.integers(200, 280)))
    for _ in range(int(rng.integers(3, 6))):  # furniture and frames: features for ECC/SIFT
        x, y = int(rng.integers(0, WIDTH - 200)), int(rng.integers(top, HEIGHT - 120))
        w, h = int(rng.integers(60, 200)), int(rng.integers(40, 140))
        color = tuple(float(c) for c in rng.uniform(0.05, 0.5, 3))
        cv2.rectangle(rad, (x, y), (x + w, y + h), color, -1)
        cv2.rectangle(rad, (x, y), (x + w, y + h), (0.03, 0.03, 0.03), 2)
    texture = 1 + 0.12 * _smooth_noise(rng, (HEIGHT, WIDTH), 40) + 0.05 * _smooth_noise(rng, (HEIGHT, WIDTH), 6)
    return rad * texture[..., None]


def _ghost_bbox(rng_x0: int, index: int) -> list[int]:
    w, h = 90, 260
    return [rng_x0 + index * 70, HEIGHT - h - 60, w, h]


def _draw_ghost(rad: np.ndarray, bbox: list[int]) -> np.ndarray:
    out = rad.copy()
    x, y, w, h = bbox
    cv2.ellipse(out, (x + w // 2, y + h // 2), (w // 2, h // 2), 0, 0, 360, (0.05, 0.08, 0.35), -1)
    return out


# ---------- geometry ----------

def _random_shake(rng: np.random.Generator) -> np.ndarray:
    angle = np.deg2rad(rng.uniform(-0.4, 0.4))
    tx, ty = rng.uniform(-6, 6, 2)
    c, s = np.cos(angle), np.sin(angle)
    cx, cy = WIDTH / 2, HEIGHT / 2
    rot = np.array([[c, -s, cx - c * cx + s * cy], [s, c, cy - s * cx - c * cy], [0, 0, 1]])
    persp = np.array([[1, 0, 0], [0, 1, 0], rng.uniform(-2e-6, 2e-6, 3) * [1, 1, 0] + [0, 0, 1]])
    return np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]]) @ rot @ persp


def _source_maps(homography: np.ndarray, lens: Lens, roll_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """For every output pixel, where to sample the undistorted, unrolled scene."""
    xs, ys = np.meshgrid(np.arange(WIDTH, dtype=np.float64), np.arange(HEIGHT, dtype=np.float64))
    pts = np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)])
    pre_shake = np.linalg.inv(homography) @ pts
    pre_shake = (pre_shake[:2] / pre_shake[2]).T.reshape(-1, 1, 2)
    if lens.k1 or lens.k2:
        k = lens.camera_matrix()
        pre_shake = cv2.undistortPoints(pre_shake, k, lens.dist_coeffs(), P=k, criteria=UNDISTORT_CRITERIA)
    pre_lens = pre_shake.reshape(-1, 2)
    roll = cv2.getRotationMatrix2D((WIDTH / 2, HEIGHT / 2), roll_deg, 1.0)
    inv_roll = cv2.invertAffineTransform(roll)
    scene = pre_lens @ inv_roll[:, :2].T + inv_roll[:, 2]
    return (scene[:, 0].reshape(HEIGHT, WIDTH).astype(np.float32),
            scene[:, 1].reshape(HEIGHT, WIDTH).astype(np.float32))


# ---------- exposure and output ----------

@dataclass(frozen=True)
class Profile:
    """How the simulated camera turns radiance into a JPEG.

    clean: exact gamma 2.2, low noise. It matches the sRGB assumption inside
           merge/window, so it is the easy case.
    phone: per-channel S tone curve (like a phone's "look"), stronger noise,
           heavier JPEG, and real exposures off by up to 5 % from the EXIF value.
    """
    s_curve: tuple[float, float, float] = (0.0, 0.0, 0.0)  # B, G, R blend toward smoothstep
    noise_sigma: float = NOISE_SIGMA
    jpeg_quality: int = JPEG_QUALITY
    exposure_jitter: float = 0.0


PROFILES = {
    "clean": Profile(),
    "phone": Profile(s_curve=(0.30, 0.38, 0.45), noise_sigma=5.0, jpeg_quality=75, exposure_jitter=0.05),
}


def expose(radiance: np.ndarray, ev: float, rng: np.random.Generator,
           profile: Profile = PROFILES["clean"]) -> np.ndarray:
    factor = 2.0 ** ev
    if profile.exposure_jitter:  # no extra draw for "clean": keeps its images identical to before
        factor *= 1 + rng.uniform(-profile.exposure_jitter, profile.exposure_jitter)
    tone = np.clip(radiance * EXPOSURE_GAIN * factor, 0, 1) ** (1 / 2.2)
    if any(profile.s_curve):
        s = tone * tone * (3 - 2 * tone)
        amount = np.asarray(profile.s_curve, np.float32)
        tone = (1 - amount) * tone + amount * s
    ldr = tone * 255 + rng.normal(0, profile.noise_sigma, radiance.shape)
    return np.clip(ldr, 0, 255).astype(np.uint8)


def _save_jpeg(path: Path, bgr: np.ndarray, ev: float, exposure_s: float, taken_at: datetime,
               quality: int = JPEG_QUALITY) -> None:
    exif = Image.Exif()
    exif[271], exif[272] = "theMnexus", "Lumen Synth"
    sub = exif.get_ifd(0x8769)
    sub[36867] = taken_at.strftime("%Y:%m:%d %H:%M:%S")
    sub[37521] = f"{taken_at.microsecond:06d}"
    sub[33434] = IFDRational(Fraction(exposure_s).limit_denominator(100000))
    sub[37380] = IFDRational(Fraction(ev).limit_denominator(100))
    sub[37386] = IFDRational(Fraction(24, 1))
    Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).save(path, "JPEG", quality=quality, exif=exif)


def make_set(out_dir: Path, set_index: int, seed: int, profile: Profile = PROFILES["clean"]) -> SetTruth:
    rng = np.random.default_rng(seed * 1000 + set_index)
    scene = _draw_room(rng)
    lens = Lens(k1=float(rng.choice([0.0, -0.08, -0.15, -0.25])), k2=0.0, fx=FOCAL_PX, cx=WIDTH / 2, cy=HEIGHT / 2)
    roll_deg = float(rng.uniform(-3, 3))
    has_ghost = bool(set_index % 2)
    ghost_x0 = int(rng.integers(100, WIDTH - 400))
    reference = len(EVS) // 2
    set_id = f"set{set_index:02d}"
    frames = []
    for i, ev in enumerate(EVS):
        shake = np.eye(3) if i == reference else _random_shake(rng)
        bbox = _ghost_bbox(ghost_x0, i) if has_ghost else None
        rad = _draw_ghost(scene, bbox) if bbox else scene
        map_x, map_y = _source_maps(shake, lens, roll_deg)
        warped = cv2.remap(rad, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        exposure_s = BASE_EXPOSURE_S * 2.0 ** ev
        taken_at = START_TIME + timedelta(seconds=set_index * SET_GAP_S + i * FRAME_GAP_S)
        name = f"{set_id}_f{i}.jpg"
        _save_jpeg(out_dir / name, expose(warped, ev, rng, profile), ev, exposure_s, taken_at, profile.jpeg_quality)
        frames.append(FrameTruth(name, ev, exposure_s, taken_at.isoformat(timespec="microseconds"),
                                 shake.tolist(), bbox))
    return SetTruth(set_id, reference, roll_deg, lens, frames)


def generate(out_dir: str | Path, sets: int = 6, seed: int = 0, profile: str = "clean") -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    camera = PROFILES[profile]
    truth = {"version": 1, "seed": seed, "profile": profile, "width": WIDTH, "height": HEIGHT,
             "sets": [asdict(make_set(out, i, seed, camera)) for i in range(sets)]}
    (out / "truth.json").write_text(json.dumps(truth, indent=2))
    return truth


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic bracket sets with truth.json")
    parser.add_argument("out_dir")
    parser.add_argument("--sets", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="clean")
    args = parser.parse_args()
    truth = generate(args.out_dir, args.sets, args.seed, args.profile)
    print(f"wrote {sum(len(s['frames']) for s in truth['sets'])} frames + truth.json to {args.out_dir}")


if __name__ == "__main__":
    main()
