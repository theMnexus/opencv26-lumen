import json

import cv2
import numpy as np
import pytest

from lumen import synth
from lumen.io import load_frame


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    out = tmp_path_factory.mktemp("synth")
    truth = synth.generate(out, sets=2, seed=7)
    return out, truth


def test_writes_frames_and_truth(generated):
    out, truth = generated
    on_disk = json.loads((out / "truth.json").read_text())
    assert on_disk == truth
    assert len(truth["sets"]) == 2
    for s in truth["sets"]:
        assert len(s["frames"]) == len(synth.EVS)
        for f in s["frames"]:
            assert (out / f["file"]).exists()


def test_exif_matches_truth(generated):
    out, truth = generated
    f = truth["sets"][1]["frames"][0]
    exif = load_frame(out / f["file"]).exif
    assert exif.exposure_bias_ev == pytest.approx(f["ev"])
    assert exif.exposure_time_s == pytest.approx(f["exposure_time_s"], rel=1e-3)
    assert exif.taken_at.isoformat(timespec="microseconds") == f["taken_at"]


def test_frames_get_brighter_with_ev(generated):
    out, truth = generated
    means = [load_frame(out / f["file"]).image.mean() for f in truth["sets"][0]["frames"]]
    assert means == sorted(means)


def test_reference_frame_has_identity_homography(generated):
    _, truth = generated
    for s in truth["sets"]:
        ref = s["frames"][s["reference_index"]]
        assert np.allclose(ref["homography"], np.eye(3))


def test_ghost_only_on_odd_sets(generated):
    _, truth = generated
    assert all(f["ghost_bbox"] is None for f in truth["sets"][0]["frames"])
    boxes = [f["ghost_bbox"] for f in truth["sets"][1]["frames"]]
    assert all(boxes) and len({b[0] for b in boxes}) == len(boxes)  # object moves


def test_shake_homography_is_exact():
    """Warping the reference by the truth homography reproduces the shaken frame."""
    rng = np.random.default_rng(3)
    scene = synth._draw_room(rng)
    lens = synth.Lens(0.0, 0.0, synth.FOCAL_PX, synth.WIDTH / 2, synth.HEIGHT / 2)
    shake = synth._random_shake(rng)
    ref = cv2.remap(scene, *synth._source_maps(np.eye(3), lens, 0.0), cv2.INTER_LINEAR)
    moved = cv2.remap(scene, *synth._source_maps(shake, lens, 0.0), cv2.INTER_LINEAR)
    predicted = cv2.warpPerspective(ref, shake, (synth.WIDTH, synth.HEIGHT))
    inner = (slice(40, -40), slice(40, -40))
    assert np.abs(predicted[inner] - moved[inner]).mean() < 0.01


def test_lens_truth_undistorts_back():
    """initUndistortRectifyMap with the truth k1 recovers the straight scene."""
    rng = np.random.default_rng(5)
    scene = synth._draw_room(rng)
    lens = synth.Lens(-0.25, 0.0, synth.FOCAL_PX, synth.WIDTH / 2, synth.HEIGHT / 2)
    no_lens = synth.Lens(0.0, 0.0, lens.fx, lens.cx, lens.cy)
    straight = cv2.remap(scene, *synth._source_maps(np.eye(3), no_lens, 0.0), cv2.INTER_LINEAR)
    bent = cv2.remap(scene, *synth._source_maps(np.eye(3), lens, 0.0), cv2.INTER_LINEAR)
    k = lens.camera_matrix()
    mx, my = cv2.initUndistortRectifyMap(k, lens.dist_coeffs(), None, k, (synth.WIDTH, synth.HEIGHT), cv2.CV_32FC1)
    fixed = cv2.remap(bent, mx, my, cv2.INTER_LINEAR)
    inner = (slice(150, -150), slice(200, -200))
    before = np.abs(bent[inner] - straight[inner]).mean()
    after = np.abs(fixed[inner] - straight[inner]).mean()
    assert after < before * 0.2


def test_dark_frame_shows_window_detail(generated):
    """The window clips at ev 0 but keeps detail at ev -2 (needed by the window step)."""
    out, truth = generated
    frames = truth["sets"][0]["frames"]
    dark = load_frame(out / frames[0]["file"]).image
    mid = load_frame(out / frames[1]["file"]).image
    clipped_mid = (mid.min(axis=2) >= 250).mean()  # all channels blown = no detail left
    clipped_dark = (dark.max(axis=2) >= 250).mean()
    assert clipped_mid > 0.03
    assert clipped_dark < clipped_mid * 0.1
