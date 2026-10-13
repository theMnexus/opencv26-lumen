import cv2
import numpy as np
import pytest

from lumen import synth
from lumen.align import MAX_RESIDUAL_PX, align_pair, align_set
from lumen.io import load_frame
from lumen.qc import homography_error_px

SEED = 3
TARGET_ERROR_PX = 1.0  # alignment target on synthetic brackets


@pytest.fixture(scope="module")
def sets(tmp_path_factory):
    out = tmp_path_factory.mktemp("align")
    truth = synth.generate(out, sets=6, seed=SEED)
    loaded = []
    for s in truth["sets"]:
        images = [load_frame(out / f["file"]).image for f in s["frames"]]
        truths = [np.array(f["homography"]) for f in s["frames"]]
        loaded.append((s, images, truths))
    return loaded


@pytest.fixture(scope="module")
def results(sets):
    return [align_set(images, s["reference_index"]) for s, images, _ in sets]


def test_homography_matches_truth(sets, results):
    errors = [
        homography_error_px(fa.homography, truth, synth.WIDTH, synth.HEIGHT)
        for (_, _, truths), result in zip(sets, results)
        for fa, truth in zip(result.frames, truths)
    ]
    assert max(errors) < TARGET_ERROR_PX, errors


def test_ecc_is_used_on_normal_brackets(results):
    methods = {fa.method for r in results for fa in r.frames} - {"reference"}
    assert methods == {"ecc"}


def test_residual_is_small_without_truth(results):
    """Ghost sets included: the residual must not be fooled by a moving object."""
    assert all(r.worst_residual_px <= r.residual_limit_px for r in results)


def test_high_resolution_uses_a_scaled_limit(sets):
    """4x upscale (~15 MP): the same brackets must still pass, error scales with size."""
    s, images, truths = sets[1]
    factor = 4
    big = [cv2.resize(im, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC) for im in images]
    result = align_set(big, s["reference_index"])
    assert result.residual_limit_px > MAX_RESIDUAL_PX
    assert result.worst_residual_px <= result.residual_limit_px
    scale = np.diag([factor, factor, 1.0])
    for fa, truth in zip(result.frames, truths):
        big_truth = scale @ truth @ np.linalg.inv(scale)
        err = homography_error_px(fa.homography, big_truth, synth.WIDTH * factor, synth.HEIGHT * factor)
        assert err < TARGET_ERROR_PX * factor


def test_reference_is_identity_and_shapes_match(sets, results):
    _, images, _ = sets[0]
    ref = results[0].frames[results[0].reference_index]
    assert ref.method == "reference" and np.allclose(ref.homography, np.eye(3))
    assert all(a.shape == images[0].shape for a in results[0].aligned)


def test_big_motion_still_aligns(sets):
    """Extra 30 px shift + 1 degree on top of the dark frame's own shake."""
    s, images, truths = sets[0]
    angle = np.deg2rad(1.0)
    c, sn = np.cos(angle), np.sin(angle)
    extra = np.array([[c, -sn, 30.0], [sn, c, -18.0], [0, 0, 1]])
    moved = cv2.warpPerspective(images[0], extra, (synth.WIDTH, synth.HEIGHT), borderMode=cv2.BORDER_REPLICATE)
    truth = extra @ truths[0]  # reference -> dark frame -> moved
    result = align_set([images[s["reference_index"]], moved], reference_index=0)
    err = homography_error_px(result.frames[1].homography, truth, synth.WIDTH, synth.HEIGHT)
    assert err < 2 * TARGET_ERROR_PX, (err, result.frames[1])


def test_sift_takes_over_when_ecc_fails(sets, monkeypatch):
    import lumen.align as align_mod

    def broken_ecc(*_args):
        raise cv2.error("forced: ECC did not converge")

    monkeypatch.setattr(align_mod, "_ecc", broken_ecc)
    s, images, truths = sets[0]
    result = align_set(images, s["reference_index"])
    for fa, truth in zip(result.frames, truths):
        if fa.method == "reference":
            continue
        assert fa.method == "sift" and "ecc failed" in fa.note
        assert homography_error_px(fa.homography, truth, synth.WIDTH, synth.HEIGHT) < 2 * TARGET_ERROR_PX


def test_different_rooms_report_bad_residual(sets):
    """Two different rooms must not look aligned: the agent needs that signal."""
    ref = cv2.cvtColor(sets[0][1][1], cv2.COLOR_BGR2GRAY)
    other = cv2.cvtColor(sets[1][1][1], cv2.COLOR_BGR2GRAY)
    fa = align_pair(ref, other, 1.0, 1)
    assert fa.residual_px > MAX_RESIDUAL_PX
    assert fa.note  # the trace says why
