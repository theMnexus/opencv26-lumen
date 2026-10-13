import cv2
import numpy as np
import pytest

from lumen import synth
from lumen.align import align_set
from lumen.io import load_frame
from lumen.merge import exposure_lut, ghost_mask, merge_set, remove_ghosts
from lumen.qc import clip_ratio, mef_ssim

SEED = 4  # seed 3 was used while debugging the ghost step


@pytest.fixture(scope="module")
def sets(tmp_path_factory):
    out = tmp_path_factory.mktemp("merge")
    truth = synth.generate(out, sets=4, seed=SEED)
    loaded = []
    for s in truth["sets"]:
        images = [load_frame(out / f["file"]).image for f in s["frames"]]
        aligned = align_set(images, s["reference_index"]).aligned
        loaded.append((s, aligned))
    return loaded


def _truth_ghost(s) -> np.ndarray:
    mask = np.zeros((synth.HEIGHT, synth.WIDTH), np.uint8)
    for f in s["frames"]:
        if f["ghost_bbox"]:
            x, y, w, h = f["ghost_bbox"]
            cv2.ellipse(mask, (x + w // 2, y + h // 2), (w // 2, h // 2), 0, 0, 360, 255, -1)
    return mask > 0


def _union_mask(s, aligned) -> np.ndarray:
    ref = s["reference_index"]
    masks = [ghost_mask(a, aligned[ref]) > 0 for i, a in enumerate(aligned) if i != ref]
    return np.logical_or.reduce(masks)


def test_fusion_beats_single_frame(sets):
    s, aligned = sets[0]
    result = merge_set(aligned, s["reference_index"])
    assert result.image.shape == aligned[0].shape and result.image.dtype == np.uint8
    assert result.mef_ssim > mef_ssim(aligned, aligned[s["reference_index"]]) + 0.01


def test_debevec_keeps_window_detail(sets):
    s, aligned = sets[0]
    times = [f["exposure_time_s"] for f in s["frames"]]
    mertens = merge_set(aligned, s["reference_index"])
    debevec = merge_set(aligned, s["reference_index"], "debevec", times)
    assert debevec.engine == "debevec"
    assert debevec.clip_ratio < mertens.clip_ratio


def test_debevec_without_times_falls_back(sets):
    s, aligned = sets[0]
    result = merge_set(aligned, s["reference_index"], "debevec", [None] * len(aligned))
    assert result.engine == "mertens" and "ExposureTime" in result.note


def test_unknown_engine_raises(sets):
    s, aligned = sets[0]
    with pytest.raises(ValueError):
        merge_set(aligned, s["reference_index"], "magic")


def test_no_ghost_on_still_scenes(sets):
    for s, aligned in sets[0::2]:  # even sets have no moving object
        assert _union_mask(s, aligned).mean() < 0.003  # quantile LUT: <= 0.24 % measured


def test_moving_object_is_found_and_removed(sets):
    for s, aligned in sets[1::2]:
        truth = _truth_ghost(s)
        found = _union_mask(s, aligned)
        assert (found & truth).sum() / truth.sum() >= 0.6  # recall
        far = ~(cv2.dilate(truth.astype(np.uint8), np.ones((25, 25), np.uint8)) > 0)
        assert (found & far).mean() < 0.003  # false alarms (quantile LUT: <= 0.24 % measured)
        cleaned, ratio = remove_ghosts(aligned, s["reference_index"])
        assert ratio > 0.01
        assert _union_mask(s, cleaned).mean() < 0.01


def test_exposure_lut_recovers_known_ratio():
    rng = np.random.default_rng(0)
    ref = rng.integers(30, 200, (64, 64, 3), dtype=np.uint8)
    lin = (ref / 255.0) ** 2.2
    brighter = np.clip(255 * np.clip(lin * 2.5, 0, 1) ** (1 / 2.2), 0, 255).round().astype(np.uint8)
    lut = exposure_lut(brighter, ref)
    mapped = np.stack([lut[c][brighter[..., c]] for c in range(3)], axis=2).astype(int)
    ok = brighter < 243
    assert np.abs(mapped - ref.astype(int))[ok].max() <= 2


def test_mef_ssim_range():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
    img = cv2.GaussianBlur(img, (0, 0), 3)
    noise = rng.integers(0, 255, img.shape, dtype=np.uint8)
    assert mef_ssim([img], img) == pytest.approx(1.0, abs=1e-6)
    assert mef_ssim([img], noise) < 0.3


def test_clip_ratio():
    img = np.full((10, 10, 3), 128, np.uint8)
    img[:2] = 255
    img[-1] = 0
    assert clip_ratio(img) == pytest.approx(0.3)
