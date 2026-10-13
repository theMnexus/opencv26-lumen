import numpy as np
import pytest

from lumen import synth
from lumen.io import load_frame
from lumen.lens import _camera, _work_copy, correct_lens, line_pieces, straightness

SEED = 5  # seeds 3-4 were used to choose the cost function and MIN_GAIN


@pytest.fixture(scope="module")
def references(tmp_path_factory):
    out = tmp_path_factory.mktemp("lens")
    truth = synth.generate(out, sets=8, seed=SEED)
    return [(load_frame(out / s["frames"][s["reference_index"]]["file"]).image, s["lens"]) for s in truth["sets"]]


def _plumb(image: np.ndarray) -> float:
    grey, _ = _work_copy(image)
    return straightness(line_pieces(grey), 0.0, _camera(grey.shape, grey.shape[1]))


def test_k1_within_10_percent_when_focal_is_known(references):
    for image, lens in references:
        result = correct_lens(image, focal_px=lens["fx"])
        assert abs(result.k1 - lens["k1"]) <= max(0.1 * abs(lens["k1"]), 0.01), (lens["k1"], result)


def test_corrected_image_has_straighter_lines(references):
    image, lens = next((im, ln) for im, ln in references if ln["k1"] <= -0.15)
    result = correct_lens(image)  # focal guessed as image width
    assert result.k1 < 0
    assert result.curvature_after_px < 0.6 * result.curvature_before_px
    assert _plumb(result.image) < 0.6 * _plumb(image)


def test_focal_guess_gives_nearly_the_same_straightness(references):
    image, lens = next((im, ln) for im, ln in references if ln["k1"] <= -0.15)
    known = correct_lens(image, focal_px=lens["fx"])
    guessed = correct_lens(image)
    assert guessed.curvature_after_px <= known.curvature_after_px * 1.2


def test_straight_scene_is_left_alone(tmp_path):
    truth = synth.generate(tmp_path, sets=8, seed=6)
    straight = [s for s in truth["sets"] if s["lens"]["k1"] == 0.0]
    assert straight, "seed 6 should contain an undistorted set"
    s = straight[0]
    image = load_frame(tmp_path / s["frames"][s["reference_index"]]["file"]).image
    result = correct_lens(image)
    assert result.k1 == 0.0 and np.array_equal(result.image, image) and "not corrected" in result.note


def test_no_lines_is_left_alone():
    rng = np.random.default_rng(0)
    blurry = (rng.random((400, 600, 3)) * 30 + 100).astype(np.uint8)
    result = correct_lens(blurry)
    assert result.k1 == 0.0 and np.array_equal(result.image, blurry) and "pieces" in result.note
