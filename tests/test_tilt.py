import cv2
import numpy as np
import pytest

from lumen import synth
from lumen.io import load_frame
from lumen.lens import correct_lens
from lumen.tilt import MAX_KEYSTONE, straighten

SEED = 5  # seed 3 was used while building the step
TARGET_DEG = 0.5  # tilt target after straightening


@pytest.fixture(scope="module")
def lens_corrected(tmp_path_factory):
    out = tmp_path_factory.mktemp("tilt")
    truth = synth.generate(out, sets=6, seed=SEED)
    result = []
    for s in truth["sets"]:
        image = load_frame(out / s["frames"][s["reference_index"]]["file"]).image
        result.append((correct_lens(image).image, s["roll_deg"]))
    return result


@pytest.fixture(scope="module")
def straight_room():
    rng = np.random.default_rng(11)
    scene = synth._draw_room(rng)
    maps = synth._source_maps(np.eye(3), synth.Lens(0, 0, synth.FOCAL_PX, synth.WIDTH / 2, synth.HEIGHT / 2), 0.0)
    return synth.expose(cv2.remap(scene, *maps, cv2.INTER_LINEAR), 0.0, rng)


def _pitch(image: np.ndarray, degrees: float) -> np.ndarray:
    f = synth.FOCAL_PX
    k = np.array([[f, 0, synth.WIDTH / 2], [0, f, synth.HEIGHT / 2], [0, 0, 1]])
    a = np.radians(degrees)
    r = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    return cv2.warpPerspective(image, k @ r @ np.linalg.inv(k), (synth.WIDTH, synth.HEIGHT),
                               borderMode=cv2.BORDER_REPLICATE)


def test_roll_matches_truth(lens_corrected):
    for image, roll in lens_corrected:
        result = straighten(image)
        assert abs(result.roll_deg - roll) < 0.3, (roll, result)
        assert result.tilt_after_deg < TARGET_DEG


def test_keystone_from_pitch_is_removed(straight_room):
    result = straighten(_pitch(straight_room, 8.0))
    expected = np.tan(np.radians(8.0)) * max(synth.WIDTH, synth.HEIGHT) / synth.FOCAL_PX
    assert result.keystone == pytest.approx(expected, rel=0.1)
    assert result.tilt_before_deg > 1.5 and result.tilt_after_deg < 0.3
    assert 1.0 < result.zoom < 1.2


def test_extreme_pitch_asks_for_review(straight_room):
    result = straighten(_pitch(straight_room, 25.0))
    assert result.keystone > MAX_KEYSTONE and "needs review" in result.note
    assert result.zoom == 1.0


def test_upright_image_is_left_alone(straight_room):
    result = straighten(straight_room)
    assert result.note == "already upright" and np.array_equal(result.image, straight_room)


def test_tilt_measure_ignores_a_few_slanted_objects():
    """Long upright architecture plus a short slanted chair leg: the median stays near zero."""
    from lumen.tilt import tilt_deg
    upright = [[100.0 * i, 0, 100.0 * i, 400] for i in range(1, 6)]
    slanted = [[50, 0, 50 + 400 * np.tan(np.radians(8)), 400]]
    assert tilt_deg(np.array(upright + slanted)) < 0.01
    assert tilt_deg(np.array(slanted)) == pytest.approx(8.0, abs=0.01)


def test_no_lines_is_left_alone():
    flat = np.full((400, 600, 3), 128, np.uint8)
    result = straighten(flat)
    assert "vertical segments" in result.note and np.array_equal(result.image, flat)
