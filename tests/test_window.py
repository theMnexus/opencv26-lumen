import cv2
import numpy as np
import pytest

from lumen import synth
from lumen.align import align_set
from lumen.io import load_frame
from lumen.merge import merge_set
from lumen.qc import foreign_edge_share
from lumen.window import pull_windows, window_mask

SEED = 5  # seed 3 was used while building the step


@pytest.fixture(scope="module")
def merged_sets(tmp_path_factory):
    out = tmp_path_factory.mktemp("window")
    truth = synth.generate(out, sets=3, seed=SEED)
    result = []
    for s in truth["sets"]:
        images = [load_frame(out / f["file"]).image for f in s["frames"]]
        aligned = align_set(images, s["reference_index"]).aligned
        result.append((merge_set(aligned, s["reference_index"]).image, aligned[0]))
    return result


def test_window_clipping_drops(merged_sets):
    for merged, dark in merged_sets:
        result = pull_windows(merged, dark)
        assert result.window_ratio > 0.01
        assert result.clip_before > 0.5
        assert result.clip_after < 0.05


def test_pixels_outside_window_unchanged(merged_sets):
    merged, dark = merged_sets[0]
    result = pull_windows(merged, dark)
    near = cv2.dilate(window_mask(merged, dark), np.ones((15, 15), np.uint8)) > 0
    diff = np.abs(result.image.astype(int) - merged.astype(int)).max(axis=2)
    assert diff[~near].max() <= 1


def test_window_gains_detail(merged_sets):
    merged, dark = merged_sets[0]
    result = pull_windows(merged, dark)
    # Only the pixels that were blown: the mullions inside the region already vary.
    blown = (window_mask(merged, dark) > 0) & (merged.min(axis=2) >= 245)
    grey = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)[blown].astype(float)  # noqa: E731
    assert grey(merged).std() < 3
    assert grey(result.image).std() > 8


def test_no_blown_region_is_a_no_op():
    img = np.full((200, 300, 3), 120, np.uint8)
    result = pull_windows(img, img // 4)
    assert result.window_ratio == 0 and np.array_equal(result.image, img) and result.note


def test_dark_frame_also_blown_is_skipped():
    merged = np.full((200, 300, 3), 120, np.uint8)
    merged[50:150, 100:200] = 255
    result = pull_windows(merged, merged.copy())
    assert result.window_ratio == 0 and np.array_equal(result.image, merged)


def test_unblown_pixels_inside_window_come_from_the_fill():
    # Branches in front of a blown sky are not blown themselves. Gating each pixel on
    # merged luminance kept them from the merged image while the sky came from the dark
    # frame: tree-shaped blotches on a real room (13/10). The whole region is one source now.
    merged = np.full((200, 300, 3), 120, np.uint8)
    merged[40:160, 60:240] = 255
    merged[90:110, 60:240] = 215  # a bright branch across the window, not blown
    darker = [np.full_like(merged, 40), np.full_like(merged, 90)]
    result = pull_windows(merged, darker[0], darker)
    branch, sky = result.image[100, 150].astype(int), result.image[60, 150].astype(int)
    assert abs(branch - sky).max() <= 2
    assert result.gain == 1.0


def test_seam_between_two_sources_counts_as_foreign_edges():
    rng = np.random.default_rng(0)
    frame = np.full((200, 300, 3), 200, np.uint8)
    frame[:, 150:] = 60  # a real edge every frame has
    frames = [frame, (frame * 0.5).astype(np.uint8)]
    mask = np.zeros((200, 300), np.uint8)
    mask[20:180, 20:280] = 255
    assert foreign_edge_share(frame, frames, mask) == 0.0
    seamy = frame.copy()
    seamy[60:140, 40:110] = 90  # a blotch no frame recorded
    seamy = np.clip(seamy + rng.normal(0, 1, seamy.shape), 0, 255).astype(np.uint8)
    assert foreign_edge_share(seamy, frames, mask) > 0.3


def test_window_result_reports_foreign_edges(merged_sets):
    merged, dark = merged_sets[0]
    assert pull_windows(merged, dark).foreign_edges <= 0.05
