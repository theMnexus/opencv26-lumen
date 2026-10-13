from dataclasses import replace

import pytest

from lumen import synth
from lumen.group import exact_match_rate, group_frames
from lumen.io import Exif, load_frame

HELD_OUT_SEED = 2  # thresholds were measured on seed 1


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    out = tmp_path_factory.mktemp("group")
    truth = synth.generate(out, sets=8, seed=HELD_OUT_SEED)
    frames, truth_sets = [], []
    for s in truth["sets"]:
        ids = []
        for f in s["frames"]:
            ids.append(len(frames))
            frames.append(load_frame(out / f["file"]))
        truth_sets.append(ids)
    return frames, truth_sets


def _strip(frames, keep_ev=False):
    return [replace(f, exif=Exif(exposure_bias_ev=f.exif.exposure_bias_ev if keep_ev else None)) for f in frames]


def test_full_exif_groups_all_sets(dataset):
    frames, truth = dataset
    assert exact_match_rate(group_frames(frames).sets, truth) == 1.0


def test_image_only_groups_all_sets(dataset):
    """No time, no EV: phase correlation alone must split the rooms."""
    frames, truth = dataset
    assert exact_match_rate(group_frames(_strip(frames)).sets, truth) == 1.0


def test_no_time_but_ev_groups_all_sets(dataset):
    """Without timestamps, input order is capture order; an EV repeat also splits."""
    frames, truth = dataset
    assert exact_match_rate(group_frames(_strip(frames, keep_ev=True)).sets, truth) == 1.0


def test_sets_are_ordered_dark_to_bright(dataset):
    frames, _ = dataset
    for s in group_frames(frames).sets:
        evs = [frames[i].exif.exposure_bias_ev for i in s]
        assert evs == sorted(evs)


def test_sets_ordered_by_brightness_when_ev_missing(dataset):
    frames, _ = dataset
    stripped = _strip(frames)
    for s in group_frames(stripped).sets:
        means = [stripped[i].image.mean() for i in s]
        assert means == sorted(means)


def test_every_link_has_a_reason(dataset):
    frames, truth = dataset
    links = group_frames(frames).links
    assert len(links) == len(frames) - 1
    assert sum(not link.joined for link in links) == len(truth) - 1
    assert all(link.reason for link in links)


def test_long_exposures_do_not_split_a_bracket(dataset):
    """Museum1-like timing: a 4 s frame, then the next shot starts 8.7 s later."""
    from datetime import datetime, timedelta

    frames, _ = dataset
    start = datetime(2026, 10, 9, 10, 0, 0)
    timeline = [(0.0, 1.0), (3.0, 4.0), (11.7, 8.0)]  # (start offset s, exposure s)
    shots = [replace(frames[i], exif=Exif(taken_at=start + timedelta(seconds=t), exposure_time_s=e,
                                          exposure_bias_ev=float(i)))
             for i, (t, e) in enumerate(timeline)]
    result = group_frames(shots)
    assert result.sets == [[0, 1, 2]], [link.reason for link in result.links]


def test_empty_input():
    result = group_frames([])
    assert result.sets == [] and result.links == []


def test_exact_match_rate():
    assert exact_match_rate([[0, 1], [2]], [[1, 0], [2]]) == 1.0
    assert exact_match_rate([[0, 1, 2]], [[0, 1], [2]]) == 0.0
    assert exact_match_rate([[0, 1], [2, 3]], [[0, 1], [2], [3]]) == pytest.approx(1 / 3)


def _manual(frames, ids, times, start_s=0.0, bias=0.0):
    """Manual-mode bracket: every frame has the same exposure bias, only the shutter time changes."""
    from datetime import datetime, timedelta

    start = datetime(2026, 10, 10, 9, 0, 0)
    out, t = [], start_s
    for i, e in zip(ids, times):
        out.append(replace(frames[i], exif=Exif(taken_at=start + timedelta(seconds=t), exposure_time_s=e,
                                                exposure_bias_ev=bias)))
        t += e + 0.5
    return out


def test_manual_bracket_with_equal_bias_is_one_set(dataset):
    """A Cambridge camera bracket (bias 0.0 on every frame, 1/8 s to 2 s) split into single frames."""
    frames, truth = dataset
    shots = _manual(frames, truth[0], [0.5, 0.125, 2.0])  # shot out of brightness order on purpose
    result = group_frames(shots)
    assert len(result.sets) == 1, [link.reason for link in result.links]
    assert [shots[i].exif.exposure_time_s for i in result.sets[0]] == [0.125, 0.5, 2.0]  # dark to bright


def test_same_bias_and_same_shutter_starts_a_new_set(dataset):
    frames, truth = dataset
    shots = _manual(frames, truth[0][:1] + truth[0][:1], [0.5, 0.5])  # the same exposure taken twice
    assert len(group_frames(shots).sets) == 2
