import json
from dataclasses import replace

import cv2
import numpy as np
import pytest

import lumen.agent as agent
from lumen import synth
from lumen.agent import run_folder, run_set, trace_dict
from lumen.io import load_frame
from lumen.tilt import tilt_deg, vertical_segments
from lumen.window import WindowResult

SEED = 5  # seed 3 was used while building the agent


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    out = tmp_path_factory.mktemp("agent")
    truth = synth.generate(out, sets=4, seed=SEED)
    sets = [[load_frame(out / f["file"]) for f in s["frames"]] for s in truth["sets"]]
    return truth, sets


@pytest.fixture(scope="module")
def results(dataset):
    _, sets = dataset
    return [run_set(frames) for frames in sets]


def _tools(state):
    return [(s.tool, s.decision) for s in state.trace]


def test_normal_brackets_finish_ok(results):
    assert all(r.status == "ok" for r in results)


def test_final_images_are_upright(results):
    for r in results:
        grey = cv2.cvtColor(r.image, cv2.COLOR_BGR2GRAY)
        assert tilt_deg(vertical_segments(grey)) < 0.5


def test_lens_step_follows_the_distortion(dataset, results):
    truth, _ = dataset
    for s, r in zip(truth["sets"], results):
        decision = next(st.decision for st in r.trace if st.tool == "correct_lens")
        assert decision == ("accept" if s["lens"]["k1"] != 0 else "skip"), (s["lens"], r.trace)


def test_window_step_runs_and_helps(results):
    for r in results:
        step = next(st for st in r.trace if st.tool == "pull_windows")
        assert step.decision == "accept" and step.result["clip_after"] < 0.05


def test_frame_from_another_room_is_dropped(dataset):
    _, sets = dataset
    frames = [sets[0][0], sets[0][1], sets[2][2]]  # bright frame swapped for another room's
    state = run_set(frames)
    drop = next(st for st in state.trace if st.tool == "drop_frame")
    assert drop.observe["worst_residual_px"] == float("inf")
    assert drop.params["drop_frame"] == 2
    assert state.status == "ok" and len(state.frames) == 2


def test_stranger_in_the_reference_slot_is_dropped(dataset):
    """Middle frame from another room: swap the reference first, then drop the stranger."""
    _, sets = dataset
    state = run_set([sets[0][0], sets[2][1], sets[0][2]])
    tools = [st.tool for st in state.trace]
    assert tools.index("swap_reference") < tools.index("drop_frame")
    drop = next(st for st in state.trace if st.tool == "drop_frame")
    assert drop.params["drop_frame"] == 1
    assert state.status == "ok" and [f.path for f in state.frames] == [sets[0][0].path, sets[0][2].path]


def test_three_strangers_in_six_frames_still_finish(dataset):
    """Each dropped frame costs two steps; the budget grows with the bracket instead of a flat 10."""
    _, sets = dataset
    frames = [sets[0][0], sets[0][1], sets[0][2], sets[1][0], sets[2][1], sets[3][2]]
    state = run_set(frames)
    assert state.status == "ok", (state.reason, [(s.tool, s.decision) for s in state.trace])
    assert len(state.trace) > 10 and len(state.frames) == 3


def test_no_matching_frames_asks_a_person(dataset):
    _, sets = dataset
    state = run_set([sets[0][1], sets[1][1]])
    assert state.status == "needs_review" and state.trace[-1].decision == "review"


def test_failed_window_pull_falls_back_to_debevec(dataset, monkeypatch):
    _, sets = dataset

    def useless_pull(merged, dark, darker=(), recorded=()):
        return WindowResult(merged, 0.04, 0.9, 0.9, 1.0, "forced: no improvement")

    monkeypatch.setattr(agent, "pull_windows", useless_pull)
    state = run_set(sets[0])
    tools = _tools(state)
    assert ("pull_windows", "revert") in tools
    assert ("merge_set", "accept") in tools[tools.index(("pull_windows", "revert")):]
    assert state.merged.engine == "debevec"


def test_window_pull_with_seams_is_reverted(dataset, monkeypatch):
    # Less clipping is not enough: a pull that adds edges no frame recorded made blotches (13/10).
    _, sets = dataset

    def seamy_pull(merged, dark, darker=(), recorded=()):
        return WindowResult(merged, 0.04, 0.9, 0.01, 1.0, "", foreign_edges=0.11)

    monkeypatch.setattr(agent, "pull_windows", seamy_pull)
    tools = _tools(run_set(sets[0]))
    assert ("pull_windows", "revert") in tools


def test_no_window_found_is_a_skip_not_a_failure(dataset, monkeypatch):
    """Real Empa FORTH3: clipping existed but no window to pull; that must not trigger Debevec."""
    _, sets = dataset

    def nothing_to_pull(merged, dark, darker=(), recorded=()):
        return WindowResult(merged, 0.0, 0.0, 0.0, 1.0, "no blown window with dark-frame detail")

    monkeypatch.setattr(agent, "pull_windows", nothing_to_pull)
    state = run_set(sets[0])
    assert ("pull_windows", "skip") in _tools(state)
    assert not any(st.rule == "debevec_fallback" for st in state.trace)
    assert state.merged.engine == "mertens"


def test_debevec_that_loses_structure_is_reverted(dataset, monkeypatch):
    """Less clipping alone is not enough: MEF-SSIM may not drop more than MAX_MEF_SSIM_LOSS."""
    _, sets = dataset
    real_merge = agent.merge_set

    def useless_pull(merged, dark, darker=(), recorded=()):
        return WindowResult(merged, 0.04, 0.9, 0.9, 1.0, "forced: no improvement")

    def poor_debevec(frames, ref, engine="mertens", times=None, deghost=True):
        result = real_merge(frames, ref, engine, times, deghost)
        return replace(result, mef_ssim=result.mef_ssim - 0.2) if engine == "debevec" else result

    monkeypatch.setattr(agent, "pull_windows", useless_pull)
    monkeypatch.setattr(agent, "merge_set", poor_debevec)
    state = run_set(sets[0])
    step = next(st for st in state.trace if st.rule == "debevec_fallback")
    assert step.decision == "revert" and state.merged.engine == "mertens"


def test_straightening_that_measures_worse_is_reverted(dataset, monkeypatch):
    from lumen.tilt import TiltResult
    _, sets = dataset

    def worse(image):
        return TiltResult(image.copy(), 0.5, 0.01, 0.47, 0.95, 1.01, 20, "")

    monkeypatch.setattr(agent, "straighten", worse)
    state = run_set(sets[0])
    step = next(st for st in state.trace if st.tool == "straighten")
    assert step.decision == "revert" and state.status == "ok"
    assert state.image is not None


def test_extreme_pitch_asks_a_person(dataset):
    _, sets = dataset
    f = synth.FOCAL_PX
    k = np.array([[f, 0, synth.WIDTH / 2], [0, f, synth.HEIGHT / 2], [0, 0, 1]])
    a = np.radians(25)
    pitch = k @ np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]]) @ np.linalg.inv(k)
    tilted = [replace(fr, image=cv2.warpPerspective(fr.image, pitch, (synth.WIDTH, synth.HEIGHT),
                                                     borderMode=cv2.BORDER_REPLICATE)) for fr in sets[0]]
    state = run_set(tilted)
    assert state.status == "needs_review" and state.trace[-1].tool == "straighten"


def test_single_frame_is_not_a_bracket(dataset):
    _, sets = dataset
    assert run_set([sets[0][0]]).status == "needs_review"


def test_same_input_same_trace(dataset):
    _, sets = dataset
    assert trace_dict(run_set(sets[1])) == trace_dict(run_set(sets[1]))


def test_trace_is_json(dataset, results):
    _, sets = dataset
    state = run_set([sets[0][0], sets[0][1], sets[2][2]])  # contains an inf residual
    text = json.dumps(trace_dict(state), allow_nan=False)
    assert '"inf"' in text and "steps" in json.loads(text)
    json.dumps(trace_dict(results[0]), allow_nan=False)


def test_folder_groups_then_runs_each_set(dataset):
    _, sets = dataset
    frames = [f for s in sets[:3] for f in s]
    result = run_folder(frames)
    assert len(result.sets) == 3 and all(s.status == "ok" for s in result.sets)
    assert sum(not link["joined"] for link in result.grouping) == 2


def test_turned_off_steps_are_traced_and_not_run(dataset):
    """A person can turn optional steps off at upload; the trace still says so."""
    _, sets = dataset
    state = run_set(sets[0], off={"lens", "tilt", "deghost", "window"})
    steps = {s.tool: s for s in state.trace}
    assert state.status == "ok"
    assert steps["correct_lens"].decision == "off" and steps["straighten"].decision == "off"
    assert steps["merge_set"].params["deghost"] is False
    assert state.image is state.merged.image  # with every optional step off, nothing changes the merge


def test_unknown_step_cannot_be_turned_off(dataset):
    _, sets = dataset
    with pytest.raises(ValueError):
        run_set(sets[0], off={"merge"})
