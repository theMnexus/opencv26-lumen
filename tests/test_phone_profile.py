"""The 'phone' camera profile breaks the exact-gamma assumption: S tone curve,
noise sigma 5, JPEG 75, exposure off by up to 5 %. These tests keep the fixes
for it (gradient pre-blur, ECC cc limit 0.8) from regressing."""

import numpy as np
import pytest

from lumen import synth
from lumen.agent import run_set
from lumen.align import align_set
from lumen.io import load_frame
from lumen.merge import ghost_mask
from lumen.qc import homography_error_px

SEED = 6


@pytest.fixture(scope="module")
def phone(tmp_path_factory):
    out = tmp_path_factory.mktemp("phone")
    truth = synth.generate(out, sets=4, seed=SEED, profile="phone")
    return truth, [[load_frame(out / f["file"]) for f in s["frames"]] for s in truth["sets"]]


def test_profile_is_recorded(phone):
    truth, _ = phone
    assert truth["profile"] == "phone"


def test_noisy_dark_frames_still_use_ecc(phone):
    truth, sets = phone
    for s, frames in zip(truth["sets"], sets):
        result = align_set([f.image for f in frames], s["reference_index"])
        for fa, f in zip(result.frames, s["frames"]):
            assert fa.method in ("reference", "ecc"), fa
            assert homography_error_px(fa.homography, np.array(f["homography"]), synth.WIDTH, synth.HEIGHT) < 1.0


def test_no_false_ghost_on_still_phone_sets(phone):
    truth, sets = phone
    for s, frames in list(zip(truth["sets"], sets))[0::2]:
        aligned = align_set([f.image for f in frames], s["reference_index"]).aligned
        ref = s["reference_index"]
        flagged = max(ghost_mask(a, aligned[ref]).mean() / 255 for i, a in enumerate(aligned) if i != ref)
        assert flagged < 0.003


def test_agent_drops_a_stranger_on_phone_images(phone):
    _, sets = phone
    state = run_set([sets[0][0], sets[0][1], sets[2][2]])
    assert state.status == "ok" and any(st.tool == "drop_frame" for st in state.trace)
