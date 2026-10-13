"""Archive member names decide where files land: only checked JPEG stack paths are kept."""

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("fetch_cambridge", Path(__file__).parents[1] / "bench" / "fetch_cambridge.py")
fc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fc)


def test_jpeg_stack_members_map_to_scene_folders():
    assert fc.scene_path("exposure_stacks/complex/image_set1/ghosted/jpg/setup1_complex_1.jpg") == \
        ("complex-1", "ghosted", "setup1_complex_1.jpg")
    assert fc.scene_path("x/lolm/image_set4/ground_truth/jpg/a_gt2.JPG") == ("lolm-4", "ground_truth", "a_gt2.JPG")


def test_raw_files_and_unsafe_names_are_skipped():
    for member in ["exposure_stacks/complex/image_set1/ghosted/raw/a.cr2",
                   "exposure_stacks/complex/image_set1/other/jpg/a.jpg",
                   "exposure_stacks/../image_set1/ghosted/jpg/a.jpg",
                   "exposure_stacks/complex/image_set1/ghosted/jpg/._a.jpg",
                   "exposure_stacks/complex/no_number/ghosted/jpg/a.jpg"]:
        assert fc.scene_path(member) is None, member
