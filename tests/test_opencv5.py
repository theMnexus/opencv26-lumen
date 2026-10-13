"""The entry must run on OpenCV 5 and every OpenCV function Lumen relies on must exist."""

import cv2

REQUIRED = [
    # group
    "dct", "phaseCorrelate", "resize",
    # align
    "createAlignMTB", "findTransformECCMultiScale", "ECCParameters", "MOTION_HOMOGRAPHY", "SIFT_create",
    "findHomography", "warpPerspective",
    # merge
    "createMergeMertens", "createCalibrateDebevec", "createMergeDebevec",
    "createTonemapReinhard", "createTonemapMantiuk", "createTonemapDrago", "LUT", "dilate",
    # ghost / window
    "absdiff", "threshold", "morphologyEx", "connectedComponentsWithStats",
    "inRange", "GaussianBlur",
    # lens
    "Canny", "findContours", "undistortPoints", "initUndistortRectifyMap",
    "remap", "getOptimalNewCameraMatrix",
    # tilt / qc
    "createLineSegmentDetector", "perspectiveTransform", "pointPolygonTest", "Laplacian",
    "calcHist", "Sobel",
]


def test_opencv_major_version_is_5():
    assert cv2.__version__.split(".")[0] == "5"


def test_required_functions_exist():
    missing = [name for name in REQUIRED if not hasattr(cv2, name)]
    assert missing == []


def test_fisheye_module_exists():
    assert hasattr(cv2.fisheye, "initUndistortRectifyMap")
