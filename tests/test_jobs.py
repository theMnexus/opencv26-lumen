

def test_sets_shrink_to_the_lambda_memory_budget(tmp_path, monkeypatch):
    """A 3008 MB Lambda ran out of memory on 7 x 16 MP; frames must shrink to fit, evenly."""
    import cv2
    import numpy as np

    from lumen.jobs import load_set, set_megapixel_budget

    paths = []
    for i in range(4):
        p = tmp_path / f"f{i}.jpg"
        cv2.imwrite(str(p), np.full((2000, 3000, 3), 40 * i, np.uint8))  # 6 MP each
        paths.append(p)

    monkeypatch.delenv("AWS_LAMBDA_FUNCTION_MEMORY_SIZE", raising=False)
    assert set_megapixel_budget() is None
    assert all(f.image.shape[:2] == (2000, 3000) for f in load_set(paths))  # local: untouched

    monkeypatch.setenv("AWS_LAMBDA_FUNCTION_MEMORY_SIZE", "1800")  # (1800 - 400) / 70 = 20 MP x frames
    frames = load_set(paths)
    sizes = {f.image.shape for f in frames}
    assert len(sizes) == 1  # every frame the same size, so alignment still works
    h, w, _ = sizes.pop()
    assert h * w <= 5.0e6 * 1.01 and abs(w / h - 1.5) < 0.01
