"""MCP server: tools work on real files, paths stay inside the root, and the stdio server answers a client."""

import json
import os
import sys

import anyio
import pytest

pytest.importorskip("mcp")

import lumen.mcp_server as srv  # noqa: E402
from lumen import synth  # noqa: E402


@pytest.fixture()
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("LUMEN_MCP_ROOT", str(tmp_path))
    truth = synth.generate(tmp_path / "photos", sets=2, seed=5)
    paths = [[f"photos/{f['file']}" for f in s["frames"]] for s in truth["sets"]]
    return tmp_path, paths


def test_group_photos_splits_rooms(root):
    _, paths = root
    result = srv.group_photos(paths[0] + paths[1])
    assert result["sets"] == paths and len(result["links"]) == 5


def test_a_folder_stands_for_its_images(root):
    _, paths = root
    assert srv.group_photos(["photos"])["sets"] == paths  # truth.json in the folder is skipped
    with pytest.raises(ValueError):
        srv.group_photos([".."])


def test_align_bracket_reports_residuals(root):
    _, paths = root
    result = srv.align_bracket(paths[0])
    assert [f["method"] for f in result["frames"]].count("reference") == 1
    assert all(f["residual_px"] <= result["residual_limit_px"] for f in result["frames"])


def test_merge_lens_tilt_and_qc_write_inside_the_root(root):
    tmp_path, paths = root
    merged = srv.merge_bracket(paths[0], output_name="room one")
    assert merged["output"] == "lumen-out/room_one.jpg" and (tmp_path / merged["output"]).is_file()
    qc = srv.qc_image(merged["output"])
    assert set(qc) >= {"clip_ratio", "tilt_deg", "curvature_px", "sharpness"}
    lens = srv.correct_lens(merged["output"])
    assert lens["output"] is None or (tmp_path / lens["output"]).is_file()
    upright = srv.straighten(lens["output"] or merged["output"])
    assert upright["tilt_after_deg"] < 0.5


def test_run_agent_returns_a_trace(root):
    tmp_path, paths = root
    result = srv.run_agent(paths[1], output_name="../escape")
    assert result["status"] == "ok" and result["trace"][-1]["decision"] == "done"
    assert result["output"] == "lumen-out/escape.jpg"  # "../" cannot leave lumen-out
    assert (tmp_path / result["output"]).is_file()


@pytest.mark.parametrize("bad", ["../outside.jpg", "/etc/passwd", "photos/truth.json", "photos/missing.jpg"])
def test_paths_must_be_images_inside_the_root(root, bad):
    with pytest.raises(ValueError):
        srv.qc_image(bad)


def test_path_count_is_limited(root):
    _, paths = root
    with pytest.raises(ValueError):
        srv.group_photos(paths[0] * 25)
    with pytest.raises(ValueError):
        srv.align_bracket(paths[0][:1])


def test_stdio_server_answers_a_client(root):
    from mcp import ClientSession, StdioServerParameters, stdio_client

    tmp_path, paths = root
    params = StdioServerParameters(command=sys.executable, args=["-m", "lumen.mcp_server"],
                                   env={**os.environ, "LUMEN_MCP_ROOT": str(tmp_path)})

    async def session():
        async with stdio_client(params) as (read, write), ClientSession(read, write) as client:
            await client.initialize()
            names = {t.name for t in (await client.list_tools()).tools}
            ok = await client.call_tool("qc_image", {"path": paths[0][1]})
            bad = await client.call_tool("qc_image", {"path": "../outside.jpg"})
            return names, ok, bad

    names, ok, bad = anyio.run(session)
    assert names == {fn.__name__ for fn in srv.TOOLS}
    assert not ok.is_error and "clip_ratio" in json.loads(ok.content[0].text)
    assert bad.is_error and "outside the server root" in bad.content[0].text
