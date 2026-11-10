# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every example still runs against the current API: headless, at a tiny size."""

import importlib
import runpy
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA unavailable")]


@pytest.mark.parametrize("name", ["minimal", "curves", "mixed_geometry", "motion_blur", "raytracing"])
def test_ray_tracing_examples_write_their_images(name, tmp_path, monkeypatch):
    output = tmp_path / f"{name}.bmp"
    argv = [name] if name == "minimal" else [name, "--width", "32", "--height", "16", "--output", str(output)]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.chdir(tmp_path)
    runpy.run_path(str(EXAMPLES / f"example_{name}.py"), run_name="__main__")
    if name != "minimal":
        assert output.stat().st_size > 0


@pytest.fixture
def pathtracing(monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLES))
    return importlib.import_module("example_pathtracing")


def test_path_tracing_scenes_build_and_render(pathtracing):
    from warp_optix.pathtracing import PathTracer

    # The downloaded scenes need the network; their loader's cache handling is tested below.
    args = SimpleNamespace(hair_count=40, arrow_count=20, contact_count=64)
    for scene in ("chess", "hairball", "arrowball", "contact_lines"):
        api = PathTracer(width=16, height=16, denoiser="none", **pathtracing.SCENE_OPTIONS.get(scene, {}))
        try:
            api.initialize()
            pathtracing.SCENES[scene](api, args)
            api.render_frame()
            assert np.isfinite(api.get_frame()).all(), scene
        finally:
            api.close()


def test_scene_downloads_stay_in_their_cache_and_publish_only_complete_scenes(pathtracing, tmp_path, monkeypatch):
    from scenes import mitsuba

    monkeypatch.setenv("WARP_OPTIX_CACHE", str(tmp_path))
    members = {"dining-room/scene_v3.xml": "<scene/>", "../../escape.obj": "untrusted"}

    def download(url, archive):
        with zipfile.ZipFile(archive, "w") as zf:
            for name, text in members.items():
                zf.writestr(name, text)

    monkeypatch.setattr(mitsuba.urllib.request, "urlretrieve", download)
    # An archive member or a scene asset that escapes its directory is rejected and nothing is cached.
    with pytest.raises(ValueError, match="escapes its directory"):
        mitsuba.fetch("dining-room")
    assert not (tmp_path / "scenes" / "dining-room").exists()
    with pytest.raises(ValueError, match="Unknown bundled scene"):
        mitsuba.fetch("../../escape")

    # An interrupted extraction is not published either.
    del members["../../escape.obj"]
    extract = zipfile.ZipFile.extractall

    def interrupted(archive, directory):
        raise OSError("disk full")

    monkeypatch.setattr(zipfile.ZipFile, "extractall", interrupted)
    with pytest.raises(OSError, match="disk full"):
        mitsuba.fetch("dining-room")
    assert not (tmp_path / "scenes" / "dining-room").exists()

    # A complete download is published once and reused.
    monkeypatch.setattr(zipfile.ZipFile, "extractall", extract)
    xml = mitsuba.fetch("dining-room")
    assert xml.read_text() == "<scene/>" and mitsuba.fetch("dining-room") == xml
    for name in ("../secret.obj", str(tmp_path / "secret.obj")):
        with pytest.raises(ValueError, match="escapes its directory"):
            mitsuba._asset_path(xml.parent, name)
