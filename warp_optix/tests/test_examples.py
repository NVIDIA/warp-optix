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


def test_cluster_example_animates_rebuilds_and_accumulates(monkeypatch):
    import warp_optix as wo

    if not wo.cluster_capabilities()["flags"]:
        pytest.skip("Cluster API unsupported")
    monkeypatch.syspath_prepend(str(EXAMPLES))
    module = importlib.import_module("example_clusters")
    args = SimpleNamespace(
        width=64,
        height=48,
        along=16,
        around=4,
        cluster_size=4,
        truncate=0,
        distance=5.8,
        time=0.0,
        view="clusters",
        animate=1.0,
        samples=1,
    )
    app = module.Example(args)
    image = wp.empty(args.width * args.height, dtype=wp.uint32)
    try:
        app.draw(image, 0, 0.0)
        initial = app.points.numpy()
        assert np.count_nonzero(app.hit_ids.numpy() >= 0) > 50
        app.draw(image, 1, 1.0)
        assert np.max(np.abs(app.points.numpy() - initial)) > 0.001
        args.along = 24
        app.draw(image, 2, 2.0)
        ids = app.hit_ids.numpy()
        assert app.grid_sizes.size == args.along * args.around
        assert np.unique(ids[ids >= 0]).size > 4
        assert ids.max() < app.grid_sizes.size
        assert np.isfinite(app.points.numpy()).all()
        app.paused = True
        app.draw(image, 3, 3.0)
        assert app.accumulated_frames == 2
        app.yaw += 0.1
        app.draw(image, 4, 4.0)
        assert app.accumulated_frames == 1
    finally:
        app.close()


@pytest.mark.parametrize("mode", ["clusters", "triangles", "warp"])
def test_deforming_objects_example_updates_all_bodies(mode, monkeypatch):
    import warp_optix as wo

    if mode == "clusters" and not wo.cluster_capabilities()["flags"]:
        pytest.skip("Cluster API unsupported")
    monkeypatch.syspath_prepend(str(EXAMPLES))
    example = importlib.import_module("example_deforming_objects")
    bodies, frames = 8, 2
    distances = example.run(mode, bodies, frames)
    phase = 0.013 * np.arange(bodies) + 0.1 * frames
    # The ray lands 64% of the way between the vertices at x=0.25 and x=0.50.
    expected = 2.0 - 0.15 * (0.36 * np.sin(phase + 0.1) + 0.64 * np.sin(phase + 0.2))
    np.testing.assert_allclose(distances, expected, atol=2e-5)


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


def test_material_gallery_renders_and_wraps_pages(monkeypatch):
    from warp_optix.pathtracing import PathTracer

    from warp_optix import procedural_textures

    monkeypatch.syspath_prepend(str(EXAMPLES))
    example = importlib.import_module("example_material_gallery")
    api = PathTracer(width=96, height=56, denoiser="none", enable_set=False)
    try:
        gallery = example.Gallery(api, texture_size=48)
        overview = example._select_materials(procedural_textures.reference_ids(), None, "", False)
        assert example.PAGE_SIZE == 45 and len(overview) == len(set(overview))
        assert {"Concrete008", "Concrete015", "Concrete036", "Tiles071"} <= set(overview)
        assert "BlueCeramicTile" not in overview and "SageCeramicTile" not in overview
        assets = procedural_textures.reference_ids()
        gallery.load(0, None)
        api.render_frame()
        first = api.get_frame().copy()
        assert gallery.visible_ids == assets[: example.PAGE_SIZE]
        assert np.isfinite(first).all() and first[..., :3].std() > 0.02

        gallery.load(-1, 42)
        api.render_frame()
        last = api.get_frame()
        assert gallery.visible_ids == assets[(gallery.page_count - 1) * example.PAGE_SIZE :]
        assert gallery.seed == 42
        assert api.scene.texture_count == 3 * len(gallery.visible_ids)
        assert np.isfinite(last).all() and not np.allclose(first, last)

        controls = importlib.import_module("example_pathtracing")
        symbols = "W A S D Q E UP DOWN LEFT RIGHT LSHIFT RSHIFT".split()
        keys = SimpleNamespace(**dict(zip(symbols, range(len(symbols)), strict=True)))
        window = SimpleNamespace(push_handlers=lambda _handler: None)
        viewer = SimpleNamespace(window=window, pyglet=SimpleNamespace(window=SimpleNamespace(key=keys)))
        home = controls.View(gallery.home_position, 180.0, -example.CAMERA_PITCH, gallery.home_fov)
        camera = controls.FreeCameraController(viewer, api, home, use_arrows=False)
        name = gallery.visible_ids[0]
        label_x = {caption: x for caption, x, _ in gallery.annotations()}[name]
        camera.keys_down = {keys.RIGHT}
        camera.update(0.1)
        np.testing.assert_allclose(api.viewer.camera.position, home.position)
        camera.keys_down = {keys.D, keys.E, keys.W}
        camera.update(0.1)
        x, y, z = api.viewer.camera.position
        assert x > 0 and y > home.position[1] and z < home.position[2]
        assert {caption: x for caption, x, _ in gallery.annotations()}[name] < label_x
        gallery.resize(100, 64, reframe=False)
        np.testing.assert_allclose(api.viewer.camera.position, (x, y, z))
        camera.set_view(home)
        np.testing.assert_allclose(api.viewer.camera.position, home.position)
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
