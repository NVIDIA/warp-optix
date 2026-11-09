# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

from warp_optix_pathtracing import PathTracer, Renderer
from warp_optix_pathtracing import pathtracing_viewer as renderer_module


def _renderer(monkeypatch, mode):
    renderer = Renderer.__new__(Renderer)
    renderer.width, renderer.height = 64, 48
    renderer._render_width, renderer._render_height = 64, 48
    renderer.denoiser = mode
    renderer.optix_upscale = False
    renderer._denoiser_explicit = True
    renderer._device = object()
    renderer._render_stream = object()
    renderer._optix_denoiser = None
    renderer._sync_prev_camera_matrices_to_current = lambda: None
    renderer._set_render_resolution = lambda w, h: (
        setattr(renderer, "_render_width", w),
        setattr(renderer, "_render_height", h),
    )
    monkeypatch.setattr(renderer_module.wp, "synchronize_stream", lambda stream: None)
    return renderer


def test_auto_selects_optix(monkeypatch):
    monkeypatch.setattr(renderer_module.wo, "Denoiser", lambda *args, **kwargs: SimpleNamespace(close=lambda: None))
    renderer = _renderer(monkeypatch, "auto")
    renderer._init_denoiser()
    assert renderer.active_denoiser == "optix"
    assert renderer.denoiser_error is None

    def unavailable(*args, **kwargs):
        raise RuntimeError("OptiX unavailable")

    monkeypatch.setattr(renderer_module.wo, "Denoiser", unavailable)
    renderer = _renderer(monkeypatch, "auto")
    renderer._init_denoiser()
    assert renderer.active_denoiser == "none"
    assert "OptiX unavailable" in renderer.denoiser_error
    renderer.denoiser = "optix"
    with pytest.raises(RuntimeError, match="OptiX unavailable"):
        renderer._init_denoiser()


def test_failed_switch_restores_previous_backend(monkeypatch):
    renderer = _renderer(monkeypatch, "none")

    def create(*args, **kwargs):
        if not kwargs["upscale"]:
            raise RuntimeError("unavailable configuration")
        return SimpleNamespace(close=lambda: None)

    monkeypatch.setattr(renderer_module.wo, "Denoiser", create)
    renderer.set_denoiser("optix", optix_upscale=True)
    assert renderer.active_denoiser == "optix"
    assert (renderer._render_width, renderer._render_height) == (32, 24)
    with pytest.raises(RuntimeError, match="unavailable configuration"):
        renderer.set_denoiser("optix", optix_upscale=False)
    assert renderer.denoiser == renderer.active_denoiser == "optix"
    assert renderer.optix_upscale is True
    assert renderer._denoiser_reset_history is True
    assert renderer.sample_index == 0
    renderer.width = 63
    renderer._init_denoiser()
    assert renderer._render_width == 32
    # A closed renderer only records the choice; it creates no denoiser.
    renderer._device = None
    renderer.set_denoiser("optix", optix_upscale=False)


requires_gpu = pytest.mark.skipif(not wp.is_cuda_available(), reason="requires a CUDA device")


def _cornell_api(**kwargs):
    api = PathTracer(enable_set=False, **kwargs)
    api.initialize()
    api.scene.create_cornell_box()
    api.build_scene()
    return api


@pytest.mark.gpu
@pytest.mark.render
@requires_gpu
def test_renderer_gpu_optix_samples_motion_switch_and_resize():
    # OptiX launches cannot be graph-captured, so a requested graph stays inactive.
    api = _cornell_api(width=64, height=48, denoiser="optix", optix_upscale=True, enable_cuda_graphs=True)
    try:
        scene = api.scene
        # A camera placed before the first frame produces no motion on that frame.
        start = api.viewer.camera.position.copy()
        api.set_camera_look_at(start + (0.0, 0.5, 0.0), api.viewer.camera.target)
        frames = []
        for _ in range(3):
            api.render_frame()
            if not frames:
                assert np.abs(api.viewer._motion_buffer.numpy()).max() < 1.0e-3
            frames.append(api.viewer._color_buffer.numpy().copy())
        assert api.viewer.sample_index == 3
        assert not api.cuda_graph_active
        assert not np.array_equal(frames[-1], frames[-2]), "Ray samples froze"
        assert api.get_frame().shape == (48, 64, 4)

        previous = api.viewer.camera.position.copy()
        api.set_camera_look_at(previous + (1.0, 0.0, 0.0), api.viewer.camera.target)
        api.render_frame()
        assert not np.array_equal(api.viewer._color_buffer.numpy(), frames[-1]), "Camera changes did not reach OptiX"
        assert float(np.max(np.abs(api.viewer._motion_buffer.numpy()))) > 0.1, "Camera movement produced no motion"

        api.reset_temporal_history()
        api.set_denoiser("none")
        api.render_frame()
        assert api.active_denoiser == "none"
        api.set_denoiser("optix", optix_upscale=False)
        api.resize(80, 60)
        api.render_frame()
        assert api.scene is scene
        assert api.active_denoiser == "optix"
        assert api.get_frame().shape == (60, 80, 4)
        assert np.isfinite(api.get_frame()).all()
        api.set_denoiser("optix", optix_upscale=True)
        api.resize(81, 61)
        api.render_frame()
        assert api.get_frame().shape == (61, 81, 4)
    finally:
        api.close()


@pytest.mark.gpu
@requires_gpu
def test_optix_gpu_frame_is_ordered_when_called_from_another_stream():
    def first_frame(caller):
        api = PathTracer(
            width=384,
            height=256,
            denoiser="optix",
            enable_set=False,
            enable_cuda_graphs=False,
        )
        try:
            api.initialize()
            api.scene.create_cornell_box()
            api.build_scene()
            with wp.ScopedStream(caller):
                previous = wp.get_stream("cuda")
                api.render_frame()
                assert wp.get_stream("cuda") == previous
            wp.synchronize_device()
            return api.get_frame().copy()
        finally:
            api.close()

    expected = first_frame(None)
    actual = first_frame(wp.Stream("cuda"))
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
