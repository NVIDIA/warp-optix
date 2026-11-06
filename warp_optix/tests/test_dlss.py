# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DLSS Ray Reconstruction through the driver's built-in runtime (NGX header-only)."""

from pathlib import Path

import numpy as np
import pytest
import warp as wp
from warp_optix.pathtracing import DenoiserInputs, DlssDenoiser, dlss_rr_available, dlss_rr_requirements
from warp_optix.pathtracing import denoising as denoising_module
from warp_optix.pathtracing.pathtracing_viewer import Renderer

from warp_optix import native

DLSS_BUILT = bool(getattr(native, "DLSS_ENABLED", False))


def _dlss_runs() -> bool:
    return DLSS_BUILT and wp.is_cuda_available() and dlss_rr_available("cuda:0")


requires_dlss = pytest.mark.skipif(
    not _dlss_runs(), reason="needs a DLSS build and a driver with the built-in runtime (R620+)"
)
VIEW = np.eye(4, dtype=np.float32)
PROJECTION = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, -1], [0, 0, -0.1, 0]], np.float32)


def _noisy_inputs(width, height, seed=0):
    rng = np.random.default_rng(seed)
    color = np.full((height, width, 4), 0.5, np.float32)
    color[..., :3] *= rng.uniform(0.0, 2.0, (height, width, 3)).astype(np.float32)
    normal = np.zeros((height, width, 4), np.float32)
    normal[..., 2] = 1.0
    normal[..., 3] = 0.5  # roughness
    return DenoiserInputs(
        color=wp.array(color, dtype=wp.vec4, device="cuda:0"),
        normal=wp.array(normal, dtype=wp.vec4, device="cuda:0"),
        albedo=wp.full((height, width), wp.vec4(0.8), dtype=wp.vec4, device="cuda:0"),
        motion=wp.zeros((height, width), dtype=wp.vec2, device="cuda:0"),
        depth=wp.full((height, width), 5.0, dtype=float, device="cuda:0"),
        specular_albedo=wp.full((height, width), wp.vec4(0.04), dtype=wp.vec4, device="cuda:0"),
    )


def test_package_ships_no_dlss_binary():
    # The Ray Reconstruction runtime comes from the driver; nothing NGX is installed.
    package_dir = Path(native.__file__).resolve().parent if hasattr(native, "__file__") else None
    for root in filter(None, (package_dir, Path(denoising_module.__file__).resolve().parents[1])):
        assert not list(root.glob("*nvngx*")) and not list(root.glob("*dlss*.dll"))


def test_build_flag_matches_bindings():
    assert DLSS_BUILT == hasattr(native, "DlssRRContext")
    if DLSS_BUILT:
        assert native.DLSS_SDK_VERSION > 0
        assert hasattr(native, "dlssRRQueryRequirements")


def test_requirements_report_missing_build(monkeypatch):
    monkeypatch.setattr(native, "DLSS_ENABLED", False, raising=False)
    info = dlss_rr_requirements("cuda:0")
    assert info["supported"] is False and "WARP_OPTIX_ENABLE_DLSS" in info["reason"]
    assert dlss_rr_available("cuda:0") is False
    with pytest.raises(RuntimeError, match="built without DLSS"):
        DlssDenoiser(64, 64)


def test_requirements_reject_cpu():
    if not DLSS_BUILT:
        pytest.skip("requires a DLSS build")
    info = dlss_rr_requirements("cpu")
    assert info["supported"] is False and "CUDA" in info["reason"]


@pytest.mark.skipif(not (DLSS_BUILT and wp.is_cuda_available()), reason="requires a DLSS build and CUDA")
def test_requirements_query_returns_diagnostics():
    # Works on any driver: older ones report unsupported with a reason instead of raising.
    info = dlss_rr_requirements("cuda:0")
    assert {"supported", "result", "result_string", "feature_supported"} <= set(info)
    if not info["supported"]:
        assert "R620" in info["reason"]


def test_quality_and_size_validation(monkeypatch):
    with pytest.raises(ValueError, match="quality"):
        denoising_module._dlss_quality("ultra")
    if not DLSS_BUILT:
        return
    with pytest.raises(ValueError, match="positive"):
        DlssDenoiser(0, 32)
    with pytest.raises(ValueError, match="at least the render size"):
        DlssDenoiser(64, 64, output_size=(32, 32))


def test_renderer_accepts_dlss_mode_and_validates_quality():
    assert Renderer._normalize_denoiser("DLSS") == "dlss"
    with pytest.raises(ValueError, match="dlss"):
        Renderer._normalize_denoiser("ngx")
    with pytest.raises(ValueError, match="dlss_quality"):
        Renderer(width=8, height=8, dlss_quality="best")


class _FakeStream:
    device = "cuda:0"


def test_renderer_auto_falls_back_when_dlss_is_unsupported(monkeypatch):
    module = __import__("warp_optix.pathtracing.pathtracing_viewer", fromlist=["x"])
    monkeypatch.setattr(module, "dlss_rr_requirements", lambda device: {"supported": False, "reason": "too old"})
    renderer = Renderer.__new__(Renderer)
    renderer._render_stream = _FakeStream()
    renderer.width, renderer.height = 64, 48
    renderer.dlss_quality = "dlaa"
    renderer.denoiser = "auto"
    renderer.denoiser_error = None
    renderer._dlss_denoiser = None
    sizes = []
    renderer._set_render_resolution = lambda w, h: sizes.append((w, h))
    renderer._init_dlss_denoiser()
    # Missing DLSS support is normal in 'auto': no error, and the next backend sizes the render.
    assert renderer._dlss_denoiser is None
    assert renderer.denoiser_error is None and sizes == []
    renderer.denoiser = "dlss"  # an explicit request must fail loudly instead
    with pytest.raises(RuntimeError, match="too old"):
        renderer._init_dlss_denoiser()


@requires_dlss
@pytest.mark.gpu
@pytest.mark.parametrize("quality", ["dlaa", "balanced"])
def test_denoises_and_upscales_with_driver_runtime(quality):
    output_size = (256, 192)
    width, height = DlssDenoiser.optimal_render_size(output_size, quality)
    if quality == "dlaa":
        assert (width, height) == output_size
    else:
        assert width < output_size[0] and height < output_size[1]
    denoiser = DlssDenoiser(width, height, output_size=output_size, quality=quality, device="cuda:0")
    try:
        for frame in range(6):
            output = denoiser.apply(
                _noisy_inputs(width, height, seed=frame),
                view_matrix=VIEW,
                projection_matrix=PROJECTION,
                reset=frame == 0,
            ).numpy()
    finally:
        denoiser.close()
    rgb = output[..., :3]
    assert output.shape == (output_size[1], output_size[0], 4)
    assert np.isfinite(output).all()
    # Uniform noise around 0.5 (std ~0.29) reconstructs to a flat 0.5.
    assert abs(rgb.mean() - 0.5) < 0.02 and rgb.std() < 0.05


@requires_dlss
@pytest.mark.gpu
def test_denoiser_requires_every_guide_and_rejects_use_after_close():
    denoiser = DlssDenoiser(64, 48, device="cuda:0")
    inputs = _noisy_inputs(64, 48)
    missing = DenoiserInputs(inputs.color, inputs.normal, inputs.albedo, inputs.motion)
    with pytest.raises(ValueError, match="depth"):
        denoiser.apply(missing, view_matrix=VIEW, projection_matrix=PROJECTION)
    wrong = _noisy_inputs(32, 48)
    with pytest.raises(ValueError, match="shape"):
        denoiser.apply(wrong, view_matrix=VIEW, projection_matrix=PROJECTION)
    with pytest.raises(ValueError, match="4x4"):
        denoiser.apply(inputs, view_matrix=np.eye(3), projection_matrix=PROJECTION)
    denoiser.close()
    denoiser.close()  # idempotent
    with pytest.raises(RuntimeError, match="closed"):
        denoiser.apply(inputs, view_matrix=VIEW, projection_matrix=PROJECTION)
