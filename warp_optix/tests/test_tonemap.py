# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import warp as wp
from warp_optix.pathtracing import tonemap
from warp_optix.pathtracing.tonemap import Tonemapper, _adapt_auto_exposure

requires_gpu = pytest.mark.skipif(not wp.is_cuda_available(), reason="requires a CUDA device")


@wp.kernel
def _compress_test_colors(colors: wp.array(dtype=wp.vec3), output: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    output[i] = tonemap.compress_highlights(colors[i])


def test_highlight_compression_is_bounded_monotonic_and_preserves_color_ratios():
    colors = np.array(
        [[0, 0, 0], [-1, -2, -3], [0.2, 0.1, 0.05], [2, 1, 0.5], [2e20, 1e20, 5e19]],
        dtype=np.float32,
    )
    output = wp.empty(len(colors), dtype=wp.vec3, device="cpu")
    wp.launch(
        _compress_test_colors,
        dim=len(colors),
        inputs=[wp.array(colors, dtype=wp.vec3, device="cpu"), output],
        device="cpu",
    )
    result = output.numpy()
    assert np.isfinite(result).all()
    assert ((result >= 0) & (result <= 1)).all()
    np.testing.assert_array_equal(result[:2], 0)
    assert (np.diff(result[2:, 0]) > 0).all()
    np.testing.assert_allclose(result[2:, 0] / result[2:, 1], 2.0)
    np.testing.assert_allclose(result[2:, 1] / result[2:, 2], 2.0)


@pytest.mark.gpu
@requires_gpu
def test_tonemap_preserves_pixels_and_orientation_on_non_square_image():
    # Distinct channels and corners catch transposition and missing pixels.
    image = np.arange(9 * 17 * 4, dtype=np.float32).reshape(9, 17, 4) / 1000.0
    mapper = Tonemapper(17, 9)
    mapper.is_active = False
    mapper.exposure = 2.0
    mapper.process(wp.array(image, dtype=wp.vec4, device="cuda"))
    expected = np.flipud(image).copy()
    expected[..., :3] *= 2.0
    expected[..., 3] = 1.0
    np.testing.assert_array_equal(mapper.get_numpy(), expected)


@pytest.mark.gpu
@requires_gpu
def test_debug_normals_preserve_orientation_when_upscaled():
    image = np.arange(3 * 5 * 4, dtype=np.float32).reshape(3, 5, 4) / 100.0
    normal = wp.array(image, dtype=wp.vec4, device="cuda")
    scalar = wp.zeros((3, 5), dtype=wp.float32, device="cuda")
    motion = wp.zeros((3, 5), dtype=wp.vec2, device="cuda")
    mapper = Tonemapper(17, 9)
    mapper.process_debug(
        tonemap.OUTPUT_NORMAL,
        normal,
        scalar,
        motion,
        normal,
        normal,
        normal,
        scalar,
        5,
        3,
    )
    sx = np.arange(17) * 5 // 17
    sy = 2 - np.arange(9) * 3 // 9
    expected = image[sy[:, None], sx[None, :]].copy()
    expected[..., :3] = expected[..., :3] * 0.5 + 0.5
    expected[..., 3] = 1.0
    np.testing.assert_array_equal(mapper.get_numpy(), expected)


@pytest.mark.gpu
@requires_gpu
def test_texture_input_matches_array_tonemapping():
    image = np.random.default_rng(7).uniform(0.0, 8.0, (9, 17, 4)).astype(np.float32)
    array = wp.array(image, dtype=wp.vec4, device="cuda")
    texture = wp.Texture2D(image, filter_mode=wp.TextureFilterMode.CLOSEST, device="cuda")
    for auto_exposure in (False, True):
        outputs = []
        for source in (array, texture):
            mapper = Tonemapper(17, 9)
            mapper.exposure = 0.68
            mapper.contrast = 1.08
            mapper.saturation = 1.1
            mapper.vignette = 0.1
            mapper.configure_auto_exposure(auto_exposure)
            mapper.process(source)
            outputs.append(mapper.get_numpy().copy())
        np.testing.assert_allclose(outputs[0], outputs[1], rtol=1e-6, atol=1e-6, err_msg=f"auto exposure {auto_exposure}")


@pytest.mark.parametrize(
    ("measured_luminance", "min_ev", "expected_ev"),
    [
        (0.01, -6.0, np.log2(18.0)),  # the first valid meter result initializes exposure immediately
        (2.0, 0.0, 0.0),  # a zero floor restricts adaptation to brightening
    ],
)
def test_auto_exposure_adaptation(measured_luminance, min_ev, expected_ev):
    stats = wp.array([np.log2(measured_luminance), 1.0], dtype=wp.float32, device="cpu")
    exposure_ev = wp.zeros(1, dtype=wp.float32, device="cpu")
    wp.launch(
        _adapt_auto_exposure,
        dim=1,
        inputs=[stats, exposure_ev, 0.18, min_ev, 6.0, 0.6, 1.2, 1.0 / 60.0, 1],
        device="cpu",
    )
    np.testing.assert_allclose(exposure_ev.numpy(), [expected_ev], rtol=1.0e-6)
