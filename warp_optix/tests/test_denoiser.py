# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import warp as wp

import warp_optix as wo

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not wp.is_cuda_available(), reason="requires a CUDA device"),
]


def _inputs(width=64, height=48, guide_dtype=wp.vec4):
    rng = np.random.default_rng(42)
    noisy = np.ones((height, width, 4), dtype=np.float32)
    noisy[..., :3] += rng.normal(0.0, 0.15, (height, width, 3)).astype(np.float32)
    channels = 4 if guide_dtype is wp.vec4 else 3
    normals = np.zeros((height, width, channels), dtype=np.float32)
    normals[..., 2] = 1.0
    return noisy, {
        "color": wp.array(noisy, dtype=wp.vec4, device="cuda"),
        "albedo": wp.ones((height, width), dtype=guide_dtype, device="cuda"),
        "normal": wp.array(normals, dtype=guide_dtype, device="cuda"),
        "motion": wp.zeros((height, width), dtype=wp.vec2, device="cuda"),
    }


@pytest.mark.parametrize("upscale", [False, True])
def test_denoises_and_resets_history(upscale):
    noisy, inputs = _inputs()
    with wo.Denoiser(64, 48, upscale=upscale, device="cuda") as denoiser:
        color = inputs.pop("color")
        first = denoiser.apply(color, **inputs).numpy()
        scale = 2 if upscale else 1
        assert (denoiser.output_height, denoiser.output_width) == (48 * scale, 64 * scale)
        assert first.shape == (48 * scale, 64 * scale, 4)
        assert np.isfinite(first).all()
        assert np.mean((first[..., :3] - 1.0) ** 2) < np.mean((noisy[..., :3] - 1.0) ** 2) * 0.5
        denoiser.apply(color, **inputs)
        reset = denoiser.apply(color, **inputs, reset=True).numpy()
        np.testing.assert_allclose(reset, first, rtol=1e-5, atol=1e-5)
    assert denoiser.closed


def test_float3_guides_match_float4_guides():
    _, vec4_inputs = _inputs()
    _, vec3_inputs = _inputs(guide_dtype=wp.vec3)
    results = []
    for inputs in (vec4_inputs, vec3_inputs):
        with wo.Denoiser(64, 48, device="cuda") as denoiser:
            results.append(denoiser.apply(inputs.pop("color"), **inputs).numpy())
    np.testing.assert_allclose(results[1], results[0], rtol=1e-5, atol=1e-5)


def test_rejects_mismatched_inputs_and_use_after_close():
    _, inputs = _inputs()
    color = inputs.pop("color")
    denoiser = wo.Denoiser(64, 48, device="cuda")
    with pytest.raises(ValueError, match="color must be a vec4f array of shape"):
        denoiser.apply(color[:24], **inputs)
    with pytest.raises(ValueError, match="normal must be a vec3f or vec4f array"):
        denoiser.apply(color, **{**inputs, "normal": wp.zeros((48, 64), dtype=wp.vec2, device="cuda")})
    with pytest.raises(ValueError, match="motion must be contiguous"):
        motion = wp.zeros((48, 128), dtype=wp.vec2, device="cuda")[:, ::2]
        denoiser.apply(color, **{**inputs, "motion": motion})
    denoiser.close()
    denoiser.close()
    assert denoiser.closed
    with pytest.raises(RuntimeError, match="closed"):
        denoiser.apply(color, **inputs)
    with pytest.raises(ValueError, match="positive"):
        wo.Denoiser(0, 48, device="cuda")


def test_uses_the_given_stream():
    stream = wp.Stream("cuda")
    with wo.Denoiser(64, 48, stream=stream) as denoiser:
        assert denoiser.stream is stream
        assert denoiser.device == stream.device
    with pytest.raises(TypeError, match="Warp Stream"):
        wo.Denoiser(64, 48, stream=object())


def test_upscaling_allocates_scratch_beyond_two_gib():
    if wp.get_device("cuda").free_memory < 6 * 2**30:
        pytest.skip("needs about 3 GiB of free device memory")
    with wo.Denoiser(1920, 1080, upscale=True, device="cuda") as denoiser:
        assert denoiser._scratch_size > 2**31
