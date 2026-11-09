# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ctypes
import sys

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
        extreme = np.full((48, 64, 2), 1.0e9, dtype=np.float32)
        extreme[0, 0] = np.nan
        jump = {**inputs, "motion": wp.array(extreme, dtype=wp.vec2, device="cuda")}
        for frame_inputs in (jump, inputs, jump, inputs):
            output = denoiser.apply(color, **frame_inputs).numpy()
            assert np.isfinite(output).all()
            assert np.mean((output[..., :3] - 1.0) ** 2) < np.mean((noisy[..., :3] - 1.0) ** 2) * 0.5

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


@pytest.mark.gpu
@pytest.mark.parametrize(("upscale", "other_context"), [(False, False), (True, True)])
def test_optix_gpu_apply_preserves_cuda_context(upscale, other_context):
    cuda = ctypes.CDLL("nvcuda.dll" if sys.platform == "win32" else "libcuda.so.1")
    cuda.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    cuda.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
    cuda.cuCtxCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int]
    cuda.cuCtxDestroy_v2.argtypes = [ctypes.c_void_p]

    def current_context():
        context = ctypes.c_void_p()
        assert cuda.cuCtxGetCurrent(ctypes.byref(context)) == 0
        return context.value

    denoiser = wo.Denoiser(64, 48, upscale=upscale, device="cuda")
    inputs = dict(
        color=wp.ones((48, 64), dtype=wp.vec4, device=denoiser.stream.device),
        normal=wp.zeros((48, 64), dtype=wp.vec4, device=denoiser.stream.device),
        albedo=wp.ones((48, 64), dtype=wp.vec4, device=denoiser.stream.device),
        motion=wp.zeros((48, 64), dtype=wp.vec2, device=denoiser.stream.device),
    )
    original_context = current_context()
    caller_context = ctypes.c_void_p()
    try:
        # A second context on the same GPU exercises this without requiring two GPUs.
        if other_context:
            assert cuda.cuCtxCreate_v2(ctypes.byref(caller_context), 0, denoiser.stream.device.ordinal) == 0
        # Exercise fresh bindings, both cached output parities, and a history reset.
        for reset in (False, False, False, True):
            assert cuda.cuCtxSetCurrent(caller_context) == 0
            denoiser.apply(**inputs, reset=reset)
            assert current_context() == caller_context.value
    finally:
        assert cuda.cuCtxSetCurrent(original_context) == 0
        denoiser.close()
        if caller_context.value is not None:
            assert cuda.cuCtxDestroy_v2(caller_context) == 0
        assert cuda.cuCtxSetCurrent(original_context) == 0


def test_apply_refuses_capture_before_changing_history():
    _, inputs = _inputs()
    stream = wp.Stream("cuda")
    with wo.Denoiser(64, 48, stream=stream) as denoiser:
        denoiser.apply(**inputs)
        parity = denoiser._parity
        with wp.ScopedCapture(stream=stream) as capture:
            with pytest.raises(RuntimeError, match="cannot be recorded into CUDA graphs"):
                denoiser.apply(**inputs, reset=True)
        assert denoiser._history_valid
        assert denoiser._parity == parity
        wp.capture_launch(capture.graph, stream=stream)
        assert np.isfinite(denoiser.apply(**inputs).numpy()).all()


def test_close_releases_cached_inputs_without_synchronizing(monkeypatch):
    import gc
    import weakref

    _, inputs = _inputs()
    stream = wp.Stream("cuda")
    denoiser = wo.Denoiser(64, 48, stream=stream)
    result = denoiser.apply(**inputs)
    refs = [weakref.ref(array) for array in inputs.values()]
    del inputs
    gc.collect()
    assert all(ref() is not None for ref in refs)

    def unexpected_sync(*args, **kwargs):
        raise AssertionError("close must not synchronize the host")

    with monkeypatch.context() as patch:
        patch.setattr(wp, "synchronize_stream", unexpected_sync)
        denoiser.close()
    gc.collect()
    assert all(ref() is None for ref in refs)
    with wp.ScopedStream(stream):
        assert np.isfinite(result.numpy()).all()
