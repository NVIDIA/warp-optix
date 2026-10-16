# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
from dataclasses import replace

import numpy as np
import pytest
import warp as wp
from neural_texture import (
    NetworkLayer,
    NeuralTextureAsset,
    TextureChannel,
    decode_texel,
    pack_latents,
    upload_asset,
)
from neural_texture.device import (
    NeuralTextureView,
    neural_texture_sample,
    neural_texture_sample_texel,
)

import warp_optix as wo
from warp_optix import native


@wp.kernel(enable_backward=False, module="unique")
def neural_raygen(output: wp.array(dtype=wp.float32), texture: NeuralTextureView):
    value = neural_texture_sample_texel(texture, 2, 3)
    uv_value = neural_texture_sample(texture, wp.vec2(2.5 / 8.0, 3.5 / 8.0))
    for channel in range(3):
        output[channel] = wp.float32(value[channel])
        output[channel + 3] = wp.float32(uv_value[channel])


def _asset():
    grid0 = ((np.arange(4 * 4 * 8) % 16) * (2.0 / 15.0) - 1.0).reshape(4, 4, 8)
    grid1 = ((np.arange(2 * 2 * 8) * 3 % 16) * (2.0 / 15.0) - 1.0).reshape(2, 2, 8)
    output_weights = np.zeros((16, 32), dtype=np.float16)
    output_weights[:3, :3] = np.eye(3, dtype=np.float16)
    return NeuralTextureAsset(
        width=8,
        height=8,
        channels=(TextureChannel("color", 0, 3),),
        latent_features=8,
        latent_mips=(pack_latents(grid0), pack_latents(grid1)),
        layers=(
            NetworkLayer(np.eye(32, dtype=np.float16), np.zeros(32, dtype=np.float16)),
            NetworkLayer(np.eye(32, dtype=np.float16), np.zeros(32, dtype=np.float16)),
            NetworkLayer(output_weights, np.zeros(16, dtype=np.float16), "none"),
        ),
    )


@pytest.mark.gpu
def test_neural_texture_device_inference():
    if not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    device = "cuda:0"
    context = wo.create_context(device)
    if not context.getProperty(native.DEVICE_PROPERTY_COOP_VEC):
        pytest.skip("cooperative vectors unsupported")
    pipeline = wo.create_pipeline(context, neural_raygen, device=device)
    output = wp.zeros(6, dtype=wp.float32, device=device)

    def decode_on_device(asset):
        output.zero_()
        runtime = upload_asset(asset, context, device=device)
        wo.launch(pipeline, dim=(1,), inputs=[output, runtime.device_view()])
        return output.numpy()

    asset = _asset()
    actual = decode_on_device(asset)
    expected = decode_texel(asset, 2, 3)
    np.testing.assert_allclose(actual[:3], expected, atol=3e-3)
    np.testing.assert_allclose(actual[3:], expected, atol=3e-3)

    single_level = replace(asset, latent_mips=asset.latent_mips[:1])
    actual = decode_on_device(single_level)
    expected = decode_texel(single_level, 2, 3)
    np.testing.assert_allclose(actual[:3], expected, atol=3e-3)
    np.testing.assert_allclose(actual[3:], expected, atol=3e-3)

    if importlib.util.find_spec("warp_nn") is not None:
        from neural_texture import compress_texture

        x = np.arange(8, dtype=np.float32)[None, :] / 7.0
        y = np.arange(8, dtype=np.float32)[:, None] / 7.0
        image = np.empty((8, 8, 3), dtype=np.float32)
        image[..., 0] = x
        image[..., 1] = y
        image[..., 2] = 0.5 + 0.25 * np.sin(2.0 * np.pi * (x + y))
        trained = compress_texture(image, steps=2, refinement_steps=1, batch_size=32, device=device, seed=17).asset
        np.testing.assert_allclose(decode_on_device(trained)[:3], decode_texel(trained, 2, 3), atol=5e-3)
