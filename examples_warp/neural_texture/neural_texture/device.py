# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Warp device functions for the standard 8-feature neural-texture decoder.

The implementation has no warp-nn dependency.
"""

from __future__ import annotations

import warp as wp

# Registers the wp.optix_coop_vec_* builtins used below.
import warp_optix  # noqa: F401

Vec32h = wp.types.vector(length=32, dtype=wp.float16)
Vec16h = wp.types.vector(length=16, dtype=wp.float16)


@wp.struct
class NeuralTextureView:
    """Compact device view stored in an OptiX launch-parameter struct."""

    latents: wp.array(dtype=wp.uint16)
    mip_offsets: wp.array(dtype=wp.int32)
    mip_widths: wp.array(dtype=wp.int32)
    mip_heights: wp.array(dtype=wp.int32)
    matrices: wp.uint64
    biases: wp.uint64
    weight_offsets: wp.vec3ui
    bias_offsets: wp.vec3ui
    width: wp.int32
    height: wp.int32
    latent_level_count: wp.int32


@wp.func
def _wrap_index(value: int, size: int):
    return ((value % size) + size) % size


@wp.func
def _latent_value(
    latents: wp.array(dtype=wp.uint16),
    mip_offset: int,
    width: int,
    height: int,
    x: int,
    y: int,
    feature: int,
):
    words_per_texel = 2
    px = _wrap_index(x, width)
    py = _wrap_index(y, height)
    word_index = mip_offset + (py * width + px) * words_per_texel + feature // 4
    shift = wp.uint32((feature % 4) * 4)
    nibble = (wp.uint32(latents[word_index]) >> shift) & wp.uint32(15)
    return wp.float16(wp.float32(nibble) * (2.0 / 15.0) - 1.0)


@wp.func
def _sample_mip(
    latents: wp.array(dtype=wp.uint16),
    mip_offsets: wp.array(dtype=wp.int32),
    mip_widths: wp.array(dtype=wp.int32),
    mip_heights: wp.array(dtype=wp.int32),
    mip: int,
    uv: wp.vec2,
    output_offset: int,
    output: Vec32h,
):
    width = mip_widths[mip]
    height = mip_heights[mip]
    fx = uv[0] * wp.float32(width) - 0.5
    fy = uv[1] * wp.float32(height) - 0.5
    x0 = int(wp.floor(fx))
    y0 = int(wp.floor(fy))
    tx = fx - wp.float32(x0)
    ty = fy - wp.float32(y0)
    for feature in range(8):
        s00 = wp.float32(_latent_value(latents, mip_offsets[mip], width, height, x0, y0, feature))
        s10 = wp.float32(_latent_value(latents, mip_offsets[mip], width, height, x0 + 1, y0, feature))
        s01 = wp.float32(_latent_value(latents, mip_offsets[mip], width, height, x0, y0 + 1, feature))
        s11 = wp.float32(_latent_value(latents, mip_offsets[mip], width, height, x0 + 1, y0 + 1, feature))
        value = s00 * (1.0 - tx) * (1.0 - ty) + s10 * tx * (1.0 - ty) + s01 * (1.0 - tx) * ty + s11 * tx * ty
        output[output_offset + feature] = wp.float16(value)
    return output


@wp.func
def neural_texture_input_8(
    latents: wp.array(dtype=wp.uint16),
    mip_offsets: wp.array(dtype=wp.int32),
    mip_widths: wp.array(dtype=wp.int32),
    mip_heights: wp.array(dtype=wp.int32),
    x: int,
    y: int,
    image_width: int,
    image_height: int,
    mip: int,
    mip_count: int,
):
    result = Vec32h(wp.float16(0.0))
    mip_width = wp.max(image_width >> mip, 1)
    mip_height = wp.max(image_height >> mip, 1)
    uv = wp.vec2(
        (wp.float32(x) + 0.5) / wp.float32(mip_width),
        (wp.float32(y) + 0.5) / wp.float32(mip_height),
    )
    latent_mip = wp.min(mip, mip_count - 1)
    result = _sample_mip(latents, mip_offsets, mip_widths, mip_heights, latent_mip, uv, 0, result)
    result = _sample_mip(
        latents,
        mip_offsets,
        mip_widths,
        mip_heights,
        wp.min(latent_mip + 1, mip_count - 1),
        uv,
        8,
        result,
    )
    px = wp.float32(x) / wp.float32(mip_width)
    py = wp.float32(y) / wp.float32(mip_height)
    for wave in range(3):
        index = 16 + wave * 4
        result[index] = wp.float16((px - wp.floor(px)) * 2.0 - 1.0)
        result[index + 1] = wp.float16((py - wp.floor(py)) * 2.0 - 1.0)
        result[index + 2] = wp.float16((px + 0.25 - wp.floor(px + 0.25)) * 2.0 - 1.0)
        result[index + 3] = wp.float16((py + 0.25 - wp.floor(py + 0.25)) * 2.0 - 1.0)
        px *= 2.0
        py *= 2.0
    normalized_lod = wp.float32(mip) / wp.float32(wp.max(mip_count - 1, 1))
    result[28] = wp.float16(normalized_lod)
    result[29] = wp.float16(normalized_lod)
    return result


@wp.func
def _hgelu32(value: Vec32h):
    tmp = Vec32h()
    clamped = Vec32h()
    result = Vec32h()
    wp.optix_coop_vec_ffma(value, Vec32h(wp.float16(1.0 / 3.0)), Vec32h(wp.float16(0.5)), tmp)
    wp.optix_coop_vec_max_scalar(tmp, wp.float16(0.0), clamped)
    wp.optix_coop_vec_min_scalar(clamped, wp.float16(1.0), tmp)
    wp.optix_coop_vec_min_scalar(value, wp.float16(3.0), clamped)
    wp.optix_coop_vec_mul(clamped, tmp, result)
    return result


@wp.func
def neural_texture_infer_8x32x32x16(
    inputs: Vec32h,
    matrices: wp.uint64,
    biases: wp.uint64,
    weight_offsets: wp.vec3ui,
    bias_offsets: wp.vec3ui,
):
    hidden0 = Vec32h()
    hidden1 = Vec32h()
    output = Vec16h()
    wp.optix_coop_vec_matmul_bias_fp8_e4m3(
        inputs,
        matrices,
        weight_offsets[0],
        biases,
        bias_offsets[0],
        wp.uint32(0),
        hidden0,
    )
    hidden0 = _hgelu32(hidden0)
    wp.optix_coop_vec_matmul_bias_fp8_e4m3(
        hidden0,
        matrices,
        weight_offsets[1],
        biases,
        bias_offsets[1],
        wp.uint32(0),
        hidden1,
    )
    hidden1 = _hgelu32(hidden1)
    wp.optix_coop_vec_matmul_bias_fp8_e4m3(
        hidden1,
        matrices,
        weight_offsets[2],
        biases,
        bias_offsets[2],
        wp.uint32(0),
        output,
    )
    return output


@wp.func
def neural_texture_sample_texel(texture: NeuralTextureView, x: int, y: int):
    """Decode all 16 output channels for one integer texel."""
    inputs = neural_texture_input_8(
        texture.latents,
        texture.mip_offsets,
        texture.mip_widths,
        texture.mip_heights,
        x,
        y,
        int(texture.width),
        int(texture.height),
        0,
        int(texture.latent_level_count),
    )
    return neural_texture_infer_8x32x32x16(
        inputs,
        texture.matrices,
        texture.biases,
        texture.weight_offsets,
        texture.bias_offsets,
    )


@wp.func
def neural_texture_sample(texture: NeuralTextureView, uv: wp.vec2):
    """Decode the base-level texel at a normalized, repeat-wrapped UV."""
    width = int(texture.width)
    height = int(texture.height)
    u = uv[0] - wp.floor(uv[0])
    v = uv[1] - wp.floor(uv[1])
    x = wp.min(int(wp.floor(u * wp.float32(width))), width - 1)
    y = wp.min(int(wp.floor(v * wp.float32(height))), height - 1)
    return neural_texture_sample_texel(texture, x, y)
