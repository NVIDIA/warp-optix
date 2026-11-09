# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NumPy reference decoder used for validation and offline tooling."""

from __future__ import annotations

import numpy as np

from .format import NeuralTextureAsset, unpack_latents


def _sample_wrap(grid: np.ndarray, uv: tuple[float, float]) -> np.ndarray:
    height, width, _ = grid.shape
    x = uv[0] * width - 0.5
    y = uv[1] * height - 0.5
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    fx = x - x0
    fy = y - y0
    return (
        grid[y0 % height, x0 % width] * (1.0 - fx) * (1.0 - fy)
        + grid[y0 % height, (x0 + 1) % width] * fx * (1.0 - fy)
        + grid[(y0 + 1) % height, x0 % width] * (1.0 - fx) * fy
        + grid[(y0 + 1) % height, (x0 + 1) % width] * fx * fy
    )


def network_input(
    asset: NeuralTextureAsset,
    x: int,
    y: int,
    mip_level: int = 0,
) -> np.ndarray:
    """Construct the decoder input for one texel using two adjacent latent mips."""
    if mip_level < 0:
        raise ValueError("mip_level must be non-negative")
    image_width = max(asset.width >> mip_level, 1)
    image_height = max(asset.height >> mip_level, 1)
    uv = ((x + 0.5) / image_width, (y + 0.5) / image_height)
    result = np.zeros(asset.layers[0].weights.shape[1], dtype=np.float32)
    latent_level = min(mip_level, len(asset.latent_mips) - 1)
    first = unpack_latents(asset.latent_mips[latent_level], asset.latent_features)
    second_level = min(latent_level + 1, len(asset.latent_mips) - 1)
    second = unpack_latents(asset.latent_mips[second_level], asset.latent_features)
    features = asset.latent_features
    if result.size < features * 2 + 2:
        raise ValueError("decoder input is too small for two latent samples and position")
    result[:features] = _sample_wrap(first, uv)
    result[features : features * 2] = _sample_wrap(second, uv)

    position = np.array([x / image_width, y / image_height], dtype=np.float32)
    index = features * 2
    while index + 4 <= result.size - 2:
        phase = position - np.floor(position)
        quarter = position + 0.25
        quarter -= np.floor(quarter)
        result[index : index + 4] = (
            phase[0] * 2.0 - 1.0,
            phase[1] * 2.0 - 1.0,
            quarter[0] * 2.0 - 1.0,
            quarter[1] * 2.0 - 1.0,
        )
        index += 4
        position *= 2.0
    if index + 2 <= result.size:
        normalized_lod = mip_level / max(len(asset.latent_mips) - 1, 1)
        result[index : index + 2] = normalized_lod
    return result


def hgelu(value: np.ndarray) -> np.ndarray:
    """Hardware-friendly GELU approximation used by NVIDIA NTC decoders."""
    return np.minimum(value, 3.0) * np.clip(value / 3.0 + 0.5, 0.0, 1.0)


def _fp8_e4m3_values() -> np.ndarray:
    values = [mantissa * 2.0**-9 for mantissa in range(8)]
    for exponent in range(1, 15):
        values.extend((1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7) for mantissa in range(8))
    values.extend((1.0 + mantissa / 8.0) * 2.0**8 for mantissa in range(7))
    return np.asarray(values, dtype=np.float32)


_FP8_E4M3_POSITIVE = _fp8_e4m3_values()


def _quantize_fp8_e4m3(values: np.ndarray) -> np.ndarray:
    """Emulate OptiX's saturating round-to-nearest-even E4M3 conversion."""
    values = np.asarray(values, dtype=np.float32)
    magnitudes = np.minimum(np.abs(values), _FP8_E4M3_POSITIVE[-1])
    upper = np.searchsorted(_FP8_E4M3_POSITIVE, magnitudes, side="left")
    upper = np.minimum(upper, _FP8_E4M3_POSITIVE.size - 1)
    lower = np.maximum(upper - 1, 0)
    lower_distance = magnitudes - _FP8_E4M3_POSITIVE[lower]
    upper_distance = _FP8_E4M3_POSITIVE[upper] - magnitudes
    choose_upper = (upper_distance < lower_distance) | ((upper_distance == lower_distance) & ((upper & 1) == 0))
    rounded = _FP8_E4M3_POSITIVE[np.where(choose_upper, upper, lower)]
    return np.copysign(rounded, values).astype(np.float32, copy=False)


def decode_texel(
    asset: NeuralTextureAsset,
    x: int,
    y: int,
    mip_level: int = 0,
) -> np.ndarray:
    """Decode a texel with NumPy; useful as a portable correctness oracle."""
    value = network_input(asset, x, y, mip_level)
    for layer in asset.layers:
        value = _quantize_fp8_e4m3(value)
        value = layer.weights.astype(np.float32) @ value + layer.bias.astype(np.float32)
        if layer.activation == "hgelu":
            value = hgelu(value)
    return value[: asset.channel_count]


def _sample_wrap_batch(grid: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    height, width, _ = grid.shape
    px = u * width - 0.5
    py = v * height - 0.5
    x0 = np.floor(px).astype(np.int64)
    y0 = np.floor(py).astype(np.int64)
    fx = (px - x0)[:, None]
    fy = (py - y0)[:, None]
    return (
        grid[y0 % height, x0 % width] * (1.0 - fx) * (1.0 - fy)
        + grid[y0 % height, (x0 + 1) % width] * fx * (1.0 - fy)
        + grid[(y0 + 1) % height, x0 % width] * (1.0 - fx) * fy
        + grid[(y0 + 1) % height, (x0 + 1) % width] * fx * fy
    )


def decode_image(
    asset: NeuralTextureAsset,
    mip_level: int = 0,
    *,
    batch_size: int = 65536,
) -> np.ndarray:
    """Decode a full image in bounded NumPy batches."""
    result = _decode_pixels(asset, mip_level, batch_size=batch_size)
    return result.reshape(
        max(asset.height >> mip_level, 1),
        max(asset.width >> mip_level, 1),
        asset.channel_count,
    )


def _decode_pixels(
    asset: NeuralTextureAsset,
    mip_level: int = 0,
    *,
    batch_size: int = 65536,
    pixel_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Decode all pixels or selected flattened pixel indices in bounded batches.

    Latent mips are unpacked once, which makes this suitable for offline quality
    evaluation while :func:`decode_texel` remains the simple scalar oracle.
    """
    if mip_level < 0:
        raise ValueError("mip_level must be non-negative")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    width = max(asset.width >> mip_level, 1)
    height = max(asset.height >> mip_level, 1)
    latent_level = min(mip_level, len(asset.latent_mips) - 1)
    second_level = min(latent_level + 1, len(asset.latent_mips) - 1)
    first = unpack_latents(asset.latent_mips[latent_level], asset.latent_features)
    second = unpack_latents(asset.latent_mips[second_level], asset.latent_features)
    count = height * width if pixel_indices is None else len(pixel_indices)
    result = np.empty((count, asset.channel_count), dtype=np.float32)
    input_width = asset.layers[0].weights.shape[1]

    for begin in range(0, count, batch_size):
        end = min(begin + batch_size, count)
        indices = np.arange(begin, end, dtype=np.int64) if pixel_indices is None else pixel_indices[begin:end]
        x = indices % width
        y = indices // width
        u = (x.astype(np.float32) + 0.5) / width
        v = (y.astype(np.float32) + 0.5) / height
        inputs = np.zeros((end - begin, input_width), dtype=np.float32)
        features = asset.latent_features
        inputs[:, :features] = _sample_wrap_batch(first, u, v)
        inputs[:, features : features * 2] = _sample_wrap_batch(second, u, v)

        position = np.stack((x / width, y / height), axis=1).astype(np.float32)
        column = features * 2
        while column + 4 <= input_width - 2:
            phase = position - np.floor(position)
            quarter = position + 0.25
            quarter -= np.floor(quarter)
            inputs[:, column : column + 4] = (
                np.stack((phase[:, 0], phase[:, 1], quarter[:, 0], quarter[:, 1]), axis=1) * 2.0 - 1.0
            )
            column += 4
            position *= 2.0
        if column + 2 <= input_width:
            inputs[:, column : column + 2] = mip_level / max(len(asset.latent_mips) - 1, 1)

        value = inputs
        for layer in asset.layers:
            value = _quantize_fp8_e4m3(value)
            value = value @ layer.weights.astype(np.float32).T
            value += layer.bias.astype(np.float32)
            if layer.activation == "hgelu":
                value = hgelu(value)
        result[begin:end] = value[:, : asset.channel_count]
    return result
