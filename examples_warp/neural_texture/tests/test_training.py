# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("warp_nn", reason="offline compression is an optional extra")
pytestmark = pytest.mark.training

from neural_texture import (
    compress_texture,
    compress_texture_set,
    decode_image,
    load_asset,
    save_asset,
)
from neural_texture.training import _quantize_fp8_e4m3


def test_compress_texture_exports_quantized_asset(tmp_path):
    image = np.empty((8, 8, 3), dtype=np.float32)
    image[..., 0] = 0.2
    image[..., 1] = 0.5
    image[..., 2] = 0.8

    result = compress_texture(
        image,
        steps=2,
        refinement_steps=1,
        learning_rate=2.0e-2,
        device="cuda:0",
        seed=7,
        channel_name="albedo",
    )

    assert len(result.losses) == 3
    assert np.all(np.isfinite(result.losses))
    assert np.isfinite(result.psnr)
    assert result.asset.channels[0].name == "albedo"
    assert result.asset.metadata["compressor"] == "warp-nn"
    assert result.asset.metadata["weight_type"] == "fp8_e4m3"
    assert result.asset.metadata["batch_size"] == 64

    decoded = decode_image(result.asset, batch_size=7)
    expected_psnr = -10.0 * np.log10(np.mean((decoded - image) ** 2))
    # Actual OptiX FP16/FP8 arithmetic differs slightly from the float32 oracle.
    np.testing.assert_allclose(result.psnr, expected_psnr, atol=0.02, rtol=0.0)

    path = tmp_path / "compressed.wnt"
    save_asset(path, result.asset)
    loaded = load_asset(path)
    assert loaded.storage_bytes == result.asset.storage_bytes


def test_fp8_projection_uses_optix_tie_to_even_rounding():
    values = np.array([-152.0, -136.0, 136.0, 152.0], dtype=np.float32)
    np.testing.assert_array_equal(
        _quantize_fp8_e4m3(values),
        np.array([-160.0, -128.0, 128.0, 160.0], dtype=np.float32),
    )


def test_compression_seed_is_repeatable_and_preserves_numpy_rng():
    image = np.linspace(0.0, 1.0, 4 * 4 * 3, dtype=np.float32).reshape(4, 4, 3)
    state = np.random.get_state()
    try:
        np.random.seed(1234)
        np.random.random()
        first = compress_texture(
            image,
            steps=2,
            refinement_steps=1,
            batch_size=8,
            seed=17,
        )
        observed_next = np.random.random()
        np.random.seed(1234)
        np.random.random()
        expected_next = np.random.random()
    finally:
        np.random.set_state(state)

    second = compress_texture(
        image,
        steps=2,
        refinement_steps=1,
        batch_size=8,
        seed=17,
    )
    assert observed_next == expected_next
    np.testing.assert_allclose(first.losses, second.losses)
    for first_mip, second_mip in zip(first.asset.latent_mips, second.asset.latent_mips):
        np.testing.assert_array_equal(first_mip, second_mip)
    for first_layer, second_layer in zip(first.asset.layers, second.asset.layers):
        np.testing.assert_array_equal(first_layer.weights, second_layer.weights)


def test_compress_texture_maps_more_than_four_outputs():
    image = np.zeros((2, 2, 5), dtype=np.float32)
    result = compress_texture(image, steps=0, refinement_steps=0, seed=9)
    assert [(c.name, c.first_channel, c.channel_count) for c in result.asset.channels] == [
        ("texture", 0, 4),
        ("texture_1", 4, 1),
    ]


def test_compress_texture_set_preserves_named_channel_ranges():
    albedo = np.full((4, 6, 3), 0.25, dtype=np.float32)
    normal = np.full((4, 6, 3), 0.5, dtype=np.float32)
    roughness = np.full((4, 6), 0.75, dtype=np.float32)
    result = compress_texture_set(
        {"albedo": albedo, "normal": normal, "roughness": roughness},
        color_spaces={"albedo": "srgb"},
        channel_weights={"normal": 2.0, "roughness": [0.5]},
        steps=2,
        refinement_steps=1,
        batch_size=12,
        seed=11,
    )

    assert [(c.name, c.first_channel, c.channel_count, c.color_space) for c in result.asset.channels] == [
        ("albedo", 0, 3, "srgb"),
        ("normal", 3, 3, "linear"),
        ("roughness", 6, 1, "linear"),
    ]
    assert decode_image(result.asset).shape == (4, 6, 7)
    assert set(result.psnr_by_texture) == {"albedo", "normal", "roughness"}
    assert np.all(np.isfinite(list(result.psnr_by_texture.values())))


def test_compress_texture_set_validates_shape_and_options():
    rgb = np.zeros((2, 3, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="same height and width"):
        compress_texture_set({"a": rgb, "b": np.zeros((3, 2), dtype=np.float32)}, steps=0)
    with pytest.raises(ValueError, match="unknown textures"):
        compress_texture_set({"a": rgb}, color_spaces={"missing": "srgb"}, steps=0)
    with pytest.raises(ValueError, match="one value per channel"):
        compress_texture_set({"a": rgb}, channel_weights={"a": [1.0, 2.0]}, steps=0)


def test_compress_texture_set_accepts_integer_images():
    result = compress_texture_set(
        {
            "albedo": np.full((2, 2, 3), 128, dtype=np.uint8),
            "roughness": np.full((2, 2), 32768, dtype=np.uint16),
        },
        steps=0,
        refinement_steps=0,
    )
    assert result.asset.channel_count == 4


@pytest.mark.parametrize(
    ("image", "kwargs", "message"),
    [
        (np.empty((0, 2, 3), dtype=np.float32), {}, "dimensions"),
        (
            np.zeros((2, 2, 3), dtype=np.float32),
            {"learning_rate": 0.0},
            "learning_rate",
        ),
        (np.zeros((2, 2, 3), dtype=np.float32), {"batch_size": 0}, "batch_size"),
    ],
)
def test_compress_texture_validates_configuration(image, kwargs, message):
    with pytest.raises(ValueError, match=message):
        compress_texture(image, steps=0, refinement_steps=0, **kwargs)
