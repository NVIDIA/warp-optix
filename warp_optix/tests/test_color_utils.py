# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from warp_optix.pathtracing.color_utils import (
    srgb_to_linear_rgb,
    srgb_to_linear_u8,
)
from warp_optix.pathtracing.scene import Scene


def test_u8_srgb_lookup_matches_float_conversion():
    values = np.arange(256, dtype=np.uint8)
    expected = np.clip(
        srgb_to_linear_rgb(values.astype(np.float32) * (1.0 / 255.0)) * 255.0 + 0.5,
        0.0,
        255.0,
    ).astype(np.uint8)

    np.testing.assert_array_equal(srgb_to_linear_u8(values), expected)


def test_parallel_texture_preparation_preserves_order_and_alpha():
    color = np.asarray([[[128, 64, 32, 17]]], dtype=np.uint8)
    data = np.asarray([[[1.0, 0.5, 0.25, 0.75]]], dtype=np.float32)
    scene = Scene(None)

    scene.set_gltf_textures([color, data], srgb_texture_indices={0})

    np.testing.assert_array_equal(
        scene._gltf_textures[0][0, 0, :3], srgb_to_linear_u8(color[0, 0, :3])
    )
    assert scene._gltf_textures[0][0, 0, 3] == 17
    np.testing.assert_array_equal(
        scene._gltf_textures[1],
        np.asarray([[[255, 128, 64, 191]]], dtype=np.uint8),
    )


def test_texture_mipmaps_are_opt_in():
    assert not Scene(None).enable_texture_mipmaps
    assert Scene(None, enable_texture_mipmaps=True).enable_texture_mipmaps
