# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from warp_optix import procedural_textures as textures


def test_plastic_override_and_seeded_maps():
    shape = (512, 640)
    trim = textures.PlasticSurface("pebbled", color=(0.61, 0.13, 0.08), roughness=0.62)
    seed = 7
    assert textures.reference_maps(trim) == ("color", "normal", "roughness", "displacement")
    packed, properties = textures.bake_gltf(trim, shape, device="cpu", seed=seed, first_texture_index=5)
    repeat, _ = textures.bake_gltf(trim, shape, device="cpu", seed=seed)
    np.testing.assert_array_equal(packed, repeat)
    assert properties["normal_texture"]["index"] == 6
    assert np.all(packed[0][..., 0] > packed[0][..., 1])
    assert abs(np.mean(packed[2][..., 1]) / 255 - 0.62) < 0.06
    assert not packed[2][..., 2].any()
    height = textures.bake(trim, shape, map_type="displacement", device="cpu", seed=seed).numpy()[..., 0]
    for axis, channel in ((1, 0), (0, 1)):
        slope = np.roll(height, 1, axis=axis) - np.roll(height, -1, axis=axis)
        assert np.corrcoef(slope.ravel(), packed[1][..., channel].ravel())[0, 1] > 0.97
    other, _ = textures.bake_gltf(trim, shape, device="cpu", seed=seed + 1)
    assert all(not np.array_equal(a, b) for a, b in zip(packed, other, strict=True))


def test_gloss_and_textured_plastic_finishes_have_distinct_relief():
    shape = (512, 512)
    glossy, _ = textures.bake_gltf("PlasticGlossABS", shape, device="cpu")
    dash, _ = textures.bake_gltf("PlasticGrainedDash", shape, device="cpu")
    assert np.std(dash[1][..., :2]) > 4 * np.std(glossy[1][..., :2])
    assert np.min(dash[2][..., 1]) > np.max(glossy[2][..., 1])
    with pytest.raises(ValueError, match="Unknown plastic finish"):
        textures.PlasticSurface("unknown")
