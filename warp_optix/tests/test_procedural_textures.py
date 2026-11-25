# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from warp_optix.pathtracing import Scene

from warp_optix import procedural_textures as textures


def test_new_material_families_provide_aligned_maps():
    ids = textures.reference_ids()
    assert len(ids) == 230
    assert sum(asset.startswith("WoodFloor") for asset in ids) == 74
    for material in ("Leather037", "Fabric004", "DiamondPlate009", "Bamboo001C", "Ice003", "Cork004"):
        assert {"color", "normal", "roughness", "displacement"} <= set(textures.reference_maps(material))


def test_reference_normals_have_unit_length():
    normals = textures.bake("Bricks113A", (64, 96), map_type="normal", device="cpu").numpy()
    np.testing.assert_allclose(np.linalg.norm(2.0 * normals - 1.0, axis=-1), 1.0, atol=1.0e-5)
    assert np.std(normals[..., 0]) > 0.01


def test_gltf_textures_pack_metalness_and_keep_existing_textures():
    scene = Scene()
    sentinel = np.zeros((1, 1, 4), dtype=np.uint8)
    scene.set_gltf_textures([sentinel])
    shape = (33, 47)
    images, kwargs = textures.bake_gltf("Metal064", shape, device="cpu", first_texture_index=1)
    scene.set_gltf_textures(images, append=True)
    material_id = scene.materials.add_gltf_material(**kwargs)

    np.testing.assert_array_equal(scene._gltf_textures[0], sentinel)
    assert material_id == 0
    assert kwargs["normal_texture"] == {"index": 2}
    assert kwargs["metallic_roughness_texture"] == {"index": 3}
    metal = textures.bake("Metal064", shape, map_type="metalness", device="cpu").numpy()[..., 0]
    np.testing.assert_array_equal(scene._gltf_textures[3][..., 2], (metal * 255.0 + 0.5).astype(np.uint8))


def test_gltf_opacity_uses_base_color_alpha():
    shape = (33, 47)
    images, kwargs = textures.bake_gltf("RoofingTiles013A", shape, device="cpu")
    opacity = textures.bake("RoofingTiles013A", shape, map_type="opacity", device="cpu").numpy()[..., 0]
    assert kwargs["alpha_mode"] == "MASK"
    np.testing.assert_array_equal(images[0][..., 3], (opacity * 255.0 + 0.5).astype(np.uint8))


def test_seeded_variants_repeat_and_keep_maps_aligned():
    shape = (48, 72)
    a, _ = textures.bake_gltf("Bricks113A", shape, device="cpu", seed=42)
    b, _ = textures.bake_gltf("Bricks113A", shape, device="cpu", seed=42)
    c, _ = textures.bake_gltf("Bricks113A", shape, device="cpu", seed=43)
    for first, same, other in zip(a, b, c, strict=True):
        np.testing.assert_array_equal(first, same)
        assert not np.array_equal(first, other)
    normal = textures.bake("Bricks113A", shape, map_type="normal", device="cpu", seed=42).numpy()
    np.testing.assert_array_equal(a[1][..., :3], (normal * 255.0 + 0.5).astype(np.uint8))


def test_seeded_paving_normals_follow_height_slopes():
    shape = (128, 160)
    normal = textures.bake("PavingStones153", shape, map_type="normal", device="cpu", seed=17).numpy()
    height = textures.bake("PavingStones153", shape, map_type="displacement", device="cpu", seed=17).numpy()[..., 0]
    dx = np.roll(height, -1, axis=1) - np.roll(height, 1, axis=1)
    dy = np.roll(height, -1, axis=0) - np.roll(height, 1, axis=0)
    assert np.corrcoef(-dx.ravel(), normal[..., 0].ravel())[0, 1] > 0.6
    assert np.corrcoef(dy.ravel(), normal[..., 1].ravel())[0, 1] > 0.6
