# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import warp as wp
from warp_optix.pathtracing import Scene

from warp_optix import procedural_textures as textures


def test_new_material_families_provide_aligned_maps():
    ids = textures.reference_ids()
    assert len(ids) == len(set(ids))
    assert {
        "LimewashWall",
        "WhiteSubwayTile",
        "GraphiteConcrete",
        "Wood087",
        "PaintedDrywall",
        "AsphaltShingles",
        "GalvanizedZinc",
        "EngineeredQuartz",
        "PlasticGlossABS",
        "WoodFloor071",
        "Fabric004",
        "Leather037",
        "Metal064",
        "Bricks113A",
        "RoofingTiles013A",
    } <= set(ids)
    if "Cork004" not in ids:
        return
    assert sum(asset.startswith("WoodFloor") for asset in ids) == 74
    for material in ("Leather037", "Fabric004", "DiamondPlate009", "Bamboo001C", "Ice003", "Cork004"):
        assert {"color", "normal", "roughness", "displacement"} <= set(textures.reference_maps(material))


def test_reference_normals_have_unit_length():
    normals = textures.bake("Bricks113A", (64, 96), map_type="normal", device="cpu").numpy()
    np.testing.assert_allclose(np.linalg.norm(2.0 * normals - 1.0, axis=-1), 1.0, atol=1.0e-5)
    assert np.std(normals[..., 0]) > 0.01


@pytest.mark.gpu
@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA unavailable")
def test_2k_bake_adds_fine_surface_detail():
    image = textures.bake("Concrete049", (2048, 2048), device="cuda").numpy()
    assert image.shape == (2048, 2048, 3)
    groups = image[..., 0].reshape(512, 4, 512, 4)
    average = groups.mean(axis=(1, 3), keepdims=True)
    assert np.sqrt(np.mean((groups - average) ** 2)) > 0.01


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
    normal[..., 1] = 1.0 - normal[..., 1]
    np.testing.assert_array_equal(a[1][..., :3], (normal * 255.0 + 0.5).astype(np.uint8))


@pytest.mark.parametrize(
    "material",
    ["PavingStones153", "Concrete049", "Wood099", "WoodFloor071", "Bricks101", "WhiteSubwayTile", "LimewashWall"],
)
def test_analytic_normals_follow_seeded_height(material):
    if material not in textures.reference_ids():
        pytest.skip("Material requires the full catalog")
    shape = (128, 160)
    normal = textures.bake(material, shape, map_type="normal", device="cpu", seed=17).numpy()
    height = textures.bake(material, shape, map_type="displacement", device="cpu", seed=17).numpy()[..., 0]
    dx = np.roll(height, -1, axis=1) - np.roll(height, 1, axis=1)
    dy = np.roll(height, -1, axis=0) - np.roll(height, 1, axis=0)
    assert np.corrcoef(-dx.ravel(), normal[..., 0].ravel())[0, 1] > 0.99
    assert np.corrcoef(dy.ravel(), normal[..., 1].ravel())[0, 1] > 0.99


@pytest.mark.parametrize(
    "material",
    ["SheetMetal002", "Fabric084", "Ground111", "Bark014", "Wicker010A", "Planks037A", "Fence001", "MetalWalkway010"],
)
def test_analytic_packed_maps_match_individual_bakes(material):
    shape = (81, 112)
    seed = 71
    maps, properties = textures.bake_gltf(material, shape, device="cpu", seed=seed)
    channels = {
        "color": (0, slice(0, 3)),
        "normal": (1, slice(0, 3)),
        "roughness": (2, 1),
        "occlusion": (2, 0),
        "metalness": (2, 2),
        "opacity": (0, 3),
    }
    for name in textures.reference_maps(material):
        if name == "displacement":
            continue
        image, channel = channels[name]
        expected = textures.bake(material, shape, map_type=name, device="cpu", seed=seed).numpy()
        if name == "normal":
            expected[..., 1] = 1.0 - expected[..., 1]
        if isinstance(channel, int):
            expected = expected[..., 0]
        np.testing.assert_array_equal(maps[image][..., channel], (255.0 * expected + 0.5).astype(np.uint8))
    if "opacity" in textures.reference_maps(material):
        assert properties["alpha_mode"] == "MASK"
        assert np.min(maps[0][..., 3]) < 128 < np.max(maps[0][..., 3])


def test_material_mask_preserves_endpoints_and_blends_in_between():
    shape = (48, 64)
    base, overlay = "Concrete008", "Ground094C"
    a, _ = textures.bake_gltf(base, shape, device="cpu", seed=3)
    b, _ = textures.bake_gltf(overlay, shape, device="cpu", seed=3)
    mask = np.full(shape, 0.5, dtype=np.float32)
    mask[:, :16] = 0
    mask[:, -16:] = 1
    maps, properties = textures.bake_layered_gltf(base, overlay, shape, device="cpu", seed=3, mask=mask)
    assert "occlusion_texture" in properties
    for actual, first, second in zip(maps, a, b, strict=True):
        np.testing.assert_array_equal(actual[:, :16], first[:, :16])
        np.testing.assert_array_equal(actual[:, -16:], second[:, -16:])
    average = (a[0][:, 16:-16, :3].astype(float) + b[0][:, 16:-16, :3]) / 2
    np.testing.assert_allclose(maps[0][:, 16:-16, :3], average, atol=0.5)
    norm = np.linalg.norm(2 * maps[1][..., :3].astype(float) / 255 - 1, axis=-1)
    np.testing.assert_allclose(norm, 1, atol=0.008)
    procedural, _ = textures.bake_layered_gltf(base, overlay, shape, device="cpu", seed=3, height_blend=0.5)
    assert not np.array_equal(procedural[0], maps[0])


def test_dark_foam_packs_color_with_sub_byte_precision():
    shape = (256, 293)
    images, properties = textures.bake_gltf("AcousticFoam001", shape, device="cpu")
    actual = images[0][..., :3].astype(np.float32) * (properties["base_color"][0] / 255.0)
    reference = textures.bake("AcousticFoam001", shape, device="cpu").numpy()
    np.testing.assert_allclose(actual, reference, atol=0.5 / (8.0 * 255.0) + 1e-7)
    assert len(np.unique(images[0][..., 0])) > 3


def test_configurable_metal_uses_the_same_bake_api():
    shape = (64, 80)
    polished = textures.MetalSurface("copper", finish="polished")
    worn = textures.MetalSurface("copper", finish="scratched", scratches=1.0, direction=32)
    assert "titanium" in textures.metal_names()
    assert textures.reference_maps(worn) == ("color", "normal", "roughness", "displacement", "metalness")
    polished_roughness = textures.bake(polished, shape, map_type="roughness", device="cpu").numpy()
    worn_roughness = textures.bake(worn, shape, map_type="roughness", device="cpu", seed=7).numpy()
    assert np.max(polished_roughness) < np.min(worn_roughness)
    maps, kwargs = textures.bake_gltf(worn, shape, device="cpu", seed=7)
    normal = textures.bake(worn, shape, map_type="normal", device="cpu", seed=7).numpy()
    gltf_normal = normal.copy()
    gltf_normal[..., 1] = 1.0 - gltf_normal[..., 1]
    np.testing.assert_array_equal(maps[1][..., :3], (255.0 * gltf_normal + 0.5).astype(np.uint8))
    assert kwargs["metallic_roughness_texture"] == {"index": 2}
    assert (maps[2][..., 2] == 255).all()
    assert not np.array_equal(normal, textures.bake(worn, shape, map_type="normal", device="cpu", seed=8).numpy())
    layered, _ = textures.bake_layered_gltf(polished, worn, shape, mask=np.ones(shape), device="cpu", seed=7)
    np.testing.assert_array_equal(layered, maps)
    with pytest.raises(ValueError, match="roughness"):
        textures.MetalSurface(roughness=2.0)


def test_metal_scratches_change_relief_and_normal_together():
    shape = (256, 384)
    clean = textures.MetalSurface("aluminum", finish="polished", scratches=0.0)
    worn = textures.MetalSurface("aluminum", finish="polished", scratches=1.0, direction=27)
    height = textures.bake(worn, shape, map_type="displacement", device="cpu", seed=19).numpy()[..., 0]
    normals = 2.0 * textures.bake(worn, shape, map_type="normal", device="cpu", seed=19).numpy() - 1.0
    clean_normal = textures.bake(clean, shape, map_type="normal", device="cpu").numpy()
    np.testing.assert_array_equal(clean_normal, np.broadcast_to((0.5, 0.5, 1.0), (*shape, 3)))
    assert height.min() < 0.54 and height.max() > 0.55
    assert np.ptp(normals[..., 0]) > 0.02 and np.ptp(normals[..., 1]) > 0.02
    dx = np.roll(height, 1, axis=1) - np.roll(height, -1, axis=1)
    dy = np.roll(height, -1, axis=0) - np.roll(height, 1, axis=0)
    assert np.corrcoef(dx.ravel(), normals[..., 0].ravel())[0, 1] > 0.97
    assert np.corrcoef(dy.ravel(), normals[..., 1].ravel())[0, 1] > 0.97
    maps, _ = textures.bake_gltf(worn, shape, device="cpu", seed=19)
    # Ignore slopes smaller than one 8-bit step around the flat normal.
    for axis, gradient in enumerate((dx, -dy)):
        component = maps[1][..., axis].astype(float) - 127.5
        pronounced = np.abs(component) > 1.0
        assert pronounced.any()
        assert np.mean(np.sign(component[pronounced]) == np.sign(gradient[pronounced])) > 0.98
