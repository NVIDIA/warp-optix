# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from warp_optix.pathtracing.materials import MaterialManager


def test_material_defaults():
    manager = MaterialManager()
    entries = manager.get_material_entries()
    ground = entries[manager.add_default()]
    pbr = entries[manager.add_pbr(base_color=(0.8, 0.1, 0.02))]
    gltf = entries[manager.add_gltf_material()]

    # Unassigned surfaces use a realistic neutral reflectance.
    assert ground["pbrBaseColorFactor"] == pytest.approx((0.4479884, 0.4479884, 0.4479884, 1.0))
    # Programmatic PBR keeps its base color unmodified.
    assert pbr["pbrDiffuseFactor"][:2] == pytest.approx((0.0, 0.0))
    # The checker overlay and angular opacity are disabled unless requested.
    assert (gltf["uSubdiv"], gltf["vSubdiv"], gltf["baseColorScale"]) == pytest.approx((0.0, 0.0, 0.75))
    assert gltf["opacityFresnelLow"] == pytest.approx(-1.0)
    assert gltf["transmissionColorFactor"] == pytest.approx((-1.0, -1.0, -1.0))


def test_gltf_material_preserves_authored_controls():
    manager = MaterialManager()
    material = manager.get_material_entries()[
        manager.add_gltf_material(
            alpha_mode="MASK",
            alpha_cutoff=0.35,
            normal_texture={"index": 0, "scale": (0.75, -0.75)},
            occlusion_texture={"index": 4, "texCoord": 1},
            occlusion_strength=0.7,
            specular=0.45,
            specular_color=(0.8, 0.7, 0.6),
            clearcoat=0.85,
            clearcoat_roughness=0.12,
            thickness=2.0,
            base_color_add=0.1,
            base_color_desaturation=0.2,
            opacity_fresnel=(0.05, 0.2, 2.0),
            transmission_color=(1.0, 1.0, 1.0),
            u_subdiv=12.5,
            v_subdiv=7.25,
            base_color_scale=0.4,
        )
    ]

    assert material["alphaMode"] == 1
    assert material["alphaCutoff"] == pytest.approx(0.35)
    np.testing.assert_array_equal(material["normalTextureScale"], (0.75, -0.75))
    assert material["occlusionTexture"]["index"] == 4
    assert material["occlusionTexture"]["texCoord"] == 1
    assert material["occlusionStrength"] == pytest.approx(0.7)
    assert material["specularFactor"] == pytest.approx(0.45)
    assert material["specularColorFactor"] == pytest.approx((0.8, 0.7, 0.6))
    assert material["clearcoatFactor"] == pytest.approx(0.85)
    assert material["clearcoatRoughness"] == pytest.approx(0.12)
    assert material["thicknessFactor"] == pytest.approx(2.0)
    assert material["pbrDiffuseFactor"][:2] == pytest.approx((0.1, 0.2))
    assert (material["opacityFresnelLow"], material["opacityFresnelHigh"]) == pytest.approx((0.05, 0.2))
    assert material["opacityFresnelFalloff"] == pytest.approx(2.0)
    assert material["transmissionColorFactor"] == pytest.approx((1.0, 1.0, 1.0))
    assert (material["uSubdiv"], material["vSubdiv"], material["baseColorScale"]) == pytest.approx((12.5, 7.25, 0.4))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"u_subdiv": -1.0}, "u_subdiv and v_subdiv"),
        ({"v_subdiv": -1.0}, "u_subdiv and v_subdiv"),
        ({"base_color_scale": -0.1}, "base_color_scale"),
        ({"base_color_scale": 1.1}, "base_color_scale"),
    ],
)
def test_checker_overlay_rejects_out_of_range_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        MaterialManager().add_gltf_material(**kwargs)
