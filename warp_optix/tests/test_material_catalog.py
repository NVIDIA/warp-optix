# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import numpy as np
import pytest

from warp_optix import procedural_textures as textures


def test_discover_materials_and_parameters():
    entries = textures.list_materials()
    assert len({entry["id"] for entry in entries}) == len(entries)
    assert {"metal", "plastic", "wood", "wall"} <= set(textures.material_families())
    assert {entry["id"] for entry in textures.list_materials(family="plastic", query="automotive")} >= {
        "plastic/pebbled",
        "plastic/grained",
    }
    metal = textures.describe_material("metal/copper")
    assert "metalness" in metal["maps"]
    assert "scratched" in metal["parameters"]["properties"]["finish"]["enum"]
    assert textures.describe_material("PaintedDrywall")["parameters"]["properties"] == {}
    assert json.loads(json.dumps(metal)) == metal


def test_generate_resolves_ids_and_applies_parameters():
    shape = (32, 48)
    seed = 19
    overrides = {"finish": "scratched", "scratch_density": 5}
    configured = textures.MetalSurface("copper", **overrides)
    direct, direct_kwargs = textures.bake_gltf(configured, shape, device="cpu", seed=seed, first_texture_index=4)
    packed, kwargs = textures.generate_gltf(
        "metal/copper", shape, parameters=overrides, device="cpu", seed=seed, first_texture_index=4
    )
    np.testing.assert_array_equal(packed, direct)
    assert kwargs == direct_kwargs

    maps = textures.generate("Plastic013A", shape, maps="color", parameters={"color": [0.2, 0.6, 0.1]}, device="cpu")
    color = maps["color"].numpy()
    assert np.all(color[..., 1] > color[..., 0])
    assert tuple(textures.generate("PaintedDrywall", shape, device="cpu")) == (
        "color",
        "normal",
        "roughness",
        "displacement",
    )
    with pytest.raises(ValueError, match="Unsupported parameters"):
        textures.generate("PaintedDrywall", shape, parameters={"roughness": 0.5})
    with pytest.raises(KeyError, match="Available maps"):
        textures.generate("plastic/pebbled", shape, maps="metalness")
    with pytest.raises(KeyError, match="list_materials"):
        textures.describe_material("metal/unknown")
