# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discover materials and generate tileable Warp or packed path tracer textures."""

from warp_optix._src.procedural_materials.bake import (
    bake,
    bake_gltf,
    bake_layered_gltf,
    brick,
    marble,
    reference_ids,
    reference_maps,
    wood,
)
from warp_optix._src.procedural_materials.catalog import (
    describe_material,
    generate,
    generate_gltf,
    list_materials,
    material_families,
)
from warp_optix._src.procedural_materials.configurable_metals import MetalSurface, metal_names
from warp_optix._src.procedural_materials.plastic_surfaces import PlasticSurface

__all__ = [
    "MetalSurface",
    "PlasticSurface",
    "bake",
    "bake_gltf",
    "bake_layered_gltf",
    "brick",
    "describe_material",
    "generate",
    "generate_gltf",
    "list_materials",
    "marble",
    "material_families",
    "metal_names",
    "reference_ids",
    "reference_maps",
    "wood",
]
