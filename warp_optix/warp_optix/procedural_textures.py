# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tileable material textures generated with standard Warp launches."""

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
from warp_optix._src.procedural_materials.configurable_metals import MetalSurface, metal_names
from warp_optix._src.procedural_materials.plastic_surfaces import PlasticSurface


__all__ = [
    "MetalSurface", "PlasticSurface", "bake", "bake_gltf", "bake_layered_gltf", "brick",
    "marble", "metal_names", "reference_ids", "reference_maps", "wood",
]
