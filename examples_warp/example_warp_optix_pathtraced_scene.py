# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared palette and scene styling for path-traced geometry examples."""

from __future__ import annotations

import numpy as np


RAINBOW_SRGB = np.array(
    [
        (0.86, 0.06, 0.02),
        (0.92, 0.22, 0.01),
        (0.94, 0.42, 0.01),
        (0.88, 0.62, 0.01),
        (0.78, 0.72, 0.01),
        (0.46, 0.72, 0.01),
        (0.03, 0.60, 0.12),
        (0.00, 0.58, 0.38),
        (0.00, 0.62, 0.68),
        (0.00, 0.53, 0.88),
        (0.06, 0.30, 0.88),
        (0.22, 0.12, 0.78),
    ],
    dtype=np.float32,
)


def srgb_to_linear(colors: np.ndarray) -> np.ndarray:
    colors = np.asarray(colors, dtype=np.float32)
    return np.where(
        colors <= 0.04045,
        colors / 12.92,
        ((colors + 0.055) / 1.055) ** 2.4,
    )


def rainbow_height_slots(roots: np.ndarray) -> np.ndarray:
    """Assign palette slots in gently rippled bottom-to-top bands."""
    roots = np.asarray(roots, dtype=np.float32).reshape(-1, 3)
    directions = roots / np.linalg.norm(roots, axis=1, keepdims=True)
    height = 0.5 * (directions[:, 1] + 1.0)
    azimuth = np.arctan2(directions[:, 2], directions[:, 0])
    position = np.clip(height + 0.045 * np.sin(3.0 * azimuth + 0.5), 0.0, 1.0)
    return np.minimum(
        (position * len(RAINBOW_SRGB)).astype(np.uint32),
        len(RAINBOW_SRGB) - 1,
    )


def create_rainbow_materials(api) -> np.ndarray:
    """Create the shared saturated PBR palette and return its material IDs."""
    return np.array(
        [
            api.create_pbr_material(
                color,
                roughness=0.7,
                metallic=0.0,
                ior=1.46,
                specular=0.15,
                clearcoat=0.0,
                base_color_scale=1.0,
            )
            for color in srgb_to_linear(RAINBOW_SRGB)
        ],
        dtype=np.uint32,
    )


def add_checker_ground(api, height: float, size: float = 1000.0) -> int:
    """Add the large light checker ground used by the path-traced demos."""
    material = api.create_pbr_material(
        srgb_to_linear(np.array((0.7, 0.7, 0.7), dtype=np.float32)),
        roughness=0.8,
        metallic=0.0,
        ior=1.46,
        specular=0.75,
        clearcoat=0.03,
        clearcoat_roughness=0.4,
        u_subdiv=size,
        v_subdiv=size,
    )
    half_extent = 0.5 * size
    geometry = api.create_mesh(
        np.array(
            [
                (-half_extent, height, -half_extent),
                (half_extent, height, -half_extent),
                (half_extent, height, half_extent),
                (-half_extent, height, half_extent),
            ],
            dtype=np.float32,
        ),
        np.array(((0, 2, 1), (0, 3, 2)), dtype=np.uint32),
        normals=np.tile((0.0, 1.0, 0.0), (4, 1)).astype(np.float32),
        uvs=np.array(
            ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
            dtype=np.float32,
        ),
        material_id=material,
    )
    return api.create_instance(geometry)


def configure_demo_sky(api) -> None:
    """Apply the shared Kapla-inspired procedural sky."""
    api.set_use_procedural_sky(True)
    api.set_sky_parameters(
        (-0.55, 0.78, 0.30),
        multiplier=1.25,
        haze=0.18,
        saturation=0.9,
        sun_disk_intensity=1.2,
        sun_glow_intensity=0.8,
    )
