# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pressed crumb rubber with blue-grey granules and fine binder pores."""

import warp as wp

from .mineral_surfaces import _stone_chips
from .sampling import periodic_noise as _noise


@wp.func
def crumb_rubber(uv: wp.vec2, seed: int, resolution: float):
    rough_grain = _noise(uv, seed + 10, 177, 193)
    fine = _noise(uv, seed + 11, 363, 381) * wp.smoothstep(420.0, 1200.0, resolution)
    pores = _noise(uv, seed + 12, 617, 601) * wp.smoothstep(740.0, 1900.0, resolution)
    uv_warp = 0.0011 * wp.vec2(rough_grain, _noise(uv, seed + 13, 179, 167))
    mask, tag, _ = _stone_chips(uv + uv_warp, seed + 14, 177, resolution)
    fleck = mask * wp.smoothstep(0.70, 0.85, tag)
    binder = wp.vec3(0.069, 0.073, 0.079) + wp.vec3(0.038 * rough_grain + 0.055 * fine)
    crumb = wp.vec3(0.097, 0.107, 0.118) * (0.68 + 0.55 * tag + 0.25 * rough_grain)
    pigment = wp.lerp(wp.vec3(0.17, 0.20, 0.22), wp.vec3(0.35, 0.37, 0.40), tag * tag)
    pigment += wp.vec3(0.067 * fine + 0.033 * pores)
    color = wp.lerp(wp.lerp(binder, crumb, mask), pigment, fleck)
    # Both crumb colors share the same compressed granule structure.
    height = 0.855 + mask * (0.027 + 0.067 * tag) + 0.105 * rough_grain + 0.067 * fine + 0.035 * pores
    roughness = 0.60 + 0.16 * rough_grain + 0.075 * fine - 0.047 * mask + 0.048 * fleck
    return color, height, roughness, 1.0, 0.0, 1.0
