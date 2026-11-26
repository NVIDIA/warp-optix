# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interlocking dark roof tiles and fine-grained brown upholstery."""

import warp as wp

from .sampling import periodic_noise as _noise
from .weathered_surfaces import _soil_cell


@wp.func
def dark_roof_tiles(uv: wp.vec2, seed: int, resolution: float):
    uv = uv - wp.vec2(wp.floor(uv[0]), wp.floor(uv[1]))
    count = 13
    weather = _noise(uv, seed, 16, 22)
    clay = _noise(uv, seed + 3, 85, 103)
    grit = _noise(uv, seed + 4, 393, 437) * wp.smoothstep(240.0, 1300.0, resolution)
    u = float(count) * uv[0] + 0.081 + 0.047 * _noise(uv, seed + 11, 9, 15)
    x = u - wp.floor(u)
    col = (int(wp.floor(u)) % count + count) % count
    column = wp.rand_init(seed + 13, col)
    # Separate courses seat at slightly different heights and bow at the nose.
    v = float(count) * uv[1] + 0.635 + wp.randf(column, -0.052, 0.052)
    v += 0.029 * _noise(uv, seed + 17, 19, 13) + 0.017 * wp.cos(2.0 * wp.pi * x)
    y = v - wp.floor(v)
    tile = col + count * ((int(wp.floor(v)) % count + count) % count)
    state = wp.rand_init(seed + 19, tile)
    dye = wp.randf(state, -0.012, 0.012)

    # Continue the rounded flange through the join into the pan of the adjacent
    # tile. As with the vertical overlap, its profile agrees at both endpoints.
    aa = wp.min(0.68 * float(count) / resolution, 0.105)
    shoulder = 1.0 - wp.smoothstep(-aa, 0.190 + aa, x)
    crown = wp.pow(wp.smoothstep(0.428, 0.973, x), 1.48)
    flange = crown + shoulder - wp.smoothstep(1.0 - aa, 1.190 + aa, x)
    toe = 0.033 + 0.010 * clay + aa
    lap = y + 1.0 - wp.smoothstep(-toe, toe, y) - wp.smoothstep(1.0 - toe, 1.0 + toe, y)
    underlap = wp.smoothstep(0.015, 0.050, y) * (1.0 - wp.smoothstep(0.055, 0.188, y))
    recess = wp.smoothstep(0.060, 0.145, x) * (1.0 - wp.smoothstep(0.149, 0.283, x))
    pore = wp.smoothstep(0.19, 0.68, -clay + 0.72 * grit)
    deposit = wp.smoothstep(0.32, 0.69, 0.68 * weather + 0.58 * clay + 0.29 * grit + 0.17 * underlap)

    height = 0.291 + 0.386 * flange + 0.295 * lap + 0.014 * weather + 0.011 * grit - 0.013 * pore
    color = wp.vec3(0.195, 0.191, 0.211) + wp.vec3(
        dye + 0.031 * weather + 0.025 * clay + 0.018 * grit + 0.009 * flange - 0.014 * underlap - 0.014 * pore
    )
    color += wp.vec3(0.064, 0.054, 0.022) * deposit
    roughness = 0.453 + 0.104 * flange + 0.068 * lap + 0.134 * weather + 0.098 * clay
    roughness += 0.047 * pore + 0.053 * deposit
    ao = wp.clamp(0.77 + 0.136 * lap + 0.117 * flange - 0.199 * recess - 0.185 * underlap, 0.0, 1.0)

    # The reference represents a continuous field of overlapping tiles.
    return color, height, roughness, ao, 0.0, 1.0


@wp.func
def grained_leather(uv: wp.vec2, seed: int, resolution: float):
    uv = uv - wp.vec2(wp.floor(uv[0]), wp.floor(uv[1]))
    count = 87
    crease_drift = wp.vec2(_noise(uv, seed + 2, 79, 92), _noise(uv, seed + 3, 95, 81))
    crease_drift += 0.31 * wp.vec2(_noise(uv, seed + 4, 171, 199), _noise(uv, seed + 5, 189, 164))
    border, tint = _soil_cell(uv + 0.0045 * crease_drift, seed + 19, count)
    fine = _noise(uv, seed + 7, 483, 411) * wp.smoothstep(280.0, 1500.0, resolution)
    grain = _noise(uv, seed + 11, 242, 211)
    pores = wp.smoothstep(0.15, 0.58, -fine + 0.38 * grain)
    variation = wp.smoothstep(110.0, 465.0, resolution)
    aa = 12.0 / resolution
    shoulder = wp.smoothstep(0.005, 0.140 + aa, border)
    cushion = wp.smoothstep(0.030, 0.370 + aa, border)
    height = 0.59 + variation * (-0.390 + 0.390 * shoulder + 0.155 * cushion + 0.061 * tint * cushion)
    height += 0.037 * grain * variation - 0.026 * pores

    # The finish is fairly uniform in color. The network shows primarily in
    # the relief and in the rougher valleys between soft rounded grain.
    tone = 0.014 * (height - 0.59) + 0.005 * grain * variation + 0.004 * fine
    color = wp.vec3(0.391, 0.219, 0.159) + wp.vec3(0.80, 0.89, 1.25) * tone
    roughness = 0.371 + variation * (0.162 * (1.0 - shoulder) + 0.043 * grain) + 0.026 * pores
    return color, height, roughness, 1.0, 0.0, 1.0
