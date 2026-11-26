# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interlocking dark roof tiles and fine-grained brown upholstery."""

import warp as wp

from .roof_and_mosaic import _noise
from .weathered_surfaces import _soil_cell


@wp.func
def dark_roof_tiles(uv: wp.vec2, seed: int, resolution: float):
    uv = uv - wp.vec2(wp.floor(uv[0]), wp.floor(uv[1]))
    count = 13
    mottling = _noise(uv, seed, 17, 19)
    clay = _noise(uv, seed + 3, 95, 109)
    grit = _noise(uv, seed + 4, 355, 379) * wp.smoothstep(240.0, 1300.0, resolution)
    position = float(count) * uv + wp.vec2(
        0.081 + 0.040 * _noise(uv, seed + 11, 13, 19),
        0.635 + 0.044 * _noise(uv, seed + 13, 19, 11),
    )
    col, row = int(wp.floor(position[0])), int(wp.floor(position[1]))
    x, y = position[0] - float(col), position[1] - float(row)
    tile = (col % count + count) % count + count * ((row % count + count) % count)
    state = wp.rand_init(seed + 19, tile)
    dye = wp.randf(state, -0.018, 0.018)
    depth = wp.randf(state, 0.292, 0.310)
    toe = wp.randf(state, 0.033, 0.048)

    # The reference shows a broad flat pan, a rounded right-hand flange,
    # and a steep slip into the neighboring pan.
    aa = wp.clamp(0.70 * float(count) / resolution, 0.0, 0.11)
    flange = wp.smoothstep(0.40, 0.86, x) - wp.smoothstep(0.88 - aa, 1.0, x)
    flange *= 0.384
    joint = wp.smoothstep(1.0 - toe - aa, 1.0, y)
    lap = depth * (y - joint)
    pan_edge = 1.0 - wp.smoothstep(0.0, 0.140, x)
    seam = (1.0 - wp.smoothstep(0.012, 0.128, y)) * (1.0 - 0.26 * pan_edge)
    edge = wp.smoothstep(0.69, 0.84, x) * (1.0 - wp.smoothstep(0.94, 1.0, x))
    grain = 0.031 * mottling + 0.020 * clay + 0.010 * grit
    pore = wp.smoothstep(0.17, 0.64, -clay + 0.55 * grit)
    deposit = wp.smoothstep(0.39, 0.80, 0.53 * mottling + 0.62 * clay + 0.56 * grit)
    height = 0.309 + flange + lap + 0.006 * (clay + grit) - 0.008 * pore
    color = wp.vec3(0.196, 0.193, 0.212) + wp.vec3(dye + grain - 0.025 * seam - 0.010 * pan_edge)
    color += wp.vec3(0.066, 0.056, 0.027) * deposit
    roughness = 0.485 + 0.085 * edge - 0.053 * seam + 0.118 * mottling + 0.095 * clay + 0.030 * pore
    ao = (0.847 - 0.279 * pan_edge + 0.098 * edge) * (1.0 - 0.280 * seam - 0.081 * joint)

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
