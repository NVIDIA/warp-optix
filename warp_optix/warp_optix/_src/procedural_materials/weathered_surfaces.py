# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Weathered paper and ground surfaces based on ambientCG references."""

import warp as wp

from .sampling import periodic_noise as _noise


@wp.func
def _cardboard(uv: wp.vec2, seed: int, resolution: float):
    tear_field = _noise(uv, seed, 5, 7)
    torn_edge = 0.30 * _noise(uv, seed + 1, 27, 35) + 0.12 * _noise(uv, seed + 2, 147, 89)
    fiber = _noise(uv, seed + 3, 760, 174) * wp.smoothstep(400.0, 1440.0, resolution)
    machine_marks = _noise(uv, seed + 4, 340, 12)
    torn_edge += 0.04 * fiber
    torn = wp.smoothstep(0.16 - 3.0 / resolution, 0.20 + 3.0 / resolution, tear_field + torn_edge)
    exposed = wp.smoothstep(0.17, 0.25, tear_field + torn_edge)
    fuzz = 0.11 * _noise(uv, seed + 5, 188, 331) + 0.095 * fiber
    flute = wp.abs(wp.sin(93.0 * wp.pi * 2.0 * uv[0]))
    pigment = 0.033 * _noise(uv, seed + 6, 20, 29) + 0.018 * machine_marks
    liner = wp.vec3(0.686, 0.504, 0.248) + wp.vec3(pigment + 0.035 * fiber)
    core = wp.vec3(0.559, 0.474, 0.313) + wp.vec3(pigment + fuzz - 0.039 * flute)
    color = wp.lerp(liner, core, torn) + wp.vec3(0.07 * wp.max(0.0, torn - exposed))
    height = 0.55 - 0.082 * exposed + 0.036 * (torn - exposed) + 0.020 * machine_marks
    height += 0.018 * fiber + exposed * (0.021 * flute + 0.028 * fuzz)
    return color, height, 0.77 + 0.11 * exposed + 0.05 * fiber, 1.0


@wp.func
def _soil_cell(uv: wp.vec2, seed: int, count: int):
    p = uv * float(count)
    x, y = int(wp.floor(p[0])), int(wp.floor(p[1]))
    nearest, next_nearest = float(10.0), float(10.0)
    offset = wp.vec2(0.0)
    next_offset = wp.vec2(0.0)
    tint = float(0.0)
    for j in range(-1, 2):
        for i in range(-1, 2):
            col, row = x + i, y + j
            wrapped = (col % count + count) % count + count * ((row % count + count) % count)
            state = wp.rand_init(seed, wrapped)
            center = wp.vec2(float(col), float(row)) + wp.vec2(wp.randf(state, 0.10, 0.90), wp.randf(state, 0.10, 0.90))
            delta = center - p
            distance = wp.dot(delta, delta)
            if distance < nearest:
                next_nearest, next_offset = nearest, offset
                nearest, offset = distance, delta
                tint = wp.randf(state, -1.0, 1.0)
            elif distance < next_nearest:
                next_nearest, next_offset = distance, delta
    border = 0.5 * (next_nearest - nearest) / wp.max(0.01, wp.length(next_offset - offset))
    return border, tint


@wp.func
def _dry_soil(uv: wp.vec2, seed: int, resolution: float):
    p = uv + 0.018 * wp.vec2(_noise(uv, seed, 17, 21), _noise(uv, seed + 1, 19, 21))
    border, tint = _soil_cell(p, seed + 5, 19)
    grit = _noise(uv, seed + 3, 629, 717) * wp.smoothstep(340.0, 1450.0, resolution)
    gravel = _noise(uv, seed + 6, 171, 188)
    coarse = _noise(uv, seed + 4, 101, 118)
    width = 0.031 + 0.019 * coarse
    crack = 1.0 - wp.smoothstep(width * 0.55, width + 10.0 / resolution, border)
    bevel = 1.0 - wp.smoothstep(width, width + 0.065, border)
    inclusion = wp.smoothstep(0.26, 0.50, gravel + 0.38 * grit)
    patch = _noise(uv, seed + 2, 12, 11)
    tone = 0.043 * tint + 0.075 * coarse + 0.060 * grit + 0.14 * inclusion + 0.11 * patch
    soil = wp.vec3(0.545, 0.515, 0.477) + wp.vec3(tone)
    color = wp.lerp(soil, wp.vec3(0.255, 0.249, 0.225) + wp.vec3(0.079 * grit), 0.79 * crack)
    height = 0.58 + 0.055 * coarse + 0.014 * grit + 0.067 * inclusion - 0.26 * bevel
    return color, height, 0.83 - 0.10 * inclusion + 0.08 * crack, 1.0 - 0.40 * bevel


@wp.func
def _damp_ripples(uv: wp.vec2, seed: int, resolution: float):
    drift = _noise(uv, seed, 4, 3)
    scour = _noise(uv, seed + 1, 10, 13)
    bunched = _noise(uv, seed + 2, 31, 27)
    x = 19.0 * uv[1] + 1.34 * drift + 0.88 * scour + 0.23 * bunched
    ridge = 0.5 + 0.5 * wp.sin(2.0 * wp.pi * (x + 0.19 * wp.sin(2.0 * wp.pi * x)))
    ridge = wp.smoothstep(0.24, 0.82, ridge + 0.31 * bunched)
    grain = _noise(uv, seed + 4, 505, 564) * wp.smoothstep(285.0, 1180.0, resolution)
    small = _noise(uv, seed + 5, 182, 219)
    damp = wp.smoothstep(0.07, 0.63, 1.02 - ridge + 0.33 * bunched)
    color = wp.lerp(wp.vec3(0.583, 0.505, 0.392), wp.vec3(0.330, 0.287, 0.224), damp)
    color += wp.vec3(0.079 * small + 0.055 * grain + 0.043 * drift)
    height = 0.35 + 0.29 * ridge + 0.080 * bunched + 0.033 * small + 0.017 * grain
    return color, height, 0.83 - 0.27 * damp + 0.070 * small, 0.89 + 0.11 * ridge


@wp.func
def weathered_surface(uv: wp.vec2, seed: int, resolution: float, style: int):
    if style == 0:
        return _cardboard(uv, seed, resolution)
    if style == 1:
        return _dry_soil(uv, seed, resolution)
    return _damp_ripples(uv, seed, resolution)
