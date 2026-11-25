# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Closely cut lawn, patterned upholstery, and polished stone surfaces."""

import warp as wp

from .weathered_surfaces import _soil_cell


@wp.func
def _noise(uv: wp.vec2, seed: int, nx: int, ny: int):
    uv -= wp.vec2(wp.floor(uv[0]), wp.floor(uv[1]))
    return wp.pnoise(wp.rand_init(seed), wp.vec2(float(nx) * uv[0], float(ny) * uv[1]), nx, ny)


@wp.func
def _blade_layer(uv: wp.vec2, seed: int, count: int, aa: float):
    grid = uv * float(count)
    col, row = int(wp.floor(grid[0])), int(wp.floor(grid[1]))
    index = (col % count + count) % count + count * ((row % count + count) % count)
    rng = wp.rand_init(seed, index)
    center = wp.vec2(wp.randf(rng, 0.35, 0.65), wp.randf(rng, 0.30, 0.70))
    q = grid - wp.vec2(float(col), float(row)) - center
    angle = wp.randf(rng, 0.0, wp.pi)
    forward = wp.vec2(wp.cos(angle), wp.sin(angle))
    along = wp.dot(q, forward)
    width = wp.randf(rng, 0.083, 0.165) * wp.clamp(1.0 - 2.4 * along, 0.22, 1.4)
    cross = wp.abs(wp.dot(q, wp.vec2(-forward[1], forward[0])) + 0.26 * along * along)
    length = wp.randf(rng, 0.27, 0.40)
    blade = 1.0 - wp.smoothstep(width - aa, width + aa, cross)
    blade *= 1.0 - wp.smoothstep(length * 0.78, length, wp.abs(along))
    return blade, wp.randf(rng)


@wp.func
def lawn_surface(uv: wp.vec2, seed: int, resolution: float):
    worms = _noise(uv, seed, 71, 56)
    growth = wp.smoothstep(-0.40, 0.43, _noise(uv, seed + 1, 17, 15) + 0.41 * worms)
    dirt = wp.vec3(0.264, 0.290, 0.113) + wp.vec3(worms * 0.055)
    color = wp.lerp(dirt, wp.vec3(0.368, 0.477, 0.141), 0.64 + 0.28 * growth)
    height = 0.52 + 0.077 * worms
    exposed = float(0.0)
    for band in range(4):
        frequency = 77 + 43 * band
        offset = wp.vec2(0.1547 * float(band), -0.2123 * float(band))
        blade, pick = _blade_layer(uv + offset, seed + 51 * band, frequency, 0.47 * float(frequency) / resolution)
        gain = wp.smoothstep(float(frequency), 2.8 * float(frequency), resolution)
        blade *= gain
        straw = wp.vec3(0.60, 0.58, 0.286)
        green = wp.vec3(0.336, 0.497, 0.163)
        pigment = wp.lerp(green, straw, wp.smoothstep(0.76, 1.0, pick))
        pigment *= 0.75 + 0.29 * growth + 0.33 * pick
        color = wp.lerp(color, pigment, blade)
        height = wp.lerp(height, 0.62 + pick * 0.12 + 0.08 * worms, blade)
        exposed = wp.max(exposed, blade)
    return color, height, 0.88 - 0.12 * exposed, 0.85 + 0.15 * exposed, 0.0, 1.0


@wp.func
def leather_panel(uv: wp.vec2, seed: int, resolution: float):
    p = 4.0 * wp.vec2(uv[0] + uv[1], uv[1] - uv[0])
    local = p - wp.vec2(wp.floor(p[0]), wp.floor(p[1]))
    seam = wp.min(wp.min(local[0], 1.0 - local[0]), wp.min(local[1], 1.0 - local[1]))
    grain = _noise(uv, seed, 365, 355) * wp.smoothstep(285.0, 1150.0, resolution)
    fine = _noise(uv, seed + 1, 672, 679) * wp.smoothstep(400.0, 1600.0, resolution)
    inflated = 0.42 + 0.14 * wp.smoothstep(0.0, 0.35, seam)
    indent = 1.0 - wp.smoothstep(0.008, 0.019 + 2.1 / resolution, seam)
    stripe = 1.0 - wp.smoothstep(0.005, 0.014 + 2.1 / resolution, wp.abs(seam - 0.105))
    stitch_run = p[1]
    if wp.min(local[1], 1.0 - local[1]) < wp.min(local[0], 1.0 - local[0]):
        stitch_run = p[0]
    z = 30.0 * stitch_run
    stitch = wp.smoothstep(0.12, 0.28, wp.abs(z - wp.floor(z) - 0.5))
    thread = stripe * stitch
    patina = _noise(uv, seed + 8, 13, 15)
    color = wp.vec3(0.814, 0.727, 0.585) + wp.vec3(0.026 * grain + 0.019 * fine + 0.022 * patina)
    color -= wp.vec3(0.23, 0.22, 0.19) * (0.56 * thread + 0.43 * indent)
    height = inflated + 0.011 * grain + 0.006 * fine - 0.045 * indent + 0.017 * thread
    return color, height, 0.49 + 0.073 * grain + 0.18 * thread, 0.99 - 0.29 * indent, 0.0, 1.0


@wp.func
def onyx_layers(uv: wp.vec2, seed: int, resolution: float):
    fold = _noise(uv, seed, 4, 3) + 0.35 * _noise(uv, seed + 1, 9, 7)
    shear = _noise(uv, seed + 2, 13, 11)
    bed = uv[1] + uv[0] + 0.21 * fold + 0.035 * shear
    coarse = 0.5 + 0.5 * wp.sin(2.0 * wp.pi * (5.0 * bed + 0.42 * shear))
    layer = 47.0 * bed + 0.87 * _noise(uv, seed + 3, 18, 16) + 0.31 * _noise(uv, seed + 4, 73, 41)
    lamina = 1.0 - wp.smoothstep(0.065, 0.16 + 23.0 / resolution, wp.abs(layer - wp.floor(layer + 0.5)))
    crystal = _noise(uv, seed + 7, 355, 371) * wp.smoothstep(310.0, 1240.0, resolution)
    inclusion = wp.smoothstep(0.18, 0.48, _noise(uv, seed + 5, 96, 98) + 0.42 * crystal)
    light = wp.vec3(0.884, 0.875, 0.858)
    dark = wp.vec3(0.575, 0.551, 0.521)
    color = wp.lerp(light, dark, (0.46 + 0.23 * shear) * coarse + (0.24 + 0.24 * fold) * lamina + 0.22 * inclusion)
    color += wp.vec3(0.043 * crystal + 0.045 * shear)
    height = 0.58 + 0.035 * coarse - 0.015 * lamina - 0.020 * inclusion + 0.005 * crystal
    return color, height, 0.235 + 0.09 * coarse + 0.041 * crystal, 1.0, 0.0, 1.0


@wp.func
def fractured_marble(uv: wp.vec2, seed: int, resolution: float):
    haze = _noise(uv, seed, 9, 7)
    q = uv + 0.12 * wp.vec2(_noise(uv, seed + 1, 5, 7), _noise(uv, seed + 2, 7, 5))
    q += 0.015 * wp.vec2(_noise(uv, seed + 3, 44, 53), _noise(uv, seed + 4, 51, 39))
    q = wp.vec2(q[0] + q[1], q[1] - q[0])
    boundary, tone = _soil_cell(q, seed + 19, 6)
    border = _noise(uv, seed + 5, 84, 81)
    branch, unused = _soil_cell(q, seed + 20, 17)
    fracture = 1.0 - wp.smoothstep(0.008, 0.037 + 3.0 / resolution, boundary + 0.044 * border)
    fracture *= wp.smoothstep(-0.25, 0.22, haze + 0.23 * border)
    thread = (1.0 - wp.smoothstep(0.004, 0.038 + 5.0 / resolution, branch + 0.088 * border))
    thread *= wp.smoothstep(-0.10, 0.27, border + haze)
    weather = wp.smoothstep(0.02, 0.34, border + 0.63 * _noise(uv, seed + 9, 166, 141))
    weather *= 1.0 - wp.smoothstep(0.026, 0.19, boundary + 0.09 * haze)
    partial = wp.min(0.76, 0.66 * fracture + 0.43 * thread + 0.67 * weather)
    fleck = _noise(uv, seed + 6, 391, 382) * wp.smoothstep(350.0, 1150.0, resolution)
    matrix = wp.vec3(0.683, 0.685, 0.715) + wp.vec3(0.085 * haze + 0.016 * tone * fracture + 0.064 * border + 0.061 * fleck)
    color = wp.lerp(matrix, wp.vec3(0.408, 0.409, 0.443), partial)
    height = 0.57 + 0.019 * haze - 0.027 * partial + 0.006 * fleck
    return color, height, 0.31 + 0.084 * partial + 0.048 * haze, 1.0, 0.0, 1.0
