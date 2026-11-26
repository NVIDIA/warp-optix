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
def _blade_layer(
    uv: wp.vec2,
    seed: int,
    count: int,
    aa: float,
    direction: float,
):
    grid = uv * float(count) - wp.vec2(0.5)
    col, row = int(wp.floor(grid[0])), int(wp.floor(grid[1]))
    coverage = float(0.0)
    shade = float(0.5)
    lift = float(0.0)
    # Check adjacent cells so a long blade survives crossing its source cell.
    # Centers stay within 0.16 cells of the lattice and blades reach <0.82.
    for dy in range(2):
        for dx in range(2):
            x, y = col + dx, row + dy
            index = (x % count + count) % count + count * ((y % count + count) % count)
            rng = wp.rand_init(seed, index)
            center = wp.vec2(float(x) + wp.randf(rng, -0.16, 0.16), float(y) + wp.randf(rng, -0.16, 0.16))
            q = grid - center
            if wp.dot(q, q) < 0.70:
                angle = direction + wp.randf(rng, -1.03, 1.03)
                tangent = wp.vec2(wp.cos(angle), wp.sin(angle))
                length = wp.randf(rng, 0.40, 0.65)
                along = wp.dot(q, tangent) / length
                curl = wp.randf(rng, -0.22, 0.22) * along * along
                across = wp.dot(q, wp.vec2(-tangent[1], tangent[0])) - curl
                breadth = wp.randf(rng, 0.088, 0.15)
                width = breadth * wp.smoothstep(-1.12, -0.52, along) * (1.0 - wp.smoothstep(-0.02, 1.0, along))
                body = 1.0 - wp.smoothstep(width - aa, width + aa, wp.abs(across))
                body *= 1.0 - wp.smoothstep(0.86, 1.0, wp.abs(along))
                if body > coverage:
                    coverage = body
                    crown = 1.0 - wp.smoothstep(0.0, breadth, wp.abs(across))
                    shade = wp.randf(rng) + 0.27 * crown - 0.16 * along
                    lift = 0.09 + crown * 0.07 - 0.025 * along
    return coverage, shade, lift


@wp.func
def lawn_surface(uv: wp.vec2, seed: int, resolution: float):
    patch = _noise(uv, seed, 13, 9)
    flow = _noise(uv, seed + 1, 24, 19)
    grain = _noise(uv, seed + 2, 283, 277) * wp.smoothstep(360.0, 1240.0, resolution)
    color = wp.vec3(0.355, 0.479, 0.140) + wp.vec3(0.070 * patch + 0.038 * grain)
    height = 0.52 + 0.048 * patch + 0.018 * grain
    air = float(0.0)
    for band in range(4):
        frequency = 47 + 15 * band
        offset = wp.vec2(0.1727 * float(band), -0.2131 * float(band))
        direction = 2.0 * flow + 0.82 * float(band)
        aa = 0.47 * float(frequency) / resolution
        blade, shade, lift = _blade_layer(uv + offset, seed + 51 * band, frequency, aa, direction)
        blade *= wp.smoothstep(2.8 * float(frequency), 7.8 * float(frequency), resolution)
        dark = wp.vec3(0.295, 0.401, 0.110)
        light = wp.vec3(0.477, 0.604, 0.209)
        pigment = wp.lerp(dark, light, wp.clamp(shade, 0.0, 1.0)) + wp.vec3(0.043 * patch + 0.027 * grain)
        color = wp.lerp(color, pigment, blade)
        height = wp.lerp(height, 0.48 + lift + 0.035 * patch, blade)
        air = wp.max(air, blade)
    return color, height, 0.87 - 0.12 * air, 0.92 + 0.08 * air, 0.0, 1.0


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
    thread = 1.0 - wp.smoothstep(0.004, 0.038 + 5.0 / resolution, branch + 0.088 * border)
    thread *= wp.smoothstep(-0.10, 0.27, border + haze)
    weather = wp.smoothstep(0.02, 0.34, border + 0.63 * _noise(uv, seed + 9, 166, 141))
    weather *= 1.0 - wp.smoothstep(0.026, 0.19, boundary + 0.09 * haze)
    partial = wp.min(0.76, 0.66 * fracture + 0.43 * thread + 0.67 * weather)
    fleck = _noise(uv, seed + 6, 391, 382) * wp.smoothstep(350.0, 1150.0, resolution)
    matrix = wp.vec3(0.683, 0.685, 0.715) + wp.vec3(
        0.085 * haze + 0.016 * tone * fracture + 0.064 * border + 0.061 * fleck
    )
    color = wp.lerp(matrix, wp.vec3(0.408, 0.409, 0.443), partial)
    height = 0.57 + 0.019 * haze - 0.027 * partial + 0.006 * fleck
    return color, height, 0.31 + 0.084 * partial + 0.048 * haze, 1.0, 0.0, 1.0


@wp.func
def lake_ice(uv: wp.vec2, seed: int, resolution: float):
    drift = wp.vec2(_noise(uv, seed, 6, 7), _noise(uv, seed + 1, 7, 6))
    shear = wp.vec2(_noise(uv, seed + 2, 25, 21), _noise(uv, seed + 4, 21, 27))
    warped = uv + 0.062 * drift + 0.013 * shear
    boundary, unused_tone = _soil_cell(warped, seed + 9, 10)
    chipped = _noise(uv, seed + 11, 83, 71)
    inner = 1.0 - wp.smoothstep(0.004, 0.029 + 5.5 / resolution, boundary + 0.013 * chipped)
    mist = 1.0 - wp.smoothstep(0.016, 0.18, boundary + 0.079 * chipped)
    surge = _noise(uv, seed + 13, 14, 12)
    vein = inner * wp.smoothstep(-0.34, 0.27, surge + 0.37 * chipped)
    cloud = mist * wp.smoothstep(-0.14, 0.25, surge + 0.30 * chipped)
    frost = _noise(uv, seed + 17, 431, 477) * wp.smoothstep(270.0, 1300.0, resolution)
    crystals = wp.smoothstep(0.22, 0.59, frost + 0.42 * chipped)
    dark = wp.vec3(0.205, 0.278, 0.294) + wp.vec3(0.15 * surge + 0.036 * frost)
    haze = wp.vec3(0.59, 0.66, 0.666) + wp.vec3(0.071 * frost)
    color = wp.lerp(dark, haze, wp.clamp(0.38 * cloud + 0.71 * vein + 0.27 * crystals, 0.0, 1.0))
    height = 0.62 - 0.029 * vein + 0.008 * surge + 0.007 * crystals
    return color, height, 0.17 + 0.135 * cloud + 0.105 * crystals, 1.0, 0.0, 1.0
