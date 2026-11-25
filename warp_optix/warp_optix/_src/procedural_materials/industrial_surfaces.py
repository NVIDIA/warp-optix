# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Twisted rope, oxidized steel, and partially exposed wire reinforcement."""

import warp as wp


@wp.func
def _noise(uv: wp.vec2, seed: int, nx: int, ny: int) -> float:
    return wp.pnoise(wp.rand_init(seed), wp.vec2(uv[0] * float(nx), uv[1] * float(ny)), nx, ny)


@wp.func
def _fraction(x: float) -> float:
    return x - wp.floor(x)


@wp.func
def _rope(uv: wp.vec2, seed: int, resolution: float):
    # Integer turns of both the large strand and its inner yarns keep the
    # diagonal winding periodic on a rectangular tile.
    drift = _noise(uv, seed, 4, 6)
    cord = _fraction(8.0 * uv[0] - 6.0 * uv[1] + 0.028 * drift)
    seam_distance = wp.min(cord, 1.0 - cord)
    groove = 1.0 - wp.smoothstep(0.022, 0.25, seam_distance)
    bulk = wp.sin(wp.pi * cord) ** 0.48
    fiber = _noise(uv, seed + 7, 403, 739) * wp.smoothstep(390.0, 1540.0, resolution)
    strand = _noise(uv, seed + 19, 65, 121) * wp.smoothstep(64.0, 245.0, resolution)
    yarn_phase = 63.0 * uv[1] - 34.0 * uv[0] + 0.24 * bulk + 0.07 * drift + 0.16 * strand
    yarn = 0.5 + 0.5 * wp.cos(2.0 * wp.pi * yarn_phase)
    yarn = wp.lerp(0.48, yarn, wp.smoothstep(42.0, 170.0, resolution))
    thread = yarn * yarn
    dye = _noise(uv, seed + 2, 11, 13)
    tuft = wp.smoothstep(0.11, 0.39, strand + 0.31 * fiber)
    base = wp.vec3(0.108, 0.124, 0.059) + wp.vec3(0.054 * dye + 0.022 * fiber)
    twine = wp.vec3(0.334, 0.281, 0.126) + wp.vec3(0.083 * dye + 0.041 * fiber + 0.09 * tuft)
    color = wp.lerp(base, twine, 0.22 * bulk + 0.67 * thread * (1.0 - 0.80 * groove))
    height = 0.36 + 0.30 * bulk - 0.047 * groove + 0.094 * yarn + 0.015 * fiber + 0.047 * drift
    rough = 0.63 + 0.22 * groove - 0.14 * thread + 0.055 * dye
    return color, height, rough, 1.0 - 0.38 * groove, 0.0, 1.0


@wp.func
def _rust(uv: wp.vec2, seed: int, resolution: float):
    haze = _noise(uv, seed, 21, 19)
    grain = _noise(uv, seed + 3, 218, 267)
    flakes = _noise(uv, seed + 5, 85, 109)
    grit = _noise(uv, seed + 7, 517, 483) * wp.smoothstep(360.0, 1430.0, resolution)
    scratches = _noise(uv, seed + 13, 223, 27)
    stain = 0.61 * haze + 0.43 * flakes + 0.34 * grain + 0.25 * grit
    oxide = wp.smoothstep(-0.06, 0.33, stain + 0.10 * scratches)
    pale = wp.smoothstep(0.135, 0.43, stain * 0.47 + 0.58 * grit + 0.26 * grain)
    pit_field = -0.37 * flakes - 0.69 * grain - 0.34 * grit
    pit = wp.smoothstep(0.13, 0.44, pit_field)
    dark_scratch = wp.smoothstep(0.36, 0.63, scratches + 0.32 * flakes) * (1.0 - oxide)
    base = wp.vec3(0.265, 0.137, 0.072) + wp.vec3(0.078 * haze + 0.056 * grain + 0.034 * grit)
    fresh = wp.vec3(0.463, 0.222, 0.087) + wp.vec3(0.13 * pale + 0.074 * flakes)
    color = wp.lerp(base, fresh, oxide) - wp.vec3(0.10, 0.065, 0.030) * (0.65 * pit + 0.45 * dark_scratch)
    height = 0.49 + 0.053 * haze + 0.064 * flakes + 0.041 * grain + 0.023 * grit
    height += 0.058 * oxide + 0.031 * pale - 0.11 * pit - 0.05 * dark_scratch
    roughness = 0.84 + 0.078 * oxide + 0.060 * pit - 0.14 * dark_scratch
    return color, height, roughness, 1.0 - 0.22 * pit, 0.07 * dark_scratch, 1.0


@wp.func
def _reinforcement(uv: wp.vec2, seed: int, resolution: float):
    warp_x = _noise(uv, seed, 9, 13)
    warp_y = _noise(uv, seed + 2, 13, 9)
    concrete = _noise(uv, seed + 3, 5, 7) + 0.37 * _noise(uv, seed + 5, 11, 10)
    sand = _noise(uv, seed + 7, 117, 125)
    speckle = _noise(uv, seed + 13, 647, 693) * wp.smoothstep(360.0, 1350.0, resolution)
    gx = _fraction(16.0 * uv[0] + 0.071 * warp_x)
    gy = _fraction(16.0 * uv[1] + 0.071 * warp_y)
    x_distance = wp.min(gx, 1.0 - gx)
    y_distance = wp.min(gy, 1.0 - gy)
    distance = wp.min(x_distance, y_distance)
    contact = wp.max(x_distance, y_distance)
    joint = 1.0 - wp.smoothstep(0.015, 0.08, contact)
    unbroken = wp.smoothstep(-0.18, 0.19, concrete + 0.083 * sand + 0.027 * speckle)
    exposed = 1.0 - unbroken

    # The exposed rod is round. Its steel core sits above a narrow cavity,
    # with rust on the rod and a separate, softer stain in the concrete.
    aa = wp.min(0.09, 7.0 / resolution)
    radius = 0.033 + 0.007 * sand + 0.005 * joint
    wire = 1.0 - wp.smoothstep(radius - aa, radius + aa, distance)
    cross_section = wp.sqrt(wp.max(0.0, 1.0 - (distance / radius) * (distance / radius)))
    trench = (1.0 - wp.smoothstep(0.052, 0.106 + 1.4 * aa, distance + 0.025 * sand)) * exposed
    stain = (1.0 - wp.smoothstep(0.058, 0.21, distance + 0.052 * sand)) * exposed
    aggregate = wp.smoothstep(0.14, 0.37, sand + 0.34 * speckle)
    cement = wp.vec3(0.441, 0.416, 0.384) + wp.vec3(0.067 * sand + 0.085 * concrete + 0.045 * speckle)
    cement -= wp.vec3(0.096, 0.081, 0.058) * aggregate
    stain_color = wp.vec3(0.331, 0.210, 0.147) + wp.vec3(0.092 * sand + 0.082 * speckle)
    color = wp.lerp(cement, stain_color, 0.69 * stain)
    color = wp.lerp(color, wp.vec3(0.175, 0.151, 0.136), 0.53 * trench)

    oxide_field = _noise(uv, seed + 21, 147, 145) + 0.49 * speckle
    bare = wp.smoothstep(0.06, 0.46, oxide_field + 0.17 * joint) * wp.smoothstep(0.05, 0.92, cross_section)
    rust = wp.vec3(0.334, 0.196, 0.122) + wp.vec3(0.121 * sand + 0.106 * speckle)
    steel = wp.vec3(0.408, 0.396, 0.373) + wp.vec3(0.088 * speckle)
    coverage = wire * exposed
    color = wp.lerp(color, wp.lerp(rust, steel, bare), coverage)

    height = 0.39 + 0.19 * concrete + 0.15 * unbroken + 0.048 * sand + 0.023 * speckle
    height -= 0.118 * trench + 0.024 * aggregate
    shaft_height = 0.43 + 0.142 * cross_section + 0.02 * joint + 0.027 * oxide_field
    height = wp.lerp(height, shaft_height, coverage)
    ao = 0.98 - 0.36 * trench * (1.0 - 0.83 * coverage) - 0.048 * aggregate
    rough = wp.lerp(0.75 + 0.075 * aggregate + 0.062 * sand, 0.85 - 0.32 * bare, coverage)
    return color, height, rough, ao, coverage * 0.72 * bare, 1.0


@wp.func
def industrial_surface(uv: wp.vec2, seed: int, resolution: float, style: int):
    uv -= wp.vec2(wp.floor(uv[0]), wp.floor(uv[1]))
    if style == 0:
        return _rope(uv, seed, resolution)
    if style == 1:
        return _rust(uv, seed, resolution)
    return _reinforcement(uv, seed, resolution)
