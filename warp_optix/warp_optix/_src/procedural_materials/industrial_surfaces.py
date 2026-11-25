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
    bands = _noise(uv, seed, 17, 5)
    islands = _noise(uv, seed + 3, 21, 17)
    runoff = _noise(uv, seed + 7, 84, 14) + 0.32 * _noise(uv, seed + 8, 201, 33)
    rough_edges = _noise(uv, seed + 9, 165, 236)
    powder = _noise(uv, seed + 11, 710, 687) * wp.smoothstep(340.0, 1450.0, resolution)
    scale = 0.55 * bands + 0.69 * islands + 0.32 * runoff + 0.19 * rough_edges
    fresh = wp.smoothstep(0.049, 0.355, scale + 0.075 * powder)
    spalls = _noise(uv, seed + 15, 47, 74)
    pits = wp.smoothstep(0.19, 0.55, -0.61 * rough_edges - 0.82 * spalls - 0.13 * powder + 0.14 * bands)
    older = wp.vec3(0.326, 0.163, 0.066) + wp.vec3(0.09 * bands + 0.094 * runoff + 0.034 * powder)
    fresh_oxide = wp.vec3(0.608, 0.247, 0.093) + wp.vec3(0.10 * rough_edges + 0.081 * runoff + 0.08 * powder)
    color = wp.lerp(older, fresh_oxide, fresh) - wp.vec3(0.113, 0.061, 0.025) * pits
    height = 0.39 + 0.17 * bands + 0.103 * runoff + 0.063 * islands + 0.095 * fresh - 0.178 * pits
    height += 0.034 * powder
    rough = wp.min(1.0, 0.82 + 0.13 * fresh + 0.11 * pits - 0.25 * wp.smoothstep(0.02, 0.42, runoff))
    metallic = 0.03 + 0.097 * (1.0 - fresh) * wp.smoothstep(-0.19, 0.37, -runoff)
    return color, height, rough, 1.0 - 0.20 * pits, metallic, 1.0


@wp.func
def _reinforcement(uv: wp.vec2, seed: int, resolution: float):
    # Rust bleeds from exposed wires. The same crumbling islands conceal
    # both those wires and their corresponding stains and metalness.
    warp_x = _noise(uv, seed, 9, 13)
    warp_y = _noise(uv, seed + 2, 13, 9)
    concrete = _noise(uv, seed + 3, 5, 7) + 0.37 * _noise(uv, seed + 5, 11, 10)
    sand = _noise(uv, seed + 7, 117, 125)
    speckle = _noise(uv, seed + 13, 647, 693) * wp.smoothstep(360.0, 1350.0, resolution)
    gx = _fraction(16.0 * uv[0] + 0.071 * warp_x)
    gy = _fraction(16.0 * uv[1] + 0.071 * warp_y)
    distance = wp.min(wp.min(gx, 1.0 - gx), wp.min(gy, 1.0 - gy))
    distance += 0.032 * sand + 0.010 * speckle
    unbroken = wp.smoothstep(-0.07, 0.29, concrete + 0.095 * sand)
    aa = wp.min(0.10, 8.4 / resolution)
    wire_width = 0.031 + 0.009 * sand
    wire = (1.0 - wp.smoothstep(wire_width - aa, wire_width + aa, distance)) * (1.0 - unbroken)
    trench = (1.0 - wp.smoothstep(0.036, 0.11, distance + 0.016 * sand)) * (1.0 - unbroken)
    stain = (1.0 - wp.smoothstep(0.071, 0.21, distance + 0.047 * sand)) * (1.0 - unbroken)
    aggregate = wp.smoothstep(0.14, 0.37, sand + 0.34 * speckle)
    cement = wp.vec3(0.414, 0.380, 0.353) + wp.vec3(0.051 * sand + 0.089 * concrete + 0.040 * speckle)
    cement -= wp.vec3(0.096, 0.081, 0.058) * aggregate
    oxide = wp.vec3(0.288, 0.158, 0.093) + wp.vec3(0.142 * sand + 0.078 * speckle)
    color = wp.lerp(cement, oxide, wp.clamp(0.52 * stain + 0.52 * wire, 0.0, 1.0))
    height = 0.323 + 0.31 * concrete + 0.26 * unbroken + 0.049 * sand + 0.018 * speckle
    height = wp.max(0.0, height + 0.121 * wire - 0.069 * trench - 0.023 * aggregate)
    ao = 0.94 - 0.17 * trench - 0.04 * aggregate + 0.023 * unbroken
    rough = 0.68 - 0.15 * wire + 0.085 * aggregate + 0.077 * sand
    bare_wire = wire * wp.smoothstep(0.24, 0.5, speckle + 0.38 * sand)
    return color, height, rough, ao, 0.013 * wire + 0.095 * bare_wire, 1.0


@wp.func
def industrial_surface(uv: wp.vec2, seed: int, resolution: float, style: int):
    uv -= wp.vec2(wp.floor(uv[0]), wp.floor(uv[1]))
    if style == 0:
        return _rope(uv, seed, resolution)
    if style == 1:
        return _rust(uv, seed, resolution)
    return _reinforcement(uv, seed, resolution)
