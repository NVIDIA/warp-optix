# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Procedural mineral, stamped-metal, and tactile paving surfaces."""

import warp as wp


@wp.func
def _noise(uv: wp.vec2, seed: int, nx: int, ny: int) -> float:
    return wp.pnoise(wp.rand_init(seed), wp.cw_mul(uv, wp.vec2(float(nx), float(ny))), nx, ny)


@wp.func
def _cell_seed(seed: int, x: int, y: int, n: int):
    return wp.rand_init(seed, (x % n + n) % n + n * ((y % n + n) % n))


@wp.func
def _stone_chips(uv: wp.vec2, seed: int, n: int, resolution: float):
    grid = uv * float(n)
    ix, iy = int(wp.floor(grid[0])), int(wp.floor(grid[1]))
    nearest = float(100.0)
    second = float(100.0)
    mineral = float(0.0)
    size = float(0.0)
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            rng = _cell_seed(seed, ix + dx, iy + dy, n)
            center = wp.vec2(float(ix + dx), float(iy + dy))
            center += wp.vec2(wp.randf(rng, 0.07, 0.93), wp.randf(rng, 0.07, 0.93))
            delta = grid - center
            distance = wp.dot(delta, delta)
            if distance < nearest:
                second = nearest
                nearest = distance
                mineral = wp.randf(rng)
                size = wp.randf(rng, 0.12, 0.63)
            elif distance < second:
                second = distance
    # Keep each heterogeneous stone inside its own angular cell.
    dist = wp.min(size - wp.sqrt(nearest), 0.85 * (second - nearest) - 0.025)
    aa = 0.47 * float(n) / resolution
    return wp.smoothstep(-aa, aa, dist), mineral, second - nearest


@wp.func
def terrazzo(uv: wp.vec2, seed: int, resolution: float):
    color = wp.vec3(0.902, 0.894, 0.882)
    polish = float(0.31)
    relief = float(0.74)
    fine = _noise(uv, seed + 1, 505, 523) * wp.smoothstep(500.0, 1200.0, resolution)
    stain = _noise(uv, seed + 21, 25, 29)
    for layer in range(2):
        frequency = 112 - layer * 49
        q = uv + wp.vec2(0.0022 * _noise(uv, seed + 5, 141, 133), 0.0022 * _noise(uv, seed + 6, 133, 141))
        mask, variation, edge = _stone_chips(q, seed + 71 * layer, frequency, resolution)
        mineral = wp.lerp(wp.vec3(0.62, 0.623, 0.637), wp.vec3(0.94, 0.938, 0.916), variation)
        mineral += wp.vec3(0.045 * stain + 0.033 * fine + 0.038 * edge)
        color = wp.lerp(color, mineral, mask)
        relief = wp.lerp(relief, 0.75 + 0.02 * variation, mask)
        polish = wp.lerp(polish, 0.20 + 0.22 * variation, mask)
    return color + wp.vec3(0.027 * fine), relief + 0.009 * fine, polish + 0.06 * stain


@wp.func
def perforated_sheet(uv: wp.vec2, seed: int, resolution: float):
    grid = wp.cw_mul(uv, wp.vec2(16.0, 18.0))
    base_row = int(wp.floor(grid[1]))
    distance = float(10.0)
    for offset in range(-1, 2):
        row = base_row + offset
        x = grid[0] - 0.5 * float(row & 1)
        delta = wp.vec2(x - wp.floor(x) - 0.5, (grid[1] - float(row) - 0.5) * (16.0 / 18.0))
        distance = wp.min(distance, wp.length(delta))
    aa = 8.0 / resolution
    opacity = wp.smoothstep(0.381 - aa, 0.381 + aa, distance)
    rim = wp.smoothstep(0.365, 0.44, distance)
    q = uv + 0.003 * wp.vec2(_noise(uv, seed + 1, 142, 147), _noise(uv, seed + 2, 147, 142))
    unused_mask, zinc, boundary = _stone_chips(q, seed + 9, 60, resolution)
    crystal = wp.smoothstep(0.01, 0.07, boundary)
    scratch = _noise(uv, seed + 3, 302, 98)
    tone = 0.805 + 0.19 * (zinc - 0.5) * crystal + 0.033 * scratch
    return wp.vec3(tone, tone, tone + 0.013), 0.17 + 0.67 * rim, 0.15 + 0.16 * zinc + 0.045 * scratch, opacity


@wp.func
def tactile_paving(uv: wp.vec2, seed: int, resolution: float):
    grid = wp.cw_mul(uv, wp.vec2(5.0, 4.0))
    p = grid - wp.vec2(wp.floor(grid[0]), wp.floor(grid[1]))
    edge = wp.min(wp.min(p[0], 1.0 - p[0]), wp.min(p[1], 1.0 - p[1]))
    tile = wp.smoothstep(0.009 - 2.0 / resolution, 0.027 + 2.0 / resolution, edge)
    bar = p[1] * 12.0
    across = wp.abs(bar - wp.floor(bar) - 0.5)
    rib = 1.0 - wp.smoothstep(0.20, 0.37 + 20.0 / resolution, across)
    rib *= wp.smoothstep(0.03, 0.065, wp.min(p[0], 1.0 - p[0])) * tile
    mineral = _noise(uv, seed + 1, 184, 171)
    grit = _noise(uv, seed + 2, 570, 560) * wp.smoothstep(320.0, 1140.0, resolution)
    dirt = wp.smoothstep(0.09, 0.50, _noise(uv, seed + 3, 33, 38) + 0.33 * mineral)
    tone = 0.07 * mineral + 0.052 * grit - (0.035 + 0.05 * dirt) * (1.0 - rib)
    color = wp.lerp(wp.vec3(0.39, 0.36, 0.32), wp.vec3(0.655, 0.622, 0.574), tile) + wp.vec3(tone)
    return (
        color,
        0.28 + tile * 0.25 + 0.29 * rib + 0.019 * grit,
        0.43 + 0.26 * dirt - 0.14 * rib,
        0.76 + 0.18 * tile + 0.06 * rib,
    )


@wp.func
def riveted_metal(uv: wp.vec2, seed: int, resolution: float):
    rows = uv[1] * 6.0
    row = int(wp.floor(rows))
    columns = uv[0] * 4.0 + 0.5 * float(row & 1)
    col = int(wp.floor(columns))
    state = _cell_seed(seed, col, row, 12)
    p = wp.vec2(columns - wp.floor(columns), rows - wp.floor(rows))
    seam = wp.min(wp.min(p[0], 1.0 - p[0]), wp.min(p[1], 1.0 - p[1]))
    panel = wp.smoothstep(0.0, 0.007 + 2.0 / resolution, seam)
    rivet_region = wp.min(p[1], 1.0 - p[1])
    across = wp.abs(p[0] - 0.5)
    x = (across + 0.022) * 23.0
    rivet_distance = wp.length(wp.vec2(x - wp.floor(x) - 0.5, (rivet_region - 0.058) * 15.0))
    rivet = 1.0 - wp.smoothstep(0.08, 0.26 + 7.5 / resolution, rivet_distance)
    rivet *= 1.0 - wp.smoothstep(0.13, 0.144, across)
    smudge = _noise(uv, seed + 8, 32, 27)
    brushed = _noise(uv, seed + 4, 650, 30) * wp.smoothstep(350.0, 1300.0, resolution)
    tone = wp.randf(state, -0.021, 0.019) + 0.038 * smudge + 0.026 * brushed
    color = wp.vec3(0.511, 0.527, 0.55) + wp.vec3(tone - 0.17 * rivet - 0.15 * (1.0 - panel))
    return color, 0.37 + 0.034 * panel + 0.045 * rivet + 0.011 * smudge, 0.265 + 0.052 * smudge + 0.088 * rivet
