# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Periodic crumpled foil and overlapping moss shoots."""

import warp as wp

from .sampling import wrap_index, wrap_uv, wrapped_noise


@wp.func
def _fold_corner(x: int, y: int, count: int, seed: int):
    x = wrap_index(x, count)
    y = wrap_index(y, count)
    return wp.randf(wp.rand_init(seed, x + count * y), -1.0, 1.0)


@wp.func
def _folds(uv: wp.vec2, count: int, seed: int):
    grid = uv * float(count)
    col = int(wp.floor(grid[0]))
    row = int(wp.floor(grid[1]))
    u = grid[0] - float(col)
    v = grid[1] - float(row)
    a = _fold_corner(col, row, count, seed)
    b = _fold_corner(col + 1, row, count, seed)
    c = _fold_corner(col, row + 1, count, seed)
    d = _fold_corner(col + 1, row + 1, count, seed)
    # A random diagonal joins four shared fold vertices. Keeping each face
    # planar produces real angle discontinuities across the crease.
    diagonal = _fold_corner(col, row, count, seed + 71)
    peak = float(0.0)
    seam = float(0.0)
    if diagonal > 0.0:
        seam = wp.abs(u - v)
        if u > v:
            peak = (1.0 - u) * a + (u - v) * b + v * d
        else:
            peak = (1.0 - v) * a + (v - u) * c + u * d
    else:
        seam = wp.abs(u + v - 1.0)
        if u + v < 1.0:
            peak = (1.0 - u - v) * a + u * b + v * c
        else:
            peak = (1.0 - v) * b + (1.0 - u) * c + (u + v - 1.0) * d
    seam = wp.min(seam, wp.min(wp.min(u, 1.0 - u), wp.min(v, 1.0 - v)))
    return peak, 1.0 - wp.smoothstep(0.017, 0.11, seam)


@wp.func
def _foil(uv: wp.vec2, seed: int, resolution: float):
    sway = wp.vec2(wrapped_noise(uv, seed + 17, 6, 6), wrapped_noise(uv, seed + 23, 7, 7))
    wrinkle = wp.vec2(wrapped_noise(uv, seed + 29, 17, 17), wrapped_noise(uv, seed + 31, 19, 19))
    bent = uv + 0.073 * sway + 0.024 * wrinkle
    broad, _ = _folds(bent, 9, seed + 37)
    cross = wp.vec2(bent[0] + bent[1], bent[1] - bent[0])
    angular, edge = _folds(cross, 22, seed + 43)
    small, fine_edge = _folds(bent + 0.012 * wrinkle, 51, seed + 53)
    fine = wp.smoothstep(100.0, 350.0, resolution)
    heights = 0.496 + 0.201 * broad + 0.222 * angular + 0.101 * fine * small
    brushed = wrapped_noise(uv, seed + 59, 193, 193) * wp.smoothstep(230.0, 880.0, resolution)
    color = wp.vec3(0.50 + 0.011 * brushed + 0.0045 * angular)
    roughness = 0.111 + 0.017 * brushed + 0.042 * fine_edge + 0.027 * edge
    occlusion = 0.996 - 0.057 * edge * (1.0 - wp.smoothstep(-0.55, 0.15, angular)) - 0.016 * fine_edge
    return color, heights, roughness, occlusion, 1.0, 1.0


@wp.func
def _moss_shoots(uv: wp.vec2, seed: int, resolution: float):
    count = 72
    grid = uv * float(count)
    origin_x = int(wp.floor(grid[0]))
    origin_y = int(wp.floor(grid[1]))
    local = grid - wp.vec2(float(origin_x), float(origin_y))
    aa = 0.40 * float(count) / wp.max(128.0, resolution)
    canopy = float(0.0)
    tint = float(0.0)
    tip = float(0.0)
    age = float(0.0)
    for y in range(-1, 2):
        for x in range(-1, 2):
            col = (origin_x + x + count) % count
            row = (origin_y + y + count) % count
            state = wp.rand_init(seed, col + count * row)
            center = wp.vec2(float(x) + wp.randf(state, 0.07, 0.93), float(y) + wp.randf(state, 0.07, 0.93))
            radius = wp.randf(state, 0.57, 0.94)
            d = (local - center) / radius
            d[0] *= wp.randf(state, 0.82, 1.25)
            r = wp.length(d)
            if r < 1.2:
                theta = wp.atan2(d[1], d[0]) + wp.randf(state, 0.0, 2.0 * wp.PI)
                branches = wp.randi(state, 4, 9)
                twist = wp.randf(state, -0.53, 0.57)
                phase = (theta * float(branches) + twist * r * r) / (2.0 * wp.PI)
                branch = int(wp.floor(phase + 0.5))
                branch_id = wrap_index(branch, branches)
                frond = wp.rand_init(seed + 97, 9 * (col + count * row) + branch_id)
                length = wp.randf(frond, 0.62, 1.14)
                width = wp.randf(frond, 0.24, 0.48)
                angular = wp.abs(phase - float(branch))
                taper = wp.smoothstep(0.09, 0.32, r) * (1.0 - wp.smoothstep(0.64 * length, length, r))
                leaf = 1.0 - wp.smoothstep(width * taper * 0.22, width * taper + 0.06, angular)
                root = wp.randf(frond, 0.26, 0.52)
                spread = root + (length - root) * leaf
                margin = spread - r
                fresh = wp.randf(state)
                blades = wp.smoothstep(-aa, aa + 0.06, margin)
                # A raised stalk and curved blades emerge from a common root.
                lifted = blades * (0.47 + 0.14 * leaf + 0.30 * (1.0 - r) + 0.13 * fresh)
                if lifted > canopy:
                    canopy = lifted
                    age = fresh
                    tint = wp.randf(state)
                    tip = 0.13 + 0.54 * wp.smoothstep(-0.02, 0.24, margin) + 0.28 * leaf
    return canopy, tint, tip, age


@wp.func
def _forest_litter(uv: wp.vec2, seed: int, resolution: float):
    # The canopy has occasional pine needles rather than a second green
    # surface. Use a long, tapered leaf with a central vein.
    count = 26
    grid = uv * float(count)
    x = int(wp.floor(grid[0]))
    y = int(wp.floor(grid[1]))
    col = wrap_index(x, count)
    row = wrap_index(y, count)
    state = wp.rand_init(seed, col + count * row)
    present = wp.randf(state)
    mask = float(0.0)
    vein = float(0.0)
    if present > 0.79:
        center = wp.vec2(wp.randf(state, 0.32, 0.68), wp.randf(state, 0.32, 0.68))
        delta = grid - wp.vec2(float(x), float(y)) - center
        angle = wp.randf(state, 0.0, 2.0 * wp.PI)
        axis = wp.vec2(wp.cos(angle), wp.sin(angle))
        side = wp.vec2(-axis[1], axis[0])
        along = wp.dot(delta, axis)
        across = wp.dot(delta, side) + 0.30 * along * along
        length = wp.randf(state, 0.18, 0.33)
        width = wp.randf(state, 0.013, 0.044)
        taper = wp.max(0.0, 1.0 - wp.abs(along) / length)
        aa = 0.45 * float(count) / wp.max(128.0, resolution)
        edge = width * taper - wp.abs(across)
        mask = wp.smoothstep(-aa, aa, edge) * wp.smoothstep(0.0, 0.12, taper)
        vein = 1.0 - wp.smoothstep(0.003, width, wp.abs(across))
    return mask, vein


@wp.func
def _moss(uv: wp.vec2, seed: int, resolution: float):
    patches = wrapped_noise(uv, seed + 7, 7, 7) + 0.57 * wrapped_noise(uv, seed + 11, 16, 16)
    deposit = wrapped_noise(uv, seed + 13, 39, 39)
    direction = wp.vec2(wrapped_noise(uv, seed + 17, 13, 13), wrapped_noise(uv, seed + 19, 11, 11))
    cover, tint, tips, age = _moss_shoots(uv + 0.009 * direction, seed + 23, resolution)
    soil = wp.vec3(0.263, 0.243, 0.176) + wp.vec3(0.098 * deposit + 0.059 * patches)
    grit = wrapped_noise(uv, seed + 29, 329, 329) * wp.smoothstep(230.0, 730.0, resolution)
    pores = wrapped_noise(uv, seed + 37, 603, 603) * wp.smoothstep(480.0, 1630.0, resolution)
    leaf_micro = 0.13 * grit + 0.11 * pores
    healthy = wp.smoothstep(-0.42, 0.45, patches + 0.80 * tint - 0.42)
    wet = wp.smoothstep(0.02, 0.36, patches + 0.4 * deposit)
    pale = wp.lerp(wp.vec3(0.556, 0.517, 0.332), wp.vec3(0.531, 0.587, 0.318), healthy)
    tinted = wp.lerp(wp.vec3(0.333, 0.337, 0.201), pale, 0.34 + 0.64 * age)
    flash = wp.smoothstep(0.72, 1.0, age) * wp.smoothstep(0.53, 0.91, tips)
    growth = tinted + wp.vec3(0.12 * tips + 0.083 * healthy + 0.15 * flash + leaf_micro)
    mask = wp.smoothstep(0.015, 0.51, cover + 0.19 * patches + 0.10 * deposit)
    undergrowth = wp.smoothstep(-0.18, 0.32, 0.46 * patches + 0.43 * deposit + 0.65 * grit + 0.51 * pores)
    soil = wp.lerp(soil, wp.vec3(0.481, 0.468, 0.319), 0.50 * undergrowth)
    color = wp.lerp(soil + wp.vec3(0.85 * leaf_micro), growth, 0.30 + 0.70 * mask)
    height = 0.308 + 0.238 * cover + 0.094 * patches + 0.051 * deposit + 0.039 * tips + 0.029 * grit
    roughness = 0.681 - 0.172 * mask - 0.083 * wet + 0.092 * grit
    ao = 0.31 + 0.685 * wp.smoothstep(-0.018, 0.29, cover + 0.26 * patches + 0.10 * deposit)
    debris, vein = _forest_litter(uv, seed + 83, resolution)
    debris *= 0.42 + 0.58 * wp.smoothstep(0.48, 0.84, 1.0 - cover)
    needle = wp.vec3(0.54, 0.372, 0.251) + wp.vec3(0.24, 0.22, 0.16) * vein
    color = wp.lerp(color, needle, debris)
    height += 0.107 * debris
    roughness = wp.lerp(roughness, 0.666 - 0.082 * vein, debris)
    color = wp.vec3(wp.clamp(color[0], 0.0, 1.0), wp.clamp(color[1], 0.0, 1.0), wp.clamp(color[2], 0.0, 1.0))
    return color, height, roughness, ao, 0.0, 1.0


@wp.func
def natural_or_foil(uv: wp.vec2, seed: int, resolution: float, style: int):
    """Return sRGB color, height, roughness, occlusion, metalness, and opacity."""
    uv = wrap_uv(uv)
    if style == 0:
        return _foil(uv, seed, resolution)
    return _moss(uv, seed, resolution)
