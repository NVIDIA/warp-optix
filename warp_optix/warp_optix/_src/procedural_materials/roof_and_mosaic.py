# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tileable interlocking roof courses and glazed hexagonal mosaic."""

import warp as wp


@wp.func
def _noise(uv: wp.vec2, seed: int, nx: int, ny: int) -> float:
    return wp.pnoise(wp.rand_init(seed), wp.vec2(uv[0] * float(nx), uv[1] * float(ny)), nx, ny)


@wp.func
def _roof_course(uv: wp.vec2, seed: int, resolution: float, dark: bool):
    columns, courses = 19, 11
    course_phase = 0.66
    if dark:
        columns, courses = 8, 8
        course_phase = 0.015

    broad = _noise(uv, seed, 7, 9)
    dirt = _noise(uv, seed + 3, 63, 61)
    clay = _noise(uv, seed + 5, 203, 169) * wp.smoothstep(120.0, 410.0, resolution)
    grain = _noise(uv, seed + 13, 710, 648) * wp.smoothstep(350.0, 1280.0, resolution)
    pattern_x = uv[0] * float(columns) + 0.012 * dirt
    col = int(wp.floor(pattern_x))
    column_seed = wp.rand_init(seed + 17, (col % columns + columns) % columns)
    pattern_y = uv[1] * float(courses) + course_phase + wp.randf(column_seed, -0.015, 0.015) + 0.013 * dirt
    row = int(wp.floor(pattern_y))
    x, y = pattern_x - wp.floor(pattern_x), pattern_y - wp.floor(pattern_y)
    tile = (col % columns + columns) % columns + columns * ((row % courses + courses) % courses)
    pigment_seed = wp.rand_init(seed + 29, tile)
    pigment = wp.randf(pigment_seed, -0.049, 0.049)

    # The downslope lip of each course projects out over the course below.
    # Widen its bevel just enough to keep the edge stable at smaller sizes.
    lap_bevel = wp.clamp(1.45 * float(courses) / resolution, 0.062, 0.48)
    lap = y * (1.0 - wp.smoothstep(0.993 - lap_bevel, 0.993, y))
    vertical_join = wp.min(x, 1.0 - x)
    rolled_edge = wp.sin(wp.pi * x) ** 1.58
    rough = 0.69 + 0.08 * broad + 0.082 * clay - 0.13 * rolled_edge
    ao = (0.63 + 0.36 * rolled_edge) * (0.76 + 0.24 * wp.smoothstep(0.0, 0.09, y))
    height = 0.275 + 0.435 * rolled_edge + 0.255 * lap
    color = wp.vec3(0.563, 0.360, 0.268)
    clay_tone = pigment + 0.032 * dirt + 0.044 * clay + 0.035 * grain
    color += wp.vec3(1.0, 0.87, 0.70) * clay_tone
    color *= 0.91 + 0.09 * wp.smoothstep(0.01, 0.105, vertical_join)

    if dark:
        folded = 0.5 + 0.5 * wp.cos(2.0 * wp.pi * (x - 0.704))
        rolled_edge = wp.smoothstep(0.15, 0.92, folded)
        height = 0.07 + 0.58 * rolled_edge + 0.35 * lap
        # Dark growth collects in sheltered valleys and survives as scattered
        # patina on exposed ridges. It follows the same profile as the relief.
        patina = _noise(uv, seed + 23, 33, 43)
        wet_edge = 1.0 - wp.smoothstep(0.018, 0.20 + 0.13 * dirt, y)
        exposed = wp.smoothstep(0.16, 0.85, rolled_edge + 0.52 * broad + 0.36 * patina + 0.25 * dirt + 0.14 * clay)
        exposed *= 1.0 - 0.58 * wet_edge
        clay_tone = 1.38 * pigment + 0.075 * broad + 0.055 * dirt + 0.061 * clay
        valley = wp.vec3(0.092, 0.062, 0.029) + wp.vec3(0.030 * broad + 0.026 * clay)
        crest = wp.vec3(0.407, 0.233, 0.145) + wp.vec3(clay_tone)
        color = wp.lerp(valley, crest, exposed) + 0.036 * wp.vec3(grain)
        rough = 0.68 - 0.13 * exposed + 0.103 * dirt + 0.075 * clay
        ao = (0.48 + 0.51 * rolled_edge) * (0.76 + 0.24 * wp.smoothstep(0.0, 0.08, y))

    return color, height + 0.017 * clay + 0.012 * grain, rough, ao


@wp.func
def _hex_mosaic(uv: wp.vec2, seed: int, resolution: float):
    columns, rows = 16, 18
    row_spacing = 0.8660254
    p = wp.vec2(uv[0] * float(columns), uv[1] * float(rows) * row_spacing)
    row_center = int(wp.floor(p[1] / row_spacing + 0.5))
    best_dist = float(10.0)
    tile_x, tile_y = int(0), int(0)
    local = wp.vec2(0.0)
    for offset in range(-1, 2):
        row = row_center + offset
        stagger = 0.5 * float(row & 1)
        col = int(wp.floor(p[0] - stagger + 0.5))
        delta = p - wp.vec2(float(col) + stagger, float(row) * row_spacing)
        dist = wp.dot(delta, delta)
        if dist < best_dist:
            best_dist = dist
            local = delta
            tile_x, tile_y = col, row

    tile = (tile_x % columns + columns) % columns + columns * ((tile_y % rows + rows) % rows)
    state = wp.rand_init(seed + 19, tile)
    dye = wp.randf(state, -0.070, 0.057)
    tilt = wp.vec2(wp.randf(state, -0.24, 0.24), wp.randf(state, -0.24, 0.24))
    tile_height = wp.randf(state, -0.145, 0.145)
    glaze = _noise(uv, seed + 37, 39, 41) + 0.41 * _noise(uv, seed + 39, 131, 129)
    speckle = _noise(uv, seed + 41, 627, 641) * wp.smoothstep(310.0, 1300.0, resolution)
    edge = 0.5 - wp.max(wp.abs(local[0]), 0.5 * wp.abs(local[0]) + row_spacing * wp.abs(local[1]))
    glaze_chip = 0.022 * wp.smoothstep(0.25, 0.60, glaze + 0.7 * speckle)
    edge -= glaze_chip
    aa = wp.min(0.11, 8.3 / resolution)
    coverage = wp.smoothstep(0.016 - aa, 0.016 + aa, edge)
    bevel = wp.smoothstep(0.009, 0.078 + aa, edge)
    ceramic = wp.vec3(0.943, 0.941, 0.938) + wp.vec3(dye + 0.077 * glaze + 0.021 * speckle)
    grout = wp.vec3(0.752, 0.727, 0.695) + wp.vec3(0.063 * glaze + 0.083 * speckle)
    surface = 0.695 + tile_height + wp.dot(tilt, local) + 0.068 * glaze + 0.009 * speckle
    height = wp.lerp(0.19 + 0.03 * speckle, surface, bevel)
    roughness = wp.lerp(0.75 + 0.041 * speckle, 0.48 + 0.134 * glaze - 0.48 * dye, coverage)
    return wp.lerp(grout, ceramic, coverage), height, roughness, 0.60 + 0.40 * bevel


@wp.func
def roof_or_mosaic(uv: wp.vec2, seed: int, resolution: float, style: int):
    uv -= wp.vec2(wp.floor(uv[0]), wp.floor(uv[1]))
    if style == 2:
        return _hex_mosaic(uv, seed, resolution)
    return _roof_course(uv, seed, resolution, style == 0)
