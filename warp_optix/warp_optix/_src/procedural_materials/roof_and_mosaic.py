# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tileable interlocking roof courses and glazed hexagonal mosaic."""

import warp as wp

from .sampling import cell_index, wrap_index, wrap_uv
from .sampling import periodic_noise as _noise


@wp.func
def _roof_course(uv: wp.vec2, seed: int, resolution: float, dark: bool):
    columns, courses = 19, 11
    course_phase = 0.66
    if dark:
        columns, courses = 8, 8
        course_phase = 0.015

    mottling = _noise(uv, seed, 23, 29)
    patches = _noise(uv, seed + 3, 61, 68)
    clay = _noise(uv, seed + 5, 199, 211) * wp.smoothstep(115.0, 440.0, resolution)
    grain = _noise(uv, seed + 13, 591, 643) * wp.smoothstep(300.0, 1260.0, resolution)
    pattern_x = uv[0] * float(columns) + 0.023 * _noise(uv, seed + 2, 11, 17) + 0.007 * patches
    col = int(wp.floor(pattern_x))
    column_seed = wp.rand_init(seed + 17, wrap_index(col, columns))
    x = pattern_x - wp.floor(pattern_x)
    pattern_y = uv[1] * float(courses) + course_phase + wp.randf(column_seed, -0.014, 0.014)
    # Each molded nose sits a little forward at the crown of the tile.
    nose = 0.017 if not dark else 0.006
    pattern_y += nose * wp.cos(2.0 * wp.pi * x) + nose * _noise(uv, seed + 7, 13, 19)
    row = int(wp.floor(pattern_y))
    y = pattern_y - wp.floor(pattern_y)
    tile = cell_index(col, row, columns, courses)
    material_seed = wp.rand_init(seed + 29, tile)
    pigment = wp.randf(material_seed, -0.024, 0.024)
    thickness = wp.randf(material_seed, 0.273, 0.291)

    # A full course rises gradually over its neighbor, ending in a short
    # rounded lap joint. The small ceramic chips interrupt only that lip.
    chip_field = _noise(uv, seed + 31, 247, 219)
    chipped = wp.smoothstep(0.22, 0.48, chip_field + 0.35 * patches)
    bevel_width = wp.clamp(1.7 * float(courses) / resolution, 0.019, 0.21)
    bevel = wp.smoothstep(1.0 - bevel_width - 0.019 * chipped, 1.0, y)
    lap = y - bevel
    lower_tile = 1.0 - wp.smoothstep(0.0, 0.052, y)
    lip = wp.smoothstep(0.91, 0.981, y) * (1.0 - bevel)
    room = wp.smoothstep(0.015, 0.14, y) * wp.smoothstep(0.015, 0.14, 1.0 - y)
    pits = wp.smoothstep(0.23, 0.57, -clay + 0.54 * grain)
    mineral = wp.smoothstep(0.31, 0.60, clay + 0.48 * grain)

    # The curved cross-sections are low-order approximations of the measured
    # reference relief. The cleaner barrel is broad and rounded. The older
    # interlocking pantile has a pronounced trough and an asymmetric crest.
    phase = 2.0 * wp.pi * x
    profile = -0.202 * wp.cos(phase) - 0.048 * wp.cos(2.0 * phase) - 0.012 * wp.cos(3.0 * phase)
    crown = wp.smoothstep(-0.25, 0.24, profile)
    height = 0.548 + profile + thickness * lap
    height += 0.007 * (clay + grain) - 0.021 * pits - 0.022 * chipped * lip
    seam = 1.0 - wp.smoothstep(0.008, 0.041, wp.min(x, 1.0 - x))
    temper = pigment + 0.047 * mottling + 0.027 * patches + 0.022 * clay + 0.017 * grain
    color = wp.vec3(0.563, 0.357, 0.263) + wp.vec3(1.0, 0.87, 0.78) * temper
    color += wp.vec3(0.022 * lip + 0.031 * mineral - 0.039 * pits - 0.030 * seam - 0.029 * lower_tile)
    rough = 0.674 + 0.098 * mottling + 0.079 * clay + 0.070 * pits - 0.062 * lip
    ao = (0.72 + 0.27 * crown) * (1.0 - 0.20 * lower_tile - 0.10 * bevel)

    if dark:
        profile = 0.335 * wp.cos(phase + 1.935) + 0.023 * wp.cos(2.0 * phase + 2.723)
        profile += 0.037 * wp.cos(3.0 * phase + 2.778) + 0.014 * wp.cos(4.0 * phase + 3.026)
        crown = wp.smoothstep(-0.30, 0.28, profile)
        height = 0.347 + profile + (thickness + 0.039) * lap + 0.013 * room * mottling
        # Water and dirt gather below the rounded-over edge. Weathering
        # breaks into islands where the clay surface dries unevenly.
        rainfall = _noise(uv, seed + 23, 16, 19) + 0.51 * mottling + 0.15 * patches
        damp = 1.0 - wp.smoothstep(0.02, 0.125 + 0.029 * mottling, y)
        drying = 0.83 * crown + 0.38 * rainfall + 0.076 * clay - 0.36 * damp
        exposed = wp.smoothstep(0.17, 0.84, drying)
        algae = (1.0 - exposed) * wp.smoothstep(0.08, 0.50, rainfall + 0.64 * patches + 0.52 * damp)
        dark_clay = wp.vec3(0.083, 0.048, 0.016) + wp.vec3(0.017 * clay + 0.028 * mottling)
        dark_clay = wp.lerp(dark_clay, wp.vec3(0.078, 0.089, 0.025), 0.66 * algae)
        kiln = wp.vec3(0.428, 0.259, 0.169) + wp.vec3(1.0, 0.80, 0.68) * (1.21 * temper)
        color = wp.lerp(dark_clay, kiln, exposed) + wp.vec3(0.022 * grain - 0.023 * pits + 0.017 * mineral)
        height += 0.012 * algae + 0.008 * grain - 0.015 * pits - 0.019 * chipped * lip
        rough = 0.65 - 0.083 * exposed + 0.092 * patches + 0.080 * clay + 0.11 * algae
        ao = (0.57 + 0.41 * crown) * (1.0 - 0.27 * lower_tile - 0.13 * bevel)

    return color, height, rough, ao


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

    tile = cell_index(tile_x, tile_y, columns, rows)
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
    uv = wrap_uv(uv)
    if style == 2:
        return _hex_mosaic(uv, seed, resolution)
    return _roof_course(uv, seed, resolution, style == 0)
