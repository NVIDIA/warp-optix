# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Procedural ashlar masonry and oxidized sheet metal."""

import warp as wp

from .sampling import periodic_noise as _noise
from .sampling import wrap_index, wrap_uv


@wp.func
def _ashlar_course(uv: wp.vec2, seed: int):
    # Tall courses consist of wide blocks, pairs of short blocks, and split
    # stretchers. A separate offset and joint jitter stagger neighboring courses.
    course_uv = 4.0 * uv[1] + 0.095 * _noise(uv, seed + 1, 5, 4)
    course = int(wp.floor(course_uv))
    course_phase = course_uv - float(course)
    row = wrap_index(course, 4)
    row_state = wp.rand_init(seed + 11, row)
    across = 5.0 * uv[0] + wp.randf(row_state, 0.10, 2.10)
    across += 0.081 * _noise(uv, seed + 2, 5, 7)
    column = int(wp.floor(across))

    # The adjoining blocks share the same jittered endpoints.
    left = float(column) + wp.randf(wp.rand_init(seed + 34 + row, wrap_index(column, 5)), -0.31, 0.31)
    if across < left:
        column -= 1
        left = float(column) + wp.randf(wp.rand_init(seed + 34 + row, wrap_index(column, 5)), -0.31, 0.31)
    right = float(column + 1) + wp.randf(wp.rand_init(seed + 34 + row, wrap_index(column + 1, 5)), -0.31, 0.31)
    if across > right:
        column += 1
        left = right
        right = float(column + 1) + wp.randf(wp.rand_init(seed + 34 + row, wrap_index(column + 1, 5)), -0.31, 0.31)
    block_id = 5 * row + wrap_index(column, 5)
    state = wp.rand_init(seed + 49, block_id)
    split = wp.randf(state, 0.28, 0.72)
    lower = float(0.0)
    upper = float(1.0)
    shape = wp.randf(state)
    piece = int(0)
    if shape > 0.11:
        if course_phase < split:
            upper = split
        else:
            lower = split
            piece = 1
    if shape > 0.52 and ((piece + block_id) & 1) == 0:
        middle = wp.lerp(left, right, wp.randf(state, 0.29, 0.71))
        if across < middle:
            right = middle
        else:
            left = middle
            piece += 2
    distance = wp.vec2(
        wp.min(across - left, right - across) / 5.0,
        wp.min(course_phase - lower, upper - course_phase) / 4.0,
    )
    local = wp.vec2((across - left) / (right - left), (course_phase - lower) / (upper - lower))
    return distance, local, 4 * block_id + piece


@wp.func
def ashlar_masonry(uv: wp.vec2, seed: int, resolution: float):
    uv = wrap_uv(uv)
    distance, local, block_id = _ashlar_course(uv, seed)
    state = wp.rand_init(seed + 57, block_id)
    tint = wp.randf(state, -0.105, 0.081)
    warmth = wp.randf(state, -0.032, 0.031)
    base_height = wp.randf(state, 0.40, 0.69)
    stone_type = wp.randf(state)
    grain_level = wp.randf(state, 0.39, 1.0)

    # Rounded, chipped stone outlines and dirty mortar share the same edge.
    grit = _noise(uv, seed + 61, 215, 228)
    powder = _noise(uv, seed + 73, 514, 473) * wp.smoothstep(400.0, 1400.0, resolution)
    broad = _noise(uv, seed + 77, 29, 23)
    weather = _noise(uv, seed + 81, 75, 98)
    corner = wp.vec2(0.008) - distance
    outside_corner = wp.vec2(wp.max(0.0, corner[0]), wp.max(0.0, corner[1]))
    edge = 0.008 - wp.length(outside_corner) - wp.min(0.0, wp.max(corner[0], corner[1]))
    edge += 0.0029 * weather + 0.0012 * grit
    aa = 0.6 / resolution
    joint = 0.0034 + 0.0008 * broad
    stone = wp.smoothstep(joint - aa, joint + 0.0028 + aa, edge)
    bevel = wp.smoothstep(joint - aa, joint + 0.008 + aa, edge)
    worn_edge = stone * (1.0 - wp.smoothstep(joint, joint + 0.017, edge + 0.005 * weather))
    aggregate = grain_level * (0.38 * broad + 0.28 * weather + 0.24 * grit + 0.19 * powder)
    pit = wp.smoothstep(0.19, 0.54, -0.67 * grit - 0.41 * weather - 0.28 * powder)
    stain = wp.smoothstep(0.08, 0.52, 0.75 * broad + 0.47 * weather + 0.12 * grit)
    facets = wp.smoothstep(-0.28, 0.32, 0.73 * broad + 0.34 * weather) - 0.5
    grain = 0.0
    if stone_type < 0.40:
        tilt = wp.randf(state, -0.61, 0.61)
        wave = 0.18 * wp.sin(4.0 * local[0] + 3.2 * local[1])
        layers = local[1] + tilt * local[0] + wave + 0.062 * broad + 0.018 * weather
        band = 0.5 + 0.5 * wp.sin(56.0 * layers + 0.55 * wp.sin(119.0 * layers))
        grain = band * band * band
        grain *= wp.smoothstep(0.0, 0.008, edge - joint)

    mortar = wp.vec3(0.53, 0.49, 0.423) + wp.vec3(0.11 * weather + 0.097 * grit + 0.072 * powder)
    mineral = wp.vec3(0.506, 0.459, 0.385) + wp.vec3(tint + 0.23 * aggregate + 0.060 * facets - 0.089 * stain)
    mineral += wp.vec3(warmth, 0.32 * warmth, -0.42 * warmth)
    mineral += wp.vec3(0.10, 0.087, 0.064) * worn_edge - wp.vec3(0.13, 0.115, 0.089) * pit
    mineral += wp.vec3(0.083, 0.068, 0.047) * grain
    color = wp.lerp(mortar, mineral, stone)
    mortar_height = 0.175 + 0.025 * weather + 0.014 * grit
    relief = wp.lerp(mortar_height, base_height + 0.14 * aggregate + 0.13 * facets + 0.028 * grain - 0.077 * pit, bevel)
    roughness = wp.lerp(
        0.735 + 0.058 * grit, 0.572 + 0.16 * pit + 0.18 * aggregate + 0.085 * stain - 0.065 * grain, stone
    )
    occlusion = wp.lerp(0.66 + 0.080 * weather, 0.941 + 0.05 * facets - 0.18 * pit - 0.15 * worn_edge, bevel)
    return color, relief, roughness, occlusion, 0.0, 1.0


@wp.func
def rusted_sheet_metal(uv: wp.vec2, seed: int, resolution: float):
    # Low frequencies organize oxide into contiguous islands. High-frequency
    # fronts break up their edges and expose small pockets of the steel.
    uv = wrap_uv(uv)
    warp_x = _noise(uv, seed + 1, 11, 13)
    warp_y = _noise(uv, seed + 2, 13, 11)
    position = uv + 0.018 * wp.vec2(warp_x, warp_y)
    position -= wp.vec2(wp.floor(position[0]), wp.floor(position[1]))
    islands = _noise(position, seed + 5, 9, 8)
    fronts = _noise(position, seed + 7, 27, 29)
    speck = _noise(uv, seed + 17, 178, 191)
    chips = _noise(uv, seed + 19, 81, 89)
    powder = _noise(uv, seed + 21, 579, 616) * wp.smoothstep(330.0, 1450.0, resolution)
    streak = _noise(uv, seed + 23, 161, 8) + 0.41 * _noise(uv, seed + 27, 294, 23)
    stain = 0.79 * islands + 0.45 * fronts + 0.22 * chips + 0.18 * speck + 0.078 * powder
    metalness = 1.0 - wp.smoothstep(-0.153, -0.018, stain)
    oxide = 1.0 - metalness

    age = _noise(position, seed + 41, 21, 25)
    weather = 0.57 * age + 0.29 * fronts + 0.38 * chips + 0.33 * speck + 0.24 * powder
    fresh_rust = wp.smoothstep(-0.06, 0.45, weather + 0.19 * islands)
    deep_pits = wp.smoothstep(0.10, 0.42, -weather + 0.15 * fronts)
    rust = wp.vec3(0.465, 0.287, 0.226) + wp.vec3(0.133, 0.093, 0.078) * fresh_rust
    rust -= wp.vec3(0.101, 0.078, 0.058) * deep_pits
    rust += wp.vec3(0.056 * chips + 0.039 * speck + 0.023 * powder)
    steel = wp.vec3(0.410, 0.404, 0.400) + wp.vec3(0.090 * fronts + 0.067 * streak + 0.029 * speck)
    color = wp.lerp(steel, rust, oxide)
    relief = wp.lerp(
        0.765 + 0.32 * fronts + 0.18 * age + 0.09 * streak,
        0.455 + 0.38 * weather - 0.14 * deep_pits - 0.12 * islands,
        oxide,
    )
    roughness = wp.lerp(
        0.242 + 0.114 * fronts + 0.07 * age + 0.034 * streak,
        0.651 + 0.29 * weather + 0.10 * fresh_rust + 0.22 * deep_pits + 0.08 * chips,
        oxide,
    )
    return color, relief, roughness, 1.0, metalness, 1.0
