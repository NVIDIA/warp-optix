# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reclaimed wooden planks with individual grain and recessed fasteners."""

import warp as wp


@wp.func
def _grain_noise(state: wp.uint32, point: wp.vec2, nx: int, ny: int):
    wrapped = point - wp.vec2(wp.floor(point[0]), wp.floor(point[1]))
    return wp.pnoise(state, wp.vec2(float(nx) * wrapped[0], float(ny) * wrapped[1]), nx, ny)


@wp.func
def reclaimed_planks(uv: wp.vec2, seed: int, resolution: float):
    course_position = 16.0 * uv[1]
    course = int(wp.floor(course_position))
    course_seed = wp.rand_init(seed + 13, course & 15)
    stagger = wp.randf(course_seed)
    span = 2.0 * uv[0] + stagger
    board = int(wp.floor(span))
    position = wp.vec2(span - float(board), course_position - float(course))
    state = wp.rand_init(seed, 2 * (course & 15) + (board & 1))
    phase = wp.randf(state)
    tone = wp.randf(state, -0.025, 0.022)
    local = wp.vec2(0.48 * position[0], 0.14 * position[1] + 0.83 * phase)
    knot_state = wp.rand_init(seed + 17, 2 * (course & 15) + (board & 1))

    bend = float(0.0)
    knot = float(0.0)
    check = float(0.0)
    for i in range(3):
        center = wp.vec2((float(i) + wp.randf(knot_state, 0.33, 0.74)) / 3.0, wp.randf(knot_state, 0.18, 0.82))
        delta = position - center
        width = wp.randf(knot_state, 0.011, 0.028)
        knot_distance = wp.length(wp.vec2(delta[0] / width, delta[1] / (4.4 * width)))
        surrounding = 1.0 - wp.smoothstep(0.68, 4.5, knot_distance)
        bend += surrounding * delta[1] * 0.12
        heart = 1.0 - wp.smoothstep(0.53, 1.24, knot_distance)
        ring = 0.5 + 0.5 * wp.cos(18.0 * knot_distance)
        knot = wp.max(knot, 0.68 * heart + 0.32 * ring * surrounding)
        split_length = wp.randf(knot_state, 0.023, 0.087)
        split_width = wp.max(0.0, 1.0 - wp.abs(delta[0]) / split_length)
        split_center = 0.16 * delta[0] + 5.0 * delta[0] * delta[0]
        check = wp.max(check, split_width * (1.0 - wp.smoothstep(0.006, 0.024, wp.abs(delta[1] - split_center))))

    grain_uv = wp.vec2(local[0], local[1] + bend)
    curl = _grain_noise(state, grain_uv, 7, 23)
    grain_uv[1] += 0.014 * curl + 0.0033 * _grain_noise(state, grain_uv, 23, 63)
    band = 153.0 * grain_uv[1] + 2.1 * _grain_noise(state, grain_uv, 7, 14)
    distance = wp.abs(band - wp.floor(band + 0.5))
    width = 10.5 / resolution
    line = 1.0 - wp.smoothstep(0.044 - width, 0.124 + width, distance)
    strands = _grain_noise(state, grain_uv, 94, 738) * wp.smoothstep(390.0, 1220.0, resolution)
    stain = _grain_noise(wp.rand_init(seed + 19), uv, 23, 19)
    specks = _grain_noise(wp.rand_init(seed + 31), uv, 227, 171)
    soot = wp.smoothstep(0.10, 0.55, stain + 0.40 * specks + 0.23 * curl)
    grain_tone = 0.040 * curl + 0.044 * strands - 0.073 * line - 0.094 * knot - 0.10 * check
    color = wp.vec3(0.351, 0.272, 0.194) + wp.vec3(tone + grain_tone, 0.78 * (tone + grain_tone), 0.64 * (tone + grain_tone))
    color = wp.lerp(color, wp.vec3(0.152, 0.151, 0.143), 0.68 * soot)

    along_end = wp.min(position[0], 1.0 - position[0])
    along_edge = wp.min(position[1], 1.0 - position[1])
    side = along_edge + 0.018 * _grain_noise(wp.rand_init(seed + 41), uv, 200, 40)
    clearance = wp.min(side / 16.0, along_end / 2.0)
    seam = 1.0 - wp.smoothstep(0.0003, 0.0014 + 0.46 / resolution, clearance)
    bevel = 1.0 - wp.smoothstep(0.0008, 0.0040, clearance)

    # Two recessed steel nail heads at each board end, with slight placement variation.
    pin_state = wp.rand_init(seed + 43, course & 15)
    pin_y = wp.randf(pin_state, 0.18, 0.28)
    nail_offset = wp.vec2((along_end - wp.randf(pin_state, 0.019, 0.032)) / 2.0, (along_edge - pin_y) / 16.0)
    nail_distance = wp.length(nail_offset)
    nail = 1.0 - wp.smoothstep(0.00155 - 0.35 / resolution, 0.00225 + 0.35 / resolution, nail_distance)
    countersink = 1.0 - wp.smoothstep(0.00225, 0.0045, nail_distance)
    color = wp.lerp(color, wp.vec3(0.16, 0.134, 0.104), 0.76 * seam + 0.21 * countersink)
    color = wp.lerp(color, wp.vec3(0.44, 0.443, 0.417) + wp.vec3(0.080 * specks), nail)
    height = 0.654 + 0.022 * curl + 0.020 * strands - 0.025 * line - 0.08 * knot - 0.085 * check
    height -= 0.23 * bevel + 0.135 * countersink - 0.066 * nail
    roughness = wp.lerp(0.414 + 0.11 * soot + 0.070 * check + 0.04 * line, 0.33, nail)
    ao = 0.985 - 0.37 * seam - 0.13 * countersink - 0.14 * knot - 0.13 * check
    return color, height, roughness, ao, nail * 0.82, 1.0
