# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Periodic plank and chevron layouts with seeded board grain."""

import warp as wp

from .mineral_surfaces import _noise


@wp.func
def _floor_board(local: wp.vec2, state: wp.uint32, seed: int, resolution: float, count: int, knotty: bool):
    grain_uv = local + wp.vec2(wp.randf(state), wp.randf(state))
    curl = _noise(grain_uv, seed + 8, 2, 4)
    knot = float(0.0)
    split = float(0.0)
    if knotty:
        for i in range(2):
            center = wp.vec2((float(i) + wp.randf(state, 0.15, 0.85)) * 0.5, wp.randf(state, 0.16, 0.84))
            delta = local - center
            size = wp.randf(state, 0.022, 0.052)
            radius = wp.length(wp.vec2(delta[0], delta[1] * 0.48)) / size
            influence = 1.0 - wp.smoothstep(0.30, 3.6, radius)
            grain_uv[1] += 0.37 * delta[1] * influence
            heart = 1.0 - wp.smoothstep(0.22, 0.86, radius)
            rings = 0.5 + 0.5 * wp.cos(13.0 * radius + 4.0 * curl)
            knot += 0.76 * heart + 0.24 * rings * influence
            taper = 1.0 - wp.smoothstep(0.0, 2.8 * size, wp.abs(delta[0]))
            centerline = 0.10 * delta[0] + 2.2 * delta[0] * delta[0]
            split = wp.max(split, taper * (1.0 - wp.smoothstep(0.003, 0.014, wp.abs(delta[1] - centerline))))

    grain_uv[1] += 0.044 * curl + 0.016 * _noise(grain_uv, seed + 11, 5, 9)
    annual = float(count) * grain_uv[1] + 0.72 * _noise(grain_uv, seed + 13, 2, 6)
    ring = 0.5 + 0.5 * wp.cos(2.0 * wp.pi * annual)
    latewood = ring * ring * ring * wp.smoothstep(70.0, 270.0, resolution)
    grain = _noise(grain_uv, seed + 17, 6, 29)
    fibers = _noise(grain_uv, seed + 19, 17, 68) * wp.smoothstep(470.0, 1570.0, resolution)
    open_pores = _noise(grain_uv, seed + 23, 54, 135)
    pores = wp.smoothstep(0.13, 0.52, open_pores) * wp.smoothstep(810.0, 2450.0, resolution)
    tone = 0.064 * curl + 0.16 * grain + 0.090 * fibers - 0.050 * latewood - 0.063 * pores
    depth = 0.073 * grain + 0.047 * fibers - 0.051 * latewood - 0.080 * pores
    return tone, depth, knot, split


@wp.func
def wood_floor_layout(uv: wp.vec2, seed: int, resolution: float, variant: int):
    """Return aligned maps for staggered oak, weathered boards, or chevron parquet."""
    uv -= wp.vec2(wp.floor(uv[0]), wp.floor(uv[1]))
    across_distance = float(0.0)
    along_distance = float(0.0)
    local = wp.vec2(0.0)
    key = int(0)
    knotty = variant == 1
    if variant == 2:
        span = 8.0 * uv[0]
        column = int(wp.floor(span))
        x = span - float(column)
        along = x if column % 2 == 0 else 1.0 - x
        course_position = 24.0 * uv[1] + 1.5 * along
        course = int(wp.floor(course_position))
        across = course_position - float(course)
        local = wp.vec2(along, across)
        across_distance = wp.min(across, 1.0 - across) / 26.83
        along_distance = wp.min(x, 1.0 - x) / 8.0
        key = column + 8 * (course % 24)
    else:
        point = wp.vec2(uv[1], uv[0]) if knotty else uv
        course_position = 16.0 * point[1]
        course = int(wp.floor(course_position))
        stagger_state = wp.rand_init(seed + 4, course)
        span = 4.0 * point[0] + 0.25 * float(wp.randi(stagger_state, 0, 4))
        if knotty:
            span = 3.0 * point[0] + wp.randf(stagger_state)
            span += 0.24 * wp.sin(2.0 * wp.pi * point[0] + 6.28 * wp.randf(stagger_state))
        board = int(wp.floor(span))
        local = wp.vec2(span - float(board), course_position - float(course))
        boards = 3 if knotty else 4
        across_distance = wp.min(local[1], 1.0 - local[1]) / 16.0
        along_distance = wp.min(local[0], 1.0 - local[0]) / float(boards)
        key = course * boards + (board % boards + boards) % boards

    state = wp.rand_init(seed + 61, key)
    board_tone = wp.randf(state, -1.0, 1.0)
    rings = wp.randi(state, 7, 14)
    grain, depth, knot, split = _floor_board(local, state, seed, resolution, rings, knotty)
    clearance = wp.min(across_distance, along_distance)
    aa = 0.50 / resolution
    joint = 1.0 - wp.smoothstep(0.00025 - aa, 0.0008 + aa, clearance)
    bevel = 1.0 - wp.smoothstep(0.00018, 0.0027 + aa, clearance)

    base = wp.vec3(0.523, 0.385, 0.249)
    tone = 0.129 * board_tone + 0.84 * grain - 0.055 * joint
    height = 0.851 + 0.035 * board_tone + 1.05 * depth - 0.044 * bevel
    roughness = 0.529 - 0.060 * board_tone - 0.43 * grain + 0.035 * bevel
    if knotty:
        drift = _noise(uv, seed + 41, 4, 6)
        wear = _noise(local, seed + 45 + key, 4, 3)
        base = wp.vec3(0.370, 0.310, 0.239)
        tone = 0.128 * board_tone + 0.123 * wear + 0.075 * drift + 1.00 * grain
        tone = wp.max(-0.285, tone - 0.092 * knot - 0.104 * split - 0.117 * joint)
        height = 0.844 + 0.030 * board_tone + 0.54 * depth - 0.066 * knot - 0.099 * split - 0.058 * bevel
        roughness = 0.602 - 0.018 * board_tone + 0.030 * wear + 0.053 * bevel + 0.026 * split
    elif variant == 2:
        base = wp.vec3(0.610, 0.485, 0.366)
        tone = 0.074 * board_tone + 0.72 * grain - 0.038 * joint
        height = 0.75 + 0.038 * board_tone + 0.71 * depth - 0.033 * bevel
        roughness = 0.339 + 0.026 * board_tone - 0.150 * grain + 0.063 * bevel

    color = base + wp.vec3(tone, 0.87 * tone, 0.73 * tone)
    ao = 0.998 - 0.18 * joint - 0.035 * bevel - 0.060 * knot - 0.083 * split - 0.14 * wp.max(0.0, -depth)
    return color, height, roughness, ao, 0.0, 1.0
