# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Open-cell sponge and pressed paper grain."""

import warp as wp

from .sampling import periodic_noise as _noise


@wp.func
def _wrapped_cell(x: int, y: int, count: int) -> int:
    return (x % count + count) % count + count * ((y % count + count) % count)


@wp.func
def _pore_layer(uv: wp.vec2, seed: int, count: int, resolution: float):
    grid = uv * float(count)
    col, row = int(wp.floor(grid[0])), int(wp.floor(grid[1]))
    depth = float(0.0)
    opening = float(0.0)
    aa = wp.clamp(0.64 * float(count) / resolution, 0.01, 0.14)
    # Cells overlap neighboring grid squares and have independently rotated
    # cross sections. Rounded rims connect the openings to the surface.
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            x, y = col + dx, row + dy
            state = wp.rand_init(seed, _wrapped_cell(x, y, count))
            mid = wp.vec2(float(x), float(y)) + wp.vec2(wp.randf(state, 0.15, 0.85), wp.randf(state, 0.15, 0.85))
            delta = grid - mid
            if wp.dot(delta, delta) < 0.285:
                size = wp.randf(state, 0.075, 0.46)
                radius = 0.068 + 0.39 * size * size + 0.48 * size
                angle = wp.randf(state, 0.0, wp.pi)
                tangent = wp.vec2(wp.cos(angle), wp.sin(angle))
                shape = wp.randf(state, 1.0, 2.9)
                across = wp.dot(delta, wp.vec2(-tangent[1], tangent[0]))
                along = wp.dot(delta, tangent)
                dist = wp.length(wp.vec2(along, shape * across + 0.42 * along * along))
                cover = 1.0 - wp.smoothstep(radius - aa, radius + aa, dist)
                wall = 1.0 - wp.smoothstep(0.13 * radius, radius + aa, dist)
                opening = wp.max(opening, cover)
                depth = wp.max(depth, wp.randf(state, 0.64, 1.0) * wall)
    return depth, opening


@wp.func
def _pinholes(uv: wp.vec2, seed: int, resolution: float) -> float:
    grid = uv * 161.0
    cell = wp.vec2(wp.floor(grid[0]), wp.floor(grid[1]))
    state = wp.rand_init(seed, _wrapped_cell(int(cell[0]), int(cell[1]), 161))
    center = wp.vec2(wp.randf(state, 0.26, 0.74), wp.randf(state, 0.26, 0.74))
    radius = wp.randf(state, 0.087, 0.192)
    relative = grid - cell - center
    relative[1] = 1.23 * relative[1] + 0.35 * relative[0]
    aa = wp.clamp(62.0 / resolution, 0.019, 0.063)
    pit = 1.0 - wp.smoothstep(0.52 * radius - aa, radius + aa, wp.length(relative))
    return pit * wp.randf(state, 0.38, 0.98) * wp.smoothstep(230.0, 910.0, resolution)


@wp.func
def _sponge(uv: wp.vec2, seed: int, resolution: float):
    grain = _noise(uv, seed + 2, 187, 201)
    skin = _noise(uv, seed + 3, 621, 671) * wp.smoothstep(310.0, 1230.0, resolution)
    flatness = _noise(uv, seed + 4, 27, 33) + 0.51 * _noise(uv, seed + 6, 61, 65)
    bend = 0.004 * wp.vec2(_noise(uv, seed + 7, 38, 41), _noise(uv, seed + 8, 43, 39))
    major_depth, major = _pore_layer(uv + bend, seed + 17, 25, resolution)
    small_depth, small = _pore_layer(uv, seed + 25, 79, resolution)
    pinhole = _pinholes(uv, seed + 31, resolution)
    opening = wp.max(0.89 * major, small)
    color = wp.vec3(0.720, 0.561, 0.148) + wp.vec3(0.058 * flatness + 0.051 * skin + 0.031 * grain)
    color -= wp.vec3(0.085, 0.093, 0.022) * (0.55 * major_depth + 0.53 * small_depth + 0.23 * pinhole)
    color += wp.vec3(0.028, 0.026, 0.034) * wp.max(0.0, opening - 0.90 * (major_depth + small_depth))
    height = 0.674 + 0.11 * flatness - 0.38 * major_depth - 0.19 * small_depth - 0.16 * pinhole
    height += 0.037 * skin + 0.059 * grain
    roughness = 0.844 - 0.055 * major_depth - 0.042 * small_depth + 0.019 * skin
    ao = wp.max(0.045, 0.946 - 0.47 * major_depth - 0.39 * small_depth - 0.25 * pinhole)
    return color, wp.max(0.0, height), roughness, ao, 0.0, 1.0


@wp.func
def _pressed_fiber(uv: wp.vec2, seed: int, count: int, resolution: float):
    grid = uv * float(count)
    col, row = int(wp.floor(grid[0])), int(wp.floor(grid[1]))
    state = wp.rand_init(seed, _wrapped_cell(col, row, count))
    center = wp.vec2(float(col), float(row)) + wp.vec2(wp.randf(state, 0.35, 0.65), wp.randf(state, 0.35, 0.65))
    relative = grid - center
    angle = wp.randf(state, -0.76, 0.76) + 0.36
    direction = wp.vec2(wp.cos(angle), wp.sin(angle))
    along = wp.dot(relative, direction)
    across = wp.dot(relative, wp.vec2(-direction[1], direction[0]))
    curvature = wp.randf(state, -1.7, 1.7)
    along /= wp.randf(state, 0.23, 0.35)
    across += 0.073 * curvature * along * along
    across /= wp.randf(state, 0.055, 0.125)
    dist = wp.length(wp.vec2(along, across))
    aa = wp.clamp(1.14 * float(count) / resolution, 0.13, 0.45)
    fiber = 1.0 - wp.smoothstep(0.19, 1.0 + aa, dist)
    local = grid - wp.vec2(float(col), float(row))
    edge = wp.min(wp.min(local[0], 1.0 - local[0]), wp.min(local[1], 1.0 - local[1]))
    return fiber * wp.randf(state, 0.72, 1.0) * wp.smoothstep(0.0, 0.045, edge)


@wp.func
def _paper(uv: wp.vec2, seed: int, resolution: float):
    # Slightly directional stock has crimped, overlapping fibers under
    # the pressed surface and subtle mottling from changes in pulp density.
    formation = _noise(uv, seed + 2, 54, 61) + 0.44 * _noise(uv, seed + 3, 133, 171)
    cross_grain = 0.44 * _noise(uv, seed + 7, 151, 263) + 0.48 * _noise(uv, seed + 9, 287, 361)
    short_fiber = _pressed_fiber(uv, seed + 23, 188, resolution)
    long_fiber = _pressed_fiber(uv + wp.vec2(0.1037, 0.217), seed + 37, 143, resolution)
    fiber = (0.48 * short_fiber + 0.52 * long_fiber) * wp.smoothstep(145.0, 580.0, resolution)
    dust = _noise(uv, seed + 6, 649, 735) * wp.smoothstep(320.0, 1240.0, resolution)
    tone = 0.024 * formation + 0.037 * cross_grain + 0.042 * fiber + 0.025 * dust
    color = wp.vec3(0.957, 0.951, 0.944) + wp.vec3(tone)
    height = 0.581 + 0.095 * formation + 0.29 * cross_grain + 0.151 * fiber + 0.063 * dust
    return color, height, 0.632 + 0.079 * formation + 0.128 * cross_grain + 0.029 * fiber, 1.0, 0.0, 1.0


@wp.func
def sponge_or_paper(uv: wp.vec2, seed: int, resolution: float, style: int):
    uv -= wp.vec2(wp.floor(uv[0]), wp.floor(uv[1]))
    if style == 0:
        return _sponge(uv, seed, resolution)
    return _paper(uv, seed, resolution)
