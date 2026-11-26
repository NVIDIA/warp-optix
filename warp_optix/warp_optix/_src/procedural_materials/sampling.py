# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warp as wp


@wp.func
def wrap_uv(uv: wp.vec2) -> wp.vec2:
    return uv - wp.vec2(wp.floor(uv[0]), wp.floor(uv[1]))


@wp.func
def wrap_index(index: int, period: int) -> int:
    return (index % period + period) % period


@wp.func
def cell_index(x: int, y: int, columns: int, rows: int) -> int:
    return wrap_index(x, columns) + columns * wrap_index(y, rows)


@wp.func
def texel_uv(row: int, col: int, height: int, width: int) -> wp.vec2:
    return wp.vec2((float(col) + 0.5) / float(width), (float(row) + 0.5) / float(height))


@wp.func
def periodic_noise(uv: wp.vec2, seed: int, nx: int, ny: int) -> float:
    return wp.pnoise(wp.rand_init(seed), wp.vec2(uv[0] * float(nx), uv[1] * float(ny)), nx, ny)


@wp.func
def wrapped_noise(uv: wp.vec2, seed: int, nx: int, ny: int) -> float:
    return periodic_noise(wrap_uv(uv), seed, nx, ny)


@wp.func
def nearest_cells(uv: wp.vec2, seed: int, count: int, lo: float, hi: float, column_major: bool = False):
    """Return squared distances, offsets and post-jitter RNG states of the nearest two sites."""
    grid = uv * float(count)
    x, y = int(wp.floor(grid[0])), int(wp.floor(grid[1]))
    nearest, second = float(100.0), float(100.0)
    offset, next_offset = wp.vec2(0.0), wp.vec2(0.0)
    state, next_state = wp.uint32(0), wp.uint32(0)
    for row in range(-1, 2):
        for col in range(-1, 2):
            # Retain the search order for seeded recipes, including ties.
            cx = x + (row if column_major else col)
            cy = y + (col if column_major else row)
            rng = wp.rand_init(seed, cell_index(cx, cy, count, count))
            center = wp.vec2(float(cx) + wp.randf(rng, lo, hi), float(cy) + wp.randf(rng, lo, hi))
            delta = grid - center
            distance = wp.dot(delta, delta)
            if distance < nearest:
                second, next_offset, next_state = nearest, offset, state
                nearest, offset, state = distance, delta, rng
            elif distance < second:
                second, next_offset, next_state = distance, delta, rng
    return nearest, second, offset, next_offset, state, next_state
