# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared height-to-normal conversion for procedural materials."""

import warp as wp


@wp.func
def _normal_from_relief(relief: wp.array2d(dtype=float), row: int, col: int, strength: float) -> wp.vec3:
    h, w = relief.shape[0], relief.shape[1]
    west, east = (col + w - 1) % w, (col + 1) % w
    north, south = (row + h - 1) % h, (row + 1) % h
    dx = relief[row, west] - relief[row, east]
    dy = relief[south, col] - relief[north, col]
    if wp.min(h, w) <= 1536:
        # Average a little of the adjacent rows and columns at preview
        # resolutions. This damps one-pixel staircase edges on diagonal relief.
        nw, ne = relief[north, west], relief[north, east]
        sw, se = relief[south, west], relief[south, east]
        dx = 0.84 * dx + 0.08 * ((nw - ne) + (sw - se))
        dy = 0.84 * dy + 0.08 * ((sw - nw) + (se - ne))
    n = wp.normalize(wp.vec3(strength * float(w) * dx, strength * float(h) * dy, 1.0))
    return 0.5 * n + wp.vec3(0.5)
