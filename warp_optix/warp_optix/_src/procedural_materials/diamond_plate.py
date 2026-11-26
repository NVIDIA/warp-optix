# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Weathered stamped steel with alternating raised lozenges."""

import warp as wp

from .mineral_surfaces import _noise


@wp.func
def _tread_profile(uv: wp.vec2, seed: int, resolution: float):
    cols, rows = 28, 58
    grid = wp.cw_mul(uv, wp.vec2(float(cols), float(rows))) + wp.vec2(0.07, 0.24)
    row = int(wp.floor(grid[1]))
    shoulder = float(0.0)
    crown = float(0.0)
    wear_rim = float(0.0)
    aa = 0.53 * float(cols) / resolution
    for i in range(-1, 2):
        y = row + i
        odd = y & 1
        x = grid[0] - 0.5 * float(odd)
        col = int(wp.floor(x))
        key = (col % cols + cols) % cols + cols * ((y % rows + rows) % rows)
        state = wp.rand_init(seed + 31, key)
        px = x - float(col) - 0.5 + wp.randf(state, -0.012, 0.012)
        py = (grid[1] - float(y) - 0.5) * (float(cols) / float(rows)) + wp.randf(state, -0.009, 0.009)
        slope = 1.0 - 2.0 * float(odd)
        along = (px + slope * py) * 0.70710678
        across = (py - slope * px) * 0.70710678
        length = wp.randf(state, 0.402, 0.437)
        width = wp.randf(state, 0.126, 0.148)
        phase = wp.abs(along) / length
        local_width = width * wp.max(0.0, 1.0 - phase * phase)
        edge = wp.min(local_width - wp.abs(across), length - wp.abs(along))
        base = wp.smoothstep(-aa - 0.005, 0.052 + aa, edge)
        crest = wp.smoothstep(0.033 - aa, 0.085 + aa, edge)
        shoulder = wp.max(shoulder, base)
        crown = wp.max(crown, crest)
        # Foot traffic exposes irregular glints mainly along the shoulders.
        band = wp.smoothstep(-0.010 - aa, 0.016 + aa, edge) * (1.0 - wp.smoothstep(0.026 - aa, 0.056 + aa, edge))
        band *= wp.smoothstep(-0.035, 0.012, across)
        wear_rim = wp.max(wear_rim, band)
    return shoulder, crown, wear_rim


@wp.func
def diamond_plate(uv: wp.vec2, seed: int, resolution: float):
    uv = uv - wp.vec2(wp.floor(uv[0]), wp.floor(uv[1]))
    shoulder, crown, rim = _tread_profile(uv, seed, resolution)
    patina = _noise(uv, seed + 3, 11, 13) + 0.60 * _noise(uv, seed + 5, 31, 33)
    grain = _noise(uv, seed + 7, 118, 129)
    pits = _noise(uv, seed + 11, 255, 233) * wp.smoothstep(320.0, 1170.0, resolution)
    grit = _noise(uv, seed + 17, 497, 531) * wp.smoothstep(550.0, 1720.0, resolution)
    corrosion = wp.smoothstep(-0.19, 0.45, 0.78 * patina + 0.35 * grain + 0.19 * pits)
    broken_rim = rim * wp.smoothstep(-0.12, 0.38, 0.53 * grain + 0.64 * pits + 0.25 * grit)
    warm = wp.vec3(0.439, 0.387, 0.327)
    rust = wp.vec3(0.058, 0.034, 0.013) * corrosion
    base = warm + rust + wp.vec3(0.045 * patina + 0.080 * grain + 0.068 * pits + 0.045 * grit)
    color = base - wp.vec3(0.110, 0.096, 0.076) * shoulder + wp.vec3(0.19, 0.18, 0.15) * broken_rim
    relief = 0.286 + 0.344 * shoulder + 0.156 * crown + (0.042 - 0.021 * shoulder) * grain + 0.035 * pits + 0.027 * grit
    roughness = 0.565 + 0.089 * patina + 0.166 * grain + 0.13 * pits + 0.09 * corrosion - 0.17 * shoulder
    roughness -= 0.18 * broken_rim
    ao = 0.934 + 0.033 * patina + 0.044 * crown - 0.10 * rim + 0.021 * pits
    metalness = 0.84 * broken_rim * (1.0 - 0.45 * corrosion)
    return color, relief, roughness, ao, metalness, 1.0
