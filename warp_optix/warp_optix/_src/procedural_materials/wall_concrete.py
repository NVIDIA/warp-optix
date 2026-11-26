# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Procedural cast panels, trowelled concrete and weathered cement plaster."""

import warp as wp

from .sampling import periodic_noise as _noise


@wp.func
def _cell_random(seed: int, cell: wp.vec2, sx: int, sy: int):
    col = (int(cell[0]) % sx + sx) % sx
    row = (int(cell[1]) % sy + sy) % sy
    return wp.rand_init(seed, col + sx * row)


@wp.func
def _crack_distance(uv: wp.vec2, seed: int):
    """Distance to fractured boundaries in UV units."""
    frequency = 4
    p = float(frequency) * uv
    cell = wp.vec2(wp.floor(p[0]), wp.floor(p[1]))
    first = float(8.0)
    second = float(8.0)
    for j in range(-1, 2):
        for i in range(-1, 2):
            neighbor = cell + wp.vec2(float(i), float(j))
            rng = _cell_random(seed, neighbor, frequency, frequency)
            point = neighbor + wp.vec2(wp.randf(rng, 0.14, 0.86), wp.randf(rng, 0.14, 0.86))
            delta = p - point
            distance = wp.dot(delta, delta)
            if distance < first:
                second = first
                first = distance
            elif distance < second:
                second = distance
    return 0.5 * (wp.sqrt(second) - wp.sqrt(first)) / float(frequency)


@wp.func
def _pinholes(uv: wp.vec2, seed: int, sx: int, sy: int, resolution: float):
    p = wp.cw_mul(uv, wp.vec2(float(sx), float(sy)))
    cell = wp.vec2(wp.floor(p[0]), wp.floor(p[1]))
    rng = _cell_random(seed, cell, sx, sy)
    center = wp.vec2(wp.randf(rng, 0.27, 0.73), wp.randf(rng, 0.27, 0.73))
    radius = wp.randf(rng, 0.018, 0.14)
    amount = wp.smoothstep(0.58, 0.97, wp.randf(rng))
    delta = p - cell - center
    delta[1] *= wp.randf(rng, 0.92, 1.42)
    aa = 0.55 * float(wp.max(sx, sy)) / resolution
    return amount * (1.0 - wp.smoothstep(radius * 0.44, radius + aa, wp.length(delta)))


@wp.func
def _cast_panels(uv: wp.vec2, seed: int, resolution: float):
    # Each panel retains two circular recesses from its form ties. The cones,
    # seams and fractures all contribute to the same colored relief.
    p = wp.cw_mul(uv, wp.vec2(3.0, 4.0))
    cell = wp.vec2(wp.floor(p[0]), wp.floor(p[1]))
    local = p - cell
    rng = _cell_random(seed, cell, 3, 4)
    tone = wp.randf(rng, -0.024, 0.024)
    placement = wp.vec2(wp.randf(rng, -0.014, 0.014), wp.randf(rng, -0.017, 0.017))
    tie = local - wp.vec2(0.72, 0.31) - placement
    tie[0] = wp.min(wp.abs(tie[0]), wp.abs(tie[0] + 0.60))
    distance = wp.length(wp.cw_mul(tie, wp.vec2(1.0 / 3.0, 1.0 / 4.0)))
    aa = 0.55 / resolution
    inlet = 1.0 - wp.smoothstep(0.0049 - aa, 0.0063 + aa, distance)
    hole = 1.0 - wp.smoothstep(0.0026 - aa, 0.0046 + aa, distance)

    edge = wp.min(wp.min(local[0], 1.0 - local[0]) / 3.0, wp.min(local[1], 1.0 - local[1]) / 4.0)
    seam = 1.0 - wp.smoothstep(0.00045, 0.0015 + aa, edge)
    bleed = (1.0 - wp.smoothstep(0.0018, 0.0074, edge)) * 0.013
    broad = _noise(uv, seed + 3, 16, 13)
    mottles = _noise(uv, seed + 5, 62, 71)
    streaks = _noise(uv, seed + 36, 154, 8)
    grit = _noise(uv, seed + 32, 522, 513) * wp.smoothstep(290.0, 1100.0, resolution)
    distort = wp.vec2(_noise(uv, seed + 10, 29, 29), _noise(uv, seed + 11, 31, 27))
    crack_dist = _crack_distance(uv + 0.005 * distort, seed + 87)
    weather = wp.smoothstep(-0.14, 0.43, _noise(uv, seed + 53, 7, 8))
    crack = (1.0 - wp.smoothstep(0.00020, 0.00085 + aa, crack_dist)) * weather
    pits = _pinholes(uv, seed + 41, 135, 129, resolution)
    color = wp.vec3(0.630, 0.632, 0.627)
    color += wp.vec3(tone + 0.041 * broad + 0.044 * mottles + 0.038 * grit + 0.022 * streaks + bleed)
    color -= wp.vec3(0.108 * seam + 0.075 * crack + 0.14 * pits + 0.105 * inlet + 0.36 * hole)
    height = 0.926 - 0.019 * seam - 0.014 * crack - 0.055 * inlet - 0.185 * hole - 0.036 * pits
    height += 0.007 * (_noise(uv, seed + 66, 301, 331) * wp.smoothstep(350.0, 1200.0, resolution) + broad)
    roughness = 0.50 + 0.11 * mottles + 0.16 * seam + 0.19 * hole + 0.12 * pits
    return color, height, roughness, 0.99 - 0.16 * seam - 0.35 * inlet - 0.25 * hole - 0.15 * pits


@wp.func
def _trowelled(uv: wp.vec2, seed: int, resolution: float):
    # Broad trowel pressure changes form long swaths and leave faint curved
    # drag marks. Finer open pores interrupt those marks at high resolutions.
    pressure = _noise(uv, seed + 6, 8, 11)
    flow = uv + wp.vec2(0.0, 0.024 * _noise(uv, seed + 65, 8, 7))
    burnish = wp.smoothstep(-0.30, 0.60, _noise(flow, seed + 3, 16, 27) + 0.45 * pressure)
    stroke = (_noise(flow, seed + 53, 7, 127) + 0.31 * _noise(flow, seed + 56, 11, 242)) * burnish
    pinholes = _pinholes(uv, seed, 119, 173, resolution)
    hairs = _noise(flow, seed + 74, 52, 395) * wp.smoothstep(180.0, 800.0, resolution)
    fine = _noise(uv, seed + 73, 547, 671) * wp.smoothstep(330.0, 1340.0, resolution)
    color = wp.vec3(0.328, 0.344, 0.351)
    color += wp.vec3(0.050 * pressure + 0.049 * burnish + 0.045 * stroke + 0.015 * hairs + 0.034 * fine)
    color -= wp.vec3(0.092 * pinholes)
    height = 0.61 + 0.074 * pressure + 0.024 * stroke + 0.014 * hairs + 0.016 * fine - 0.102 * pinholes
    return color, height, 0.785 - 0.090 * burnish + 0.12 * pinholes, 0.98 - 0.36 * pinholes


@wp.func
def _weathered_plaster(uv: wp.vec2, seed: int, resolution: float):
    # Erosion peels thin layers from uneven patches. The exposed substrate
    # carries more pores and a different mineral color and roughness.
    warp = wp.vec2(_noise(uv, seed + 5, 11, 14), _noise(uv, seed + 7, 13, 17))
    warped = uv + 0.022 * warp
    water = _noise(warped, seed + 10, 6, 12)
    scatter = _noise(warped, seed + 20, 31, 35)
    erosion = water + 0.58 * scatter + 0.25 * _noise(uv, seed + 32, 92, 97)
    edge_width = 0.065 + 10.0 / resolution
    exposed = wp.smoothstep(0.23 - edge_width, 0.23 + edge_width, erosion)
    shallow = wp.smoothstep(0.079 - edge_width, 0.079 + edge_width, erosion) - exposed
    exposed *= 0.63 + 0.37 * wp.smoothstep(-0.41, 0.24, _noise(uv, seed + 71, 9, 7))
    pits = _pinholes(uv, seed + 9, 217, 209, resolution) * (0.36 + 0.64 * exposed)
    fine = _noise(uv, seed + 55, 498, 521) * wp.smoothstep(235.0, 1100.0, resolution)
    grains = _noise(uv, seed + 15, 182, 201)
    lime = wp.smoothstep(0.34, 0.58, grains + 0.58 * fine + 0.38 * scatter)
    skim = wp.vec3(0.601, 0.582, 0.54) + wp.vec3(0.098 * water + 0.075 * scatter)
    undercoat = wp.vec3(0.522, 0.520, 0.487) + wp.vec3(0.123 * scatter)
    color = wp.lerp(skim, undercoat, exposed)
    color += shallow * wp.vec3(0.033, 0.030, 0.022) + wp.vec3(0.08 * fine + 0.048 * grains + 0.14 * lime - 0.11 * pits)
    height = 0.75 - 0.11 * exposed - 0.04 * shallow + 0.051 * scatter + 0.025 * fine - 0.10 * pits
    return color, height, 0.54 + 0.17 * exposed + 0.09 * pits + 0.08 * fine, 0.99 - 0.15 * exposed - 0.20 * pits


@wp.func
def wall_concrete(uv: wp.vec2, seed: int, resolution: float, style: int):
    if style == 0:
        return _cast_panels(uv, seed, resolution)
    if style == 1:
        return _trowelled(uv, seed, resolution)
    return _weathered_plaster(uv, seed, resolution)
