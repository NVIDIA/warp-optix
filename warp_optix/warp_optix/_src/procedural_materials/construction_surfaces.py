# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Seamless plywood edge, drywall, asphalt shingles, zinc, and quartz."""

import warp as wp

from .mineral_surfaces import _cell_seed, _stone_chips
from .sampling import periodic_noise as _noise


@wp.func
def _aggregate_uv(uv: wp.vec2, seed: int) -> wp.vec2:
    warp = wp.vec2(_noise(uv, seed, 126, 117), _noise(uv, seed + 1, 111, 129))
    return uv + 0.0038 * warp


@wp.func
def _plywood_edge(uv: wp.vec2, seed: int, resolution: float):
    period = 64
    along = float(period) * uv[1] + 0.11 * _noise(uv, seed + 1, 4, 11)
    ply = int(wp.floor(along))
    local = along - float(ply)
    state = _cell_seed(seed, ply, 0, period)
    tone = wp.randf(state, -0.066, 0.056)
    # Rotary veneer alternates cross sections and longitudinal fibers.
    across_grain = _noise(uv, seed + 4, 234, 67)
    long_grain = _noise(uv, seed + 8, 39, 279)
    layer = (ply % period + period) % period
    grain = wp.lerp(long_grain, across_grain, float(layer & 1))
    fiber = _noise(uv, seed + 7, 153, 441) * wp.smoothstep(500.0, 1450.0, resolution)
    state = _cell_seed(seed + 15, ply, 0, period)
    glue = wp.randf(state, 0.036, 0.076)
    aa = wp.min(0.024, 0.35 * float(period) / resolution)
    joint = 1.0 - wp.smoothstep(glue - aa, glue + 0.018 + aa, wp.min(local, 1.0 - local))
    lip = 1.0 - wp.smoothstep(0.045 - aa, 0.091 + aa, wp.abs(local - 0.19))
    bands = 0.103 * lip - 0.025 * float(layer & 1)
    # Occasional narrow, elongated splits within a single sheet.
    check = wp.smoothstep(0.22, 0.50, _noise(uv, seed + 45, 26, 186))
    check *= wp.smoothstep(0.02, 0.1, wp.min(local, 1.0 - local))
    stain = _noise(uv, seed + 51, 11, 13) + 0.44 * _noise(uv, seed + 54, 35, 77)
    color = wp.vec3(0.716, 0.637, 0.523) + wp.vec3(tone + bands + 0.118 * grain + 0.047 * fiber + 0.19 * stain)
    color = wp.lerp(color, wp.vec3(0.45, 0.374, 0.275), wp.clamp(0.66 * joint + 0.48 * check, 0.0, 1.0))
    relief = 0.53 - 0.13 * joint - 0.11 * check + 0.045 * grain + 0.022 * fiber
    return color, relief, 0.56 + 0.14 * joint + 0.12 * check + 0.084 * grain, 1.0 - 0.23 * joint - 0.11 * check


@wp.func
def _painted_drywall(uv: wp.vec2, seed: int, resolution: float):
    roller = _noise(uv, seed + 3, 46, 30)
    granular = _noise(uv, seed + 6, 181, 161)
    pits = wp.smoothstep(0.04, 0.52, _noise(uv, seed + 21, 312, 299))
    pits *= wp.smoothstep(440.0, 1150.0, resolution)
    stipple = wp.smoothstep(-0.16, 0.36, granular + 0.19 * roller)
    unevenness = _noise(uv, seed + 11, 8, 11)
    color = wp.vec3(0.891, 0.882, 0.861) + wp.vec3(0.015 * unevenness + 0.010 * stipple - 0.008 * pits)
    relief = 0.48 + 0.033 * roller + 0.085 * stipple - 0.029 * pits
    return color, relief, 0.69 + 0.055 * pits - 0.062 * stipple


@wp.func
def _asphalt_shingles(uv: wp.vec2, seed: int, resolution: float):
    y = 10.0 * uv[1] + 0.015 * _noise(uv, seed + 1, 12, 11)
    row = int(wp.floor(y))
    row_rng = _cell_seed(seed, row, 0, 10)
    stagger = wp.randf(row_rng)
    shoulder = wp.randf(row_rng, 0.43, 0.66)
    row_tone = wp.randf(row_rng, -0.025, 0.028)
    x = 7.0 * uv[0] + stagger
    tab = int(wp.floor(x))
    p = x - float(tab)
    q = y - float(row)
    tab_rng = wp.rand_init(seed + 5, (tab % 7 + 7) % 7 + 7 * ((row % 10 + 10) % 10))
    pigment = wp.randf(tab_rng, -0.073, 0.063)
    aa = wp.min(0.012, 4.0 / resolution)
    lap = wp.smoothstep(0.015 - aa, 0.053 + aa, wp.min(q, 1.0 - q))
    side = wp.min(p, 1.0 - p)
    below_slot_end = 1.0 - wp.smoothstep(shoulder - 0.02, shoulder + 0.07, q)
    slot = (1.0 - wp.smoothstep(0.014 - aa, 0.032 + aa, side)) * below_slot_end
    pigment *= wp.smoothstep(0.0, 0.045, side) * below_slot_end
    profile = lap * (1.0 - slot)
    stones = _aggregate_uv(uv, seed + 14)
    granule, variation, _ = _stone_chips(stones, seed + 12, 211, resolution)
    fleck, contrast, _ = _stone_chips(stones, seed + 13, 405, resolution)
    weathering = _noise(uv, seed + 10, 41, 62)
    color = wp.vec3(0.256, 0.237, 0.214) + wp.vec3(row_tone + pigment + 0.055 * weathering)
    color += wp.vec3((0.042 + 0.18 * (variation - 0.37)) * granule + 0.14 * fleck * (contrast - 0.38))
    color = wp.lerp(wp.vec3(0.107, 0.100, 0.091) + wp.vec3(0.022 * weathering), color, profile)
    relief = 0.30 + profile * (0.40 - 0.19 * q + 0.081 * granule + 0.041 * fleck)
    return color, relief, 0.81 + 0.105 * granule - 0.083 * (1.0 - profile), 0.66 + 0.34 * profile


@wp.func
def _galvanized_zinc(uv: wp.vec2, seed: int, resolution: float):
    period = 25
    grid = uv * float(period)
    x, y = int(wp.floor(grid[0])), int(wp.floor(grid[1]))
    nearest = float(100.0)
    second = float(100.0)
    delta = wp.vec2(0.0)
    crystal = float(0.0)
    next_crystal = float(0.0)
    angle = float(0.0)
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            rng = _cell_seed(seed, x + dx, y + dy, period)
            point = wp.vec2(float(x + dx) + wp.randf(rng, 0.14, 0.86), float(y + dy) + wp.randf(rng, 0.14, 0.86))
            offset = grid - point
            d = wp.dot(offset, offset)
            if d < nearest:
                second, nearest = nearest, d
                delta = offset
                next_crystal = crystal
                crystal = wp.randf(rng)
                angle = wp.randf(rng, -wp.pi, wp.pi)
            elif d < second:
                second = d
                next_crystal = wp.randf(rng)
    # Zinc solidifies as spangles with different reflectances and dendrites
    # along the three hexagonal crystal axes.
    radial = wp.atan2(delta[1], delta[0]) - angle
    facet = 0.5 + 0.5 * wp.cos(6.0 * radial)
    grain = _noise(uv, seed + 17, 245, 257)
    sector = radial - (wp.pi / 3.0) * wp.floor(radial * (3.0 / wp.pi) + 0.5)
    radius = wp.sqrt(nearest)
    across = radius * wp.abs(wp.sin(sector))
    along = radius * wp.cos(sector)
    branches = wp.sin(81.0 * (across + 0.48 * along) + 3.5 * grain)
    branches *= wp.exp(-11.0 * across) * wp.smoothstep(0.01, 0.11, radius)
    boundary = wp.smoothstep(0.0, 0.025 + 14.0 / resolution, second - nearest)
    crystal_mix = wp.lerp(0.5 * (crystal + next_crystal), crystal, boundary)
    patina = 0.057 * _noise(uv, seed + 53, 52, 67)
    tone = 0.808 + 0.206 * (crystal_mix - 0.5) + boundary * (0.023 * (facet - 0.5) + 0.031 * branches)
    tone += patina + 0.021 * grain
    relief = 0.47 + boundary * (0.017 * delta[0] + 0.009 * branches) + 0.015 * grain
    rough = 0.27 + 0.28 * (1.0 - crystal_mix) + 0.052 * grain
    return wp.vec3(tone, tone + 0.004, tone + 0.024), relief, rough


@wp.func
def _engineered_quartz(uv: wp.vec2, seed: int, resolution: float):
    stones = _aggregate_uv(uv, seed + 11)
    small, crystal, _ = _stone_chips(stones, seed + 8, 167, resolution)
    large, shade, _ = _stone_chips(stones, seed + 10, 79, resolution)
    large *= wp.smoothstep(0.59, 0.72, shade)
    haze = _noise(uv, seed + 3, 41, 43)
    matrix = wp.vec3(0.899, 0.891, 0.876) + wp.vec3(0.033 * haze)
    white = wp.vec3(0.963, 0.955, 0.934)
    mineral = wp.lerp(wp.vec3(0.56, 0.54, 0.53), white, wp.smoothstep(0.07, 0.95, crystal))
    color = wp.lerp(matrix, mineral, small)
    color = wp.lerp(color, wp.lerp(white, wp.vec3(0.655, 0.625, 0.583), shade), 0.91 * large)
    # Embedded grains read primarily in reflectance. Polishing leaves a
    # nearly planar surface, with the binder slightly below the hard chips.
    relief = 0.55 + 0.016 * small + 0.007 * large + 0.007 * haze
    return color, relief, 0.27 - 0.079 * small - 0.035 * large + 0.09 * crystal * small


@wp.func
def construction_surface(uv: wp.vec2, seed: int, resolution: float, variant: int):
    if variant == 0:
        color, relief, rough, ao = _plywood_edge(uv, seed, resolution)
        return color, relief, rough, ao, 0.0, 1.0
    if variant == 1:
        color, relief, rough = _painted_drywall(uv, seed, resolution)
        return color, relief, rough, 1.0, 0.0, 1.0
    if variant == 2:
        color, relief, rough, ao = _asphalt_shingles(uv, seed, resolution)
        return color, relief, rough, ao, 0.0, 1.0
    if variant == 3:
        color, relief, rough = _galvanized_zinc(uv, seed, resolution)
        return color, relief, rough, 1.0, 1.0, 1.0
    color, relief, rough = _engineered_quartz(uv, seed, resolution)
    return color, relief, rough, 1.0, 0.0, 1.0
