# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Analytical models of short-pile microfiber and ground with brick rubble."""

import warp as wp


@wp.func
def _periodic_noise(state: wp.uint32, uv: wp.vec2, nx: int, ny: int):
    return wp.pnoise(state, wp.vec2(uv[0] * float(nx), uv[1] * float(ny)), nx, ny)


@wp.func
def _cell_state(seed: int, x: int, y: int, nx: int, ny: int):
    return wp.rand_init(seed, (x % nx + nx) % nx + nx * ((y % ny + ny) % ny))


@wp.func
def _microfiber(uv: wp.vec2, seed: int, resolution: float):
    state = wp.rand_init(seed)
    # Brush direction changes across the nap. Warp the fibers in UV space so
    # the small strokes turn together across an otherwise uniform blue fabric.
    drift = _periodic_noise(state, uv, 7, 9)
    lift = _periodic_noise(state, uv, 23, 19)
    grain_uv = uv + wp.vec2(0.019 * drift, 0.004 * lift)
    tufts = _periodic_noise(state, grain_uv, 432, 218)
    clumps = _periodic_noise(state, grain_uv, 245, 179)
    fine = _periodic_noise(state, grain_uv, 887, 532)
    visibility = wp.smoothstep(180.0, 840.0, resolution)
    individual = wp.smoothstep(550.0, 2100.0, resolution)
    roots = wp.smoothstep(0.27, 0.57, -tufts + 0.34 * clumps) * visibility
    nap = 0.46 * tufts * visibility + 0.32 * clumps + 0.22 * fine * individual
    color = wp.vec3(0.343, 0.535, 0.714)
    color += wp.vec3(0.042, 0.055, 0.062) * nap
    color -= wp.vec3(0.062, 0.078, 0.084) * roots
    height = 0.52 + 0.13 * nap - 0.12 * roots
    return color, height, 0.795 + 0.12 * roots - 0.062 * nap, 0.98 - 0.17 * roots


@wp.func
def _fragment(uv: wp.vec2, frequency: int, seed: int, resolution: float):
    """One jittered and rotated faceted fragment per cell, with empty cells allowed."""
    p = uv * float(frequency)
    cell = wp.vec2(wp.floor(p[0]), wp.floor(p[1]))
    state = _cell_state(seed, int(cell[0]), int(cell[1]), frequency, frequency)
    radius = wp.randf(state, 0.10, 0.40)
    # Keep each fragment away from its cell perimeter. Give small fragments
    # more room to move so the placement does not expose the grid.
    jitter = 0.44 - radius
    center = wp.vec2(wp.randf(state, -jitter, jitter), wp.randf(state, -jitter, jitter))
    p -= cell + wp.vec2(0.5) + center
    angle = wp.randf(state, 0.0, 6.2831853)
    c, s = wp.cos(angle), wp.sin(angle)
    q = wp.vec2(c * p[0] + s * p[1], -s * p[0] + c * p[1])
    q[1] *= wp.randf(state, 1.18, 1.85)
    slant = wp.randf(state, 0.18, 0.68)
    distance = wp.max(wp.abs(q[0] + slant * q[1]), wp.abs(q[1]))
    distance = wp.max(distance, wp.randf(state, 0.75, 1.15) * wp.abs(q[0] - q[1]))
    distance = wp.max(distance, wp.randf(state, 0.70, 1.15) * wp.abs(q[0] + q[1]))
    aa = wp.max(0.008, 0.55 * float(frequency) / resolution)
    mask = 1.0 - wp.smoothstep(radius - aa, radius + aa, distance)
    relief = (1.0 - wp.smoothstep(radius * 0.36, radius + aa, distance)) * (0.82 - 0.43 * q[0])
    return mask, relief, wp.randf(state), wp.randf(state, -1.0, 1.0)


@wp.func
def _rubble(uv: wp.vec2, seed: int, resolution: float):
    state = wp.rand_init(seed)
    patches = _periodic_noise(state, uv, 10, 13)
    powder = _periodic_noise(state, uv, 43, 49)
    grit = _periodic_noise(state, uv, 511, 551) * wp.smoothstep(275.0, 1024.0, resolution)
    sand = _periodic_noise(state, uv, 952, 967) * wp.smoothstep(600.0, 1850.0, resolution)
    grit_density = 0.36 + 0.64 * wp.smoothstep(-0.42, 0.38, patches + 0.45 * powder)
    uv += 0.0018 * wp.vec2(_periodic_noise(state, uv, 77, 84), _periodic_noise(state, uv, 86, 79))
    color = wp.vec3(0.57, 0.49, 0.415)
    color += wp.vec3(0.105 * patches + 0.080 * powder + grit_density * (0.10 * grit + 0.13 * sand))
    height = 0.43 + 0.035 * patches + 0.036 * powder + 0.024 * grit + 0.017 * sand
    occlusion = float(0.94)
    roughness = 0.76 - 0.07 * powder
    for band in range(4):
        frequency = 19 << band
        mask, relief, chance, tone = _fragment(uv, frequency, seed + 343 + band * 23, resolution)
        if band == 3:
            mask *= wp.smoothstep(110.0, 480.0, resolution)
        threshold = 0.29 - 0.25 * patches
        if band >= 2:
            threshold += 0.36 * (1.0 - grit_density)
        if chance > threshold:
            brick = wp.vec3(0.57, 0.312, 0.237) + tone * wp.vec3(0.105, 0.072, 0.062)
            stone = wp.vec3(0.36, 0.345, 0.316) + tone * wp.vec3(0.13)
            if chance > 0.83:
                stone = wp.vec3(0.70, 0.627, 0.488) + tone * wp.vec3(0.11)
            rock = stone
            if band < 2 and chance > 0.40 and chance < 0.79:
                rock = brick
            rock += wp.vec3(0.15 * grit + 0.14 * sand + 0.076 * powder)
            dust = 0.19 + 0.24 * wp.smoothstep(-0.28, 0.50, powder + 0.45 * grit)
            rock = wp.lerp(rock, color, dust)
            color = wp.lerp(color, rock, mask)
            height += 0.078 * relief * mask / float(1 << band)
            occlusion -= 0.17 * mask * (1.0 - relief)
            roughness = wp.lerp(roughness, 0.64 + 0.11 * grit + 0.18 * dust, mask)
    fine_stones = wp.smoothstep(0.23, 0.58, grit + 0.51 * sand)
    dark_specks = wp.smoothstep(0.26, 0.60, -grit - 0.53 * sand) * grit_density
    color += wp.vec3(0.13 * fine_stones - 0.20 * dark_specks)
    height += 0.016 * fine_stones - 0.012 * dark_specks
    return color, height, roughness, wp.clamp(occlusion - 0.14 * dark_specks, 0.0, 1.0)


@wp.func
def organic_surface(uv: wp.vec2, seed: int, resolution: float, style: int):
    """Evaluate shared color, relief, roughness and occlusion fields."""
    if style == 0:
        return _microfiber(uv, seed, resolution)
    return _rubble(uv, seed, resolution)
