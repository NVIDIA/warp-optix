# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Periodic pressed cork and bamboo screen surfaces."""

import warp as wp

from .mineral_surfaces import _noise, _stone_chips


@wp.func
def cork_surface(uv: wp.vec2, seed: int, resolution: float):
    uv -= wp.vec2(wp.floor(uv[0]), wp.floor(uv[1]))
    warp = 0.0032 * wp.vec2(_noise(uv, seed + 1, 159, 133), _noise(uv, seed + 2, 137, 161))
    chips, tint, _ = _stone_chips(uv + warp, seed + 11, 121, resolution)
    crumb, crumb_tint, _ = _stone_chips(uv - 0.62 * warp, seed + 12, 238, resolution)
    crumb *= (1.0 - chips) * wp.smoothstep(280.0, 820.0, resolution)
    dust = _noise(uv, seed + 14, 456, 417) * wp.smoothstep(560.0, 1300.0, resolution)
    pores = _noise(uv, seed + 15, 294, 331) * wp.smoothstep(400.0, 1040.0, resolution)
    cloud = _noise(uv, seed + 16, 31, 37)
    pit = wp.smoothstep(0.06, 0.45, -pores + 0.43 * dust) * (1.0 - 0.68 * chips)
    darker = wp.smoothstep(0.10, 0.49, -cloud + 0.23 * dust) * (1.0 - chips)
    binder = wp.vec3(0.710, 0.464, 0.254) - wp.vec3(0.26, 0.26, 0.16) * darker
    grain = wp.lerp(wp.vec3(0.59, 0.353, 0.171), wp.vec3(0.82, 0.595, 0.358), tint)
    fibers = wp.lerp(wp.vec3(0.56, 0.337, 0.157), wp.vec3(0.75, 0.515, 0.282), crumb_tint)
    color = wp.lerp(wp.lerp(binder, fibers, crumb), grain, chips)
    color += wp.vec3(0.067 * dust + 0.026 * cloud)
    color -= wp.vec3(0.24, 0.176, 0.101) * pit
    height = 0.542 + chips * (0.087 + 0.115 * tint) + 0.096 * crumb - 0.28 * pit - 0.061 * darker + 0.069 * dust
    roughness = 0.691 - chips * (0.036 + 0.081 * tint) + 0.10 * pit + 0.126 * pores + 0.061 * cloud
    return color, height, roughness, 1.0, 0.0, 1.0


@wp.func
def bamboo_screen(uv: wp.vec2, seed: int, resolution: float):
    uv -= wp.vec2(wp.floor(uv[0]), wp.floor(uv[1]))
    count = 24
    grid = uv[0] * float(count)
    col = int(wp.floor(grid))
    best = float(-2.0)
    tube = float(0.0)
    tone = float(0.0)
    node = float(0.0)
    swell = float(0.0)
    weather = _noise(uv, seed + 2, 29, 37)
    for step in range(-1, 2):
        stalk = col + step
        key = (stalk % count + count) % count
        state = wp.rand_init(seed + 3, key)
        shift = wp.randf(state, -0.08, 0.08)
        width = wp.randf(state, 0.434, 0.568)
        hue = wp.randf(state, -1.0, 1.0)
        phase = wp.randf(state, 0.0, 2.0 * wp.pi)
        node_phase = wp.randf(state, 0.0, 1.0)
        internodes = wp.randi(state, 6, 10)
        v_angle = 2.0 * wp.pi * uv[1]
        tilt = wp.sin(v_angle + phase)
        local = grid - float(stalk) - 0.5 - shift - 0.038 * tilt
        z = float(internodes) * uv[1] + node_phase + 0.11 * tilt + 0.027 * local
        z -= wp.floor(z)
        d = wp.min(z, 1.0 - z)
        ring = 1.0 - wp.smoothstep(0.009, 0.039 + float(internodes) / resolution, d)
        shoulder = 1.0 - wp.smoothstep(0.015, 0.16, d)
        radius = width + 0.005 + 0.025 * wp.cos(2.0 * v_angle + phase) + 0.015 * shoulder
        margin = radius - wp.abs(local)
        if margin > best:
            best = margin
            tube = wp.sqrt(wp.max(0.0, 1.0 - wp.pow(wp.abs(local) / radius, 2.0)))
            tone = hue
            node = ring
            swell = shoulder

    fiber = _noise(uv, seed + 17, 257, 31)
    fine = _noise(uv, seed + 20, 491, 137) * wp.smoothstep(560.0, 1500.0, resolution)
    opening = wp.smoothstep(-0.55 * float(count) / resolution, 0.55 * float(count) / resolution, best)
    variation = 0.037 * tone + 0.045 * fiber + 0.046 * weather + 0.024 * fine
    color = wp.vec3(0.510, 0.426, 0.265) + wp.vec3(variation)
    color += wp.vec3(-0.041, 0.015, 0.052) * (swell * 0.82 + 0.29 * tone)
    color -= wp.vec3(0.089, 0.066, 0.034) * node
    height = 0.137 + tube * tube * tube * (0.516 + 0.076 * tone + 0.117 * swell - 0.036 * node)
    height += 0.009 * fiber + 0.006 * fine
    roughness = 0.513 + 0.053 * weather + 0.048 * fiber + 0.033 * swell + 0.051 * node
    ao = (0.674 + 0.31 * tube - 0.073 * node) * (0.81 + 0.19 * opening)
    return color, height, roughness, ao, 0.0, opening
