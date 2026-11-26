# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Periodic, vertically cut softwood with pin knots and resin pockets."""

import warp as wp


@wp.func
def _grain_noise(uv: wp.vec2, seed: int, across: int, along: int):
    wrapped = uv - wp.vec2(wp.floor(uv[0]), wp.floor(uv[1]))
    return wp.pnoise(wp.rand_init(seed), wp.vec2(float(across) * wrapped[0], float(along) * wrapped[1]), across, along)


@wp.func
def _knot_layout(index: int, seed: int):
    # Positions and proportions follow the landmarks in the reference tile.
    center = wp.vec2(0.087, 0.329)
    radius = wp.vec2(0.033, 0.019)
    pitch = float(-0.22)
    weather = float(0.52)
    if index == 1:
        center = wp.vec2(0.334, 0.218)
        radius = wp.vec2(0.030, 0.017)
        pitch = -0.16
        weather = 0.17
    elif index == 2:
        center = wp.vec2(0.503, 0.151)
        radius = wp.vec2(0.019, 0.012)
        pitch = -0.52
        weather = 0.98
    elif index == 3:
        center = wp.vec2(0.659, 0.398)
        radius = wp.vec2(0.015, 0.022)
        pitch = -0.38
        weather = 0.24
    elif index == 4:
        center = wp.vec2(0.715, 0.420)
        radius = wp.vec2(0.023, 0.026)
        pitch = -0.32
        weather = 0.87
    elif index == 5:
        center = wp.vec2(0.819, 0.414)
        radius = wp.vec2(0.027, 0.025)
        pitch = 0.19
        weather = 0.96
    elif index >= 6:
        rng = wp.rand_init(seed + 127, index)
        center = wp.vec2(wp.randf(rng), wp.randf(rng))
        radius = wp.vec2(wp.randf(rng, 0.0024, 0.0055), wp.randf(rng, 0.0029, 0.0065))
        pitch = wp.randf(rng, -0.38, 0.38)
        weather = wp.randf(rng, 0.28, 0.78)
    if seed != 173 and index < 6:
        rng = wp.rand_init(seed + 113, index)
        center += wp.vec2(wp.randf(rng, -0.18, 0.18), wp.randf(rng, -0.20, 0.20))
        radius *= wp.randf(rng, 0.71, 1.32)
        pitch += wp.randf(rng, -0.29, 0.29)
    return center, radius, pitch, weather


@wp.func
def knotted_softwood(uv: wp.vec2, seed: int, resolution: float):
    """Return sRGB color, height, roughness, and occlusion."""
    ring_bend = float(0.0)
    best = float(0.0)
    knot_position = wp.vec2(0.0)
    knot_radius = float(0.0)
    knot_age = float(0.0)
    knot_index = int(-1)
    split = float(0.0)
    for index in range(16):
        center, radius, pitch, weather = _knot_layout(index, seed)
        delta = uv - center
        delta -= wp.vec2(wp.floor(delta[0] + 0.5), wp.floor(delta[1] + 0.5))
        local = wp.vec2((delta[0] - pitch * delta[1]) / radius[0], (delta[1] + pitch * delta[0]) / radius[1])
        measure = wp.length(local)
        benefit = 1.0 - wp.smoothstep(0.82, 1.25, measure)
        if benefit > best:
            best = benefit
            knot_position = local
            knot_radius = measure
            knot_age = weather
            knot_index = index
        if index == 5 or index == 2:
            # Splintered knots open a tapered split out of the branch pocket.
            stretch = wp.smoothstep(0.27, 0.84, local[0]) * (1.0 - wp.smoothstep(1.03, 2.91, local[0]))
            notch = 0.30 - 0.091 * wp.clamp(local[0] - 0.7, 0.0, 2.1)
            offset = local[1] + 0.20 * local[0] - 0.26
            split += stretch * (1.0 - wp.smoothstep(0.56 * notch, notch + 0.065, wp.abs(offset)))

        # Long fibres bend around the branch and rejoin the vertical grain.
        influence = 1.0 - wp.smoothstep(0.3, 3.25, wp.length(wp.vec2(local[0], 0.47 * local[1])))
        ring_bend -= 0.91 * radius[0] * influence * local[0] / (0.53 + wp.abs(local[0]))

    sweep = _grain_noise(uv, seed + 5, 9, 3)
    ripple = _grain_noise(uv, seed + 11, 23, 5)
    grain_uv = uv + wp.vec2(0.0048 * sweep + 0.0022 * ripple + ring_bend, 0.0)
    phase = 102.0 * grain_uv[0] + 2.7 * _grain_noise(grain_uv, seed + 19, 16, 1)
    phase += 0.21 * _grain_noise(grain_uv, seed + 23, 71, 7)
    wave = wp.abs(phase - wp.floor(phase + 0.5))
    aa = 52.0 / resolution
    ring = 1.0 - wp.smoothstep(0.039 - aa, 0.12 + aa, wave)
    ring *= wp.smoothstep(88.0, 320.0, resolution)
    contrast = 0.58 + 0.42 * _grain_noise(grain_uv, seed + 39, 57, 6)

    # A second, lower-frequency set of broad darker bands follows the same
    # fibres. It represents the darker transitions in the longitudinal cut.
    growth = 11.0 * grain_uv[0] + 0.88 * _grain_noise(grain_uv, seed + 41, 11, 2)
    broad = 1.0 - wp.smoothstep(0.021, 0.092, wp.abs(growth - wp.floor(growth + 0.5)))
    strip = wp.vec3(0.074, 0.106, 0.14)
    fiber = _grain_noise(grain_uv, seed + 51, 448, 176) * wp.smoothstep(374.0, 1240.0, resolution)
    micro = _grain_noise(grain_uv, seed + 57, 793, 620) * wp.smoothstep(730.0, 2050.0, resolution)
    fiber_cut = wp.smoothstep(0.14, 0.38, 0.84 * fiber + 0.28 * micro)
    warm = _grain_noise(grain_uv, seed + 61, 27, 13)
    base = wp.vec3(0.716, 0.635, 0.545) + wp.vec3(0.017 * sweep + 0.018 * warm + 0.021 * fiber + 0.014 * micro)
    color = base - strip * (0.52 * broad + 0.49 * contrast * ring + 0.20 * fiber_cut)
    height = 0.572 - 0.012 * broad - 0.010 * contrast * ring + 0.004 * warm + 0.006 * fiber + 0.003 * micro

    # The branch's own concentric rings sit inside an irregular bark collar.
    knot_crinkle = _grain_noise(uv, seed + 83, 149, 135)
    knot_distance = knot_radius + 0.24 * knot_crinkle
    knot_phase = knot_distance * 4.8 + 0.31 * knot_crinkle + 0.16 * knot_position[0]
    knot_wave = wp.abs(knot_phase - wp.floor(knot_phase + 0.5))
    knot_ring = 1.0 - wp.smoothstep(0.06, 0.24, knot_wave)
    edge = wp.smoothstep(0.58, 0.87, knot_distance)
    collar = edge * (1.0 - wp.smoothstep(1.02, 1.18, knot_distance))
    collar *= wp.smoothstep(0.02, 0.68, knot_age + 0.43 * knot_crinkle + 0.17 * knot_position[1])
    heart = wp.vec3(0.56, 0.435, 0.29) + wp.vec3(0.034 * knot_crinkle - 0.06 * knot_ring)
    resin = wp.vec3(0.193, 0.119, 0.071) + wp.vec3(0.11 * knot_crinkle)
    if knot_index == 5:
        mineral_stain = wp.smoothstep(-0.38, 0.45, -knot_position[0] + 0.62 * knot_crinkle)
        heart = wp.lerp(heart, wp.vec3(0.414, 0.425, 0.343), 0.80 * mineral_stain)
    if knot_index == 2:
        missing = wp.smoothstep(-0.49, -0.16, knot_position[1] + 0.43 * knot_position[0] + 0.38 * knot_crinkle)
        heart = wp.lerp(resin, heart, missing)
    knot_color = wp.lerp(heart, resin, 0.91 * collar)
    color = wp.lerp(color, knot_color, best)
    color = wp.lerp(color, resin, 0.88 * split)
    height -= best * (0.041 + 0.036 * collar + 0.013 * knot_ring) + 0.028 * split
    roughness = 0.676 + 0.019 * warm + 0.027 * fiber - 0.082 * best * collar
    occlusion = 0.995 - 0.15 * best * collar - 0.072 * split
    return color, height, roughness, occlusion
