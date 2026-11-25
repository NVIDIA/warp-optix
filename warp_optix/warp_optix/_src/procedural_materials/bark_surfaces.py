# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Periodic bark surfaces with fractured cork and overlapping scales."""

import warp as wp


@wp.func
def _bark_noise(uv: wp.vec2, seed: int, columns: int, rows: int):
    uv = uv - wp.vec2(wp.floor(uv[0]), wp.floor(uv[1]))
    return wp.pnoise(
        wp.rand_init(seed),
        wp.vec2(float(columns) * uv[0], float(rows) * uv[1]),
        columns,
        rows,
    )


@wp.func
def _cork_scales(uv: wp.vec2, seed: int, columns: int, rows: int):
    """Overlapping asymmetrical, tapered flakes. Return the uppermost surface."""
    grid = wp.vec2(float(columns) * uv[0], float(rows) * uv[1])
    ox = int(wp.floor(grid[0]))
    oy = int(wp.floor(grid[1]))
    local = grid - wp.vec2(float(ox), float(oy))
    first = float(-2.0)
    second = float(-2.0)
    pigment = float(0.0)
    slope = float(0.0)
    for y in range(-1, 2):
        for x in range(-1, 2):
            i = ((ox + x) % columns + columns) % columns
            j = ((oy + y) % rows + rows) % rows
            state = wp.rand_init(seed, i + columns * j)
            center = wp.vec2(float(x) + wp.randf(state, 0.06, 0.94), float(y) + wp.randf(state, 0.06, 0.94))
            d = local - center
            rotation = wp.randf(state, -0.33, 0.33)
            width = wp.randf(state, 0.37, 0.79)
            length = wp.randf(state, 0.48, 0.98)
            along = d[1] / length
            across = (d[0] + rotation * d[1] + 0.15 * d[1] * d[1]) / width
            across *= 1.0 + 0.34 * along
            # The pointed corners and uneven shoulders break the circular
            # outlines that an ellipse would otherwise produce.
            side = wp.max(wp.abs(across) + 0.37 * wp.abs(along), 0.86 * wp.abs(along) + 0.23 * wp.abs(across - along))
            lift = wp.randf(state, -0.17, 0.19)
            value = lift + 0.83 * (1.0 - side)
            if value > first:
                second = first
                first = value
                pigment = wp.randf(state, -1.0, 1.0)
                slope = along
            elif value > second:
                second = value
    return first, second, pigment, slope


@wp.func
def _bark_grooves(uv: wp.vec2, seed: int, columns: int, rows: int):
    """Randomly spaced fissures with varying width and finite vertical spans."""
    grid = wp.vec2(float(columns) * uv[0], float(rows) * uv[1])
    ox = int(wp.floor(grid[0]))
    oy = int(wp.floor(grid[1]))
    local = grid - wp.vec2(float(ox), float(oy))
    groove = float(4.0)
    for y in range(-1, 2):
        for x in range(-1, 2):
            i = ((ox + x) % columns + columns) % columns
            j = ((oy + y) % rows + rows) % rows
            rng = wp.rand_init(seed, i + columns * j)
            cy = float(y) + wp.randf(rng, 0.1, 0.9)
            d = local[1] - cy
            cx = float(x) + wp.randf(rng, 0.07, 0.93) + d * wp.randf(rng, -0.5, 0.5)
            span = wp.randf(rng, 0.33, 0.77)
            radius = wp.randf(rng, 0.025, 0.13)
            # The line thins into the surrounding cork at either end.
            radius *= 1.0 - wp.smoothstep(0.26 * span, span, wp.abs(d))
            distance = wp.abs(local[0] - cx) - radius + 4.0 * wp.max(0.0, wp.abs(d) - span)
            groove = wp.min(groove, distance)
    return groove


@wp.func
def bark_surface(uv: wp.vec2, seed: int, resolution: float, style: int):
    """Return sRGB color, height, roughness, and occlusion from one bark profile."""
    uv = uv - wp.vec2(wp.floor(uv[0]), wp.floor(uv[1]))
    columns = 48
    rows = 12
    if style == 1:
        columns = 29
        rows = 11

    bend = _bark_noise(uv, seed + 13, 6, 4)
    sway = _bark_noise(uv, seed + 17, 17, 8)
    splinters = _bark_noise(uv, seed + 19, 144, 93) * wp.smoothstep(110.0, 460.0, resolution)
    fray = _bark_noise(uv, seed + 23, 283, 151) * wp.smoothstep(210.0, 920.0, resolution)
    warped = uv + wp.vec2(0.029 * bend + 0.013 * sway + 0.0016 * splinters + 0.00052 * fray, 0.027 * sway)
    top, under, pigment, slope = _cork_scales(warped, seed, columns, rows)
    splits = _bark_grooves(warped, seed + 29, columns, rows - 3)
    broad_top, broad_under, broad_pigment, broad_slope = _cork_scales(warped, seed + 31, columns // 3, 5)
    broad_plate = wp.smoothstep(-0.14, 0.58, broad_top + 0.12 * sway)
    broad_split = 1.0 - wp.smoothstep(0.019, 0.15, broad_top - broad_under + 0.21 * sway)
    broad_split *= 0.78 + 0.22 * wp.smoothstep(-0.7, 0.7, broad_slope)
    aa = 0.42 * float(columns) / resolution
    groove = 1.0 - wp.smoothstep(-aa, 0.047 + aa, splits + 0.093 * splinters + 0.061 * fray)
    pocket = 1.0 - wp.smoothstep(-0.17 - aa, 0.045 + aa, top + 0.25 * splinters)
    lip = 1.0 - wp.smoothstep(0.017 - aa, 0.109 + aa, top - under + 0.21 * splinters + 0.13 * fray)
    lip *= wp.smoothstep(-0.25, 0.26, sway - 0.33 * slope)
    fissure = wp.max(groove, wp.max(0.86 * pocket, 0.46 * lip))
    plate = wp.smoothstep(-0.13 - aa, 0.59 + aa, top + 0.17 * splinters + 0.11 * fray)
    # Cork on the upper face differs from the split, exposed edges. Shape
    # the color patch from its thickness so the boundaries are legible.
    flake = wp.smoothstep(0.17 - aa, 0.28 + aa, top + 0.23 * splinters + 0.12 * fray + 0.11 * pigment)
    flake *= wp.smoothstep(0.024 - aa, 0.14 + aa, top - under + 0.15 * splinters)

    grain = _bark_noise(warped, seed + 37, 510, 231) * wp.smoothstep(420.0, 1400.0, resolution)
    grit = _bark_noise(warped, seed + 41, 977, 821) * wp.smoothstep(820.0, 2600.0, resolution)
    timber = _bark_noise(warped, seed + 43, 128, 49) + 0.65 * splinters + 0.42 * fray
    peeling = wp.smoothstep(0.10, 0.27, timber + 0.21 * grain)
    shoulder = wp.smoothstep(0.045, 0.29, splits) * (1.0 - wp.smoothstep(0.31, 0.5, splits))
    peeling = wp.max(0.70 * shoulder * wp.smoothstep(-0.14, 0.28, timber + sway), peeling * 0.35)
    granules = wp.smoothstep(0.13, 0.34, 0.35 * grain + 0.69 * grit + 0.41 * fray)
    age = wp.smoothstep(-0.33, 0.36, 0.68 * timber + 0.51 * bend - 0.26 * sway)

    cork = wp.lerp(wp.vec3(0.397, 0.390, 0.350), wp.vec3(0.513, 0.467, 0.409), age)
    fresh = wp.vec3(0.647, 0.419, 0.269)
    fresh_mask = wp.smoothstep(-0.16, 0.46, pigment + 0.42 * bend)
    fresh *= 0.81 + 0.16 * age
    base = wp.lerp(cork, fresh, wp.max(0.68 * peeling, 0.79 * flake * fresh_mask))
    crevice = wp.vec3(0.315, 0.278, 0.214)
    detail = 0.085 * fray + 0.105 * grain + 0.094 * grit + 0.066 * granules
    height = 0.555 + 0.105 * plate - 0.064 * fissure + 0.022 * peeling
    roughness = 0.643 - 0.066 * plate + 0.101 * fray + 0.047 * grain
    if style == 1:
        damp = _bark_noise(uv, seed + 53, 10, 9) + 0.56 * _bark_noise(warped, seed + 59, 30, 27)
        moss = wp.smoothstep(-0.19, 0.18, damp + 0.19 * timber + 0.12 * fray + 0.08 * grain)
        moss_tone = wp.smoothstep(-0.32, 0.26, timber + 0.23 * grain + 0.39 * damp)
        lichen = wp.smoothstep(0.31, 0.49, 0.48 * splinters + 0.49 * fray + 0.56 * grit + 0.26 * damp)
        cork = wp.lerp(wp.vec3(0.372, 0.329, 0.289), wp.vec3(0.518, 0.447, 0.369), 0.72 * age + 0.21 * flake)
        growth = wp.lerp(wp.vec3(0.312, 0.331, 0.177), wp.vec3(0.495, 0.504, 0.280), 0.72 * moss_tone + 0.25 * flake)
        base = wp.lerp(cork, growth, moss)
        base += wp.vec3(0.135, 0.121, 0.066) * lichen
        base = wp.lerp(base, wp.vec3(0.55, 0.365, 0.262), 0.34 * peeling * (1.0 - moss))
        crevice = wp.vec3(0.233, 0.213, 0.150)
        height = 0.53 + 0.205 * plate - 0.13 * fissure + 0.040 * peeling
        roughness -= 0.037 * moss
    base += wp.vec3(detail + 0.028 * pigment * flake)
    crevice += wp.vec3(0.13 * granules + 0.072 * fray + 0.086 * grain)
    color = wp.lerp(base, crevice, wp.max(0.60 * fissure, 0.32 * broad_split))
    color += wp.vec3(0.025 * broad_pigment * broad_plate)
    # Longer slopes survive texture filtering on a small or distant sample.
    height += 0.10 * broad_plate - 0.072 * broad_split
    height += 0.027 * splinters + 0.021 * fray + 0.012 * grain + 0.006 * grit + 0.009 * granules
    occlusion = 0.985 - 0.30 * fissure - 0.10 * broad_split - 0.048 * (1.0 - plate)
    return color, height, roughness, occlusion
