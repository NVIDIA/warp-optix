# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Periodic bark surfaces with fissures and overlapping scales."""

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
def _bark_plates(uv: wp.vec2, seed: int, columns: int, rows: int):
    grid = wp.vec2(float(columns) * uv[0], float(rows) * uv[1])
    origin_x = int(wp.floor(grid[0]))
    origin_y = int(wp.floor(grid[1]))
    local = grid - wp.vec2(float(origin_x), float(origin_y))
    closest = float(10.0)
    second = float(10.0)
    plate_id = int(0)
    position = wp.vec2(0.0)
    for y in range(-1, 2):
        for x in range(-1, 2):
            column = ((origin_x + x) % columns + columns) % columns
            row = ((origin_y + y) % rows + rows) % rows
            cell = column + row * columns
            state = wp.rand_init(seed, cell)
            center = wp.vec2(float(x) + wp.randf(state, 0.06, 0.94), float(y) + wp.randf(state, 0.06, 0.94))
            delta = local - center
            distance = wp.dot(delta, delta)
            if distance < closest:
                second = closest
                closest = distance
                plate_id = cell
                position = delta
            elif distance < second:
                second = distance
    return wp.sqrt(second) - wp.sqrt(closest), plate_id, position


@wp.func
def bark_surface(uv: wp.vec2, seed: int, resolution: float, style: int):
    """Return sRGB color, height, roughness, and occlusion from one bark profile."""
    uv = uv - wp.vec2(wp.floor(uv[0]), wp.floor(uv[1]))
    columns = 48
    rows = 12
    if style == 1:
        columns = 31
        rows = 13

    # Bow the long fissures and chip their edges before evaluating the plates.
    wander = _bark_noise(uv, seed + 13, 9, 5)
    waviness = _bark_noise(uv, seed + 17, 23, 15)
    chips = _bark_noise(uv, seed + 19, 109, 143) * wp.smoothstep(128.0, 512.0, resolution)
    bent = uv + wp.vec2(0.018 * wander + 0.011 * waviness + 0.0027 * chips, 0.021 * waviness + 0.004 * chips)
    edge, plate_id, position = _bark_plates(bent, seed, columns, rows)
    state = wp.rand_init(seed + 29, plate_id)
    tone = wp.randf(state, -0.071, 0.072)
    width = wp.randf(state, 0.025, 0.085)
    age = wp.randf(state)
    aa = 0.45 * float(columns) / resolution
    # Several longitudinal fissures continue across many scales. The smaller
    # plate boundaries open only where a flake has broken away.
    valley_wave = _bark_noise(uv, seed + 71, 15, 9)
    channel_uv = bent[0] + 0.019 * valley_wave + 0.004 * _bark_noise(uv, seed + 79, 54, 37)
    channel = channel_uv * float(columns)
    channel_width = 0.068 + 0.13 * waviness + 0.082 * chips
    main_face = wp.smoothstep(channel_width - aa, channel_width + 0.095 + aa, wp.abs(channel - wp.floor(channel + 0.5)))
    fracture = wp.smoothstep(width - aa, width + 0.10 + aa, edge + 0.08 * chips)
    broken = wp.smoothstep(0.22, 0.62, age + 0.75 * chips + 0.55 * waviness)
    plate_face = 1.0 - 0.53 * broken * (1.0 - fracture)
    face = (0.22 + 0.78 * main_face) * plate_face
    crown = 0.52 * main_face + 0.48 * wp.smoothstep(width, 0.41 + aa, edge)
    if style == 1:
        face = (0.41 + 0.59 * main_face) * plate_face
    flakes = _bark_noise(bent, seed + 31, 89, 91) * wp.smoothstep(100.0, 360.0, resolution)
    fibers = _bark_noise(bent, seed + 37, 274, 111) * wp.smoothstep(216.0, 800.0, resolution)
    pores = _bark_noise(bent, seed + 43, 520, 361) * wp.smoothstep(416.0, 1850.0, resolution)
    peel = wp.smoothstep(0.12, 0.33, flakes + 0.64 * fibers + 0.42 * pores + 0.19 * chips)
    digest = _bark_noise(bent, seed + 101, 133, 57) + 0.62 * fibers + 0.32 * pores
    fragment = wp.smoothstep(-0.15, 0.19, digest + 0.34 * flakes)
    layer = wp.smoothstep(0.022, 0.11, edge - 0.09 * flakes) * (1.0 - wp.smoothstep(0.13, 0.29, edge))
    roughness = 0.67 - 0.12 * face + 0.10 * flakes + 0.11 * peel
    crevice = wp.vec3(0.29, 0.253, 0.20)
    plate = wp.vec3(0.549, 0.470, 0.361) + wp.vec3(0.63 * tone + 0.138 * flakes + 0.115 * fibers + 0.10 * pores)
    # Orange flakes of fresh cork along the shoulders of weathered plates.
    exposed = layer * wp.smoothstep(0.38, 0.75, age + 0.44 * flakes + 0.23 * chips)
    plate = wp.lerp(
        plate, wp.vec3(0.61, 0.393, 0.236) + wp.vec3(0.15 * flakes + 0.07 * pores),
        wp.max(0.67 * exposed, 0.46 * fragment * peel),
    )
    plate = wp.lerp(plate, wp.vec3(0.375, 0.349, 0.299) + wp.vec3(0.14 * fibers), 0.48 * (1.0 - fragment))
    height = 0.54 + 0.070 * face + 0.038 * crown + 0.019 * position[1] * face
    height += (0.033 * flakes + 0.025 * fibers + 0.017 * pores - 0.021 * peel) * (0.40 + 0.60 * face)
    if style == 1:
        moss_noise = _bark_noise(uv, seed + 53, 12, 11) + 0.33 * _bark_noise(bent, seed + 59, 44, 39)
        moss = wp.smoothstep(-0.12, 0.24, moss_noise + 0.13 * flakes + 0.06 * fibers)
        woody = wp.vec3(0.50, 0.411, 0.338) + wp.vec3(0.68 * tone + 0.141 * flakes + 0.102 * fibers + 0.066 * pores)
        green = wp.vec3(0.449, 0.455, 0.230) + wp.vec3(0.48 * tone + 0.163 * flakes + 0.103 * fibers + 0.096 * pores)
        plate = wp.lerp(woody, green, moss * (0.68 + 0.32 * fragment))
        lichen = wp.smoothstep(0.38, 0.64, digest + 0.35 * moss_noise)
        plate += wp.vec3(0.13, 0.105, 0.055) * lichen
        crevice = wp.vec3(0.195, 0.164, 0.108)
        roughness -= 0.045 * moss
        height = 0.49 + 0.16 * face + 0.087 * crown + 0.062 * position[1] * face
        height += (0.061 * flakes + 0.027 * fibers + 0.015 * pores - 0.041 * peel) * (0.4 + 0.60 * face)
    bark_grit = wp.smoothstep(0.08, 0.32, pores + 0.32 * fibers)
    crevice += wp.vec3(0.076 * bark_grit + 0.11 * chips + 0.035 * flakes)
    color = wp.lerp(crevice, plate, 0.23 + 0.77 * face)
    occlusion = 0.66 + 0.27 * face + 0.06 * crown - 0.06 * peel
    return color, height, roughness, occlusion
