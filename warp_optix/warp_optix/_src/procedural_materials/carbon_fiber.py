# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tileable 2/2 twill carbon fiber inspired by ambientCG Fabric004."""

import warp as wp


@wp.func
def _fiber_noise(uv: wp.vec2, seed: int, columns: int, rows: int):
    wrapped = uv - wp.vec2(wp.floor(uv[0]), wp.floor(uv[1]))
    grid = wp.vec2(wrapped[0] * float(columns), wrapped[1] * float(rows))
    return wp.pnoise(wp.rand_init(seed), grid, columns, rows)


@wp.func
def carbon_fiber(uv: wp.vec2, seed: int, resolution: float):
    """Return sRGB color, height, roughness, occlusion, metalness, and opacity."""
    # The reference tile contains five repeats of a four-yarn twill, with
    # each yarn crossing over two neighbours and then underneath two.
    uv = uv - wp.vec2(wp.floor(uv[0]), wp.floor(uv[1]))
    count = 20
    x_wander = _fiber_noise(uv, seed + 11, 11, 3)
    y_wander = _fiber_noise(uv, seed + 23, 3, 11)
    warp = wp.vec2(x_wander, y_wander) * 0.00086
    grid = (uv + warp) * float(count)
    col = int(wp.floor(grid[0]))
    row = int(wp.floor(grid[1]))
    phase = (col + row + 3) & 3
    wrapped_col = (col % count + count) % count
    wrapped_row = (row % count + count) % count
    across = grid[1] - float(row)
    along = 0.5 * (float(phase & 1) + grid[0] - float(col))
    width_id = wrapped_row
    along_u = uv[0]
    horizontal = phase < 2
    if not horizontal:
        across = grid[0] - float(col)
        along = 0.5 * (float(phase & 1) + grid[1] - float(row))
        width_id = wrapped_col + count
        along_u = uv[1]

    yarn = wp.rand_init(seed + 37, width_id)
    loft = wp.randf(yarn, -0.014, 0.014)
    dye = wp.randf(yarn, -0.008, 0.008)
    edge = wp.min(across, 1.0 - across)
    end = wp.min(along, 1.0 - along)
    aa = 0.35 * float(count) / wp.max(resolution, 64.0)
    side_bevel = wp.smoothstep(0.0, 0.038 + aa, edge)
    end_bevel = wp.smoothstep(0.0, 0.070 + aa, end)
    contact = side_bevel * end_bevel
    shoulder = 1.0 - wp.pow(2.0 * wp.abs(across - 0.5), 4.0)
    arch = wp.smoothstep(0.0, 0.29, end)

    # Individual parallel filaments have a common tangent and slow changes
    # in width. Fade them out below their screen sampling frequency.
    curl = 0.23 * wp.sin(2.0 * wp.PI * (5.0 * along_u + float(width_id) * 0.173))
    fine = 38.0 * across + curl
    filament = 0.5 + 0.5 * wp.cos(2.0 * wp.PI * fine)
    filament_id = int(wp.floor(fine))
    strand = wp.rand_init(seed + 53, 41 * width_id + filament_id + 1)
    strand_tone = wp.randf(strand, -0.014, 0.014)
    bundles = 0.5 + 0.5 * wp.cos(2.0 * wp.pi * (12.0 * across + 0.6 * curl))
    visible_bundle = wp.smoothstep(300.0, 810.0, resolution) * contact * (bundles - 0.5)
    micro_filter = wp.smoothstep(720.0, 1900.0, resolution)
    strand_mix = micro_filter * contact
    waviness = _fiber_noise(wp.vec2(across, along_u), seed + 67, 23, 62)
    strand_detail = strand_mix * ((filament - 0.5) * 0.013 + strand_tone + 0.014 * waviness)

    height = 0.39 + (0.072 + loft) * contact * arch + 0.028 * contact * shoulder
    height += 0.023 * visible_bundle + 0.18 * strand_detail
    tone = 0.143 + contact * (0.036 * arch + 0.015 * shoulder + dye) + 0.053 * visible_bundle + strand_detail
    yarn_polish = 0.150
    if not horizontal:
        yarn_polish = 0.303
    roughness = wp.lerp(0.354, yarn_polish + 0.048 * (1.0 - arch), contact)
    roughness -= 0.045 * visible_bundle
    roughness += strand_mix * (0.030 * (1.0 - filament) + 2.1 * strand_tone)
    occlusion = 0.87 + 0.13 * contact * wp.smoothstep(0.12, 0.83, arch)
    color = wp.vec3(tone * 1.016, tone, tone * 1.074)
    return color, height, roughness, occlusion, 1.0, 1.0
