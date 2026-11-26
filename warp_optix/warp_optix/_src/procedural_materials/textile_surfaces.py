# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Analytical foam and woven surfaces derived from ambientCG material references."""

import warp as wp


@wp.func
def _fiber_noise(uv: wp.vec2, seed: int, across: int, along: int) -> float:
    return wp.pnoise(wp.rand_init(seed), wp.cw_mul(uv, wp.vec2(float(across), float(along))), across, along)


@wp.func
def _foam(uv: wp.vec2, seed: int, resolution: float):
    wobble = _fiber_noise(uv, seed + 23, 18, 19)
    x = 16.0 * wp.pi * uv[0] + 0.08 * wobble
    y = 16.0 * wp.pi * uv[1] - 0.08 * wobble
    crest = wp.cos(x) * wp.cos(y)
    coarse = _fiber_noise(uv, seed + 5, 174, 167)
    cells = _fiber_noise(uv, seed, 382, 407)
    micro = _fiber_noise(uv, seed + 7, 822, 877) * wp.smoothstep(450.0, 1570.0, resolution)
    cavity = wp.smoothstep(0.12, 0.47, cells + 0.48 * coarse + 0.21 * micro)
    cavity *= wp.smoothstep(270.0, 900.0, resolution)
    color = wp.vec3(0.121 - 0.043 * cavity + 0.013 * coarse + 0.007 * micro)
    height = 0.50 + 0.43 * crest - 0.055 * cavity + 0.013 * coarse
    ao = 0.88 + 0.10 * crest - 0.16 * cavity
    return color, height, 0.85 + 0.076 * cavity, ao, 1.0


@wp.func
def _weave(uv: wp.vec2, seed: int, resolution: float, cloth: bool):
    columns, rows = 16, 48
    post_radius, ribbon_radius = 0.190, 0.484
    weft_color = wp.vec3(0.611, 0.466, 0.340)
    warp_color = wp.vec3(0.547, 0.416, 0.248)
    if cloth:
        columns, rows = 32, 32
        post_radius, ribbon_radius = 0.170, 0.468
        weft_color = wp.vec3(0.284, 0.208, 0.180)
        warp_color = wp.vec3(0.369, 0.325, 0.276)

    # The horizontal strand bends under every other vertical strand. One
    # parity expression fixes the crossing for the coverage, color and height.
    wander = 0.077 * _fiber_noise(uv, seed + 1, 11, 27)
    x = uv[0] * float(columns) + wander
    y = uv[1] * float(rows) + wander * 0.56
    post = int(wp.floor(x + 0.5))
    row = int(wp.floor(y + 0.5))
    across_post = wp.abs(x - float(post))
    across_ribbon = wp.abs(y - float(row))
    post_on_top = (post + row) % 2 == 0

    aa_x = wp.min(0.11, 0.55 * float(columns) / resolution)
    aa_y = wp.min(0.11, 0.55 * float(rows) / resolution)
    warp = 1.0 - wp.smoothstep(post_radius - aa_x, post_radius + aa_x, across_post)
    weft = 1.0 - wp.smoothstep(ribbon_radius - aa_y, ribbon_radius + aa_y, across_ribbon)
    opacity = wp.max(warp, weft)
    visible_warp = warp * (1.0 - weft)
    if post_on_top:
        visible_warp = warp

    # Shape is rounded cane for wicker and a flatter ribbon of yarn for cloth.
    post_section = wp.sqrt(wp.max(0.0, 1.0 - (across_post / post_radius) ** 2.0))
    ribbon_section = wp.sqrt(wp.max(0.0, 1.0 - (across_ribbon / ribbon_radius) ** (4.0 if cloth else 2.0)))
    bend = wp.cos(wp.pi * (x + float(row & 1)))
    warp_height = 0.35 + 0.22 * post_section + 0.17 * wp.cos(wp.pi * (y + float(post & 1)))
    weft_height = 0.41 + 0.26 * ribbon_section - 0.12 * bend

    # Long fibers follow their own yarn or reed, including where the two
    # directions meet. The finest band fades at lower bake resolutions.
    grain_weight = wp.smoothstep(448.0, 1500.0, resolution)
    along_weft = _fiber_noise(uv, seed + 11, 43, 713) * grain_weight
    along_warp = _fiber_noise(uv, seed + 13, 431, 51) * grain_weight
    row_seed = wp.rand_init(seed + 5, (row % rows + rows) % rows)
    post_seed = wp.rand_init(seed + 9, (post % columns + columns) % columns)
    weft_tone = wp.randf(row_seed, -0.047, 0.047) + 0.073 * along_weft
    warp_tone = wp.randf(post_seed, -0.037, 0.037) + 0.054 * along_warp
    if cloth:
        weft_tone += 0.088 * _fiber_noise(uv, seed + 17, 41, 271)
    edge_tone = 0.09 * wp.smoothstep(0.62, 0.99, across_ribbon / ribbon_radius)
    if not cloth:
        # The reed edges are gray and less saturated where the cane was cut.
        edge_tone += 0.082 * wp.smoothstep(0.35, 1.0, across_ribbon / ribbon_radius)
    weft_rgb = weft_color + wp.vec3(weft_tone) - edge_tone * wp.vec3(1.0, 0.62, 0.31)
    color = wp.lerp(weft_rgb, warp_color + wp.vec3(warp_tone), visible_warp)
    height = wp.lerp(weft_height + 0.034 * along_weft, warp_height + 0.027 * along_warp, visible_warp)
    contact = weft * warp * (1.0 - post_section)
    occlusion = 0.76 + 0.24 * opacity - 0.26 * contact - 0.12 * (1.0 - ribbon_section)
    roughness = 0.54 + 0.13 * visible_warp + 0.1 * (1.0 - ribbon_section) + 0.045 * along_weft
    return color, wp.lerp(0.1, height, opacity), roughness, occlusion, opacity


@wp.func
def textile_surface(uv: wp.vec2, seed: int, resolution: float, style: int):
    uv -= wp.vec2(wp.floor(uv[0]), wp.floor(uv[1]))
    if style == 0:
        return _foam(uv, seed, resolution)
    return _weave(uv, seed, resolution, style == 2)
