# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tileable pressed chipboard with layered oriented wood strands."""

import warp as wp

from .sampling import wrap_uv, wrapped_noise


@wp.func
def _strands(uv: wp.vec2, seed: int, count: int, resolution: float, edge_noise: float):
    grid = uv * float(count)
    x = int(wp.floor(grid[0]))
    y = int(wp.floor(grid[1]))
    local = grid - wp.vec2(float(x), float(y))
    aa = 0.45 * float(count) / wp.max(128.0, resolution)
    surface = float(0.0)
    thickness = float(0.0)
    grain_start = wp.vec2(0.0)
    pigment = float(0.0)
    grain_id = int(0)
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            i = (x + dx + count) % count
            j = (y + dy + count) % count
            strand_id = i + count * j
            state = wp.rand_init(seed, strand_id)
            center = wp.vec2(float(dx) + wp.randf(state, 0.09, 0.91), float(dy) + wp.randf(state, 0.09, 0.91))
            delta = local - center
            if wp.dot(delta, delta) < 1.1:
                angle = wp.randf(state, 0.0, wp.PI)
                along = wp.vec2(wp.cos(angle), wp.sin(angle))
                across = wp.vec2(-along[1], along[0])
                u = wp.dot(along, delta)
                v = wp.dot(across, delta)
                half_length = wp.randf(state, 0.48, 0.91)
                half_width = wp.randf(state, 0.145, 0.34)
                taper = wp.randf(state, -0.29, 0.37)
                step = u / half_length
                span = half_width * (1.0 + taper * step - 0.19 * wp.pow(wp.abs(step), 4.0))
                # Side grain runs through the strand, while snapped ends
                # fracture to different lengths across the fibres.
                side = span - wp.abs(v + 0.082 * step * step + 0.073 * edge_noise)
                splinter = wp.noise(wp.rand_init(seed + 71, strand_id), 51.0 * v)
                cut = half_length - wp.abs(u + 0.047 * edge_noise + 0.157 * splinter) - 0.19 * wp.abs(v)
                edge = wp.min(side, cut)
                mask = wp.smoothstep(-aa, 0.028 + aa, edge)
                depth = wp.randf(state, 0.48, 0.91) + 0.12 * step + 0.12 * v * v
                height = mask * depth
                if height > surface:
                    surface = height
                    thickness = mask
                    grain_start = wp.vec2(u, v)
                    pigment = wp.randf(state, -1.0, 1.0)
                    grain_id = strand_id
    return surface, thickness, grain_start, pigment, grain_id


@wp.func
def _wood_chip(grain: wp.vec2, seed: int, grain_id: int, count: int, resolution: float, pigment: float):
    state = wp.rand_init(seed + 97, grain_id)
    # Small cross-fibre changes break the nearly parallel fibres. Coarser
    # grain and individual splinters use anisotropic coordinates.
    bend = wp.noise(state, wp.vec2(3.7 * grain[0], 21.0 * grain[1]))
    fibre = wp.noise(state, wp.vec2(3.6 * grain[0], 62.0 * (grain[1] + 0.013 * bend)))
    ray = wp.noise(state, wp.vec2(1.27 * grain[0], 14.0 * (grain[1] + 0.013 * bend)))
    phase = 111.0 * grain[1] + 0.55 * bend
    pore = 1.0 - wp.smoothstep(0.04, 0.22, wp.abs(phase - wp.floor(phase + 0.5)))
    detail = wp.smoothstep(7.0 * float(count), 30.0 * float(count), resolution)
    long_split = wp.smoothstep(0.14, 0.48, fibre + 0.37 * bend)
    pitch = wp.smoothstep(-0.83, 0.88, pigment + 0.11 * bend)
    wood = wp.lerp(wp.vec3(0.501, 0.357, 0.221), wp.vec3(0.783, 0.623, 0.381), pitch)
    tone = 0.095 * bend + 0.14 * ray + detail * (0.19 * fibre - 0.069 * pore - 0.075 * long_split)
    relief = 0.024 * bend + 0.073 * ray + detail * (0.083 * fibre - 0.042 * pore - 0.023 * long_split)
    return wood + wp.vec3(tone, 0.88 * tone, 0.58 * tone), relief, long_split


@wp.func
def chipboard(uv: wp.vec2, seed: int, resolution: float):
    """Return sRGB color, height, roughness, occlusion, metalness, and opacity."""
    uv = wrap_uv(uv)
    grit = wrapped_noise(uv, seed + 13, 419, 419) * wp.smoothstep(360.0, 1500.0, resolution)
    torn = wrapped_noise(uv, seed + 19, 189, 189) * wp.smoothstep(150.0, 560.0, resolution)
    sawdust = wrapped_noise(uv, seed + 23, 76, 76)
    seam = wp.smoothstep(0.04, 0.39, 0.71 * sawdust + 0.48 * torn + 0.42 * grit)
    base = wp.vec3(0.677, 0.509, 0.308) + wp.vec3(0.085 * torn + 0.078 * grit - 0.16 * seam)
    small_h, small_face, small_uv, small_tone, small_id = _strands(uv, seed + 37, 78, resolution, torn)
    medium_h, medium_face, medium_uv, medium_tone, medium_id = _strands(uv, seed + 41, 53, resolution, torn)
    large_h, large_face, large_uv, large_tone, large_id = _strands(uv, seed + 47, 32, resolution, torn)

    # Compression embeds small strands into the backing layer. The larger
    # chips cross the smaller ones and present their own fibre direction.
    height = 0.423 + 0.037 * sawdust + 0.018 * torn + 0.016 * grit - 0.043 * seam
    color = base
    split = float(0.0)
    dominant = float(0.0)
    if small_h > 0.0:
        small_color, small_relief, small_split = _wood_chip(small_uv, seed + 37, small_id, 78, resolution, small_tone)
        height += small_face * (0.112 * small_h + small_relief)
        color = wp.lerp(color, small_color, small_face)
        split = small_split * small_face
        dominant = small_face * small_h
    if medium_h > 0.0:
        medium_color, medium_relief, medium_split = _wood_chip(
            medium_uv, seed + 41, medium_id, 53, resolution, medium_tone
        )
        overlay = medium_face * wp.smoothstep(-0.3, 0.11, medium_h - 0.74 * dominant)
        height = wp.lerp(height, 0.447 + 0.233 * medium_h + medium_relief, overlay)
        color = wp.lerp(color, medium_color, overlay)
        split = wp.lerp(split, medium_split, overlay)
        dominant = wp.lerp(dominant, medium_h, overlay)
    if large_h > 0.0:
        large_color, large_relief, large_split = _wood_chip(large_uv, seed + 47, large_id, 32, resolution, large_tone)
        overlay = large_face * wp.smoothstep(-0.19, 0.16, large_h - 0.74 * dominant)
        height = wp.lerp(height, 0.455 + 0.265 * large_h + large_relief, overlay)
        color = wp.lerp(color, large_color, overlay)
        split = wp.lerp(split, large_split, overlay)
        dominant = wp.lerp(dominant, large_h, overlay)
    color += wp.vec3(0.033 * grit + 0.024 * torn)
    roughness = 0.663 + 0.079 * seam - 0.082 * dominant + 0.088 * split + 0.066 * torn
    occlusion = 0.369 + 0.102 * dominant - 0.071 * seam + 0.045 * sawdust
    return color, height, roughness, occlusion, 0.0, 1.0
