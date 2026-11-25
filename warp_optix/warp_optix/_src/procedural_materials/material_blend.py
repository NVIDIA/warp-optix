# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Composite a continuous material mask into glTF texture maps."""

import warp as wp


@wp.func
def _unpack(pixel: wp.vec4ub) -> wp.vec4:
    return wp.vec4(float(pixel[0]), float(pixel[1]), float(pixel[2]), float(pixel[3])) * (1.0 / 255.0)


@wp.func
def _pack(pixel: wp.vec4) -> wp.vec4ub:
    value = pixel * 255.0 + wp.vec4(0.5)
    return wp.vec4ub(wp.uint8(value[0]), wp.uint8(value[1]), wp.uint8(value[2]), wp.uint8(value[3]))


@wp.kernel(enable_backward=False)
def blend_materials(
    base: wp.array3d(dtype=wp.vec4ub),
    overlay: wp.array3d(dtype=wp.vec4ub),
    mask: wp.array2d(dtype=float),
    base_height: wp.array2d(dtype=wp.vec3),
    overlay_height: wp.array2d(dtype=wp.vec3),
    mask_seed: int,
    frequency: int,
    threshold: float,
    softness: float,
    height_blend: float,
    output: wp.array3d(dtype=wp.vec4ub),
):
    row, col = wp.tid()
    weight = float(0.0)
    if mask.shape[0] > 0:
        weight = wp.clamp(mask[row, col], 0.0, 1.0)
    else:
        uv = wp.vec2(
            (float(col) + 0.5) / float(output.shape[2]),
            (float(row) + 0.5) / float(output.shape[1]),
        )
        seed = wp.rand_init(mask_seed)
        low = wp.pnoise(seed, uv * float(frequency), frequency, frequency)
        fine = wp.pnoise(seed, uv * float(3 * frequency), 3 * frequency, 3 * frequency)
        value = 0.5 + 0.36 * low + 0.14 * fine
        weight = wp.smoothstep(threshold - 0.5 * softness, threshold + 0.5 * softness, value)
    if height_blend > 0.0:
        depth = overlay_height[row, col][0] - base_height[row, col][0]
        weight = wp.clamp(weight + 4.0 * weight * (1.0 - weight) * height_blend * depth, 0.0, 1.0)

    if weight <= 0.0 or weight >= 1.0:
        for image in range(3):
            if weight <= 0.0:
                output[image, row, col] = base[image, row, col]
            else:
                output[image, row, col] = overlay[image, row, col]
        return

    base_color = _unpack(base[0, row, col])
    overlay_color = _unpack(overlay[0, row, col])
    output[0, row, col] = _pack(wp.lerp(base_color, overlay_color, weight))

    packed_base_normal = _unpack(base[1, row, col])
    packed_overlay_normal = _unpack(overlay[1, row, col])
    n0 = 2.0 * wp.vec3(packed_base_normal[0], packed_base_normal[1], packed_base_normal[2]) - wp.vec3(1.0)
    n1 = 2.0 * wp.vec3(packed_overlay_normal[0], packed_overlay_normal[1], packed_overlay_normal[2]) - wp.vec3(1.0)
    n = wp.normalize(wp.lerp(n0, n1, weight))
    output[1, row, col] = _pack(wp.vec4(0.5 * n[0] + 0.5, 0.5 * n[1] + 0.5, 0.5 * n[2] + 0.5, 1.0))

    a = _unpack(base[2, row, col])
    b = _unpack(overlay[2, row, col])
    orm = wp.lerp(a, b, weight)
    orm[1] = wp.sqrt(wp.lerp(a[1] * a[1], b[1] * b[1], weight))
    output[2, row, col] = _pack(orm)
