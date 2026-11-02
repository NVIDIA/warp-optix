# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warp as wp

from .pathtracing_warp_kernels import PhysicalSkyParams, _eval_physical_sky


@wp.kernel
def generate_physical_sky_lut(
    sky: PhysicalSkyParams, output: wp.array2d(dtype=wp.vec4)
):
    """Evaluate the procedural sky into an equirectangular lookup table."""
    x, y = wp.tid()
    width = output.shape[1]
    height = output.shape[0]
    u = (wp.float32(x) + 0.5) / wp.float32(width)
    v = (wp.float32(y) + 0.5) / wp.float32(height)
    phi = u * (2.0 * wp.pi) - wp.pi
    theta = v * wp.pi
    sin_theta = wp.sin(theta)
    direction = wp.vec3(
        wp.cos(phi) * sin_theta,
        wp.cos(theta),
        wp.sin(phi) * sin_theta,
    )
    radiance = _eval_physical_sky(sky, direction, wp.bool(False), wp.bool(False))
    output[y, x] = wp.vec4(radiance[0], radiance[1], radiance[2], 1.0)
