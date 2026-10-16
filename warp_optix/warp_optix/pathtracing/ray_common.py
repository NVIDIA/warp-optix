# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION
# SPDX-License-Identifier: Apache-2.0
#
# Modified Python/Warp translation of ray_common.h (ray payload structures and utilities).
# Adapted from NVIDIA vk_denoise_dlssrr; upstream reference:
# https://github.com/nvpro-samples/vk_denoise_dlssrr/blob/f8113a84fa952278bb1286e7899f81b7afe8f41e/dlss_rr/shaders/ray_common.glsl
# See THIRD_PARTY_NOTICES.txt and licenses/vk_denoise_dlssrr-NOTICE.txt.

from __future__ import annotations

import warp as wp

MISS_DISTANCE = float(65504.0)


@wp.struct
class RayPayload:
    """Geometry-only payload from hit shader; material evaluation happens in raygen.

    Full float precision matching the C++ RayPayload (19 payload registers).
    """

    hitT: wp.float32
    normal_x: wp.float32
    normal_y: wp.float32
    normal_z: wp.float32
    tangent_x: wp.float32
    tangent_y: wp.float32
    tangent_z: wp.float32
    uv_x: wp.float32
    uv_y: wp.float32
    materialId: wp.uint32
    bitangentSign: wp.float32
    instanceId: wp.int32
    meshId: wp.uint32
    primitiveId: wp.uint32
    bary_x: wp.float32
    bary_y: wp.float32
    bary_z: wp.float32
    uv1_x: wp.float32
    uv1_y: wp.float32


@wp.func
def init_ray_payload() -> RayPayload:
    p = RayPayload()
    p.hitT = MISS_DISTANCE
    p.normal_x = 0.0
    p.normal_y = 0.0
    p.normal_z = 1.0
    p.tangent_x = 1.0
    p.tangent_y = 0.0
    p.tangent_z = 0.0
    p.uv_x = 0.0
    p.uv_y = 0.0
    p.materialId = wp.uint32(0)
    p.bitangentSign = 1.0
    p.instanceId = -1
    p.meshId = wp.uint32(0)
    p.primitiveId = wp.uint32(0)
    p.bary_x = 0.0
    p.bary_y = 0.0
    p.bary_z = 0.0
    p.uv1_x = 0.0
    p.uv1_y = 0.0
    return p


@wp.func
def payload_get_normal(p: RayPayload) -> wp.vec3:
    return wp.vec3(p.normal_x, p.normal_y, p.normal_z)


@wp.func
def payload_get_tangent(p: RayPayload) -> wp.vec3:
    return wp.vec3(p.tangent_x, p.tangent_y, p.tangent_z)


@wp.func
def payload_get_uv(p: RayPayload) -> wp.vec2:
    return wp.vec2(p.uv_x, p.uv_y)


@wp.func
def payload_get_uv1(p: RayPayload) -> wp.vec2:
    return wp.vec2(p.uv1_x, p.uv1_y)


@wp.func
def payload_get_barycentrics(p: RayPayload) -> wp.vec3:
    return wp.vec3(p.bary_x, p.bary_y, p.bary_z)


# -----------------------------------------------------------------------------
# Mirror matrix helpers (3x3 stored as wp.mat33, row-major)
# -----------------------------------------------------------------------------


@wp.func
def build_mirror_matrix(normal: wp.vec3) -> wp.mat33:
    """M = I - 2 * n * n^T"""
    nx = normal[0]
    ny = normal[1]
    nz = normal[2]
    return wp.mat33(
        1.0 - 2.0 * nx * nx,
        -2.0 * nx * ny,
        -2.0 * nx * nz,
        -2.0 * ny * nx,
        1.0 - 2.0 * ny * ny,
        -2.0 * ny * nz,
        -2.0 * nz * nx,
        -2.0 * nz * ny,
        1.0 - 2.0 * nz * nz,
    )


@wp.func
def apply_matrix_3x3(m: wp.mat33, v: wp.vec3) -> wp.vec3:
    """Row-major mat33 * vec3."""
    return wp.vec3(
        m[0, 0] * v[0] + m[0, 1] * v[1] + m[0, 2] * v[2],
        m[1, 0] * v[0] + m[1, 1] * v[1] + m[1, 2] * v[2],
        m[2, 0] * v[0] + m[2, 1] * v[1] + m[2, 2] * v[2],
    )


@wp.func
def reinhard_max(color: wp.vec3) -> wp.vec3:
    lum = wp.max(1.0e-7, wp.max(wp.max(color[0], color[1]), color[2]))
    reinhard = lum / (lum + 1.0)
    return color * (reinhard / lum)
