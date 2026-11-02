# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION
# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION
# SPDX-License-Identifier: Apache-2.0
#
# Modified Python/Warp translation of func_common.h from the intermediate OptiX port.
# Vector, basis, and sampling utilities derive from nvpro_core:
# https://github.com/nvpro-samples/nvpro_core/blob/bc19d6ac3ef62938d0ea0e099735878457ce1b6e/nvvkhl/shaders/func.h
# Original notices: licenses/vk_denoise_dlssrr-NOTICE.txt and
# licenses/nvpro_core-NOTICE.txt.

#
# Utility functions for the Python/Warp renderer.

from __future__ import annotations

import warp as wp

# Matrix types stored as flat vectors matching C++ load_mat4_from_array layout.
Mat16f = wp.types.vector(length=16, dtype=wp.float32)
Vec6f = wp.types.vector(length=6, dtype=wp.float32)

M_PI = 3.14159265358979323846
M_TWO_PI = 6.28318530717958648
M_PI_2 = 1.57079632679489661923
M_PI_4 = 0.785398163397448309616
M_1_OVER_PI = 0.318309886183790671538


# -----------------------------------------------------------------------------
# Matrix-vector multiplication
# -----------------------------------------------------------------------------


@wp.func
def mul_cm_4x4(m: Mat16f, v: wp.vec4) -> wp.vec4:
    """Column-major 4x4 multiply: treats stored rows as columns (GLSL convention)."""
    return wp.vec4(
        m[0] * v[0] + m[4] * v[1] + m[8] * v[2] + m[12] * v[3],
        m[1] * v[0] + m[5] * v[1] + m[9] * v[2] + m[13] * v[3],
        m[2] * v[0] + m[6] * v[1] + m[10] * v[2] + m[14] * v[3],
        m[3] * v[0] + m[7] * v[1] + m[11] * v[2] + m[15] * v[3],
    )


@wp.func
def mul_cm_3x3(m: Mat16f, v: wp.vec3) -> wp.vec3:
    """Column-major 3x3 multiply using upper-left 3x3 of a flat 4x4."""
    return wp.vec3(
        m[0] * v[0] + m[4] * v[1] + m[8] * v[2],
        m[1] * v[0] + m[5] * v[1] + m[9] * v[2],
        m[2] * v[0] + m[6] * v[1] + m[10] * v[2],
    )


@wp.func
def mul_rm_4x4(m: Mat16f, v: wp.vec4) -> wp.vec4:
    """Row-major 4x4 multiply: result_i = dot(row_i, v)."""
    return wp.vec4(
        m[0] * v[0] + m[1] * v[1] + m[2] * v[2] + m[3] * v[3],
        m[4] * v[0] + m[5] * v[1] + m[6] * v[2] + m[7] * v[3],
        m[8] * v[0] + m[9] * v[1] + m[10] * v[2] + m[11] * v[3],
        m[12] * v[0] + m[13] * v[1] + m[14] * v[2] + m[15] * v[3],
    )


@wp.func
def mul_rm_3x3(m: Mat16f, v: wp.vec3) -> wp.vec3:
    """Row-major 3x3 multiply using upper-left 3x3 of a flat 4x4."""
    return wp.vec3(
        m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
        m[4] * v[0] + m[5] * v[1] + m[6] * v[2],
        m[8] * v[0] + m[9] * v[1] + m[10] * v[2],
    )


# -----------------------------------------------------------------------------
# UV transform (float2x3 stored as Vec6f: row0=[0..2], row1=[3..5])
# -----------------------------------------------------------------------------


@wp.func
def transform_uv(m: Vec6f, uv: wp.vec2) -> wp.vec2:
    return wp.vec2(
        m[0] * uv[0] + m[1] * uv[1] + m[2],
        m[3] * uv[0] + m[4] * uv[1] + m[5],
    )


# -----------------------------------------------------------------------------
# Scalar helpers
# -----------------------------------------------------------------------------


@wp.func
def square(x: wp.float32) -> wp.float32:
    return x * x


@wp.func
def saturate_f(x: wp.float32) -> wp.float32:
    return wp.min(wp.max(x, 0.0), 1.0)


@wp.func
def saturate_v3(v: wp.vec3) -> wp.vec3:
    return wp.vec3(saturate_f(v[0]), saturate_f(v[1]), saturate_f(v[2]))


@wp.func
def luminance(color: wp.vec3) -> wp.float32:
    return color[0] * 0.2126 + color[1] * 0.7152 + color[2] * 0.0722


@wp.func
def clamped_dot(x: wp.vec3, y: wp.vec3) -> wp.float32:
    return wp.max(wp.min(wp.dot(x, y), 1.0), 0.0)


@wp.func
def reflect(i: wp.vec3, n: wp.vec3) -> wp.vec3:
    return i - 2.0 * wp.dot(n, i) * n


@wp.func
def sign_f(x: wp.float32) -> wp.float32:
    if x > 0.0:
        return 1.0
    elif x < 0.0:
        return -1.0
    return 0.0


@wp.func
def any_isnan_v3(v: wp.vec3) -> bool:
    return wp.isnan(v[0]) or wp.isnan(v[1]) or wp.isnan(v[2])


@wp.func
def any_isinf_v3(v: wp.vec3) -> bool:
    return wp.isinf(v[0]) or wp.isinf(v[1]) or wp.isinf(v[2])


# -----------------------------------------------------------------------------
# Mix/lerp
# -----------------------------------------------------------------------------


@wp.func
def mix_f(a: wp.float32, b: wp.float32, t: wp.float32) -> wp.float32:
    return a + (b - a) * t


@wp.func
def mix_v3(a: wp.vec3, b: wp.vec3, t: wp.float32) -> wp.vec3:
    return a + (b - a) * t


@wp.func
def mix_v3_v3(a: wp.vec3, b: wp.vec3, t: wp.vec3) -> wp.vec3:
    return wp.vec3(
        a[0] + (b[0] - a[0]) * t[0],
        a[1] + (b[1] - a[1]) * t[1],
        a[2] + (b[2] - a[2]) * t[2],
    )


# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------


@wp.func
def orthonormal_basis(normal: wp.vec3) -> wp.vec2:
    """Returns (tangent, bitangent) packed; use orthonormal_basis_tbn instead."""
    # Placeholder - see orthonormal_basis_tbn for full version
    return wp.vec2(0.0, 0.0)


@wp.func
def orthonormal_basis_tangent(normal: wp.vec3) -> wp.vec3:
    if normal[2] < -0.99998796:
        return wp.vec3(0.0, -1.0, 0.0)
    a = 1.0 / (1.0 + normal[2])
    b = -normal[0] * normal[1] * a
    return wp.vec3(1.0 - normal[0] * normal[0] * a, b, -normal[0])


@wp.func
def orthonormal_basis_bitangent(normal: wp.vec3) -> wp.vec3:
    if normal[2] < -0.99998796:
        return wp.vec3(-1.0, 0.0, 0.0)
    a = 1.0 / (1.0 + normal[2])
    b = -normal[0] * normal[1] * a
    return wp.vec3(b, 1.0 - normal[1] * normal[1] * a, -normal[1])


@wp.func
def get_spherical_uv(v: wp.vec3) -> wp.vec2:
    gamma = wp.asin(-v[1])
    theta = wp.atan2(v[2], v[0])
    return wp.vec2(theta * M_1_OVER_PI * 0.5 + 0.5, gamma * M_1_OVER_PI + 0.5)


@wp.func
def mix_bary_v2(a: wp.vec2, b: wp.vec2, c: wp.vec2, bary: wp.vec3) -> wp.vec2:
    return a * bary[0] + b * bary[1] + c * bary[2]


@wp.func
def mix_bary_v3(a: wp.vec3, b: wp.vec3, c: wp.vec3, bary: wp.vec3) -> wp.vec3:
    return a * bary[0] + b * bary[1] + c * bary[2]


@wp.func
def mix_bary_v4(a: wp.vec4, b: wp.vec4, c: wp.vec4, bary: wp.vec3) -> wp.vec4:
    return a * bary[0] + b * bary[1] + c * bary[2]


@wp.func
def cosine_sample_hemisphere(r1: wp.float32, r2: wp.float32) -> wp.vec3:
    r = wp.sqrt(r1)
    phi = M_TWO_PI * r2
    return wp.vec3(r * wp.cos(phi), r * wp.sin(phi), wp.sqrt(1.0 - r1))


@wp.func
def power_heuristic(a: wp.float32, b: wp.float32) -> wp.float32:
    t = a * a
    return t / (b * b + t)


@wp.func
def rotate_vec(v: wp.vec3, k: wp.vec3, theta: wp.float32) -> wp.vec3:
    cos_theta = wp.cos(theta)
    sin_theta = wp.sin(theta)
    return v * cos_theta + wp.cross(k, v) * sin_theta + k * wp.dot(k, v) * (1.0 - cos_theta)
