# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ray queries for raygen programs, mirroring ``wp.mesh_query_ray``.

Both queries traverse without invoking closest-hit or miss programs and read
the result from the OptiX hit object, so a pipeline that only uses them needs
no hit programs for built-in geometry. After :func:`accel_query_ray`, further
hit data (intersection attributes, transforms) is available through the
``wp.optix_hit_object_*`` builtins until the next traversal.
"""

from __future__ import annotations

import warp as wp

from warp_optix.enums import PrimitiveType, RayFlags


@wp.struct
class AccelQueryRay:
    """Result of :func:`accel_query_ray`.

    Attributes:
        result: Whether a hit was found in ``[min_t, max_t]``. On a miss, no other field is defined.
        t: Distance of the closest hit along the ray.
        u: Triangles: barycentric u. Ribbons: the parameter along the curve. Round curves: the curve
            parameter. Custom primitives: the bits of the first reported attribute, as float. Other
            primitives (e.g. spheres): 0.
        v: Triangles: barycentric v. Ribbons: the parameter across the curve. Custom primitives: the bits
            of the second reported attribute, as float. Round curves and other primitives: 0.
        instance_id: User id of the hit instance (``0xFFFFFFFF`` when querying a geometry acceleration
            structure directly, which has no instance).
        primitive_index: Index of the hit primitive within its build input.
    """

    result: wp.bool
    t: float
    u: float
    v: float
    instance_id: wp.uint32
    primitive_index: wp.uint32


@wp.func
def accel_query_ray(
    id: wp.uint64,
    start: wp.vec3,
    dir: wp.vec3,
    max_t: float,
    min_t: float = 0.0,
    mask: wp.Int = 0xFF,
    flags: wp.Int = 0,
) -> AccelQueryRay:
    """Return the closest intersection of a ray with acceleration structure ``id`` in ``[min_t, max_t]``.

    ``mask`` is the 8-bit visibility mask; ``flags`` takes :class:`RayFlags`.
    """
    wp.optix_traverse(id, start, dir, max_t, min_t, 0.0, mask, flags)
    query = AccelQueryRay()
    query.result = wp.optix_hit_object_is_hit()
    if query.result:
        query.t = wp.optix_hit_object_get_ray_tmax()
        primitive = wp.optix_get_primitive_type_from_hit_kind(wp.optix_hit_object_get_hit_kind())
        if primitive == wp.uint32(PrimitiveType.TRIANGLE):
            uv = wp.optix_hit_object_get_triangle_barycentrics()
            query.u = uv[0]
            query.v = uv[1]
        elif primitive == wp.uint32(PrimitiveType.CUSTOM):
            query.u = wp.optix_uint32_as_float(wp.optix_hit_object_get_attribute(0))
            query.v = wp.optix_uint32_as_float(wp.optix_hit_object_get_attribute(1))
        elif primitive == wp.uint32(PrimitiveType.FLAT_QUADRATIC_BSPLINE):
            uv = wp.optix_hit_object_get_ribbon_parameters()
            query.u = uv[0]
            query.v = uv[1]
        elif (
            primitive == wp.uint32(PrimitiveType.ROUND_LINEAR)
            or primitive == wp.uint32(PrimitiveType.ROUND_QUADRATIC_BSPLINE)
            or primitive == wp.uint32(PrimitiveType.ROUND_CUBIC_BSPLINE)
            or primitive == wp.uint32(PrimitiveType.ROUND_CATMULLROM)
            or primitive == wp.uint32(PrimitiveType.ROUND_CUBIC_BEZIER)
            or (
                primitive >= wp.uint32(PrimitiveType.ROUND_QUADRATIC_BSPLINE_ROCAPS)
                and primitive <= wp.uint32(PrimitiveType.ROUND_CUBIC_BEZIER_ROCAPS)
            )
        ):
            query.u = wp.optix_hit_object_get_curve_parameter()
        # Other primitives (e.g. built-in spheres) have no hit parameters; u and v stay 0.
        query.instance_id = wp.optix_hit_object_get_instance_id()
        query.primitive_index = wp.optix_hit_object_get_primitive_index()
    return query


@wp.func
def accel_query_ray_anyhit(
    id: wp.uint64,
    start: wp.vec3,
    dir: wp.vec3,
    max_t: float,
    min_t: float = 0.0,
    mask: wp.Int = 0xFF,
) -> bool:
    """Return whether anything intersects the ray in ``[min_t, max_t]``, stopping at the first hit."""
    flags = RayFlags.TERMINATE_ON_FIRST_HIT | RayFlags.DISABLE_CLOSESTHIT | RayFlags.DISABLE_ANYHIT
    wp.optix_traverse(id, start, dir, max_t, min_t, 0.0, mask, flags)
    return wp.optix_hit_object_is_hit()
