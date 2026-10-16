# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ray queries for raygen programs.

Both queries traverse without invoking closest-hit or miss programs and read
the result from the OptiX hit object, so a pipeline that only uses them needs
no hit programs for built-in geometry. After :func:`closest_hit`, further hit
data (barycentrics, intersection attributes, transforms) is available through
the ``wp.optix_hit_object_*`` builtins until the next traversal.
"""

from __future__ import annotations

import warp as wp

from warp_optix.constants import (
    OPTIX_RAY_FLAG_DISABLE_ANYHIT,
    OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
    OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
)

MISS = wp.constant(-1.0)


@wp.struct
class Hit:
    """Closest intersection along a ray; ``t`` is :data:`MISS` when nothing was hit."""

    t: float
    instance_id: wp.uint32
    primitive_index: wp.uint32


@wp.func
def closest_hit(
    traversable: wp.uint64,
    origin: wp.vec3,
    direction: wp.vec3,
    tmin: float,
    tmax: float,
    visibility_mask: wp.uint32,
    ray_flags: wp.uint32,
) -> Hit:
    """Return the closest intersection in ``[tmin, tmax]``."""
    wp.optix_traverse(
        traversable,
        origin,
        direction,
        tmin,
        tmax,
        0.0,
        visibility_mask,
        ray_flags,
        wp.uint32(0),
        wp.uint32(1),
        wp.uint32(0),
    )
    hit = Hit()
    hit.t = MISS
    if wp.optix_hit_object_is_hit():
        hit.t = wp.optix_hit_object_get_ray_tmax()
        hit.instance_id = wp.optix_hit_object_get_instance_id()
        hit.primitive_index = wp.optix_hit_object_get_primitive_index()
    return hit


@wp.func
def any_hit(
    traversable: wp.uint64,
    origin: wp.vec3,
    direction: wp.vec3,
    tmin: float,
    tmax: float,
    visibility_mask: wp.uint32,
) -> bool:
    """Return whether anything intersects the ray in ``[tmin, tmax]``, stopping at the first hit."""
    flags = wp.uint32(OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT)
    flags |= wp.uint32(OPTIX_RAY_FLAG_DISABLE_ANYHIT)
    wp.optix_traverse(
        traversable,
        origin,
        direction,
        tmin,
        tmax,
        0.0,
        visibility_mask,
        flags,
        wp.uint32(0),
        wp.uint32(1),
        wp.uint32(0),
    )
    return wp.optix_hit_object_is_hit()
