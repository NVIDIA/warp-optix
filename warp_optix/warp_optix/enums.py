# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-side configuration enums with OptiX numeric values."""

from enum import IntEnum, IntFlag

from warp_optix._backend import get_backend

_backend = get_backend()


class GeometryFlags(IntFlag):
    NONE = int(_backend.GEOMETRY_FLAG_NONE)
    DISABLE_ANYHIT = int(_backend.GEOMETRY_FLAG_DISABLE_ANYHIT)
    REQUIRE_SINGLE_ANYHIT_CALL = int(_backend.GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL)


class PrimitiveType(IntEnum):
    CUSTOM = int(_backend.PRIMITIVE_TYPE_CUSTOM)
    ROUND_QUADRATIC_BSPLINE = int(_backend.PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE)
    ROUND_CUBIC_BSPLINE = int(_backend.PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE)
    ROUND_LINEAR = int(_backend.PRIMITIVE_TYPE_ROUND_LINEAR)
    ROUND_CATMULLROM = int(_backend.PRIMITIVE_TYPE_ROUND_CATMULLROM)
    FLAT_QUADRATIC_BSPLINE = int(_backend.PRIMITIVE_TYPE_FLAT_QUADRATIC_BSPLINE)
    ROUND_CUBIC_BEZIER = int(_backend.PRIMITIVE_TYPE_ROUND_CUBIC_BEZIER)
    TRIANGLE = int(_backend.PRIMITIVE_TYPE_TRIANGLE)


class InstanceFlags(IntFlag):
    NONE = int(_backend.INSTANCE_FLAG_NONE)
    DISABLE_TRIANGLE_FACE_CULLING = int(_backend.INSTANCE_FLAG_DISABLE_TRIANGLE_FACE_CULLING)
    FLIP_TRIANGLE_FACING = int(_backend.INSTANCE_FLAG_FLIP_TRIANGLE_FACING)
    DISABLE_ANYHIT = int(_backend.INSTANCE_FLAG_DISABLE_ANYHIT)
    ENFORCE_ANYHIT = int(_backend.INSTANCE_FLAG_ENFORCE_ANYHIT)


class BuildFlags(IntFlag):
    NONE = int(_backend.BUILD_FLAG_NONE)
    ALLOW_UPDATE = int(_backend.BUILD_FLAG_ALLOW_UPDATE)
    ALLOW_COMPACTION = int(_backend.BUILD_FLAG_ALLOW_COMPACTION)
    PREFER_FAST_TRACE = int(_backend.BUILD_FLAG_PREFER_FAST_TRACE)
    PREFER_FAST_BUILD = int(_backend.BUILD_FLAG_PREFER_FAST_BUILD)
    ALLOW_RANDOM_VERTEX_ACCESS = int(_backend.BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS)


class TraversableGraphFlags(IntFlag):
    ALLOW_ANY = int(_backend.TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY)
    ALLOW_SINGLE_GAS = int(_backend.TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS)
    ALLOW_SINGLE_LEVEL_INSTANCING = int(_backend.TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING)


class RayFlags(IntFlag):
    """Per-ray flags for ``optix_trace``/``optix_traverse`` and the ray queries; usable inside kernels."""

    NONE = int(_backend.RAY_FLAG_NONE)
    DISABLE_ANYHIT = int(_backend.RAY_FLAG_DISABLE_ANYHIT)
    ENFORCE_ANYHIT = int(_backend.RAY_FLAG_ENFORCE_ANYHIT)
    TERMINATE_ON_FIRST_HIT = int(_backend.RAY_FLAG_TERMINATE_ON_FIRST_HIT)
    DISABLE_CLOSESTHIT = int(_backend.RAY_FLAG_DISABLE_CLOSESTHIT)
    CULL_BACK_FACING_TRIANGLES = int(_backend.RAY_FLAG_CULL_BACK_FACING_TRIANGLES)
    CULL_FRONT_FACING_TRIANGLES = int(_backend.RAY_FLAG_CULL_FRONT_FACING_TRIANGLES)
    CULL_DISABLED_ANYHIT = int(_backend.RAY_FLAG_CULL_DISABLED_ANYHIT)
    CULL_ENFORCED_ANYHIT = int(_backend.RAY_FLAG_CULL_ENFORCED_ANYHIT)
    # Not exposed by PyOptiX 9.1; value from optix_types.h.
    FORCE_OPACITY_MICROMAP_2_STATE = 1 << 10


class HitKind(IntEnum):
    """Hit kinds reported for built-in triangles; usable inside kernels."""

    TRIANGLE_FRONT_FACE = int(_backend.HIT_KIND_TRIANGLE_FRONT_FACE)
    TRIANGLE_BACK_FACE = int(_backend.HIT_KIND_TRIANGLE_BACK_FACE)


class ExceptionFlags(IntFlag):
    NONE = int(_backend.EXCEPTION_FLAG_NONE)
    STACK_OVERFLOW = int(_backend.EXCEPTION_FLAG_STACK_OVERFLOW)
    TRACE_DEPTH = int(_backend.EXCEPTION_FLAG_TRACE_DEPTH)
    USER = int(_backend.EXCEPTION_FLAG_USER)


class CompileOptimizationLevel(IntEnum):
    """OptiX optimization level of a pipeline's programs."""

    DEFAULT = int(_backend.COMPILE_OPTIMIZATION_DEFAULT)
    LEVEL_0 = int(_backend.COMPILE_OPTIMIZATION_LEVEL_0)
    LEVEL_1 = int(_backend.COMPILE_OPTIMIZATION_LEVEL_1)
    LEVEL_2 = int(_backend.COMPILE_OPTIMIZATION_LEVEL_2)
    LEVEL_3 = int(_backend.COMPILE_OPTIMIZATION_LEVEL_3)


class CompileDebugLevel(IntEnum):
    """OptiX debug information kept in a pipeline's programs."""

    DEFAULT = int(_backend.COMPILE_DEBUG_LEVEL_DEFAULT)
    NONE = int(_backend.COMPILE_DEBUG_LEVEL_NONE)
    MINIMAL = int(_backend.COMPILE_DEBUG_LEVEL_MINIMAL)
    MODERATE = int(_backend.COMPILE_DEBUG_LEVEL_MODERATE)
    FULL = int(_backend.COMPILE_DEBUG_LEVEL_FULL)


del _backend
