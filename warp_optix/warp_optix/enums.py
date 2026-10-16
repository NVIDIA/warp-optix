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
    DISABLE_TRIANGLE_FACE_CULLING = int(
        _backend.INSTANCE_FLAG_DISABLE_TRIANGLE_FACE_CULLING
    )
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
    ALLOW_SINGLE_LEVEL_INSTANCING = int(
        _backend.TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING
    )


del _backend
