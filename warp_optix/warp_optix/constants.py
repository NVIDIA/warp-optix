# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""OptiX SDK constants for use in Warp kernels and host-side pipeline setup.

Values are taken from ``optix_types.h`` in the OptiX SDK 9.0.
They are plain Python ``int`` constants so they can be used both in
``@wp.kernel`` / ``@wp.func`` code (via ``wp.uint32()``) and in regular
Python (e.g. when configuring ``otk-pyoptix`` pipeline options).

Example — shadow ray with no closest-hit and early termination::

    wp.optix_trace(
        tlas,
        origin,
        direction,
        t_min,
        t_max,
        0.0,
        wp.uint32(0xFF),
        wp.uint32(OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT),
        sbt_offset,
        sbt_stride,
        miss_index,
        payload,
    )
"""

# ---------------------------------------------------------------------------
# Ray flags  (OptixRayFlags — passed to optixTrace / wp.optix_trace)
# ---------------------------------------------------------------------------

OPTIX_RAY_FLAG_NONE = 0
"""No change from the behavior configured for the individual AS."""

OPTIX_RAY_FLAG_DISABLE_ANYHIT = 1 << 0
"""Disable any-hit programs for the ray.
Overrides ``OPTIX_INSTANCE_FLAG_ENFORCE_ANYHIT``.
Mutually exclusive with ``OPTIX_RAY_FLAG_ENFORCE_ANYHIT``,
``OPTIX_RAY_FLAG_CULL_DISABLED_ANYHIT``, ``OPTIX_RAY_FLAG_CULL_ENFORCED_ANYHIT``."""

OPTIX_RAY_FLAG_ENFORCE_ANYHIT = 1 << 1
"""Force any-hit program execution for the ray.
Overrides ``OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT`` and
``OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT``.
Mutually exclusive with ``OPTIX_RAY_FLAG_DISABLE_ANYHIT``,
``OPTIX_RAY_FLAG_CULL_DISABLED_ANYHIT``, ``OPTIX_RAY_FLAG_CULL_ENFORCED_ANYHIT``."""

OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT = 1 << 2
"""Terminate the ray after the first hit and execute the closest-hit program of that hit."""

OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT = 1 << 3
"""Disable closest-hit programs for the ray (miss program still runs on a miss)."""

OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES = 1 << 4
"""Do not intersect back-facing triangles.
Respects ``OPTIX_INSTANCE_FLAG_FLIP_TRIANGLE_FACING``.
Mutually exclusive with ``OPTIX_RAY_FLAG_CULL_FRONT_FACING_TRIANGLES``."""

OPTIX_RAY_FLAG_CULL_FRONT_FACING_TRIANGLES = 1 << 5
"""Do not intersect front-facing triangles.
Respects ``OPTIX_INSTANCE_FLAG_FLIP_TRIANGLE_FACING``.
Mutually exclusive with ``OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES``."""

OPTIX_RAY_FLAG_CULL_DISABLED_ANYHIT = 1 << 6
"""Do not intersect geometry that disables any-hit programs.
Mutually exclusive with ``OPTIX_RAY_FLAG_CULL_ENFORCED_ANYHIT``,
``OPTIX_RAY_FLAG_ENFORCE_ANYHIT``, ``OPTIX_RAY_FLAG_DISABLE_ANYHIT``."""

OPTIX_RAY_FLAG_CULL_ENFORCED_ANYHIT = 1 << 7
"""Do not intersect geometry that has an enabled any-hit program.
Mutually exclusive with ``OPTIX_RAY_FLAG_CULL_DISABLED_ANYHIT``,
``OPTIX_RAY_FLAG_ENFORCE_ANYHIT``, ``OPTIX_RAY_FLAG_DISABLE_ANYHIT``."""

OPTIX_RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE = 1 << 10
"""Force 4-state opacity micromaps to behave as 2-state during traversal."""

# ---------------------------------------------------------------------------
# Instance flags  (OptixInstanceFlags — per-instance in the IAS)
# ---------------------------------------------------------------------------

OPTIX_INSTANCE_FLAG_NONE = 0
"""No special flag set."""

OPTIX_INSTANCE_FLAG_DISABLE_TRIANGLE_FACE_CULLING = 1 << 0
"""Prevent triangles from being culled due to their orientation.
Ignores ``OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES`` and
``OPTIX_RAY_FLAG_CULL_FRONT_FACING_TRIANGLES``."""

OPTIX_INSTANCE_FLAG_FLIP_TRIANGLE_FACING = 1 << 1
"""Flip triangle orientation (affects culling and the reported hit face)."""

OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT = 1 << 2
"""Disable any-hit programs for all geometries of the instance.
Can be overridden by ``OPTIX_RAY_FLAG_ENFORCE_ANYHIT``.
Mutually exclusive with ``OPTIX_INSTANCE_FLAG_ENFORCE_ANYHIT``."""

OPTIX_INSTANCE_FLAG_ENFORCE_ANYHIT = 1 << 3
"""Enable any-hit programs for all geometries of the instance.
Overrides ``OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT``.
Can be overridden by ``OPTIX_RAY_FLAG_DISABLE_ANYHIT``.
Mutually exclusive with ``OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT``."""

# ---------------------------------------------------------------------------
# Geometry flags  (OptixGeometryFlags — per build-input geometry)
# ---------------------------------------------------------------------------

OPTIX_GEOMETRY_FLAG_NONE = 0
"""No flags set."""

OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT = 1 << 0
"""Disable any-hit program invocation for this geometry.
Can be overridden by ``OPTIX_INSTANCE_FLAG_ENFORCE_ANYHIT`` and
``OPTIX_RAY_FLAG_ENFORCE_ANYHIT``."""

OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL = 1 << 1
"""Guarantee at most one any-hit invocation per primitive intersection."""

OPTIX_GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_CULLING = 1 << 2
"""Prevent triangles from being culled due to their orientation."""

# ---------------------------------------------------------------------------
# Hit kind  (OptixHitKind — returned by optixGetHitKind)
# ---------------------------------------------------------------------------

OPTIX_HIT_KIND_TRIANGLE_FRONT_FACE = 0xFE
"""Ray hit the triangle on the front face."""

OPTIX_HIT_KIND_TRIANGLE_BACK_FACE = 0xFF
"""Ray hit the triangle on the back face."""

# ---------------------------------------------------------------------------
# Build flags  (OptixBuildFlags — for acceleration structure builds)
# ---------------------------------------------------------------------------

OPTIX_BUILD_FLAG_NONE = 0
"""No special flags set."""

OPTIX_BUILD_FLAG_ALLOW_UPDATE = 1 << 0
"""Allow updating the AS with new vertex positions via subsequent ``optixAccelBuild`` calls."""

OPTIX_BUILD_FLAG_ALLOW_COMPACTION = 1 << 1
"""Allow compaction of the AS after build."""

OPTIX_BUILD_FLAG_PREFER_FAST_TRACE = 1 << 2
"""Prefer faster ray traversal (longer build time).
Mutually exclusive with ``OPTIX_BUILD_FLAG_PREFER_FAST_BUILD``."""

OPTIX_BUILD_FLAG_PREFER_FAST_BUILD = 1 << 3
"""Prefer faster build time (slower traversal).
Mutually exclusive with ``OPTIX_BUILD_FLAG_PREFER_FAST_TRACE``."""

OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS = 1 << 4
"""Allow random access to build input vertices from device code."""

OPTIX_BUILD_FLAG_ALLOW_RANDOM_INSTANCE_ACCESS = 1 << 5
"""Allow random access to instances from device code."""

OPTIX_BUILD_FLAG_ALLOW_OPACITY_MICROMAP_UPDATE = 1 << 6
"""Allow updating opacity micromap arrays and indices on refits."""

OPTIX_BUILD_FLAG_ALLOW_DISABLE_OPACITY_MICROMAPS = 1 << 7
"""Allow instances to disable opacity micromap tests via instance flags."""

# ---------------------------------------------------------------------------
# Traversable graph flags  (OptixTraversableGraphFlags — pipeline config)
# ---------------------------------------------------------------------------

OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY = 0
"""Any traversable graph is valid.
Mutually exclusive with all other traversable graph flags."""

OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS = 1 << 0
"""A single GAS without transforms is valid."""

OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING = 1 << 1
"""A single IAS directly connected to GAS traversables (no transforms) is valid."""

# ---------------------------------------------------------------------------
# Exception flags  (OptixExceptionFlags — pipeline compile options)
# ---------------------------------------------------------------------------

OPTIX_EXCEPTION_FLAG_NONE = 0
"""No exceptions enabled."""

OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW = 1 << 0
"""Enable exceptions for continuation stack overflows."""

OPTIX_EXCEPTION_FLAG_TRACE_DEPTH = 1 << 1
"""Enable exceptions for trace depth overflows."""

OPTIX_EXCEPTION_FLAG_USER = 1 << 2
"""Enable user exception programs."""

# ---------------------------------------------------------------------------
# Compile optimization level  (OptixCompileOptimizationLevel)
# ---------------------------------------------------------------------------

OPTIX_COMPILE_OPTIMIZATION_DEFAULT = 0
"""Default: all optimizations enabled."""

OPTIX_COMPILE_OPTIMIZATION_LEVEL_0 = 0x2340
"""No optimizations."""

OPTIX_COMPILE_OPTIMIZATION_LEVEL_1 = 0x2341
"""Some optimizations."""

OPTIX_COMPILE_OPTIMIZATION_LEVEL_2 = 0x2342
"""Most optimizations."""

OPTIX_COMPILE_OPTIMIZATION_LEVEL_3 = 0x2343
"""All optimizations."""

# ---------------------------------------------------------------------------
# Compile debug level  (OptixCompileDebugLevel)
# ---------------------------------------------------------------------------

OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT = 0
"""Default (currently minimal)."""

OPTIX_COMPILE_DEBUG_LEVEL_NONE = 0x2350
"""No debug information."""

OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL = 0x2351
"""Debug information that does not impact performance."""

OPTIX_COMPILE_DEBUG_LEVEL_MODERATE = 0x2353
"""Some debug information with slight performance cost."""

OPTIX_COMPILE_DEBUG_LEVEL_FULL = 0x2352
"""Full debug information."""

# ---------------------------------------------------------------------------
# Primitive type flags  (OptixPrimitiveTypeFlags — built-in input types)
# ---------------------------------------------------------------------------

OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM = 1 << 0
"""Custom primitive."""

OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_QUADRATIC_BSPLINE = 1 << 1
"""B-spline curve of degree 2 with circular cross-section."""

OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_CUBIC_BSPLINE = 1 << 2
"""B-spline curve of degree 3 with circular cross-section."""

OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_LINEAR = 1 << 3
"""Piecewise linear curve with circular cross-section."""

OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_CATMULLROM = 1 << 4
"""Catmull-Rom curve with circular cross-section."""

OPTIX_PRIMITIVE_TYPE_FLAGS_FLAT_QUADRATIC_BSPLINE = 1 << 5
"""B-spline curve of degree 2 with oriented, flat cross-section."""

OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE = 1 << 6
"""Sphere."""

OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_CUBIC_BEZIER = 1 << 7
"""Bezier curve of degree 3 with circular cross-section."""

OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE = 1 << 31
"""Triangle."""

OPTIX_PRIMITIVE_TYPE_FLAGS_DISPLACED_MICROMESH_TRIANGLE = 1 << 30
"""Triangle with an applied displacement micromap."""
