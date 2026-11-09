# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public namespace contracts."""

import subprocess
import sys

import warp_optix as wo
from warp_optix import native


def test_import_boundary_in_fresh_process(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys

import warp_optix as wo
import warp as wp
import optix
assert 'warp_nn' not in sys.modules
assert 'pyglet' not in sys.modules
assert 'imgui_bundle' not in sys.modules
assert wp._src.context.runtime is None
assert wo.__version__ == '0.2.0'
from warp_optix import native
assert native.version() == (9, 1, 0)
assert native.DeviceContext is optix.DeviceContext
assert native.Module is optix.Module
assert 'warp_optix._bindings' not in sys.modules
""",
        ],
        cwd=tmp_path,
        check=True,
    )


def test_public_surface():
    assert set(wo.__all__) == {
        "Accel",
        "AccelQueryRay",
        "BuildFlags",
        "BuildInput",
        "CompileDebugLevel",
        "CompileOptimizationLevel",
        "ExceptionFlags",
        "GeometryFlags",
        "HitGroup",
        "HitGroupHandle",
        "HitKind",
        "Instance",
        "InstanceFlags",
        "Launch",
        "Pipeline",
        "PrimitiveType",
        "ProgramKind",
        "RayFlags",
        "SbtBuilder",
        "SbtLayout",
        "TraversableGraphFlags",
        "aabbs",
        "compact",
        "accel_query_ray",
        "accel_query_ray_anyhit",
        "config",
        "curves",
        "get_context",
        "instances",
        "kernel",
        "launch",
        "make_instance",
        "triangles",
    }
    assert all(hasattr(wo, name) for name in wo.__all__)


def test_enums_match_native():
    # Not exposed by PyOptiX 9.1; their values come from optix_types.h.
    from_header = {"FORCE_OPACITY_MICROMAP_2_STATE"} | {m.name for m in wo.PrimitiveType if m.name.endswith("_ROCAPS")}
    for enum, prefix in (
        (wo.RayFlags, "RAY_FLAG_"),
        (wo.HitKind, "HIT_KIND_"),
        (wo.GeometryFlags, "GEOMETRY_FLAG_"),
        (wo.InstanceFlags, "INSTANCE_FLAG_"),
        (wo.BuildFlags, "BUILD_FLAG_"),
        (wo.ExceptionFlags, "EXCEPTION_FLAG_"),
        (wo.TraversableGraphFlags, "TRAVERSABLE_GRAPH_FLAG_"),
        (wo.PrimitiveType, "PRIMITIVE_TYPE_"),
    ):
        for member in enum.__members__.values():
            if member.name not in from_header:
                assert int(member) == int(getattr(native, prefix + member.name)), member
