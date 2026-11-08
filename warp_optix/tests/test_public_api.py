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


def test_public_enums_match_bundled_sdk():
    assert int(wo.PrimitiveType.ROUND_LINEAR) == int(native.PRIMITIVE_TYPE_ROUND_LINEAR)
    flags = wo.BuildFlags.PREFER_FAST_TRACE | wo.BuildFlags.ALLOW_UPDATE
    assert int(flags) == int(native.BUILD_FLAG_PREFER_FAST_TRACE) | int(native.BUILD_FLAG_ALLOW_UPDATE)


def test_public_surface():
    assert set(wo.__all__) == {
        "Accel",
        "AccelQueryRay",
        "BuildFlags",
        "BuildInput",
        "ExceptionFlags",
        "GeometryFlags",
        "HitGroup",
        "HitGroupHandle",
        "HitKind",
        "Instance",
        "InstanceFlags",
        "Launch",
        "LaunchParams",
        "Pipeline",
        "PrimitiveType",
        "ProgramKind",
        "RayFlags",
        "SbtBuilder",
        "SbtLayout",
        "TraversableGraphFlags",
        "aabbs",
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
    removed = [
        "create_context",
        "destroy_context",
        "Context",
        "build_accel",
        "update_accel",
        "create_pipeline",
        "write_launch_params",
        "create_launch_params_buffer",
        "LaunchParamsBuffer",
        "optix_kernel",
        "OptixKernelType",
        "closest_hit",
        "any_hit",
        "Hit",
        "MISS",
        "constants",
    ]
    assert [name for name in removed if hasattr(wo, name)] == []
    assert wo.config.validation is False or wo.config.validation is True
    assert isinstance(wo.config.log_level, int)
    assert [kind.name for kind in wo.ProgramKind] == [
        "RAYGEN",
        "MISS",
        "CLOSEST_HIT",
        "ANY_HIT",
        "INTERSECTION",
        "EXCEPTION",
        "DIRECT_CALLABLE",
        "CONTINUATION_CALLABLE",
    ]


def test_ray_flags_and_hit_kinds_match_native():
    for member in wo.RayFlags.__members__.values():
        if member.name == "FORCE_OPACITY_MICROMAP_2_STATE":
            assert int(member) == 1 << 10  # From optix_types.h; not exposed by PyOptiX 9.1.
            continue
        assert int(member) == int(getattr(native, f"RAY_FLAG_{member.name}")), member
    assert set(wo.RayFlags.__members__) >= {
        "NONE",
        "DISABLE_ANYHIT",
        "ENFORCE_ANYHIT",
        "TERMINATE_ON_FIRST_HIT",
        "DISABLE_CLOSESTHIT",
        "CULL_BACK_FACING_TRIANGLES",
        "CULL_FRONT_FACING_TRIANGLES",
        "CULL_DISABLED_ANYHIT",
        "CULL_ENFORCED_ANYHIT",
    }
    assert [member.name for member in wo.HitKind] == ["TRIANGLE_FRONT_FACE", "TRIANGLE_BACK_FACE"]
    for member in wo.HitKind:
        assert int(member) == int(getattr(native, f"HIT_KIND_{member.name}"))
    for enum, prefix in (
        (wo.GeometryFlags, "GEOMETRY_FLAG_"),
        (wo.InstanceFlags, "INSTANCE_FLAG_"),
        (wo.BuildFlags, "BUILD_FLAG_"),
        (wo.ExceptionFlags, "EXCEPTION_FLAG_"),
        (wo.TraversableGraphFlags, "TRAVERSABLE_GRAPH_FLAG_"),
        (wo.PrimitiveType, "PRIMITIVE_TYPE_"),
    ):
        for member in enum.__members__.values():
            assert int(member) == int(getattr(native, prefix + member.name)), member
