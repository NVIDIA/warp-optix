# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import warp as wp

import warp_optix as wo


def test_custom_primitive_builtins_register_all_attribute_arities():
    calls = []
    fake_warp = ModuleType("warp")
    fake_warp.build_experimental = SimpleNamespace(add_builtin=lambda name, **kwargs: calls.append((name, kwargs)))
    fake_warp.bool = bool
    fake_warp.float32 = np.float32
    fake_warp.uint32 = np.uint32
    fake_warp.Int = type("Int", (), {})
    fake_warp.uint64 = np.uint64
    fake_warp.int32 = np.int32
    fake_warp.vec2 = type("vec2", (), {})
    fake_warp.vec3 = type("vec3", (), {})
    fake_warp.vec3ui = type("vec3ui", (), {})
    fake_warp.mat33 = type("mat33", (), {})
    fake_warp.mat44 = type("mat44", (), {})
    fake_warp.types = SimpleNamespace(matrix=lambda *, shape, dtype: type(f"mat{shape[0]}{shape[1]}", (), {}))

    module_name = "warp_optix_builtins_under_test"
    builtins_path = Path(__file__).parents[1] / "warp_optix" / "_src" / "builtins.py"
    spec = importlib.util.spec_from_file_location(module_name, builtins_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    previous_warp = sys.modules.get("warp")
    sys.modules["warp"] = fake_warp
    try:
        spec.loader.exec_module(module)
        module.register_addon_builtins()
    finally:
        if previous_warp is None:
            sys.modules.pop("warp", None)
        else:
            sys.modules["warp"] = previous_warp

    reports = [kwargs for name, kwargs in calls if name == "optix_report_intersection"]
    assert [len(call["input_types"]) for call in reports] == list(range(2, 11))
    registered_names = {name for name, _ in calls}
    assert "optix_get_object_ray_origin" in registered_names
    assert "optix_get_object_ray_direction" in registered_names
    assert "optix_get_ray_tmin" in registered_names
    assert "optix_get_hit_kind" in registered_names
    assert "optix_get_triangle_vertex_data" in registered_names
    assert "optix_get_curve_parameter" in registered_names
    assert {
        "optix_is_triangle_hit",
        "optix_is_triangle_front_face_hit",
        "optix_is_triangle_back_face_hit",
        "optix_get_linear_curve_vertex_data",
        "optix_get_quadratic_bspline_vertex_data",
        "optix_get_cubic_bspline_vertex_data",
        "optix_get_catmull_rom_vertex_data",
        "optix_get_cubic_bezier_vertex_data",
        "optix_get_ribbon_vertex_data",
        "optix_get_ribbon_parameters",
        "optix_get_ribbon_normal",
        "optix_get_object_to_world_matrix",
        "optix_get_world_to_object_matrix",
        "optix_get_transform_list_size",
        "optix_get_transform_list_handle",
        "optix_get_transform_type_from_handle",
    } <= registered_names
    assert "optix_direct_call" in registered_names
    assert "optix_continuation_call" in registered_names
    assert "optix_get_exception_code" in registered_names
    assert "optix_get_exception_detail" in registered_names
    assert all(f"optix_get_exception_detail_{i}" in registered_names for i in range(8))
    exception_overloads = [kwargs for name, kwargs in calls if name == "optix_throw_exception"]
    assert [len(call["input_types"]) for call in exception_overloads] == list(range(1, 10))
    assert {"optix_get_attribute", "optix_hit_object_get_attribute", "optix_get_payload", "optix_set_payload"} <= (
        registered_names
    )
    assert all(f"optix_get_attribute_{i}" in registered_names for i in range(8))
    assert all(f"optix_get_payload_{i}" in registered_names for i in range(32))
    # Trailing ray arguments may be omitted: one overload per argument count (payload-free traversal too).
    traces = [kwargs for name, kwargs in calls if name == "optix_trace"]
    assert [len(call["input_types"]) for call in traces] == list(range(5, 13))
    traversals = sorted(len(kwargs["input_types"]) for name, kwargs in calls if name == "optix_traverse")
    assert traversals == sorted([*range(4, 12), *range(5, 13)])
    assert {
        "optix_get_ray_time",
        "optix_get_ray_flags",
        "optix_get_ray_visibility_mask",
        "optix_get_instance_index",
        "optix_get_sbt_gas_index",
        "optix_get_primitive_type",
        "optix_get_gas_traversable_handle",
        "optix_is_front_face_hit",
        "optix_is_back_face_hit",
        "optix_transform_point_object_to_world",
        "optix_transform_point_world_to_object",
        "optix_transform_vector_world_to_object",
        "optix_transform_normal_world_to_object",
        "optix_hit_object_transform_point_object_to_world",
        "optix_float_as_uint32",
        "optix_uint32_as_float",
        "optix_get_primitive_type_from_hit_kind",
        "optix_hit_object_get_curve_parameter",
        "optix_hit_object_get_ribbon_parameters",
    } <= registered_names


@wp.struct
class HitPayload:
    value: float


@wp.kernel(module="unique")
def write_mixed_instances(triangle_gas: wp.uint64, sphere_gas: wp.uint64, records: wp.array(dtype=wo.Instance)):
    i = wp.tid()
    # Geometry type g owns hit records [2 * g, 2 * g + 1], one per ray type.
    gas = triangle_gas if i == 0 else sphere_gas
    records[i] = wo.make_instance(wp.transform(wp.vec3(10.0 * float(i), 0.0, 0.0), wp.quat_identity()), gas, i, 2 * i)


@wp.kernel(module="unique")
def mixed_raygen(
    scene: wp.uint64, xs: wp.array(dtype=float), centers: wp.array(dtype=wp.vec3), values: wp.array2d(dtype=float)
):
    i = wp.tid()
    for ray_type in range(2):
        payload = HitPayload()
        payload.value = 999.0
        wp.optix_trace(
            scene,
            wp.vec3(xs[i], 0.0, 5.0),
            wp.vec3(0.0, 0.0, -1.0),
            100.0,
            payload,
            0.0,
            0.0,
            0xFF,
            wo.RayFlags.NONE,
            ray_type,
            2,
            0,
        )
        values[i, ray_type] = payload.value


@wp.func
def hit_code() -> float:
    instance = float(wp.optix_get_instance_id())
    record = float(wp.optix_get_sbt_gas_index())
    return 1.0 + 100.0 * instance + 10.0 * record + float(wp.optix_get_primitive_index())


@wp.kernel(module="unique")
def primary_hit():
    payload = HitPayload()
    payload.value = hit_code()
    wp.optix_store_payload(payload)


@wp.kernel(module="unique")
def shadow_hit():
    payload = HitPayload()
    payload.value = -hit_code()
    wp.optix_store_payload(payload)


@wp.kernel(module="unique")
def sphere_intersection(centers: wp.array(dtype=wp.vec3)):
    origin = wp.optix_get_object_ray_origin() - centers[int(wp.optix_get_primitive_index())]
    direction = wp.optix_get_object_ray_direction()
    b = wp.dot(origin, direction)
    disc = b * b - wp.dot(origin, origin) + 0.25
    if disc >= 0.0:
        wp.optix_report_intersection(-b - wp.sqrt(disc), 0)


@pytest.mark.gpu
@pytest.mark.skipif(not wp.is_cuda_available(), reason="requires a CUDA device")
def test_mixed_geometry_selects_hit_records_by_geometry_primitive_and_ray_type():
    device = wp.get_device("cuda")
    vertices = wp.array([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]], dtype=wp.vec3, device=device)
    indices = wp.array([0, 1, 2], dtype=wp.uint32, device=device)
    triangle_gas = wo.Accel([wo.triangles(vertices, indices)], compact=True, device=device)

    # Two spheres in one custom GAS; the second primitive uses the second SBT record.
    centers = wp.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=wp.vec3, device=device)
    bounds = wp.array([[-0.5, -0.5, -0.5, 0.5, 0.5, 0.5], [2.5, -0.5, -0.5, 3.5, 0.5, 0.5]], dtype=float, device=device)
    sbt_indices = wp.array([0, 1], dtype=wp.uint32, device=device)
    flags = [wo.GeometryFlags.DISABLE_ANYHIT] * 2
    sphere_gas = wo.Accel([wo.aabbs(bounds=bounds, flags=flags, sbt_indices=sbt_indices)], device=device)

    records = wp.empty(2, dtype=wo.Instance, device=device)
    wp.launch(write_mixed_instances, dim=2, inputs=[triangle_gas.id, sphere_gas.id, records], device=device)
    ias = wo.Accel([wo.instances(records, [triangle_gas, sphere_gas])], device=device)

    triangle = {"primary": wo.HitGroup(closest_hit=primary_hit), "shadow": wo.HitGroup(closest_hit=shadow_hit)}
    sphere = {
        "primary": wo.HitGroup(closest_hit=primary_hit, intersection=sphere_intersection),
        "shadow": wo.HitGroup(closest_hit=shadow_hit, intersection=sphere_intersection),
    }
    pipeline = wo.Pipeline(
        mixed_raygen,
        sbt=wo.SbtLayout(
            ray_types=("primary", "shadow"),
            hit_groups={"triangle": triangle, "sphere 0": sphere, "sphere 1": sphere},
        ),
        payload_words=1,
        graph_flags=wo.TraversableGraphFlags.ALLOW_SINGLE_LEVEL_INSTANCING,
        device=device,
    )
    xs = wp.array([0.0, 10.0, 13.0, 20.0], dtype=float, device=device)
    values = wp.zeros((4, 2), dtype=float, device=device)
    wo.launch(pipeline, dim=(4,), inputs=[ias, xs, centers, values])
    np.testing.assert_array_equal(values.numpy(), [[1, -1], [101, -101], [112, -112], [999, 999]])
