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
    fake_warp.uint64 = np.uint64
    fake_warp.int32 = np.int32
    fake_warp.vec2 = type("vec2", (), {})
    fake_warp.vec3 = type("vec3", (), {})
    fake_warp.vec3ui = type("vec3ui", (), {})
    fake_warp.mat33 = type("mat33", (), {})
    fake_warp.mat44 = type("mat44", (), {})
    fake_warp.types = SimpleNamespace(matrix=lambda *, shape, dtype: type(f"mat{shape[0]}{shape[1]}", (), {}))

    module_name = "warp_optix_builtins_under_test"
    builtins_path = Path(__file__).parents[1] / "warp_optix" / "builtins.py"
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
        "optix_get_object_to_world_transform_matrix",
        "optix_get_world_to_object_transform_matrix",
        "optix_get_transform_list_size",
        "optix_get_transform_list_handle",
        "optix_get_transform_type_from_handle",
    } <= registered_names
    assert "optix_direct_call" in registered_names
    assert "optix_continuation_call" in registered_names
    assert "optix_get_exception_code" in registered_names
    assert all(f"optix_get_exception_detail_{i}" in registered_names for i in range(8))
    exception_overloads = [kwargs for name, kwargs in calls if name == "optix_throw_exception"]
    assert [len(call["input_types"]) for call in exception_overloads] == list(range(1, 10))
    assert all(f"optix_get_attribute_{i}" in registered_names for i in range(8))
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
        "optix_transform_point_from_object_to_world_space",
        "optix_transform_point_from_world_to_object_space",
        "optix_transform_vector_from_world_to_object_space",
        "optix_transform_normal_from_world_to_object_space",
    } <= registered_names


@wp.struct
class HitPayload:
    value: float


@wp.kernel(module="unique")
def write_mixed_instances(triangle_gas: wp.uint64, sphere_gas: wp.uint64, records: wp.array(dtype=wo.Instance)):
    i = wp.tid()
    # Geometry type g owns hit records [2 * g, 2 * g + 1], one per ray type.
    gas = triangle_gas if i == 0 else sphere_gas
    records[i] = wo.make_instance(
        wp.identity(3, float),
        wp.vec3(10.0 * float(i), 0.0, 0.0),
        gas,
        wp.uint32(i),
        wp.uint32(2 * i),
        wp.uint32(0xFF),
        wp.uint32(0),
    )


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
            0.0,
            100.0,
            0.0,
            wp.uint32(0xFF),
            wp.uint32(0),
            wp.uint32(ray_type),
            wp.uint32(2),
            wp.uint32(0),
            payload,
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
        wp.optix_report_intersection(-b - wp.sqrt(disc), wp.uint32(0))


@pytest.mark.gpu
@pytest.mark.skipif(not wp.is_cuda_available(), reason="requires a CUDA device")
def test_mixed_geometry_selects_hit_records_by_geometry_primitive_and_ray_type():
    device = wp.get_device("cuda")
    ctx = wo.create_context(device)
    vertices = wp.array([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]], dtype=wp.vec3, device=device)
    indices = wp.array([0, 1, 2], dtype=wp.uint32, device=device)
    triangle_gas = wo.build_accel(ctx, [wo.triangles(vertices, indices)], compact=True, device=device)

    # Two spheres in one custom GAS; the second primitive uses the second SBT record.
    centers = wp.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=wp.vec3, device=device)
    bounds = wp.array([[-0.5, -0.5, -0.5, 0.5, 0.5, 0.5], [2.5, -0.5, -0.5, 3.5, 0.5, 0.5]], dtype=float, device=device)
    sbt_indices = wp.array([0, 1], dtype=wp.uint32, device=device)
    flags = [wo.GeometryFlags.DISABLE_ANYHIT] * 2
    sphere_gas = wo.build_accel(ctx, [wo.aabbs(bounds, flags=flags, sbt_indices=sbt_indices)], device=device)

    records = wp.empty(2, dtype=wo.Instance, device=device)
    wp.launch(write_mixed_instances, dim=2, inputs=[triangle_gas.handle, sphere_gas.handle, records], device=device)
    ias = wo.build_accel(ctx, [wo.instances(records)], device=device)

    triangle = [wo.HitGroup(closest_hit=primary_hit), wo.HitGroup(closest_hit=shadow_hit)]
    sphere = [
        wo.HitGroup(closest_hit=primary_hit, intersection=sphere_intersection),
        wo.HitGroup(closest_hit=shadow_hit, intersection=sphere_intersection),
    ]
    pipeline = wo.create_pipeline(
        ctx,
        mixed_raygen,
        hit_groups=[*triangle, *sphere, *sphere],
        ray_types=2,
        payload_words=1,
        graph_flags=wo.TraversableGraphFlags.ALLOW_SINGLE_LEVEL_INSTANCING,
        device=device,
    )
    xs = wp.array([0.0, 10.0, 13.0, 20.0], dtype=float, device=device)
    values = wp.zeros((4, 2), dtype=float, device=device)
    wo.launch(pipeline, dim=(4,), inputs=[ias.handle, xs, centers, values])
    np.testing.assert_array_equal(values.numpy(), [[1, -1], [101, -101], [112, -112], [999, 999]])
