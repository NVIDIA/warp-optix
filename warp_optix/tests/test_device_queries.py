# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Device builtins exercised inside real OptiX programs."""

import numpy as np
import pytest
import warp as wp

import warp_optix as wo
from warp_optix import native

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not wp.is_cuda_available(), reason="requires a CUDA device")]

TRIANGLE_VERTICES = np.array([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
TRIANGLE_INDICES = np.array([0, 1, 2], dtype=np.uint32)
RANDOM_ACCESS = wo.BuildFlags.PREFER_FAST_TRACE | wo.BuildFlags.ALLOW_RANDOM_VERTEX_ACCESS


@wp.struct
class QueryParams:
    output: wp.array(dtype=wp.uint32)
    traversable: wp.uint64


@wp.struct
class QueryPayload:
    value: wp.uint32


@wp.kernel(module="unique")
def write_rotated_instance(gas: wp.uint64, records: wp.array(dtype=wo.Instance)):
    # Object-to-world: rotate 90 degrees about z, scale x by 2, translate by (2, 0, 0).
    rotation = wp.mat33(0.0, -2.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    records[0] = wo.make_instance(
        rotation, wp.vec3(2.0, 0.0, 0.0), gas, wp.uint32(77), wp.uint32(0), wp.uint32(0xFF), wp.uint32(0)
    )


@wp.kernel(module="unique")
def query_raygen(params: QueryParams):
    payload = QueryPayload()
    wp.optix_trace(
        params.traversable,
        wp.vec3(2.0, 0.0, 3.0),
        wp.vec3(0.0, 0.0, -1.0),
        0.001,
        100.0,
        0.25,
        wp.uint32(0x5A),
        wp.uint32(1),
        wp.uint32(0),
        wp.uint32(1),
        wp.uint32(0),
        payload,
    )


@wp.kernel(module="unique")
def reorder_raygen(params: QueryParams):
    payload = QueryPayload()
    wp.optix_traverse(
        params.traversable,
        wp.vec3(2.0, 0.0, 3.0),
        wp.vec3(0.0, 0.0, -1.0),
        0.001,
        100.0,
        0.25,
        wp.uint32(0x5A),
        wp.uint32(1),
        wp.uint32(0),
        wp.uint32(1),
        wp.uint32(0),
        payload,
    )
    params.output[12] = wp.uint32(1) if wp.optix_hit_object_is_hit() else wp.uint32(0)
    params.output[13] = wp.optix_hit_object_get_instance_id()
    params.output[14] = wp.optix_hit_object_get_primitive_index()
    wp.optix_reorder()
    wp.optix_reorder(wp.uint32(1), wp.uint32(1))
    wp.optix_invoke(payload)


@wp.kernel(module="unique")
def query_miss(params: QueryParams):
    params.output[0] = wp.uint32(0xFFFFFFFF)


@wp.kernel(module="unique")
def query_closest_hit(params: QueryParams):
    params.output[0] = wp.optix_get_instance_id()
    params.output[1] = wp.optix_get_instance_index()
    params.output[2] = wp.optix_get_sbt_gas_index()
    params.output[3] = wp.optix_get_primitive_type()
    params.output[4] = wp.optix_get_ray_flags()
    params.output[5] = wp.optix_get_ray_visibility_mask()
    params.output[6] = wp.float_to_uint32(wp.optix_get_ray_time())
    params.output[7] = wp.uint32(1) if wp.optix_is_front_face_hit() else wp.uint32(0)
    params.output[8] = wp.uint32(1) if wp.optix_is_back_face_hit() else wp.uint32(0)

    gas = wp.optix_get_gas_traversable_handle()
    params.output[9] = wp.uint32(gas & wp.uint64(0xFFFFFFFF))
    params.output[10] = wp.uint32(gas >> wp.uint64(32))

    error = wp.uint32(0)
    object_origin = wp.vec3(0.0, 0.0, 0.0)
    world_origin = wp.vec3(2.0, 0.0, 0.0)
    if wp.length(wp.optix_transform_point_from_object_to_world_space(object_origin) - world_origin) > 1.0e-5:
        error |= wp.uint32(1)
    if wp.length(wp.optix_transform_point_from_world_to_object_space(world_origin) - object_origin) > 1.0e-5:
        error |= wp.uint32(2)

    vector = wp.vec3(1.0, 2.0, 3.0)
    world_vector = wp.optix_transform_vector_from_object_to_world_space(vector)
    if wp.length(wp.optix_transform_vector_from_world_to_object_space(world_vector) - vector) > 1.0e-5:
        error |= wp.uint32(4)

    normal = wp.normalize(wp.vec3(1.0, 2.0, 3.0))
    world_normal = wp.optix_transform_normal_from_object_to_world_space(normal)
    if wp.length(wp.optix_transform_normal_from_world_to_object_space(world_normal) - normal) > 1.0e-5:
        error |= wp.uint32(8)

    if not wp.optix_is_triangle_hit():
        error |= wp.uint32(128)
    if not wp.optix_is_triangle_front_face_hit():
        error |= wp.uint32(256)
    if wp.optix_is_triangle_back_face_hit():
        error |= wp.uint32(512)

    object_to_world = wp.optix_get_object_to_world_transform_matrix()
    world_to_object = wp.optix_get_world_to_object_transform_matrix()
    if (
        wp.abs(object_to_world[0, 0]) > 1.0e-5
        or wp.abs(object_to_world[0, 1] + 2.0) > 1.0e-5
        or wp.abs(object_to_world[1, 0] - 1.0) > 1.0e-5
        or wp.abs(object_to_world[0, 3] - 2.0) > 1.0e-5
        or wp.abs(object_to_world[3, 3] - 1.0) > 1.0e-5
    ):
        error |= wp.uint32(1024)
    if (
        wp.abs(world_to_object[0, 0]) > 1.0e-5
        or wp.abs(world_to_object[0, 1] - 1.0) > 1.0e-5
        or wp.abs(world_to_object[1, 0] + 0.5) > 1.0e-5
        or wp.abs(world_to_object[1, 3] - 1.0) > 1.0e-5
        or wp.abs(world_to_object[3, 3] - 1.0) > 1.0e-5
    ):
        error |= wp.uint32(2048)

    if wp.optix_get_transform_list_size() != wp.uint32(1):
        error |= wp.uint32(4096)
    else:
        transform_handle = wp.optix_get_transform_list_handle(wp.uint32(0))
        if wp.optix_get_transform_type_from_handle(transform_handle) != wp.uint32(4):
            error |= wp.uint32(8192)

    vertices = wp.optix_get_triangle_vertex_data()
    if wp.length(vertices[0] - wp.vec3(-1.0, -1.0, 0.0)) > 1.0e-5:
        error |= wp.uint32(16)
    if wp.length(vertices[1] - wp.vec3(1.0, -1.0, 0.0)) > 1.0e-5:
        error |= wp.uint32(32)
    if wp.length(vertices[2] - wp.vec3(0.0, 1.0, 0.0)) > 1.0e-5:
        error |= wp.uint32(64)
    params.output[11] = error


@wp.kernel(module="unique")
def motion_raygen(params: QueryParams):
    payload = QueryPayload()
    wp.optix_trace(
        params.traversable,
        wp.vec3(0.0, 0.0, 3.0),
        wp.vec3(0.0, 0.0, -1.0),
        0.001,
        100.0,
        0.25,
        wp.uint32(255),
        wp.uint32(0),
        wp.uint32(0),
        wp.uint32(1),
        wp.uint32(0),
        payload,
    )


@wp.kernel(module="unique")
def motion_closest_hit(params: QueryParams):
    params.output[0] = wp.float_to_uint32(wp.optix_get_ray_time())
    params.output[1] = wp.float_to_uint32(wp.optix_get_ray_tmax())


@wp.kernel(module="unique")
def curve_raygen(params: QueryParams):
    payload = QueryPayload()
    wp.optix_trace(
        params.traversable,
        wp.vec3(0.0, 0.0, 3.0),
        wp.vec3(0.0, 0.0, -1.0),
        0.001,
        100.0,
        0.0,
        wp.uint32(255),
        wp.uint32(0),
        wp.uint32(0),
        wp.uint32(1),
        wp.uint32(0),
        payload,
    )


@wp.kernel(module="unique")
def curve_closest_hit(params: QueryParams):
    params.output[0] = wp.uint32(1)
    params.output[1] = wp.optix_get_primitive_type()
    params.output[2] = wp.float_to_uint32(wp.optix_get_curve_parameter())
    world_hit = wp.optix_get_world_ray_origin() + wp.optix_get_ray_tmax() * wp.optix_get_world_ray_direction()
    params.output[3] = wp.float_to_uint32(world_hit[2])
    object_origin = wp.optix_transform_point_from_object_to_world_space(wp.vec3(0.0))
    params.output[4] = wp.float_to_uint32(object_origin[2])

    error = wp.uint32(0)
    vertices = wp.optix_get_linear_curve_vertex_data()
    if (
        wp.length(wp.vec3(vertices[0, 0], vertices[0, 1], vertices[0, 2]) - wp.vec3(0.0, -1.0, 0.0)) > 1.0e-5
        or wp.abs(vertices[0, 3] - 0.25) > 1.0e-5
    ):
        error |= wp.uint32(1)
    if (
        wp.length(wp.vec3(vertices[1, 0], vertices[1, 1], vertices[1, 2]) - wp.vec3(0.0, 1.0, 0.0)) > 1.0e-5
        or wp.abs(vertices[1, 3] - 0.25) > 1.0e-5
    ):
        error |= wp.uint32(2)
    if wp.optix_is_triangle_hit():
        error |= wp.uint32(4)
    params.output[5] = error

    # Keep every current-hit curve query in a live entry point. The primitive
    # type makes only the matching branch execute, while OptiX compiles all of
    # the device calls below.
    primitive_type = wp.optix_get_primitive_type()
    probe = wp.float32(0.0)
    if primitive_type == wp.uint32(0x2501):
        probe = wp.optix_get_quadratic_bspline_vertex_data()[0, 0]
    elif primitive_type == wp.uint32(0x2502):
        probe = wp.optix_get_cubic_bspline_vertex_data()[0, 0]
    elif primitive_type == wp.uint32(0x2504):
        probe = wp.optix_get_catmull_rom_vertex_data()[0, 0]
    elif primitive_type == wp.uint32(0x2505):
        parameters = wp.optix_get_ribbon_parameters()
        probe = wp.optix_get_ribbon_vertex_data()[0, 0] + wp.optix_get_ribbon_normal(parameters)[0]
    elif primitive_type == wp.uint32(0x2507):
        probe = wp.optix_get_cubic_bezier_vertex_data()[0, 0]
    params.output[6] = wp.float_to_uint32(probe)


@wp.kernel(module="unique")
def higher_order_curve_closest_hit(params: QueryParams):
    primitive_type = wp.optix_get_primitive_type()
    error = wp.uint32(0)
    if primitive_type == wp.uint32(0x2501):
        quadratic_vertices = wp.optix_get_quadratic_bspline_vertex_data()
        if (
            wp.abs(quadratic_vertices[0, 1] + 1.0) > 1.0e-5
            or wp.abs(quadratic_vertices[2, 1] - 1.0) > 1.0e-5
            or wp.abs(quadratic_vertices[1, 3] - 0.25) > 1.0e-5
        ):
            error = wp.uint32(1)
    elif primitive_type == wp.uint32(0x2507):
        bezier_vertices = wp.optix_get_cubic_bezier_vertex_data()
        if (
            wp.abs(bezier_vertices[0, 1] + 1.5) > 1.0e-5
            or wp.abs(bezier_vertices[3, 1] - 1.5) > 1.0e-5
            or wp.abs(bezier_vertices[2, 3] - 0.25) > 1.0e-5
        ):
            error = wp.uint32(2)
    else:
        error = wp.uint32(4)
    params.output[0] = error


@wp.kernel(module="unique")
def callable_raygen(params: QueryParams):
    wp.optix_direct_call(wp.uint32(1))
    wp.optix_direct_call(wp.uint32(0))


@wp.kernel(module="unique")
def first_callable(params: QueryParams):
    params.output[0] = params.output[1] + wp.uint32(1)


@wp.kernel(module="unique")
def second_callable(params: QueryParams):
    params.output[1] = wp.uint32(84)


@wp.kernel(module="unique")
def exception_raygen(params: QueryParams):
    wp.optix_throw_exception(wp.int32(1234), wp.uint32(77), wp.uint32(88))


@wp.kernel(module="unique")
def exception_program(params: QueryParams):
    params.output[0] = wp.uint32(wp.optix_get_exception_code())
    params.output[1] = wp.optix_get_exception_detail_0()
    params.output[2] = wp.optix_get_exception_detail_1()


@wp.kernel(module="unique")
def unbound_miss(output: wp.array(dtype=wp.uint32)):
    output[0] = wp.uint32(1)


@wp.kernel(module="unique")
def retyped_miss(params: wp.array(dtype=wp.uint32)):
    params[0] = wp.uint32(1)


@wp.kernel(module="unique")
def tid_miss(params: QueryParams):
    i = wp.tid()
    params.output[i] = wp.uint32(1)


def _launch(pipeline, traversable, output):
    params = QueryParams()
    params.output = output
    params.traversable = wp.uint64(traversable)
    output.zero_()
    wo.launch(pipeline, dim=(1,), inputs=[params])
    return output.numpy()


def test_common_device_queries_on_gpu():
    device = wp.get_device("cuda")
    ctx = wo.create_context(device)
    vertices = wp.array(TRIANGLE_VERTICES, dtype=wp.vec3, device=device)
    indices = wp.array(TRIANGLE_INDICES, dtype=wp.uint32, device=device)
    gas = wo.build_accel(ctx, [wo.triangles(vertices, indices)], flags=RANDOM_ACCESS, compact=True, device=device)
    records = wp.empty(1, dtype=wo.Instance, device=device)
    wp.launch(write_rotated_instance, dim=1, inputs=[gas.handle, records], device=device)
    ias = wo.build_accel(ctx, [wo.instances(records)], device=device)

    single_level = wo.TraversableGraphFlags.ALLOW_SINGLE_LEVEL_INSTANCING
    hit = wo.HitGroup(closest_hit=query_closest_hit)
    pipeline = wo.create_pipeline(
        ctx, query_raygen, miss=[query_miss], hit_groups=[hit], payload_words=1, graph_flags=single_level
    )
    output = wp.zeros(15, dtype=wp.uint32, device=device)
    result = _launch(pipeline, ias.handle, output)
    assert result[0] == 77
    assert result[1] == 0
    assert result[2] == 0
    assert result[3] == int(wo.PrimitiveType.TRIANGLE)
    assert result[4] == 1
    assert result[5] == 0x5A
    # Pipelines without motion blur report time zero.
    assert result[6] == np.float32(0.0).view(np.uint32)
    assert tuple(result[7:9]) == (1, 0)
    assert int(result[9]) | (int(result[10]) << 32) == gas.handle
    assert result[11] == 0

    reorder_pipeline = wo.create_pipeline(
        ctx, reorder_raygen, miss=[query_miss], hit_groups=[hit], payload_words=1, graph_flags=single_level
    )
    reorder_result = _launch(reorder_pipeline, ias.handle, output)
    assert tuple(reorder_result[:5]) == tuple(result[:5])
    assert tuple(reorder_result[7:12]) == tuple(result[7:12])
    assert tuple(reorder_result[12:15]) == (1, 77, 0)

    # Refitting a dynamic GAS keeps its handle; the moved triangle leaves the ray.
    dynamic = wo.build_accel(
        ctx, [wo.triangles(vertices, indices)], flags=RANDOM_ACCESS | wo.BuildFlags.ALLOW_UPDATE, device=device
    )
    wp.launch(write_rotated_instance, dim=1, inputs=[dynamic.handle, records], device=device)
    ias = wo.build_accel(ctx, [wo.instances(records)], device=device)
    vertices.assign(TRIANGLE_VERTICES + np.array([10.0, 0.0, 0.0], dtype=np.float32))
    assert wo.update_accel(ctx, dynamic).handle == dynamic.handle
    assert _launch(pipeline, ias.handle, output)[0] == 0xFFFFFFFF


def test_callables_and_exception_programs_on_gpu():
    device = wp.get_device("cuda")
    ctx = wo.create_context(device)
    output = wp.zeros(3, dtype=wp.uint32, device=device)

    # Callables are indexed by their position in ``callables``.
    pipeline = wo.create_pipeline(ctx, callable_raygen, callables=[first_callable, second_callable])
    assert tuple(_launch(pipeline, 0, output)[:2]) == (85, 84)

    pipeline = wo.create_pipeline(ctx, exception_raygen, exception=exception_program)
    assert tuple(_launch(pipeline, 0, output)) == (1234, 77, 88)


def test_motion_blur_triangles_on_gpu():
    device = wp.get_device("cuda")
    ctx = wo.create_context(device)
    keys = [
        wp.array(TRIANGLE_VERTICES + np.array([0.0, 0.0, dz], dtype=np.float32), dtype=wp.vec3, device=device)
        for dz in (0.0, -0.5)
    ]
    indices = wp.array(TRIANGLE_INDICES, dtype=wp.uint32, device=device)
    motion = native.MotionOptions(numKeys=2, flags=native.MOTION_FLAG_NONE, timeBegin=0.0, timeEnd=1.0)
    gas = wo.build_accel(ctx, [wo.triangles(keys, indices)], motion=motion, device=device)
    pipeline = wo.create_pipeline(
        ctx,
        motion_raygen,
        miss=[query_miss],
        hit_groups=[wo.HitGroup(closest_hit=motion_closest_hit)],
        payload_words=1,
        motion_blur=True,
    )
    result = _launch(pipeline, gas.handle, wp.zeros(2, dtype=wp.uint32, device=device))
    assert result[0] == np.float32(0.25).view(np.uint32)
    # At time 0.25 the triangle has moved a quarter of the way to z = -0.5.
    assert result[1:].view(np.float32)[0] == pytest.approx(3.125)


def test_round_linear_curve_queries_on_gpu():
    device = wp.get_device("cuda")
    ctx = wo.create_context(device)
    vertices = wp.array([[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]], dtype=wp.vec3, device=device)
    widths = wp.array([0.25, 0.25], dtype=float, device=device)
    indices = wp.array([0], dtype=wp.uint32, device=device)
    gas = wo.build_accel(ctx, [wo.curves(vertices, widths, indices)], flags=RANDOM_ACCESS, device=device)
    pipeline = wo.create_pipeline(
        ctx,
        curve_raygen,
        miss=[query_miss],
        hit_groups=[wo.HitGroup(closest_hit=curve_closest_hit, builtin=wo.PrimitiveType.ROUND_LINEAR)],
        payload_words=1,
    )
    result = _launch(pipeline, gas.handle, wp.zeros(7, dtype=wp.uint32, device=device))
    floats = result.view(np.float32)
    assert result[0] == 1
    assert result[1] == int(wo.PrimitiveType.ROUND_LINEAR)
    assert 0.0 <= floats[2] <= 1.0
    assert floats[3] == pytest.approx(0.25)
    assert floats[4] == 0.0
    assert result[5] == 0
    assert floats[6] == 0.0


@pytest.mark.parametrize(
    ("curve_type", "control_points"),
    [
        (wo.PrimitiveType.ROUND_QUADRATIC_BSPLINE, [[0.0, -1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        (
            wo.PrimitiveType.ROUND_CUBIC_BEZIER,
            [[0.0, -1.5, 0.0], [0.0, -0.5, 0.0], [0.0, 0.5, 0.0], [0.0, 1.5, 0.0]],
        ),
    ],
)
def test_higher_order_curve_vertex_data_on_gpu(curve_type, control_points):
    device = wp.get_device("cuda")
    ctx = wo.create_context(device)
    vertices = wp.array(control_points, dtype=wp.vec3, device=device)
    widths = wp.full(len(control_points), 0.25, dtype=float, device=device)
    indices = wp.array([0], dtype=wp.uint32, device=device)
    gas = wo.build_accel(
        ctx, [wo.curves(vertices, widths, indices, curve_type=curve_type)], flags=RANDOM_ACCESS, device=device
    )
    pipeline = wo.create_pipeline(
        ctx,
        curve_raygen,
        miss=[query_miss],
        hit_groups=[wo.HitGroup(closest_hit=higher_order_curve_closest_hit, builtin=curve_type)],
        payload_words=1,
    )
    assert _launch(pipeline, gas.handle, wp.zeros(1, dtype=wp.uint32, device=device))[0] == 0


@pytest.mark.parametrize(
    ("miss", "message"),
    [
        (unbound_miss, "is not a raygen argument"),
        (retyped_miss, "type differs from raygen"),
        (tid_miss, "only available in raygen"),
    ],
)
def test_programs_must_bind_raygen_arguments(miss, message):
    ctx = wo.create_context("cuda")
    with pytest.raises(TypeError, match=message):
        wo.create_pipeline(ctx, query_raygen, miss=[miss])
