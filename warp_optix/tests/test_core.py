# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests of the core API: accel builds, pipelines, launches, and ray queries."""

import numpy as np
import pytest
import warp as wp

import warp_optix as wo

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not wp.is_cuda_available(), reason="requires a CUDA device")]

QUAD_VERTICES = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], dtype=np.float32)
QUAD_INDICES = np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)
# Counter-clockwise seen from +z: a ray travelling along -z hits the front face.
TRIANGLE_VERTICES = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
TRIANGLE_INDICES = np.array([0, 1, 2], dtype=np.int32)
DOWN = (0.0, 0.0, -1.0)


def triangle_gas(device, **kwargs):
    vertices = wp.array(TRIANGLE_VERTICES, dtype=wp.vec3, device=device)
    indices = wp.array(TRIANGLE_INDICES, dtype=wp.int32, device=device)
    return wo.Accel([wo.triangles(vertices, indices)], **kwargs)


@pytest.mark.parametrize(
    "values,dtype", [([0, 1, 4], wp.int32), ([0, 1, -1], wp.int32), ([0, 1, 2], wp.float32), ([0, 1], wp.int32)]
)
def test_triangle_indices_are_validated(values, dtype):
    vertices = wp.array(QUAD_VERTICES, dtype=wp.vec3, device="cuda")
    indices = wp.array(values, dtype=dtype, device="cuda")
    with pytest.raises((ValueError, TypeError), match="indices"):
        wo.triangles(vertices, indices)


@pytest.mark.parametrize(
    "curve_type,start",
    [
        (wo.PrimitiveType.ROUND_LINEAR, 3),
        (wo.PrimitiveType.ROUND_CUBIC_BSPLINE, 1),
        (wo.PrimitiveType.ROUND_QUADRATIC_BSPLINE, -1),
    ],
)
def test_curve_indices_are_validated(curve_type, start):
    vertices = wp.array(QUAD_VERTICES, dtype=wp.vec3, device="cuda")
    widths = wp.full(4, 0.1, dtype=wp.float32, device="cuda")
    indices = wp.array([start], dtype=wp.int32, device="cuda")
    with pytest.raises(ValueError, match="indices"):
        wo.curves(vertices, widths, indices, curve_type=curve_type)


@wp.kernel(module="unique")
def depth_raygen(
    scene: wp.uint64,
    offsets: wp.array[wp.vec3],
    depth: wp.array2d[float],
    instance: wp.array2d[int],
):
    world, pixel = wp.tid()
    x = (float(pixel % 8) + 0.5) / 4.0 - 1.0
    y = (float(pixel // 8) + 0.5) / 4.0 - 1.0
    origin = wp.vec3(x, y, 5.0) + offsets[world]
    query = wo.accel_query_ray(scene, origin, wp.vec3(0.0, 0.0, -1.0), 100.0)
    depth[world, pixel] = query.t
    instance[world, pixel] = -1
    if query.result:
        instance[world, pixel] = int(query.instance_id)


@wp.kernel(module="unique")
def write_instances(
    gas: wp.uint64,
    offsets: wp.array[wp.vec3],
    heights: wp.array[float],
    records: wp.array[wo.Instance],
):
    world = wp.tid()
    translation = offsets[world] + wp.vec3(0.0, 0.0, heights[world])
    records[world] = wo.make_instance(
        wp.identity(3, float), translation, gas, wp.uint32(world), wp.uint32(0), wp.uint32(0xFF), wp.uint32(0)
    )


def test_raygen_only_pipeline_over_device_instances():
    device = wp.get_device("cuda")
    vertices = wp.array(QUAD_VERTICES, dtype=wp.vec3, device=device)
    indices = wp.array(QUAD_INDICES, dtype=wp.int32, device=device)
    gas = wo.Accel([wo.triangles(vertices, indices)], compact=True, device=device)

    # Two worlds side by side; each world's quad sits at its own height.
    nworld = 2
    offsets = wp.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=wp.vec3, device=device)
    heights = wp.array([1.0, 2.0], dtype=float, device=device)
    records = wp.empty(nworld, dtype=wo.Instance, device=device)
    wp.launch(write_instances, dim=nworld, inputs=[gas.id, offsets, heights, records], device=device)
    ias = wo.Accel([wo.instances(records)], flags=wo.BuildFlags.ALLOW_UPDATE, device=device)

    pipeline = wo.Pipeline(depth_raygen, device=device)
    depth = wp.zeros((nworld, 64), dtype=float, device=device)
    instance = wp.zeros((nworld, 64), dtype=int, device=device)
    wo.launch(pipeline, dim=(nworld, 64), inputs=[ias.id, offsets, depth, instance])
    np.testing.assert_allclose(depth.numpy(), [[4.0] * 64, [3.0] * 64], atol=1e-5)
    np.testing.assert_array_equal(instance.numpy(), [[0] * 64, [1] * 64])

    # Move the quads on the device, refit in place, and launch again on another stream.
    stream = wp.Stream(device)
    heights.assign([0.5, -1.0])
    wp.launch(write_instances, dim=nworld, inputs=[gas.id, offsets, heights, records], device=device)
    wp.synchronize_device(device)
    buffer = ias._buffer
    assert ias.refit(stream=stream) is None
    assert ias._buffer is buffer
    wo.launch(pipeline, dim=(nworld, 64), inputs=[ias, offsets, depth, instance], stream=stream)
    wp.synchronize_stream(stream)
    np.testing.assert_allclose(depth.numpy(), [[4.5] * 64, [6.0] * 64], atol=1e-5)

    # A rebuild gives the same result.
    ias.rebuild(stream=stream)
    depth.zero_()
    wo.launch(pipeline, dim=(nworld, 64), inputs=[ias, offsets, depth, instance], stream=stream)
    wp.synchronize_stream(stream)
    np.testing.assert_allclose(depth.numpy(), [[4.5] * 64, [6.0] * 64], atol=1e-5)

    # OptiX work is rejected during capture before reaching OptiX, so the capture stays usable.
    with wp.ScopedCapture(stream=stream) as capture:
        with pytest.raises(RuntimeError, match="CUDA graphs"):
            wo.launch(pipeline, dim=(nworld, 64), inputs=[ias.id, offsets, depth, instance], stream=stream)
        with pytest.raises(RuntimeError, match="CUDA graphs"):
            ias.refit(stream=stream)
        with pytest.raises(RuntimeError, match="CUDA graphs"):
            wo.Accel([wo.instances(records)], stream=stream)
        depth.fill_(0.0)
    wp.capture_launch(capture.graph, stream=stream)
    wp.synchronize_stream(stream)
    assert not depth.numpy().any()


def test_refit_requires_allow_update_and_rejects_compacted():
    static = triangle_gas("cuda")
    with pytest.raises(ValueError, match="ALLOW_UPDATE"):
        static.refit()
    static.rebuild()  # Rebuilds do not need ALLOW_UPDATE.
    compacted = triangle_gas("cuda", compact=True)
    with pytest.raises(ValueError, match="compacted"):
        compacted.rebuild()
    with pytest.raises(ValueError, match="compacted"):
        triangle_gas("cuda", compact=True, flags=wo.BuildFlags.ALLOW_UPDATE)


@wp.func
def sphere_t(origin: wp.vec3, direction: wp.vec3, radius: float) -> float:
    b = wp.dot(origin, direction)
    c = wp.dot(origin, origin) - radius * radius
    disc = b * b - c
    if disc < 0.0:
        return -1.0
    return -b - wp.sqrt(disc)


@wp.kernel(module="unique")
def sphere_intersection(radii: wp.array[float]):
    t = sphere_t(wp.optix_get_object_ray_origin(), wp.optix_get_object_ray_direction(), radii[0])
    if t > 0.0:
        wp.optix_report_intersection(t, 0)


@wp.kernel(module="unique")
def sphere_raygen(scene: wp.uint64, radii: wp.array[float], depth: wp.array[float]):
    i = wp.tid()
    origin = wp.vec3(0.0, 0.0, 5.0 + float(i))
    depth[i] = wo.accel_query_ray(scene, origin, wp.vec3(0.0, 0.0, -1.0), 100.0).t


def test_custom_primitive_intersection_program_binds_raygen_arguments_by_name():
    device = wp.get_device("cuda")
    bounds = wp.array([[-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]], dtype=float, device=device)
    gas = wo.Accel([wo.aabbs(bounds=bounds)], device=device)
    pipeline = wo.Pipeline(sphere_raygen, hit_groups=[wo.HitGroup(intersection=sphere_intersection)], device=device)
    radii = wp.array([0.5], dtype=float, device=device)
    depth = wp.zeros(3, dtype=float, device=device)
    wo.launch(pipeline, dim=(3,), inputs=[gas.id, radii, depth])
    np.testing.assert_allclose(depth.numpy(), [4.5, 5.5, 6.5], atol=1e-5)


# ---------------------------------------------------------------------------
# aabbs(lowers, uppers) and aabbs(bounds=...)
# ---------------------------------------------------------------------------


@wp.kernel(module="unique")
def moving_sphere_intersection(centers: wp.array[wp.vec3]):
    center = centers[int(wp.optix_get_primitive_index())]
    t = sphere_t(wp.optix_get_object_ray_origin() - center, wp.optix_get_object_ray_direction(), 0.5)
    if t > 0.0:
        wp.optix_report_intersection(t, 0)


@wp.kernel(module="unique")
def moving_sphere_raygen(scene: wp.uint64, xs: wp.array[float], centers: wp.array[wp.vec3], depth: wp.array[float]):
    i = wp.tid()
    query = wo.accel_query_ray(scene, wp.vec3(xs[i], 0.0, 5.0), wp.vec3(0.0, 0.0, -1.0), 100.0)
    depth[i] = -1.0
    if query.result:
        depth[i] = query.t


def test_aabbs_from_lowers_uppers_and_packed_bounds_hit_and_refit():
    device = wp.get_device("cuda")
    centers = wp.array([[0.0, 0.0, 0.0], [3.0, 0.0, -1.0]], dtype=wp.vec3, device=device)
    lowers = wp.array([[-0.5, -0.5, -0.5], [2.5, -0.5, -1.5]], dtype=wp.vec3, device=device)
    uppers = wp.array([[0.5, 0.5, 0.5], [3.5, 0.5, -0.5]], dtype=wp.vec3, device=device)
    bounds = wp.array(np.concatenate([lowers.numpy(), uppers.numpy()], axis=1), dtype=float, device=device)

    separate = wo.Accel([wo.aabbs(lowers, uppers)], flags=wo.BuildFlags.ALLOW_UPDATE, device=device)
    packed = wo.Accel([wo.aabbs(bounds=bounds)], device=device)
    pipeline = wo.Pipeline(
        moving_sphere_raygen, hit_groups=[wo.HitGroup(intersection=moving_sphere_intersection)], device=device
    )
    xs = wp.array([0.0, 3.0, 6.0, 9.0], dtype=float, device=device)
    depth = wp.zeros(4, dtype=float, device=device)
    expected = [4.5, 5.5, -1.0, -1.0]
    for accel in (separate, packed):
        depth.zero_()
        wo.launch(pipeline, dim=4, inputs=[accel, xs, centers, depth])
        np.testing.assert_allclose(depth.numpy(), expected, atol=1e-5)

    # Move both spheres by +6 in x. Without repacking lowers/uppers into the BVH on refit,
    # the rays at x = 6 and 9 would miss the old boxes.
    shift = np.array([6.0, 0.0, 0.0], dtype=np.float32)
    centers.assign(centers.numpy() + shift)
    lowers.assign(lowers.numpy() + shift)
    uppers.assign(uppers.numpy() + shift)
    separate.refit()
    depth.zero_()
    wo.launch(pipeline, dim=4, inputs=[separate, xs, centers, depth])
    np.testing.assert_allclose(depth.numpy(), [-1.0, -1.0, 4.5, 5.5], atol=1e-5)


def test_aabbs_argument_validation():
    lowers = wp.zeros(2, dtype=wp.vec3, device="cuda")
    uppers = wp.ones(2, dtype=wp.vec3, device="cuda")
    bounds = wp.zeros((2, 6), dtype=float, device="cuda")
    with pytest.raises(ValueError, match="lowers and uppers"):
        wo.aabbs(lowers)
    with pytest.raises(ValueError, match="not both"):
        wo.aabbs(lowers, uppers, bounds=bounds)
    with pytest.raises(ValueError, match="equal size"):
        wo.aabbs(lowers, wp.ones(3, dtype=wp.vec3, device="cuda"))
    with pytest.raises(ValueError, match="sextuplets"):
        wo.aabbs(bounds=wp.zeros(5, dtype=float, device="cuda"))


# ---------------------------------------------------------------------------
# make_instance overloads
# ---------------------------------------------------------------------------


@wp.kernel(module="unique")
def write_overload_instances(gas: wp.uint64, records: wp.array[wo.Instance]):
    # 0: rigid wp.transform, rotated 90 degrees about +z and lifted to z = 1.
    rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.5 * wp.pi)
    records[0] = wo.make_instance(wp.transform(wp.vec3(0.0, 0.0, 1.0), rotation), gas, 10)
    # 1: affine wp.mat44 scaling x and y by 2, translated to (10, 0, 2).
    affine = wp.mat44(2.0, 0.0, 0.0, 10.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 1.0)
    records[1] = wo.make_instance(affine, gas, wp.int64(11), wp.uint8(0), 0xFF, wo.InstanceFlags.NONE)
    # 2: rotation mat33 (90 degrees about +z) plus translation (20, 0, 3), all optional arguments omitted.
    turn = wp.mat33(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    records[2] = wo.make_instance(turn, wp.vec3(20.0, 0.0, 3.0), gas)


@wp.kernel(module="unique")
def query_points_raygen(
    scene: wp.uint64,
    points: wp.array[wp.vec3],
    depth: wp.array[float],
    ids: wp.array[int],
    any_hit: wp.array[int],
):
    i = wp.tid()
    query = wo.accel_query_ray(scene, points[i], wp.vec3(0.0, 0.0, -1.0), 100.0)
    depth[i] = -1.0
    ids[i] = -1
    if query.result:
        depth[i] = query.t
        ids[i] = int(query.instance_id)
    any_hit[i] = 0
    if wo.accel_query_ray_anyhit(scene, points[i], wp.vec3(0.0, 0.0, -1.0), 100.0):
        any_hit[i] = 1


def test_make_instance_overloads_place_instances():
    device = wp.get_device("cuda")
    gas = triangle_gas(device)
    records = wp.empty(3, dtype=wo.Instance, device=device)
    wp.launch(write_overload_instances, dim=1, inputs=[gas.id, records], device=device)
    host = records.numpy()
    np.testing.assert_array_equal(host["instance_id"], [10, 11, 0])
    np.testing.assert_array_equal(host["visibility_mask"], [0xFF, 0xFF, 0xFF])
    np.testing.assert_array_equal(host["traversable"], [gas.id] * 3)
    ias = wo.Accel([wo.instances(records)], device=device)

    # Each pair: a point covered only if the instance's transform was applied, and one that is not covered.
    points = [
        (-0.25, 0.25, 10.0),  # rotated triangle covers x < 0
        (0.25, 0.25, 10.0),  # the unrotated triangle would cover this one
        (11.5, 0.25, 10.0),  # x scaled by 2: covers x in [10, 12]
        (11.5, 1.5, 10.0),  # outside the scaled triangle (x + y > 12 after scaling)
        (19.75, 0.25, 10.0),  # mat33 rotation: covers x in [19, 20]
        (20.25, 0.25, 10.0),
    ]
    points = wp.array(points, dtype=wp.vec3, device=device)
    depth = wp.zeros(6, dtype=float, device=device)
    ids = wp.zeros(6, dtype=int, device=device)
    any_hit = wp.zeros(6, dtype=int, device=device)
    pipeline = wo.Pipeline(query_points_raygen, device=device)
    wo.launch(pipeline, dim=6, inputs=[ias, points, depth, ids, any_hit])
    np.testing.assert_allclose(depth.numpy(), [9.0, -1.0, 8.0, -1.0, 7.0, -1.0], atol=1e-5)
    np.testing.assert_array_equal(ids.numpy(), [10, -1, 11, -1, 0, -1])
    np.testing.assert_array_equal(any_hit.numpy(), [1, 0, 1, 0, 1, 0])


# ---------------------------------------------------------------------------
# accel_query_ray / accel_query_ray_anyhit
# ---------------------------------------------------------------------------


@wp.kernel(module="unique")
def triangle_query_raygen(
    scene: wp.uint64,
    min_ts: wp.array[float],
    max_ts: wp.array[float],
    results: wp.array2d[float],
    any_hit: wp.array[int],
):
    i = wp.tid()
    start = wp.vec3(0.2, 0.3, 5.0)
    query = wo.accel_query_ray(scene, start, wp.vec3(0.0, 0.0, -1.0), max_ts[i], min_ts[i])
    results[i, 0] = 0.0
    if query.result:
        results[i, 0] = 1.0
        results[i, 1] = query.t
        results[i, 2] = query.u
        results[i, 3] = query.v
        results[i, 4] = float(query.primitive_index)
    any_hit[i] = 0
    if wo.accel_query_ray_anyhit(scene, start, wp.vec3(0.0, 0.0, -1.0), max_ts[i], min_ts[i], 0xFF):
        any_hit[i] = 1


def test_accel_query_ray_reports_distance_barycentrics_and_respects_interval():
    device = wp.get_device("cuda")
    gas = triangle_gas(device)
    min_ts = wp.array([0.0, 0.0, 5.5, 4.9], dtype=float, device=device)
    max_ts = wp.array([100.0, 4.0, 100.0, 5.1], dtype=float, device=device)
    results = wp.full((4, 5), -7.0, dtype=float, device=device)
    any_hit = wp.zeros(4, dtype=int, device=device)
    pipeline = wo.Pipeline(triangle_query_raygen, device=device)
    wo.launch(pipeline, dim=4, inputs=[gas, min_ts, max_ts, results, any_hit])
    out = results.numpy()
    # The hit point (0.2, 0.3) = (1 - u - v) * v0 + u * v1 + v * v2.
    np.testing.assert_allclose(out[0], [1.0, 5.0, 0.2, 0.3, 0.0], atol=1e-5)
    np.testing.assert_allclose(out[3], [1.0, 5.0, 0.2, 0.3, 0.0], atol=1e-5)
    assert out[1, 0] == 0.0 and out[2, 0] == 0.0
    np.testing.assert_array_equal(any_hit.numpy(), [1, 0, 0, 1])


@wp.kernel(module="unique")
def ray_flag_query_raygen(scene: wp.uint64, hits: wp.array[int]):
    start = wp.vec3(0.2, 0.3, 5.0)
    direction = wp.vec3(0.0, 0.0, -1.0)
    hits[0] = int(wo.accel_query_ray(scene, start, direction, 100.0, 0.0, 0xFF, wo.RayFlags.NONE).result)
    hits[1] = int(
        wo.accel_query_ray(scene, start, direction, 100.0, 0.0, 0xFF, wo.RayFlags.CULL_FRONT_FACING_TRIANGLES).result
    )
    hits[2] = int(
        wo.accel_query_ray(scene, start, direction, 100.0, 0.0, 0xFF, wo.RayFlags.CULL_BACK_FACING_TRIANGLES).result
    )
    flags = wo.RayFlags.CULL_FRONT_FACING_TRIANGLES | wo.RayFlags.DISABLE_ANYHIT
    hits[3] = int(wo.accel_query_ray(scene, start, direction, 100.0, 0.0, 0xFF, flags).result)


def test_ray_flags_members_pass_directly_to_accel_query_ray():
    gas = triangle_gas("cuda")
    hits = wp.full(4, -1, dtype=int, device="cuda")
    wo.launch(wo.Pipeline(ray_flag_query_raygen), dim=1, inputs=[gas, hits])
    np.testing.assert_array_equal(hits.numpy(), [1, 0, 1, 0])


# ---------------------------------------------------------------------------
# optix_trace with payload and programs
# ---------------------------------------------------------------------------


@wp.struct
class ColorPayload:
    color: wp.vec3


@wp.kernel(module="unique")
def shaded_raygen(scene: wp.uint64, image: wp.array[wp.vec3]):
    i = wp.tid()
    payload = ColorPayload()
    origin = wp.vec3(-1.5 + float(i), 0.0, 5.0)
    wp.optix_trace(
        scene,
        origin,
        wp.vec3(0.0, 0.0, -1.0),
        100.0,
        payload,
        0.0,
        0.0,
        wp.uint32(0xFF),
        wp.uint32(0),
        wp.uint32(0),
        wp.uint32(1),
        wp.uint32(0),
    )
    image[i] = payload.color


@wp.kernel(module="unique")
def shaded_short_raygen(scene: wp.uint64, image: wp.array[wp.vec3]):
    i = wp.tid()
    payload = ColorPayload()
    # All optional trailing arguments omitted: min_t=0, time=0, mask=0xFF, flags=NONE, sbt 0/1, miss 0.
    wp.optix_trace(scene, wp.vec3(-1.5 + float(i), 0.0, 5.0), wp.vec3(0.0, 0.0, -1.0), 100.0, payload)
    image[i] = payload.color


@wp.kernel(module="unique")
def shaded_culled_raygen(scene: wp.uint64, image: wp.array[wp.vec3]):
    i = wp.tid()
    payload = ColorPayload()
    # Plain int literals for the mask and a RayFlags member for the flags, without casts.
    wp.optix_trace(
        scene,
        wp.vec3(-1.5 + float(i), 0.0, 5.0),
        wp.vec3(0.0, 0.0, -1.0),
        100.0,
        payload,
        0.0,
        0.0,
        0xFF,
        wo.RayFlags.CULL_FRONT_FACING_TRIANGLES,
    )
    image[i] = payload.color


@wp.kernel(module="unique")
def shaded_closest_hit():
    payload = ColorPayload()
    payload.color = wp.vec3(wp.optix_get_ray_tmax(), 1.0, 0.0)
    wp.optix_store_payload(payload)


@wp.kernel(module="unique")
def shaded_miss():
    payload = ColorPayload()
    payload.color = wp.vec3(0.0, 0.0, 1.0)
    wp.optix_store_payload(payload)


@pytest.mark.parametrize(
    "raygen,expected",
    [
        (shaded_raygen, [[0, 0, 1], [5, 1, 0], [5, 1, 0], [0, 0, 1]]),
        (shaded_short_raygen, [[0, 0, 1], [5, 1, 0], [5, 1, 0], [0, 0, 1]]),
        (shaded_culled_raygen, [[0, 0, 1]] * 4),
    ],
    ids=["all-arguments", "omitted-trailing", "ray-flags"],
)
def test_trace_with_payload_closest_hit_and_miss_programs(raygen, expected):
    device = wp.get_device("cuda")
    vertices = wp.array(QUAD_VERTICES, dtype=wp.vec3, device=device)
    indices = wp.array(QUAD_INDICES, dtype=wp.int32, device=device)
    gas = wo.Accel([wo.triangles(vertices, indices)], device=device)
    pipeline = wo.Pipeline(
        raygen,
        miss=[shaded_miss],
        hit_groups=[wo.HitGroup(closest_hit=shaded_closest_hit)],
        payload_words=3,
        device=device,
    )
    image = wp.zeros(4, dtype=wp.vec3, device=device)
    wo.launch(pipeline, dim=(4,), inputs=[gas.id, image])
    np.testing.assert_allclose(image.numpy(), expected, atol=1e-5)


def test_ray_flag_and_hit_kind_values_match_optix_headers():
    # Values from optix_types.h.
    assert int(wo.RayFlags.NONE) == 0
    assert int(wo.RayFlags.DISABLE_ANYHIT) == 1
    assert int(wo.RayFlags.TERMINATE_ON_FIRST_HIT) == 4
    assert int(wo.RayFlags.CULL_FRONT_FACING_TRIANGLES) == 32
    assert int(wo.HitKind.TRIANGLE_FRONT_FACE) == 0xFE
    assert int(wo.HitKind.TRIANGLE_BACK_FACE) == 0xFF


@wp.kernel(module="unique")
def curve_query_raygen(scene: wp.uint64, results: wp.array[float]):
    query = wo.accel_query_ray(scene, wp.vec3(0.25, 0.0, 1.0), wp.vec3(0.0, 0.0, -1.0), 100.0)
    results[0] = float(query.result)
    if query.result:
        results[1] = query.t
        results[2] = query.u
        results[3] = query.v


def test_accel_query_ray_reports_curve_parameter():
    device = wp.get_device("cuda")
    points = wp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=wp.vec3, device=device)
    widths = wp.full(2, 0.1, dtype=float, device=device)
    indices = wp.array([0], dtype=wp.int32, device=device)
    gas = wo.Accel([wo.curves(points, widths, indices)], device=device)
    pipeline = wo.Pipeline(
        curve_query_raygen, hit_groups=[wo.HitGroup(builtin=wo.PrimitiveType.ROUND_LINEAR)], device=device
    )
    results = wp.full(4, -7.0, dtype=float, device=device)
    wo.launch(pipeline, dim=1, inputs=[gas, results])
    np.testing.assert_allclose(results.numpy(), [1.0, 0.9, 0.25, 0.0], atol=1e-4)


@pytest.mark.parametrize(
    "curve_type,points",
    [
        (wo.PrimitiveType.ROUND_QUADRATIC_BSPLINE_ROCAPS, [[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        (
            wo.PrimitiveType.ROUND_CUBIC_BSPLINE_ROCAPS,
            [[-2.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        ),
    ],
)
def test_rocaps_curves_build_and_trace(curve_type, points):
    device = wp.get_device("cuda")
    points = wp.array(points, dtype=wp.vec3, device=device)
    widths = wp.full(points.size, 0.1, dtype=float, device=device)
    indices = wp.array([0], dtype=wp.int32, device=device)
    gas = wo.Accel([wo.curves(points, widths, indices, curve_type=curve_type)], device=device)
    pipeline = wo.Pipeline(curve_query_raygen, hit_groups=[wo.HitGroup(builtin=curve_type)], device=device)
    results = wp.full(4, -7.0, dtype=float, device=device)
    wo.launch(pipeline, dim=1, inputs=[gas, results])
    assert results.numpy()[0] == 1.0
    np.testing.assert_allclose(results.numpy()[1], 0.9, atol=1e-3)


def test_invalid_curve_types_hit_groups_and_miss_records_are_rejected():
    vertices = wp.array(QUAD_VERTICES, dtype=wp.vec3, device="cuda")
    widths = wp.full(4, 0.1, dtype=wp.float32, device="cuda")
    # Quadratic segments span three points, so the last segment of four points starts at index 1.
    indices = wp.array([1], dtype=wp.int32, device="cuda")
    for curve_type in (wo.PrimitiveType.ROUND_QUADRATIC_BSPLINE, wo.PrimitiveType.ROUND_QUADRATIC_BSPLINE_ROCAPS):
        wo.curves(vertices, widths, indices, curve_type=curve_type)
    with pytest.raises(ValueError, match="curve PrimitiveType"):
        wo.curves(vertices, widths, indices, curve_type=wo.PrimitiveType.TRIANGLE)
    with pytest.raises(ValueError, match="curve PrimitiveType"):
        wo.Pipeline(curve_query_raygen, hit_groups=[wo.HitGroup(builtin=wo.PrimitiveType.TRIANGLE)])
    with pytest.raises(ValueError, match="cannot be combined"):
        wo.Pipeline(
            curve_query_raygen,
            hit_groups=[wo.HitGroup(intersection=curve_query_raygen, builtin=wo.PrimitiveType.ROUND_LINEAR)],
        )
    with pytest.raises(ValueError, match="miss must contain at least one entry"):
        wo.Pipeline(curve_query_raygen, miss=[])
    with pytest.raises(ValueError, match="at least one build input"):
        wo.Accel([], device="cuda")


def test_empty_build_inputs_build_refit_and_miss():
    # Empty geometry builds a structure nothing intersects, e.g. a dynamic mesh or contact set with no elements yet.
    device = wp.get_device("cuda")
    vertices = wp.array(QUAD_VERTICES, dtype=wp.vec3, device=device)
    none = wp.empty(0, dtype=wp.int32, device=device)
    no_points = wp.empty(0, dtype=wp.vec3, device=device)
    inputs = [
        wo.triangles(vertices, none),
        wo.curves(vertices, wp.full(4, 0.1, dtype=float, device=device), none),
        wo.aabbs(no_points, no_points),
        wo.instances(wp.empty(0, dtype=wo.Instance, device=device)),
    ]
    pipeline = wo.Pipeline(curve_query_raygen, device=device)
    results = wp.full(4, -7.0, dtype=float, device=device)
    for build_input in inputs:
        accel = wo.Accel([build_input], flags=wo.BuildFlags.ALLOW_UPDATE)
        accel.refit()
        wo.launch(pipeline, dim=1, inputs=[accel, results])
        assert results.numpy()[0] == 0.0


@wp.kernel(module="unique")
def write_first(out: wp.array[float]):
    out[0] = 10.0


@wp.kernel(module="unique")
def write_second(out: wp.array[float]):
    out[1] = 20.0


@wp.kernel(module="unique")
def call_both(out: wp.array[float]):
    wp.optix_direct_call(0)
    wp.optix_continuation_call(1)


def test_continuation_callables_follow_direct_callables():
    out = wp.empty(2, dtype=float, device="cuda")
    pipeline = wo.Pipeline(call_both, callables=[write_first], continuation_callables=[write_second])
    wo.launch(pipeline, 1, [out])
    np.testing.assert_array_equal(out.numpy(), [10.0, 20.0])
