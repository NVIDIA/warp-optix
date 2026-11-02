# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests of the core API: accel builds, pipelines, and launches."""

import numpy as np
import pytest
import warp as wp

import warp_optix as wo

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not wp.is_cuda_available(), reason="requires a CUDA device")]

QUAD_VERTICES = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], dtype=np.float32)
QUAD_INDICES = np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)


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
    hit = wo.closest_hit(scene, origin, wp.vec3(0.0, 0.0, -1.0), 0.0, 100.0, wp.uint32(0xFF), wp.uint32(0))
    depth[world, pixel] = hit.t
    instance[world, pixel] = int(hit.instance_id) if hit.t != wo.MISS else -1


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
    ctx = wo.create_context(device)
    vertices = wp.array(QUAD_VERTICES, dtype=wp.vec3, device=device)
    indices = wp.array(QUAD_INDICES, dtype=wp.int32, device=device)
    gas = wo.build_accel(ctx, [wo.triangles(vertices, indices)], compact=True, device=device)

    # Two worlds side by side; each world's quad sits at its own height.
    nworld = 2
    offsets = wp.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=wp.vec3, device=device)
    heights = wp.array([1.0, 2.0], dtype=float, device=device)
    records = wp.empty(nworld, dtype=wo.Instance, device=device)
    wp.launch(write_instances, dim=nworld, inputs=[gas.handle, offsets, heights, records], device=device)
    ias = wo.build_accel(ctx, [wo.instances(records)], flags=wo.BuildFlags.ALLOW_UPDATE, device=device)

    pipeline = wo.create_pipeline(ctx, depth_raygen, device=device)
    depth = wp.zeros((nworld, 64), dtype=float, device=device)
    instance = wp.zeros((nworld, 64), dtype=int, device=device)
    wo.launch(pipeline, dim=(nworld, 64), inputs=[ias.handle, offsets, depth, instance])
    np.testing.assert_allclose(depth.numpy(), [[4.0] * 64, [3.0] * 64], atol=1e-5)
    np.testing.assert_array_equal(instance.numpy(), [[0] * 64, [1] * 64])

    # Move the quads on the device, refit without reallocating, and launch again on another stream.
    stream = wp.Stream(device)
    heights.assign([0.5, -1.0])
    wp.launch(write_instances, dim=nworld, inputs=[gas.handle, offsets, heights, records], device=device)
    wp.synchronize_device(device)
    ias = wo.update_accel(ctx, ias, stream=stream)
    wo.launch(pipeline, dim=(nworld, 64), inputs=[ias.handle, offsets, depth, instance], stream=stream)
    wp.synchronize_stream(stream)
    np.testing.assert_allclose(depth.numpy(), [[4.5] * 64, [6.0] * 64], atol=1e-5)

    # OptiX work is rejected during capture before reaching OptiX, so the capture stays usable.
    with wp.ScopedCapture(stream=stream) as capture:
        with pytest.raises(RuntimeError, match="CUDA graphs"):
            wo.launch(pipeline, dim=(nworld, 64), inputs=[ias.handle, offsets, depth, instance], stream=stream)
        with pytest.raises(RuntimeError, match="CUDA graphs"):
            wo.update_accel(ctx, ias, stream=stream)
        depth.fill_(0.0)
    wp.capture_launch(capture.graph, stream=stream)
    wp.synchronize_stream(stream)
    assert not depth.numpy().any()
    wo.destroy_context(ctx, device)


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
        wp.optix_report_intersection(t, wp.uint32(0))


@wp.kernel(module="unique")
def sphere_raygen(scene: wp.uint64, radii: wp.array[float], depth: wp.array[float]):
    i = wp.tid()
    origin = wp.vec3(0.0, 0.0, 5.0 + float(i))
    depth[i] = wo.closest_hit(scene, origin, wp.vec3(0.0, 0.0, -1.0), 0.0, 100.0, wp.uint32(0xFF), wp.uint32(0)).t


def test_custom_primitive_intersection_program_binds_raygen_arguments_by_name():
    device = wp.get_device("cuda")
    ctx = wo.create_context(device)
    bounds = wp.array([[-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]], dtype=float, device=device)
    gas = wo.build_accel(ctx, [wo.aabbs(bounds)], device=device)
    pipeline = wo.create_pipeline(
        ctx, sphere_raygen, hit_groups=[wo.HitGroup(intersection=sphere_intersection)], device=device
    )
    radii = wp.array([0.5], dtype=float, device=device)
    depth = wp.zeros(3, dtype=float, device=device)
    wo.launch(pipeline, dim=(3,), inputs=[gas.handle, radii, depth])
    np.testing.assert_allclose(depth.numpy(), [4.5, 5.5, 6.5], atol=1e-5)


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
        0.0,
        100.0,
        0.0,
        wp.uint32(0xFF),
        wp.uint32(0),
        wp.uint32(0),
        wp.uint32(1),
        wp.uint32(0),
        payload,
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


def test_trace_with_payload_closest_hit_and_miss_programs():
    device = wp.get_device("cuda")
    ctx = wo.create_context(device)
    vertices = wp.array(QUAD_VERTICES, dtype=wp.vec3, device=device)
    indices = wp.array(QUAD_INDICES, dtype=wp.int32, device=device)
    gas = wo.build_accel(ctx, [wo.triangles(vertices, indices)], device=device)
    pipeline = wo.create_pipeline(
        ctx,
        shaded_raygen,
        miss=[shaded_miss],
        hit_groups=[wo.HitGroup(closest_hit=shaded_closest_hit)],
        payload_words=3,
        device=device,
    )
    image = wp.zeros(4, dtype=wp.vec3, device=device)
    wo.launch(pipeline, dim=(4,), inputs=[gas.handle, image])
    np.testing.assert_allclose(image.numpy(), [[0, 0, 1], [5, 1, 0], [5, 1, 0], [0, 0, 1]], atol=1e-5)
