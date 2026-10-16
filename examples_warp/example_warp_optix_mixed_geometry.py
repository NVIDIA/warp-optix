# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render triangle, native curve, and analytical custom geometry together."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
import warp as wp

import warp_optix as wo


@wp.struct
class Payload:
    red: wp.uint32
    green: wp.uint32
    blue: wp.uint32


@wp.func
def _to_u8(value: float) -> wp.uint32:
    return wp.uint32(wp.clamp(value * 255.0, 0.0, 255.0))


@wp.func
def _store_color(color: wp.vec3):
    wp.optix_set_payload_0(_to_u8(color[0]))
    wp.optix_set_payload_1(_to_u8(color[1]))
    wp.optix_set_payload_2(_to_u8(color[2]))


@wp.func
def _shade(normal: wp.vec3, albedo: wp.vec3) -> wp.vec3:
    light = wp.normalize(wp.vec3(-0.4, 0.8, 0.6))
    intensity = 0.18 + 0.82 * wp.max(wp.dot(wp.normalize(normal), light), 0.0)
    return albedo * intensity


@wp.func
def _background(direction: wp.vec3) -> wp.vec3:
    scale = 2.5 / wp.max(-direction[2], 0.001)
    x = direction[0] * scale
    y = direction[1] * scale
    key = wp.exp(-0.85 * (0.65 * x * x + (y - 0.15) * (y - 0.15)))
    fill = wp.exp(-2.8 * ((x + 0.95) * (x + 0.95) + (y - 0.6) * (y - 0.6)))
    floor = wp.exp(-3.2 * (0.55 * x * x + (y + 0.9) * (y + 0.9)))
    return (
        wp.vec3(0.72, 0.75, 0.80)
        + key * wp.vec3(0.27, 0.245, 0.18)
        + fill * wp.vec3(0.025, 0.045, 0.08)
        + floor * wp.vec3(0.05, 0.035, 0.015)
    )


@wp.kernel(module="unique")
def raygen(
    scene: wp.uint64,
    curve_vertices: wp.array[wp.vec3],
    curve_indices: wp.array[wp.uint32],
    image: wp.array2d[wp.uint32],
):
    y, x = wp.tid()
    height = image.shape[0]
    width = image.shape[1]
    sx = (2.0 * (float(x) + 0.5) / float(width) - 1.0) * float(width) / float(height)
    sy = 2.0 * (float(y) + 0.5) / float(height) - 1.0
    origin = wp.vec3(0.0, 0.0, 4.5)
    direction = wp.normalize(wp.vec3(sx, sy, -3.2))

    payload = Payload()
    wp.optix_trace(
        scene,
        origin,
        direction,
        0.001,
        1.0e16,
        0.0,
        wp.uint32(255),
        wp.uint32(0),
        wp.uint32(0),
        wp.uint32(1),
        wp.uint32(0),
        payload,
    )
    image[y, x] = (
        wp.uint32(255) << wp.uint32(24) | payload.blue << wp.uint32(16) | payload.green << wp.uint32(8) | payload.red
    )


@wp.kernel(module="unique")
def miss():
    _store_color(_background(wp.normalize(wp.optix_get_world_ray_direction())))


@wp.kernel(module="unique")
def triangle_closest_hit():
    barycentrics = wp.optix_get_triangle_barycentrics()
    vertices = wp.optix_get_triangle_vertex_data()
    object_normal = wp.normalize(wp.cross(vertices[1] - vertices[0], vertices[2] - vertices[0]))
    normal = wp.optix_transform_normal_from_object_to_world_space(object_normal)
    light = wp.normalize(wp.vec3(-0.4, 0.8, 0.6))
    panel = 0.955 + 0.012 * barycentrics[0] + 0.018 * wp.max(wp.dot(wp.normalize(normal), light), 0.0)
    _store_color(_background(wp.normalize(wp.optix_get_world_ray_direction())) * panel)


@wp.func
def _report_hit(t: float, normal: wp.vec3):
    if t >= wp.optix_get_ray_tmin() and t <= wp.optix_get_ray_tmax():
        wp.optix_report_intersection(
            t,
            wp.uint32(0),
            wp.float_to_uint32(normal[0]),
            wp.float_to_uint32(normal[1]),
            wp.float_to_uint32(normal[2]),
        )


@wp.func
def _intersect_sphere(origin: wp.vec3, direction: wp.vec3, center: wp.vec3, radius: float):
    offset = origin - center
    a = wp.dot(direction, direction)
    half_b = wp.dot(offset, direction)
    c = wp.dot(offset, offset) - radius * radius
    discriminant = half_b * half_b - a * c
    if discriminant >= 0.0:
        root = wp.sqrt(discriminant)
        t0 = (-half_b - root) / a
        t1 = (-half_b + root) / a
        _report_hit(t0, (offset + t0 * direction) / radius)
        _report_hit(t1, (offset + t1 * direction) / radius)


@wp.func
def _intersect_cylinder_side(origin: wp.vec3, direction: wp.vec3, radius: float, half_height: float):
    a = direction[0] * direction[0] + direction[2] * direction[2]
    half_b = origin[0] * direction[0] + origin[2] * direction[2]
    c = origin[0] * origin[0] + origin[2] * origin[2] - radius * radius
    discriminant = half_b * half_b - a * c
    if discriminant >= 0.0 and wp.abs(a) > 1.0e-8:
        root = wp.sqrt(discriminant)
        t0 = (-half_b - root) / a
        t1 = (-half_b + root) / a
        p0 = origin + t0 * direction
        p1 = origin + t1 * direction
        if wp.abs(p0[1]) <= half_height:
            _report_hit(t0, wp.normalize(wp.vec3(p0[0], 0.0, p0[2])))
        if wp.abs(p1[1]) <= half_height:
            _report_hit(t1, wp.normalize(wp.vec3(p1[0], 0.0, p1[2])))


@wp.func
def _intersect_cylinder(origin: wp.vec3, direction: wp.vec3, radius: float, half_height: float):
    _intersect_cylinder_side(origin, direction, radius, half_height)
    if wp.abs(direction[1]) > 1.0e-8:
        bottom_t = (-half_height - origin[1]) / direction[1]
        bottom = origin + bottom_t * direction
        if bottom[0] * bottom[0] + bottom[2] * bottom[2] <= radius * radius:
            _report_hit(bottom_t, wp.vec3(0.0, -1.0, 0.0))
        top_t = (half_height - origin[1]) / direction[1]
        top = origin + top_t * direction
        if top[0] * top[0] + top[2] * top[2] <= radius * radius:
            _report_hit(top_t, wp.vec3(0.0, 1.0, 0.0))


@wp.func
def _intersect_cone(origin: wp.vec3, direction: wp.vec3, radius: float, half_height: float):
    slope = radius / (2.0 * half_height)
    slope2 = slope * slope
    q = half_height - origin[1]
    a = direction[0] * direction[0] + direction[2] * direction[2] - slope2 * direction[1] * direction[1]
    half_b = origin[0] * direction[0] + origin[2] * direction[2] + slope2 * q * direction[1]
    c = origin[0] * origin[0] + origin[2] * origin[2] - slope2 * q * q
    discriminant = half_b * half_b - a * c
    if discriminant >= 0.0 and wp.abs(a) > 1.0e-8:
        root = wp.sqrt(discriminant)
        t0 = (-half_b - root) / a
        t1 = (-half_b + root) / a
        p0 = origin + t0 * direction
        p1 = origin + t1 * direction
        if p0[1] >= -half_height and p0[1] <= half_height:
            normal0 = wp.vec3(p0[0], slope2 * (half_height - p0[1]), p0[2])
            _report_hit(t0, wp.normalize(normal0))
        if p1[1] >= -half_height and p1[1] <= half_height:
            normal1 = wp.vec3(p1[0], slope2 * (half_height - p1[1]), p1[2])
            _report_hit(t1, wp.normalize(normal1))
    if wp.abs(direction[1]) > 1.0e-8:
        cap_t = (-half_height - origin[1]) / direction[1]
        cap = origin + cap_t * direction
        if cap[0] * cap[0] + cap[2] * cap[2] <= radius * radius:
            _report_hit(cap_t, wp.vec3(0.0, -1.0, 0.0))


@wp.func
def _intersect_box(origin: wp.vec3, direction: wp.vec3, half_extent: float):
    near_t = -1.0e16
    far_t = 1.0e16
    near_normal = wp.vec3(0.0)
    far_normal = wp.vec3(0.0)
    valid = True
    for axis in range(3):
        if wp.abs(direction[axis]) < 1.0e-8:
            if origin[axis] < -half_extent or origin[axis] > half_extent:
                valid = False
        else:
            t0 = (-half_extent - origin[axis]) / direction[axis]
            t1 = (half_extent - origin[axis]) / direction[axis]
            n0 = wp.vec3(0.0)
            n1 = wp.vec3(0.0)
            if axis == 0:
                n0 = wp.vec3(-1.0, 0.0, 0.0)
                n1 = wp.vec3(1.0, 0.0, 0.0)
            elif axis == 1:
                n0 = wp.vec3(0.0, -1.0, 0.0)
                n1 = wp.vec3(0.0, 1.0, 0.0)
            else:
                n0 = wp.vec3(0.0, 0.0, -1.0)
                n1 = wp.vec3(0.0, 0.0, 1.0)
            if t0 > t1:
                swap_t = t0
                t0 = t1
                t1 = swap_t
                swap_n = n0
                n0 = n1
                n1 = swap_n
            if t0 > near_t:
                near_t = t0
                near_normal = n0
            if t1 < far_t:
                far_t = t1
                far_normal = n1
            if near_t > far_t:
                valid = False
    if valid:
        _report_hit(near_t, near_normal)
        _report_hit(far_t, far_normal)


@wp.func
def _intersect_ellipsoid(origin: wp.vec3, direction: wp.vec3, radii: wp.vec3):
    scaled_origin = wp.cw_div(origin, radii)
    scaled_direction = wp.cw_div(direction, radii)
    a = wp.dot(scaled_direction, scaled_direction)
    half_b = wp.dot(scaled_origin, scaled_direction)
    c = wp.dot(scaled_origin, scaled_origin) - 1.0
    discriminant = half_b * half_b - a * c
    if discriminant >= 0.0:
        root = wp.sqrt(discriminant)
        t0 = (-half_b - root) / a
        t1 = (-half_b + root) / a
        p0 = origin + t0 * direction
        p1 = origin + t1 * direction
        radii2 = wp.cw_mul(radii, radii)
        _report_hit(t0, wp.normalize(wp.cw_div(p0, radii2)))
        _report_hit(t1, wp.normalize(wp.cw_div(p1, radii2)))


@wp.kernel(module="unique")
def analytical_intersection():
    origin = wp.optix_get_object_ray_origin()
    direction = wp.optix_get_object_ray_direction()
    shape = int(wp.optix_get_instance_id())
    if shape == 1:
        _intersect_sphere(origin, direction, wp.vec3(0.0), 0.38)
    elif shape == 2:
        _intersect_cylinder(origin, direction, 0.31, 0.45)
    elif shape == 3:
        _intersect_cone(origin, direction, 0.40, 0.48)
    elif shape == 4:
        _intersect_box(origin, direction, 0.36)
    elif shape == 5:
        _intersect_cylinder_side(origin, direction, 0.24, 0.27)
        _intersect_sphere(origin, direction, wp.vec3(0.0, -0.27, 0.0), 0.24)
        _intersect_sphere(origin, direction, wp.vec3(0.0, 0.27, 0.0), 0.24)
    else:
        _intersect_ellipsoid(origin, direction, wp.vec3(0.43, 0.29, 0.32))


@wp.kernel(module="unique")
def analytical_closest_hit():
    object_normal = wp.vec3(
        wp.uint32_to_float(wp.optix_get_attribute_0()),
        wp.uint32_to_float(wp.optix_get_attribute_1()),
        wp.uint32_to_float(wp.optix_get_attribute_2()),
    )
    world_normal = wp.optix_transform_normal_from_object_to_world_space(object_normal)
    shape = int(wp.optix_get_instance_id())
    albedo = wp.vec3(0.20, 0.48, 0.95)
    if shape == 2:
        albedo = wp.vec3(0.95, 0.48, 0.12)
    elif shape == 3:
        albedo = wp.vec3(0.22, 0.78, 0.32)
    elif shape == 4:
        albedo = wp.vec3(0.90, 0.22, 0.25)
    elif shape == 5:
        albedo = wp.vec3(0.65, 0.30, 0.92)
    elif shape == 6:
        albedo = wp.vec3(0.15, 0.78, 0.82)
    _store_color(_shade(world_normal, albedo))


@wp.kernel(module="unique")
def curve_closest_hit(curve_vertices: wp.array[wp.vec3], curve_indices: wp.array[wp.uint32]):
    vertex = int(curve_indices[int(wp.optix_get_primitive_index())])
    u = wp.optix_get_curve_parameter()
    center = curve_vertices[vertex] * (1.0 - u) + curve_vertices[vertex + 1] * u
    world_center = wp.optix_transform_point_from_object_to_world_space(center)
    hit = wp.optix_get_world_ray_origin() + wp.optix_get_ray_tmax() * wp.optix_get_world_ray_direction()
    _store_color(_shade(wp.normalize(hit - world_center), wp.vec3(0.95, 0.72, 0.12)))


@wp.kernel
def write_instances(
    traversables: wp.array[wp.uint64],
    translations: wp.array[wp.vec3],
    sbt_offsets: wp.array[wp.uint32],
    records: wp.array[wo.Instance],
):
    i = wp.tid()
    records[i] = wo.make_instance(
        wp.identity(3, float),
        translations[i],
        traversables[i],
        wp.uint32(i),
        sbt_offsets[i],
        wp.uint32(255),
        wp.uint32(0),
    )


def _save_bmp(path: Path, pixels: np.ndarray) -> None:
    height, width = pixels.shape
    bgr = np.stack(((pixels >> 16) & 0xFF, (pixels >> 8) & 0xFF, pixels & 0xFF), axis=-1).astype(np.uint8)
    row_stride = (width * 3 + 3) & ~3
    pixel_data_size = row_stride * height
    padding = b"\x00" * (row_stride - width * 3)
    with path.open("wb") as output:
        output.write(struct.pack("<2sIHHI", b"BM", 54 + pixel_data_size, 0, 0, 54))
        output.write(struct.pack("<IiiHHIIiiII", 40, width, height, 1, 24, 0, pixel_data_size, 0, 0, 0, 0))
        for row in bgr:
            output.write(row.tobytes())
            output.write(padding)


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--output", type=Path, default=Path("mixed_geometry.bmp"))
    args = parser.parse_args()

    with wp.ScopedDevice(args.device):
        ctx = wo.create_context()

        panel_vertices = wp.array(
            [[-2.1, -1.35, -0.7], [2.1, -1.35, -0.7], [2.1, 1.35, -0.7], [-2.1, 1.35, -0.7]], dtype=wp.vec3
        )
        panel_indices = wp.array([0, 1, 2, 0, 2, 3], dtype=wp.uint32)
        triangle_gas = wo.build_accel(ctx, [wo.triangles(panel_vertices, panel_indices)], compact=True)

        shape_bounds = wp.array([[-0.55, -0.55, -0.55, 0.55, 0.55, 0.55]], dtype=wp.float32)
        shape_gas = wo.build_accel(ctx, [wo.aabbs(shape_bounds)], compact=True)

        curve_x = np.linspace(-1.85, 1.85, 17, dtype=np.float32)
        curve_vertices = wp.array(
            np.stack((curve_x, 0.18 * np.sin(2.7 * curve_x), np.full_like(curve_x, -0.35)), axis=1), dtype=wp.vec3
        )
        curve_widths = wp.full(curve_x.size, 0.045, dtype=wp.float32)
        curve_indices = wp.array(np.arange(curve_x.size - 1), dtype=wp.uint32)
        curve_type = wo.PrimitiveType.ROUND_LINEAR
        curve_gas = wo.build_accel(
            ctx, [wo.curves(curve_vertices, curve_widths, curve_indices, curve_type=curve_type)], compact=True
        )

        # Hit group i serves instances with SBT offset i: 0 triangles, 1 analytical shapes, 2 curves.
        pipeline = wo.create_pipeline(
            ctx,
            raygen,
            miss=[miss],
            hit_groups=[
                wo.HitGroup(closest_hit=triangle_closest_hit),
                wo.HitGroup(closest_hit=analytical_closest_hit, intersection=analytical_intersection),
                wo.HitGroup(closest_hit=curve_closest_hit, builtin=curve_type),
            ],
            payload_words=3,
            attribute_words=3,
            graph_flags=wo.TraversableGraphFlags.ALLOW_SINGLE_LEVEL_INSTANCING,
        )

        # Instance i has instance ID i; the analytical shapes (IDs 1-6) select their surface by ID.
        shape_positions = [(-1.15, 0.57), (0.0, 0.57), (1.15, 0.57), (-1.15, -0.57), (0.0, -0.57), (1.15, -0.57)]
        traversables = [triangle_gas.handle] + [shape_gas.handle] * 6 + [curve_gas.handle]
        translations = [(0.0, 0.0, 0.0)] + [(x, y, 0.0) for x, y in shape_positions] + [(0.0, 0.0, 0.0)]
        sbt_offsets = [0] + [1] * 6 + [2]
        records = wp.empty(len(traversables), dtype=wo.Instance)
        wp.launch(
            write_instances,
            dim=records.shape,
            inputs=[
                wp.array(traversables, dtype=wp.uint64),
                wp.array(translations, dtype=wp.vec3),
                wp.array(sbt_offsets, dtype=wp.uint32),
                records,
            ],
        )
        ias = wo.build_accel(ctx, [wo.instances(records)])

        image = wp.empty((args.height, args.width), dtype=wp.uint32)
        wo.launch(pipeline, dim=image.shape, inputs=[ias.handle, curve_vertices, curve_indices, image])

        pixels = image.numpy()
        _save_bmp(args.output, pixels)
        print(f"Wrote {args.output} (checksum {int(np.bitwise_xor.reduce(pixels, axis=None)):#010x})")
        print("Top: sphere, cylinder, cone. Bottom: cube, capsule, ellipsoid. Gold: round curve.")


if __name__ == "__main__":
    main()
