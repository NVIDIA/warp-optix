# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render a varying-width native OptiX round-linear curve with Warp shaders."""

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

    payload = Payload()
    wp.optix_trace(
        scene,
        wp.vec3(0.0, 0.0, 4.0),
        wp.normalize(wp.vec3(sx, sy, -3.0)),
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
def curve_closest_hit(curve_vertices: wp.array[wp.vec3], curve_indices: wp.array[wp.uint32]):
    primitive = int(wp.optix_get_primitive_index())
    vertex = int(curve_indices[primitive])
    u = wp.optix_get_curve_parameter()
    center = curve_vertices[vertex] * (1.0 - u) + curve_vertices[vertex + 1] * u
    world_center = wp.optix_transform_point_from_object_to_world_space(center)
    hit = wp.optix_get_world_ray_origin() + wp.optix_get_ray_tmax() * wp.optix_get_world_ray_direction()
    normal = wp.normalize(hit - world_center)

    along_curve = (float(primitive) + u) / float(curve_indices.shape[0])
    albedo = wp.vec3(0.15, 0.75, 0.95) * (1.0 - along_curve) + wp.vec3(0.95, 0.20, 0.55) * along_curve
    light = wp.normalize(wp.vec3(-0.4, 0.8, 0.6))
    intensity = 0.18 + 0.82 * wp.max(wp.dot(normal, light), 0.0)
    _store_color(albedo * intensity)


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
    parser.add_argument("--output", type=Path, default=Path("curves.bmp"))
    args = parser.parse_args()

    with wp.ScopedDevice(args.device):
        ctx = wo.create_context()

        x = np.linspace(-1.65, 1.65, 49, dtype=np.float32)
        vertices = wp.array(np.stack((x, 0.62 * np.sin(2.4 * x), 0.16 * np.cos(1.8 * x)), axis=1), dtype=wp.vec3)
        widths = wp.array(0.075 + 0.035 * (0.5 + 0.5 * np.sin(4.0 * x)), dtype=wp.float32)
        segment_indices = wp.array(np.arange(x.size - 1), dtype=wp.uint32)
        curve_type = wo.PrimitiveType.ROUND_LINEAR
        gas = wo.build_accel(ctx, [wo.curves(vertices, widths, segment_indices, curve_type=curve_type)])

        pipeline = wo.create_pipeline(
            ctx,
            raygen,
            miss=[miss],
            hit_groups=[wo.HitGroup(closest_hit=curve_closest_hit, builtin=curve_type)],
            payload_words=3,
            attribute_words=1,
        )
        image = wp.empty((args.height, args.width), dtype=wp.uint32)
        wo.launch(pipeline, dim=image.shape, inputs=[gas.handle, vertices, segment_indices, image])

        pixels = image.numpy()
        _save_bmp(args.output, pixels)
        print(f"Wrote {args.output} (checksum {int(np.bitwise_xor.reduce(pixels, axis=None)):#010x})")


if __name__ == "__main__":
    main()
