# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render vertex motion blur by tracing several shutter times per pixel."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
import warp as wp

import warp_optix as wo
from warp_optix import native


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
def raygen(scene: wp.uint64, samples: int, image: wp.array2d[wp.uint32]):
    y, x = wp.tid()
    height = image.shape[0]
    width = image.shape[1]
    sx = (2.0 * (float(x) + 0.5) / float(width) - 1.0) * float(width) / float(height)
    sy = 2.0 * (float(y) + 0.5) / float(height) - 1.0
    origin = wp.vec3(0.0, 0.0, 4.0)
    direction = wp.normalize(wp.vec3(sx, sy, -3.0))

    total = wp.vec3(0.0)
    for sample in range(samples):
        payload = Payload()
        ray_time = (float(sample) + 0.5) / float(samples)
        wp.optix_trace(
            scene,
            origin,
            direction,
            0.001,
            1.0e16,
            ray_time,
            wp.uint32(255),
            wp.uint32(0),
            wp.uint32(0),
            wp.uint32(1),
            wp.uint32(0),
            payload,
        )
        total += wp.vec3(float(payload.red), float(payload.green), float(payload.blue))

    mean = total * (1.0 / float(samples))
    image[y, x] = (
        wp.uint32(255) << wp.uint32(24)
        | wp.uint32(mean[2]) << wp.uint32(16)
        | wp.uint32(mean[1]) << wp.uint32(8)
        | wp.uint32(mean[0])
    )


@wp.kernel(module="unique")
def miss():
    _store_color(_background(wp.normalize(wp.optix_get_world_ray_direction())))


@wp.kernel(module="unique")
def closest_hit():
    color = wp.vec3(0.18, 0.22, 0.30)
    if wp.optix_get_primitive_index() == wp.uint32(0):
        ray_time = wp.optix_get_ray_time()
        color = wp.vec3(0.05, 0.80, 1.0) * (1.0 - ray_time) + wp.vec3(1.0, 0.35, 0.04) * ray_time
    _store_color(color)


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
    parser.add_argument("--samples", type=int, default=64, help="Temporal samples per pixel")
    parser.add_argument("--output", type=Path, default=Path("motion_blur.bmp"))
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be at least 1")

    with wp.ScopedDevice(args.device):
        ctx = wo.create_context()

        # Two motion keys: the top triangle slides from left to right over the shutter interval.
        moving = np.array([[-0.38, 0.12, 0.0], [0.38, 0.12, 0.0], [0.0, 0.92, 0.0]], dtype=np.float32)
        static = np.array([[-0.38, -1.0, 0.0], [0.38, -1.0, 0.0], [0.0, -0.20, 0.0]], dtype=np.float32)
        keys = [wp.array(np.concatenate((moving + [dx, 0.0, 0.0], static)), dtype=wp.vec3) for dx in (-1.15, 1.15)]
        indices = wp.array([0, 1, 2, 3, 4, 5], dtype=wp.uint32)
        motion = native.MotionOptions(numKeys=2, flags=native.MOTION_FLAG_NONE, timeBegin=0.0, timeEnd=1.0)
        gas = wo.build_accel(ctx, [wo.triangles(keys, indices)], motion=motion)

        pipeline = wo.create_pipeline(
            ctx,
            raygen,
            miss=[miss],
            hit_groups=[wo.HitGroup(closest_hit=closest_hit)],
            payload_words=3,
            motion_blur=True,
        )
        image = wp.empty((args.height, args.width), dtype=wp.uint32)
        wo.launch(pipeline, dim=image.shape, inputs=[gas.handle, args.samples, image])

        pixels = image.numpy()
        _save_bmp(args.output, pixels)
        print(f"Wrote {args.output} (checksum {int(np.bitwise_xor.reduce(pixels, axis=None)):#010x})")
        print("Top: moving triangle. Bottom: static reference.")


if __name__ == "__main__":
    main()
