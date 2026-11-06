# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Inverse rendering with fixed visibility: recover albedo and a light from one image.

OptiX does the discrete work: a G-buffer pass records which triangle each
pixel sees and where, and a shadow pass tests visibility toward the current
light estimate. Both launches are constants on the ``wp.Tape``. Shading and the
loss are ordinary Warp kernels, so gradients reach per-vertex albedo, the light
position, and (through the barycentric hit reconstruction) the vertex positions.
Silhouette and shadow-edge gradients are not computed; visibility is
re-evaluated every forward pass instead.
"""

from __future__ import annotations

import argparse
import math
import os
import struct

import numpy as np
import warp as wp
import warp.optim

import warp_optix as wo

AMBIENT = wp.constant(0.08)
LIGHT_POWER = wp.constant(60.0)


@wp.struct
class Camera:
    position: wp.vec3
    forward: wp.vec3
    right: wp.vec3
    up: wp.vec3
    tan_half_fov: float


@wp.func
def camera_ray(camera: Camera, x: int, y: int, width: int, height: int) -> wp.vec3:
    px = (2.0 * (float(x) + 0.5) / float(width) - 1.0) * camera.tan_half_fov * float(width) / float(height)
    py = (2.0 * (float(y) + 0.5) / float(height) - 1.0) * camera.tan_half_fov
    return wp.normalize(camera.forward + px * camera.right + py * camera.up)


@wp.func
def interpolate(values: wp.array[wp.vec3], indices: wp.array[wp.int32], prim: int, bary: wp.vec2) -> wp.vec3:
    a = values[indices[3 * prim + 0]]
    b = values[indices[3 * prim + 1]]
    c = values[indices[3 * prim + 2]]
    return a * (1.0 - bary[0] - bary[1]) + b * bary[0] + c * bary[1]


BACKGROUND = wp.constant(wp.vec3(0.02, 0.025, 0.03))


@wp.func
def shade_hit(
    vertices: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    indices: wp.array[wp.int32],
    albedo: wp.array[wp.vec3],
    light: wp.vec3,
    prim: int,
    bary: wp.vec2,
    visibility: float,
) -> wp.vec3:
    position = interpolate(vertices, indices, prim, bary)
    normal = wp.normalize(interpolate(normals, indices, prim, bary))
    to_light = light - position
    distance2 = wp.dot(to_light, to_light)
    cosine = wp.max(wp.dot(normal, to_light) / wp.sqrt(distance2), 0.0)
    irradiance = AMBIENT + visibility * LIGHT_POWER * cosine / (4.0 * wp.pi * distance2)
    return interpolate(albedo, indices, prim, bary) * irradiance


@wp.func
def shadow_visibility(
    scene: wp.uint64,
    vertices: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    indices: wp.array[wp.int32],
    prim: int,
    bary: wp.vec2,
    light: wp.vec3,
) -> float:
    position = interpolate(vertices, indices, prim, bary)
    origin = position + wp.normalize(interpolate(normals, indices, prim, bary)) * 1.0e-3
    to_light = light - origin
    distance = wp.length(to_light)
    if wo.any_hit(scene, origin, to_light / distance, 0.0, distance, wp.uint32(0xFF)):
        return 0.0
    return 1.0


# --- OptiX passes: discrete, non-differentiable --------------------------------


@wp.kernel(module="unique")
def gbuffer_raygen(
    scene: wp.uint64,
    camera: Camera,
    prim: wp.array2d[wp.int32],
    bary: wp.array2d[wp.vec2],
):
    y, x = wp.tid()
    direction = camera_ray(camera, x, y, prim.shape[1], prim.shape[0])
    hit = wo.closest_hit(scene, camera.position, direction, 0.0, 1.0e6, wp.uint32(0xFF), wp.uint32(0))
    prim[y, x] = -1
    if hit.t != wo.MISS:
        prim[y, x] = int(hit.primitive_index)
        bary[y, x] = wp.optix_hit_object_get_triangle_barycentrics()


@wp.kernel(module="unique")
def shadow_raygen(
    scene: wp.uint64,
    vertices: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    indices: wp.array[wp.int32],
    prim: wp.array2d[wp.int32],
    bary: wp.array2d[wp.vec2],
    light: wp.array[wp.vec3],
    visibility: wp.array2d[float],
):
    y, x = wp.tid()
    p = prim[y, x]
    visibility[y, x] = 0.0
    if p >= 0:
        visibility[y, x] = shadow_visibility(scene, vertices, normals, indices, p, bary[y, x], light[0])


# --- Warp passes: differentiable -----------------------------------------------


@wp.kernel
def shade(
    vertices: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    indices: wp.array[wp.int32],
    albedo: wp.array[wp.vec3],
    light: wp.array[wp.vec3],
    prim: wp.array2d[wp.int32],
    bary: wp.array2d[wp.vec2],
    visibility: wp.array2d[float],
    image: wp.array2d[wp.vec3],
):
    y, x = wp.tid()
    p = prim[y, x]
    if p < 0:
        image[y, x] = BACKGROUND
        return
    image[y, x] = shade_hit(vertices, normals, indices, albedo, light[0], p, bary[y, x], visibility[y, x])


@wp.kernel
def l2_loss(image: wp.array2d[wp.vec3], target: wp.array2d[wp.vec3], loss: wp.array[float]):
    y, x = wp.tid()
    d = image[y, x] - target[y, x]
    wp.atomic_add(loss, 0, wp.dot(d, d) / float(image.shape[0] * image.shape[1]))


@wp.kernel
def clamp_albedo(albedo: wp.array[wp.vec3]):
    i = wp.tid()
    a = albedo[i]
    albedo[i] = wp.vec3(wp.clamp(a[0], 0.0, 1.0), wp.clamp(a[1], 0.0, 1.0), wp.clamp(a[2], 0.0, 1.0))


# --- Scene ---------------------------------------------------------------------


def build_scene(rings: int = 24, segments: int = 48, grid: int = 24):
    """A UV sphere resting on a ground grid, with smooth vertex normals."""
    vertices, normals, indices = [], [], []
    for i in range(rings + 1):
        theta = math.pi * i / rings
        for j in range(segments + 1):
            phi = 2.0 * math.pi * j / segments
            n = (math.sin(theta) * math.cos(phi), math.cos(theta), math.sin(theta) * math.sin(phi))
            vertices.append((n[0], n[1] + 1.0, n[2]))
            normals.append(n)
    for i in range(rings):
        for j in range(segments):
            a = i * (segments + 1) + j
            b = a + segments + 1
            indices += [(a, a + 1, b), (a + 1, b + 1, b)]
    sphere_vertex_count = len(vertices)

    base = len(vertices)
    for i in range(grid + 1):
        for j in range(grid + 1):
            vertices.append((-4.0 + 8.0 * j / grid, 0.0, -4.0 + 8.0 * i / grid))
            normals.append((0.0, 1.0, 0.0))
    for i in range(grid):
        for j in range(grid):
            a = base + i * (grid + 1) + j
            b = a + grid + 1
            indices += [(a, b, a + 1), (a + 1, b, b + 1)]

    vertices = np.asarray(vertices, dtype=np.float32)
    return vertices, np.asarray(normals, dtype=np.float32), np.asarray(indices, dtype=np.int32), sphere_vertex_count


def target_albedo(vertices: np.ndarray, sphere_vertex_count: int) -> np.ndarray:
    """Hue bands on the sphere, a checkerboard on the ground."""
    albedo = np.empty_like(vertices)
    s = vertices[:sphere_vertex_count]
    phase = np.arctan2(s[:, 2], s[:, 0]) * 3.0 + s[:, 1] * 2.0
    albedo[:sphere_vertex_count] = 0.5 + 0.4 * np.stack(
        [np.sin(phase), np.sin(phase + 2.1), np.sin(phase + 4.2)], axis=1
    )
    g = vertices[sphere_vertex_count:]
    checker = (np.floor(g[:, 0]) + np.floor(g[:, 2])) % 2.0
    albedo[sphere_vertex_count:] = np.where(checker[:, None] > 0.5, [0.75, 0.72, 0.65], [0.25, 0.3, 0.4])
    return albedo.astype(np.float32)


def make_camera(position, look_at, fov_deg: float) -> Camera:
    position = np.asarray(position, dtype=np.float32)
    forward = np.asarray(look_at, dtype=np.float32) - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    camera = Camera()
    camera.position = wp.vec3(position)
    camera.forward = wp.vec3(forward)
    camera.right = wp.vec3(right)
    camera.up = wp.vec3(np.cross(right, forward))
    camera.tan_half_fov = math.tan(math.radians(fov_deg) * 0.5)
    return camera


def save_bmp(path: str, panels: list[np.ndarray]) -> None:
    """Write linear RGB panels side by side as an sRGB-ish 24-bit BMP."""
    rgb = np.concatenate(panels, axis=1)
    rgb = (np.clip(rgb, 0.0, 1.0) ** (1.0 / 2.2) * 255.0 + 0.5).astype(np.uint8)
    height, width = rgb.shape[:2]
    row_stride = (width * 3 + 3) & ~3
    with open(path, "wb") as f:
        f.write(struct.pack("<2sIHHI", b"BM", 54 + row_stride * height, 0, 0, 54))
        f.write(struct.pack("<IiiHHIIiiII", 40, width, height, 1, 24, 0, row_stride * height, 0, 0, 0, 0))
        pad = b"\x00" * (row_stride - width * 3)
        for row in rgb:  # Row 0 is the bottom of the image, as BMP expects.
            f.write(row[:, ::-1].tobytes() + pad)


class InverseRenderer:
    """Renders with an OptiX G-buffer and shadow pass, and Warp shading on the tape."""

    def __init__(self, ctx, width: int, height: int):
        vertices, normals, indices, self.sphere_vertex_count = build_scene()
        self.vertices = wp.array(vertices, dtype=wp.vec3)
        self.normals = wp.array(normals, dtype=wp.vec3)
        self.indices = wp.array(indices.reshape(-1), dtype=wp.int32)
        self.gas = wo.build_accel(ctx, [wo.triangles(self.vertices, self.indices)], compact=True)
        self.shadow_pipeline = wo.create_pipeline(ctx, shadow_raygen)
        self.shape = (height, width)

        # Primary visibility does not depend on the optimized parameters: trace it once.
        self.prim = wp.empty(self.shape, dtype=wp.int32)
        self.bary = wp.zeros(self.shape, dtype=wp.vec2)
        self.camera = camera = make_camera((0.0, 2.6, 5.5), (0.0, 0.8, 0.0), 45.0)
        wo.launch(
            wo.create_pipeline(ctx, gbuffer_raygen),
            dim=self.shape,
            inputs=[self.gas.handle, camera, self.prim, self.bary],
        )
        self.visibility = wp.empty(self.shape, dtype=float)

    def render(self, albedo: wp.array, light: wp.array, image: wp.array) -> None:
        args = [self.vertices, self.normals, self.indices]
        # The shadow launch reads the current light but is a constant on the tape.
        wo.launch(
            self.shadow_pipeline,
            dim=self.shape,
            inputs=[self.gas.handle, *args, self.prim, self.bary, light, self.visibility],
        )
        wp.launch(shade, dim=self.shape, inputs=[*args, albedo, light, self.prim, self.bary, self.visibility, image])


def fit(renderer, target, albedo, light, params, lr: float, iterations: int, label: str, report=None) -> wp.array:
    """Minimize the L2 image loss over ``params`` with Adam; return the final rendering."""
    image = wp.zeros(renderer.shape, dtype=wp.vec3, requires_grad=True)
    loss = wp.zeros(1, dtype=float, requires_grad=True)
    optimizer = warp.optim.Adam(params, lr=lr)
    for it in range(iterations):
        loss.zero_()
        with wp.Tape() as tape:
            renderer.render(albedo, light, image)
            wp.launch(l2_loss, dim=renderer.shape, inputs=[image, target, loss])
        tape.backward(loss)
        optimizer.step([p.grad for p in params])
        if albedo.requires_grad:
            wp.launch(clamp_albedo, dim=albedo.shape, inputs=[albedo])
        tape.zero()
        if it % 100 == 0 or it == iterations - 1:
            extra = f"  {report()}" if report else ""
            print(f"[{label}] iter {it:4d}  loss {loss.numpy()[0]:.3e}{extra}")
    return image


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--iterations", type=int, default=300, help="Iterations per stage")
    parser.add_argument("--output", type=str, default="inverse_rendering.bmp")
    args = parser.parse_args()

    with wp.ScopedDevice(args.device), wo.create_context() as ctx:
        renderer = InverseRenderer(ctx, args.width, args.height)
        vertex_count = renderer.vertices.shape[0]
        true_light = wp.array([[2.0, 3.5, 1.5]], dtype=wp.vec3)
        true_albedo = target_albedo(renderer.vertices.numpy(), renderer.sphere_vertex_count)

        # Two "photographs" under the same unknown light: a grey calibration scene and the object.
        grey = wp.full(vertex_count, wp.vec3(0.5, 0.5, 0.5), dtype=wp.vec3)
        calibration_photo = wp.empty(renderer.shape, dtype=wp.vec3)
        object_photo = wp.empty(renderer.shape, dtype=wp.vec3)
        renderer.render(grey, true_light, calibration_photo)
        renderer.render(wp.array(true_albedo, dtype=wp.vec3), true_light, object_photo)

        # Stage 1: recover the light from the calibration photo, whose materials are known.
        light = wp.array([[-1.5, 2.5, 2.5]], dtype=wp.vec3, requires_grad=True)
        initial = wp.empty(renderer.shape, dtype=wp.vec3)
        renderer.render(grey, light, initial)

        def light_error():
            return f"light error {np.linalg.norm(light.numpy() - true_light.numpy()):.2e}"

        fit(renderer, calibration_photo, grey, light, [light], 0.1, args.iterations, "light", light_error)

        # Stage 2: recover per-vertex albedo from the object photo under the recovered light.
        albedo = wp.full(vertex_count, wp.vec3(0.5, 0.5, 0.5), dtype=wp.vec3, requires_grad=True)
        light = wp.array(light.numpy(), dtype=wp.vec3)

        # Only vertices of triangles the camera sees receive gradients.
        seen_prims = np.unique(renderer.prim.numpy())
        seen = np.unique(renderer.indices.numpy().reshape(-1, 3)[seen_prims[seen_prims >= 0]])

        def albedo_error():
            return f"observed-albedo RMSE {np.sqrt(np.mean((albedo.numpy()[seen] - true_albedo[seen]) ** 2)):.2e}"

        result = fit(renderer, object_photo, albedo, light, [albedo], 0.02, args.iterations, "albedo", albedo_error)

        print(f"Recovered light {np.round(light.numpy()[0], 4)} (true {true_light.numpy()[0]})")
        save_bmp(args.output, [object_photo.numpy(), initial.numpy(), result.numpy()])
        print(f"Wrote photo | initial guess | recovered to {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
