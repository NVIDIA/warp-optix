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

"""Tiny Warp OptiX ray tracer with optional zero-copy OpenGL live viewer.

The raygen, miss, and closest-hit programs are ordinary Warp kernels; the
raygen arguments are the launch parameters of :func:`warp_optix.launch`.
"""

from __future__ import annotations

import argparse
import math
import os
import struct

import numpy as np
import warp as wp
from warp_optix.gl import GLInteropViewer

import warp_optix as wo


@wp.func
def pack_rgba8(color: wp.vec3) -> wp.uint32:
    r = wp.uint32(wp.max(0.0, wp.min(255.0, color[0] * 255.0)))
    g = wp.uint32(wp.max(0.0, wp.min(255.0, color[1] * 255.0)))
    b = wp.uint32(wp.max(0.0, wp.min(255.0, color[2] * 255.0)))
    a = wp.uint32(255)
    return (a << wp.uint32(24)) | (b << wp.uint32(16)) | (g << wp.uint32(8)) | r


@wp.struct
class Camera:
    position: wp.vec3
    forward: wp.vec3
    right: wp.vec3
    up: wp.vec3
    tan_half_fov: float
    aspect: float


@wp.struct
class TracePayload:
    hit: wp.uint32
    r_u8: wp.uint32
    g_u8: wp.uint32
    b_u8: wp.uint32
    rand0: wp.uint32
    rand1: wp.uint32


@wp.func
def _to_u8(v: float) -> wp.uint32:
    return wp.uint32(wp.max(0.0, wp.min(255.0, v * 255.0)))


@wp.func
def _hash_u32(x: wp.uint32) -> wp.uint32:
    x = (x ^ wp.uint32(61)) ^ (x >> wp.uint32(16))
    x *= wp.uint32(9)
    x = x ^ (x >> wp.uint32(4))
    x *= wp.uint32(0x27D4EB2D)
    x = x ^ (x >> wp.uint32(15))
    return x


@wp.func
def _u32_to_unit_float(x: wp.uint32) -> float:
    return float(x) * (1.0 / 4294967295.0)


@wp.func
def _cosine_sample_hemisphere(n: wp.vec3, u1: float, u2: float) -> wp.vec3:
    r = wp.sqrt(wp.max(0.0, u1))
    phi = 6.28318530718 * u2
    x = r * wp.cos(phi)
    y = r * wp.sin(phi)
    z = wp.sqrt(wp.max(0.0, 1.0 - u1))

    up = wp.vec3(0.0, 1.0, 0.0) if wp.abs(n[1]) < 0.999 else wp.vec3(1.0, 0.0, 0.0)
    t = wp.normalize(wp.cross(up, n))
    b = wp.cross(n, t)
    return wp.normalize(t * x + b * y + n * z)


@wp.func
def _room_radiance(p: wp.vec3, d: wp.vec3) -> wp.vec3:
    eps = 1.0e-4
    t_best = 1.0e20
    c = wp.vec3(0.0, 0.0, 0.0)

    if wp.abs(d[0]) > 1.0e-6:
        txl = (-1.0 - p[0]) / d[0]
        if txl > eps and txl < t_best:
            q = p + d * txl
            if q[1] >= -1.0 and q[1] <= 1.0 and q[2] >= -1.0 and q[2] <= 1.0:
                t_best = txl
                c = wp.vec3(0.78, 0.18, 0.14)
        txr = (1.0 - p[0]) / d[0]
        if txr > eps and txr < t_best:
            q = p + d * txr
            if q[1] >= -1.0 and q[1] <= 1.0 and q[2] >= -1.0 and q[2] <= 1.0:
                t_best = txr
                c = wp.vec3(0.16, 0.64, 0.20)

    if wp.abs(d[1]) > 1.0e-6:
        tyf = (-1.0 - p[1]) / d[1]
        if tyf > eps and tyf < t_best:
            q = p + d * tyf
            if q[0] >= -1.0 and q[0] <= 1.0 and q[2] >= -1.0 and q[2] <= 1.0:
                t_best = tyf
                c = wp.vec3(0.74, 0.74, 0.74) * 0.8
        tyc = (1.0 - p[1]) / d[1]
        if tyc > eps and tyc < t_best:
            q = p + d * tyc
            if q[0] >= -1.0 and q[0] <= 1.0 and q[2] >= -1.0 and q[2] <= 1.0:
                t_best = tyc
                if q[0] >= -0.25 and q[0] <= 0.25 and q[2] >= -0.65 and q[2] <= -0.2:
                    c = wp.vec3(3.5, 3.3, 2.9)
                else:
                    c = wp.vec3(0.74, 0.74, 0.74) * 0.7

    if wp.abs(d[2]) > 1.0e-6:
        tzb = (-1.0 - p[2]) / d[2]
        if tzb > eps and tzb < t_best:
            q = p + d * tzb
            if q[0] >= -1.0 and q[0] <= 1.0 and q[1] >= -1.0 and q[1] <= 1.0:
                t_best = tzb
                c = wp.vec3(0.72, 0.72, 0.72) * 0.75

    if t_best < 1.0e19:
        return c
    return wp.vec3(0.03, 0.04, 0.05)


@wp.func
def _segment_hits_aabb(p0: wp.vec3, p1: wp.vec3, bmin: wp.vec3, bmax: wp.vec3) -> bool:
    d = p1 - p0
    tmin = 0.0
    tmax = 1.0
    eps = 1.0e-6

    for axis in range(3):
        p = p0[axis]
        v = d[axis]
        lo = bmin[axis]
        hi = bmax[axis]
        if wp.abs(v) < eps:
            if p < lo or p > hi:
                return False
        else:
            inv_v = 1.0 / v
            t1 = (lo - p) * inv_v
            t2 = (hi - p) * inv_v
            if t1 > t2:
                tmp = t1
                t1 = t2
                t2 = tmp
            tmin = wp.max(tmin, t1)
            tmax = wp.min(tmax, t2)
            if tmax < tmin:
                return False
    return tmax > 0.0 and tmin < 1.0


@wp.kernel(module="unique")
def tiny_raygen(
    scene: wp.uint64,
    camera: Camera,
    frame_index: wp.uint32,
    seed: wp.uint32,
    accum: wp.array2d[wp.vec3],
    image: wp.array[wp.uint32],
):
    iy, ix = wp.tid()
    height = accum.shape[0]
    width = accum.shape[1]

    ndc_x = 2.0 * ((float(ix) + 0.5) / float(width)) - 1.0
    ndc_y = 2.0 * ((float(iy) + 0.5) / float(height)) - 1.0
    px = ndc_x * camera.aspect * camera.tan_half_fov
    py = ndc_y * camera.tan_half_fov

    ray_origin = camera.position
    ray_direction = wp.normalize(camera.forward + px * camera.right + py * camera.up)

    payload = TracePayload()
    pixel_idx = iy * width + ix
    rng_base = wp.uint32(pixel_idx) ^ (frame_index * wp.uint32(747796405)) ^ seed
    payload.rand0 = _hash_u32(rng_base ^ wp.uint32(2891336453))
    payload.rand1 = _hash_u32(rng_base ^ wp.uint32(1181783497))

    wp.optix_trace(
        scene,
        ray_origin,
        ray_direction,
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

    hit_color = wp.vec3(float(payload.r_u8) / 255.0, float(payload.g_u8) / 255.0, float(payload.b_u8) / 255.0)
    color = hit_color
    if payload.hit == wp.uint32(0):
        scale = 2.5 / wp.max(-ray_direction[2], 0.001)
        x = ray_direction[0] * scale
        y = ray_direction[1] * scale
        key = wp.exp(-0.85 * (0.65 * x * x + (y - 0.15) * (y - 0.15)))
        fill = wp.exp(-2.8 * ((x + 0.95) * (x + 0.95) + (y - 0.6) * (y - 0.6)))
        floor = wp.exp(-3.2 * (0.55 * x * x + (y + 0.9) * (y + 0.9)))
        display_sky = (
            wp.vec3(0.72, 0.75, 0.80)
            + key * wp.vec3(0.27, 0.245, 0.18)
            + fill * wp.vec3(0.025, 0.045, 0.08)
            + floor * wp.vec3(0.05, 0.035, 0.015)
        )
        color = wp.vec3(
            wp.pow(display_sky[0], 2.2),
            wp.pow(display_sky[1], 2.2),
            wp.pow(display_sky[2], 2.2),
        )

    prev = accum[iy, ix]
    fi = float(frame_index)
    mean = color if fi <= 0.0 else (prev * fi + color) / (fi + 1.0)
    accum[iy, ix] = mean
    display = wp.vec3(
        wp.pow(wp.max(0.0, wp.min(1.0, mean[0])), 1.0 / 2.2),
        wp.pow(wp.max(0.0, wp.min(1.0, mean[1])), 1.0 / 2.2),
        wp.pow(wp.max(0.0, wp.min(1.0, mean[2])), 1.0 / 2.2),
    )
    image[pixel_idx] = pack_rgba8(display)


@wp.kernel(module="unique")
def tiny_miss():
    wp.optix_set_payload_0(wp.uint32(0))
    wp.optix_set_payload_1(wp.uint32(0))
    wp.optix_set_payload_2(wp.uint32(0))
    wp.optix_set_payload_3(wp.uint32(0))


@wp.kernel(module="unique")
def tiny_closest_hit():
    pid = int(wp.optix_get_primitive_index())

    n = wp.vec3(0.0, 0.0, 1.0)
    albedo = wp.vec3(0.75, 0.75, 0.75)
    emissive = wp.vec3(0.0, 0.0, 0.0)

    # Room quads are emitted in fixed order in _build_cornell_box_mesh().
    if pid < 2:
        n = wp.vec3(0.0, 1.0, 0.0)  # floor
        albedo = wp.vec3(0.74, 0.74, 0.74)
    elif pid < 4:
        n = wp.vec3(0.0, -1.0, 0.0)  # ceiling
        albedo = wp.vec3(0.74, 0.74, 0.74)
    elif pid < 6:
        n = wp.vec3(0.0, 0.0, 1.0)  # back wall
        albedo = wp.vec3(0.72, 0.72, 0.72)
    elif pid < 8:
        n = wp.vec3(1.0, 0.0, 0.0)  # left wall (red)
        albedo = wp.vec3(0.78, 0.18, 0.14)
    elif pid < 10:
        n = wp.vec3(-1.0, 0.0, 0.0)  # right wall (green)
        albedo = wp.vec3(0.16, 0.64, 0.2)
    elif pid < 12:
        # Small emissive ceiling panel.
        n = wp.vec3(0.0, -1.0, 0.0)
        emissive = wp.vec3(1.0, 0.95, 0.8)
    elif pid < 24:
        face = (pid - 12) // 2  # short box
        if face == 0:
            n = wp.vec3(1.0, 0.0, 0.0)
        elif face == 1:
            n = wp.vec3(-1.0, 0.0, 0.0)
        elif face == 2:
            n = wp.vec3(0.0, 0.0, 1.0)
        elif face == 3:
            n = wp.vec3(0.0, 0.0, -1.0)
        elif face == 4:
            n = wp.vec3(0.0, 1.0, 0.0)
        else:
            n = wp.vec3(0.0, -1.0, 0.0)
        albedo = wp.vec3(0.72, 0.72, 0.72)
    elif pid < 36:
        face = (pid - 24) // 2  # tall box
        if face == 0:
            n = wp.vec3(1.0, 0.0, 0.0)
        elif face == 1:
            n = wp.vec3(-1.0, 0.0, 0.0)
        elif face == 2:
            n = wp.vec3(0.0, 0.0, 1.0)
        elif face == 3:
            n = wp.vec3(0.0, 0.0, -1.0)
        elif face == 4:
            n = wp.vec3(0.0, 1.0, 0.0)
        else:
            n = wp.vec3(0.0, -1.0, 0.0)
        albedo = wp.vec3(0.68, 0.68, 0.68)

    ro = wp.optix_get_world_ray_origin()
    rd = wp.normalize(wp.optix_get_world_ray_direction())
    t_hit = wp.optix_get_ray_tmax()
    p = ro + rd * t_hit

    light_pos = wp.vec3(0.0, 0.99, -0.42)
    p_shadow = p + n * 0.002
    to_light = light_pos - p_shadow
    dist = wp.length(to_light)
    dist2 = wp.max(dist * dist, 1.0e-4)
    l = to_light / wp.max(dist, 1.0e-4)

    occluded_short = _segment_hits_aabb(p_shadow, light_pos, wp.vec3(-0.65, -1.0, -0.35), wp.vec3(-0.15, -0.25, 0.25))
    occluded_tall = _segment_hits_aabb(p_shadow, light_pos, wp.vec3(0.2, -1.0, -0.8), wp.vec3(0.65, 0.45, -0.25))
    visibility = 0.0 if (occluded_short or occluded_tall) else 1.0

    ndotl = wp.max(wp.dot(n, l), 0.0)
    key = visibility * ndotl * 8.0 / (1.0 + 6.0 * dist2)

    fill_dir = wp.normalize(wp.vec3(-0.25, 0.9, 0.35))
    fill = wp.max(wp.dot(n, fill_dir), 0.0) * 0.08
    ambient = 0.08

    u1 = _u32_to_unit_float(wp.optix_get_payload_4())
    u2 = _u32_to_unit_float(wp.optix_get_payload_5())
    bounce_dir = _cosine_sample_hemisphere(n, u1, u2)
    bounce_p = p + n * 0.002
    bounce_radiance = _room_radiance(bounce_p, bounce_dir)
    indirect = wp.cw_mul(albedo, bounce_radiance) * 0.35

    lit = albedo * wp.min(ambient + key + fill, 1.35) + indirect + emissive
    color = wp.vec3(wp.min(lit[0], 1.0), wp.min(lit[1], 1.0), wp.min(lit[2], 1.0))

    wp.optix_set_payload_0(wp.uint32(1))
    wp.optix_set_payload_1(_to_u8(color[0]))
    wp.optix_set_payload_2(_to_u8(color[1]))
    wp.optix_set_payload_3(_to_u8(color[2]))


def _save_bmp(path: str, pixels_rgba_u32: np.ndarray, width: int, height: int):
    rgba = pixels_rgba_u32.reshape(height, width)
    bgr = np.empty((height, width, 3), dtype=np.uint8)
    bgr[..., 0] = ((rgba >> 16) & 0xFF).astype(np.uint8)
    bgr[..., 1] = ((rgba >> 8) & 0xFF).astype(np.uint8)
    bgr[..., 2] = (rgba & 0xFF).astype(np.uint8)

    row_stride = (width * 3 + 3) & ~3
    pixel_data_size = row_stride * height
    # Positive height means bottom-to-top row order, which matches our
    # kernel layout (iy=0 is the bottom of the image) -- no flip needed.
    with open(path, "wb") as f:
        f.write(struct.pack("<2sIHHI", b"BM", 54 + pixel_data_size, 0, 0, 54))
        f.write(struct.pack("<IiiHHIIiiII", 40, width, height, 1, 24, 0, pixel_data_size, 0, 0, 0, 0))
        pad = b"\x00" * (row_stride - width * 3)
        for y in range(height):
            f.write(bgr[y].tobytes())
            if pad:
                f.write(pad)


def _add_quad(vertices: list[tuple[float, float, float]], indices: list[tuple[int, int, int]], v0, v1, v2, v3):
    base = len(vertices)
    vertices.extend((v0, v1, v2, v3))
    indices.append((base, base + 1, base + 2))
    indices.append((base, base + 2, base + 3))


def _add_box(
    vertices: list[tuple[float, float, float]],
    indices: list[tuple[int, int, int]],
    x0: float,
    y0: float,
    z0: float,
    x1: float,
    y1: float,
    z1: float,
):
    p000 = (x0, y0, z0)
    p001 = (x0, y0, z1)
    p010 = (x0, y1, z0)
    p011 = (x0, y1, z1)
    p100 = (x1, y0, z0)
    p101 = (x1, y0, z1)
    p110 = (x1, y1, z0)
    p111 = (x1, y1, z1)

    # Keep quad order stable so primitive id -> normal mapping stays tiny and deterministic.
    _add_quad(vertices, indices, p100, p110, p111, p101)  # +X
    _add_quad(vertices, indices, p001, p011, p010, p000)  # -X
    _add_quad(vertices, indices, p101, p111, p011, p001)  # +Z
    _add_quad(vertices, indices, p000, p010, p110, p100)  # -Z
    _add_quad(vertices, indices, p010, p011, p111, p110)  # +Y
    _add_quad(vertices, indices, p001, p000, p100, p101)  # -Y


def _build_cornell_box_mesh() -> tuple[np.ndarray, np.ndarray]:
    vertices: list[tuple[float, float, float]] = []
    indices: list[tuple[int, int, int]] = []

    # Cornell-style room: floor, ceiling, back wall, left wall, right wall.
    _add_quad(vertices, indices, (-1.0, -1.0, -1.0), (1.0, -1.0, -1.0), (1.0, -1.0, 1.0), (-1.0, -1.0, 1.0))
    _add_quad(vertices, indices, (-1.0, 1.0, 1.0), (1.0, 1.0, 1.0), (1.0, 1.0, -1.0), (-1.0, 1.0, -1.0))
    _add_quad(vertices, indices, (-1.0, -1.0, -1.0), (-1.0, 1.0, -1.0), (1.0, 1.0, -1.0), (1.0, -1.0, -1.0))
    _add_quad(vertices, indices, (-1.0, -1.0, 1.0), (-1.0, 1.0, 1.0), (-1.0, 1.0, -1.0), (-1.0, -1.0, -1.0))
    _add_quad(vertices, indices, (1.0, -1.0, -1.0), (1.0, 1.0, -1.0), (1.0, 1.0, 1.0), (1.0, -1.0, 1.0))

    # Ceiling light panel.
    _add_quad(vertices, indices, (-0.25, 0.99, -0.65), (0.25, 0.99, -0.65), (0.25, 0.99, -0.2), (-0.25, 0.99, -0.2))

    # Two interior blocks.
    _add_box(vertices, indices, -0.65, -1.0, -0.35, -0.15, -0.25, 0.25)
    _add_box(vertices, indices, 0.2, -1.0, -0.8, 0.65, 0.45, -0.25)

    verts_np = np.asarray(vertices, dtype=np.float32)
    inds_np = np.asarray(indices, dtype=np.uint32)
    return verts_np, inds_np


def _make_camera(
    position: np.ndarray,
    yaw_deg: float,
    pitch_deg: float,
    fov_deg: float,
    width: int,
    height: int,
):
    yaw_rad = math.radians(yaw_deg)
    pitch_rad = math.radians(pitch_deg)
    cos_pitch = math.cos(pitch_rad)
    forward = np.array(
        [math.sin(yaw_rad) * cos_pitch, math.sin(pitch_rad), math.cos(yaw_rad) * cos_pitch],
        dtype=np.float32,
    )
    forward /= np.linalg.norm(forward) + 1.0e-8
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right) + 1.0e-8
    up = np.cross(right, forward)
    up /= np.linalg.norm(up) + 1.0e-8

    camera = Camera()
    camera.position = wp.vec3(position)
    camera.forward = wp.vec3(forward)
    camera.right = wp.vec3(right)
    camera.up = wp.vec3(up)
    camera.tan_half_fov = math.tan(math.radians(fov_deg) * 0.5)
    camera.aspect = float(width) / float(height)
    return camera


class FreeCameraController:
    def __init__(self, viewer: GLInteropViewer, width: int, height: int):
        self.window = viewer.window
        self.pyglet = viewer.pyglet
        self.width = int(width)
        self.height = int(height)
        self.position = np.array([0.0, 0.05, 2.5], dtype=np.float32)
        self.yaw = 180.0
        self.pitch = -4.0
        self.fov = 50.0
        self._keys_down: set[int] = set()
        self._cam_speed = 1.6
        self._look_sensitivity = 0.12
        self._dirty = True
        self.window.push_handlers(self)
        self._write(mark_dirty=False)

    def _forward_right(self) -> tuple[np.ndarray, np.ndarray]:
        yaw_rad = math.radians(self.yaw)
        pitch_rad = math.radians(self.pitch)
        cos_pitch = math.cos(pitch_rad)
        forward = np.array(
            [math.sin(yaw_rad) * cos_pitch, math.sin(pitch_rad), math.cos(yaw_rad) * cos_pitch], dtype=np.float32
        )
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(forward, world_up)
        right /= np.linalg.norm(right) + 1.0e-8
        return forward, right

    def _write(self, mark_dirty: bool = True):
        self.camera = _make_camera(self.position, self.yaw, self.pitch, self.fov, self.width, self.height)
        if mark_dirty:
            self._dirty = True

    def resize(self, width: int, height: int):
        self.width = int(width)
        self.height = int(height)
        self._write()

    def update(self, dt: float):
        key = self.pyglet.window.key

        forward, right = self._forward_right()
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        move = np.zeros(3, dtype=np.float32)
        if key.W in self._keys_down or key.UP in self._keys_down:
            move += forward
        if key.S in self._keys_down or key.DOWN in self._keys_down:
            move -= forward
        if key.A in self._keys_down or key.LEFT in self._keys_down:
            move -= right
        if key.D in self._keys_down or key.RIGHT in self._keys_down:
            move += right
        if key.Q in self._keys_down:
            move -= world_up
        if key.E in self._keys_down:
            move += world_up

        move_norm = float(np.linalg.norm(move))
        if move_norm > 1.0e-6:
            speed = self._cam_speed
            if key.LSHIFT in self._keys_down or key.RSHIFT in self._keys_down:
                speed *= 3.5
            self.position += (move / move_norm) * speed * max(0.0, min(dt, 0.1))
            self._write()

    def consume_changed(self) -> bool:
        was_dirty = self._dirty
        self._dirty = False
        return was_dirty

    def on_key_press(self, symbol, _modifiers):
        self._keys_down.add(symbol)

    def on_key_release(self, symbol, _modifiers):
        self._keys_down.discard(symbol)

    def on_mouse_drag(self, _x, _y, dx, dy, buttons, _modifiers):
        if buttons & self.pyglet.window.mouse.LEFT:
            self.yaw -= float(dx) * self._look_sensitivity
            self.pitch += float(dy) * self._look_sensitivity
            self.pitch = max(-89.0, min(89.0, self.pitch))
            self._write()

    def on_mouse_scroll(self, _x, _y, _sx, sy):
        self.fov = max(20.0, min(90.0, self.fov - float(sy) * 2.0))
        self._write()


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--output", type=str, default="tiny_optix_raytracer.bmp")
    parser.add_argument("--live", action="store_true", help="Run OpenGL interop viewer (zero-copy PBO path)")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--max-frames", type=int, default=0, help="Auto-exit after N frames in live mode (0=forever)")
    args = parser.parse_args()

    with wp.ScopedDevice(args.device):
        ctx = wo.create_context()
        vertices, indices = _build_cornell_box_mesh()
        vertices = wp.array(vertices, dtype=wp.vec3)
        indices = wp.array(indices, dtype=wp.uint32)
        gas = wo.build_accel(ctx, [wo.triangles(vertices, indices)], compact=True)
        pipeline = wo.create_pipeline(
            ctx,
            tiny_raygen,
            miss=[tiny_miss],
            hit_groups=[wo.HitGroup(closest_hit=tiny_closest_hit)],
            payload_words=6,
        )
        seed = wp.uint32(1337)
        accum = wp.zeros((args.height, args.width), dtype=wp.vec3)

        if args.live:
            sample_count = 0
            controller: FreeCameraController | None = None

            def _on_resize(width: int, height: int):
                nonlocal accum, sample_count
                accum = wp.zeros((int(height), int(width)), dtype=wp.vec3)
                controller.resize(width, height)
                sample_count = 0

            viewer = GLInteropViewer(
                width=args.width,
                height=args.height,
                device=args.device,
                title="Warp OptiX Tiny RT",
                fps=args.fps,
                on_resize=_on_resize,
            )
            controller = FreeCameraController(viewer, args.width, args.height)
            last_elapsed = 0.0

            def _render(mapped_image: wp.array, _frame_idx: int, elapsed_sec: float):
                nonlocal last_elapsed, sample_count
                dt = elapsed_sec - last_elapsed
                last_elapsed = elapsed_sec
                controller.update(dt)
                if controller.consume_changed():
                    sample_count = 0
                wo.launch(
                    pipeline,
                    dim=accum.shape,
                    inputs=[gas.handle, controller.camera, wp.uint32(sample_count), seed, accum, mapped_image],
                )
                sample_count += 1

            viewer.run(_render, max_frames=args.max_frames)
            print("Controls: WASD move, Q/E down/up, Shift faster, Left-drag look, Mouse wheel zoom.")
            print("Live viewer finished.")
        else:
            camera = _make_camera(
                np.array([0.0, 0.05, 2.5], dtype=np.float32), 180.0, -4.0, 50.0, args.width, args.height
            )
            image = wp.empty(args.width * args.height, dtype=wp.uint32)
            wo.launch(pipeline, dim=accum.shape, inputs=[gas.handle, camera, wp.uint32(0), seed, accum, image])
            pixels = image.numpy()
            _save_bmp(args.output, pixels, args.width, args.height)
            print(f"Image checksum: {int(np.bitwise_xor.reduce(pixels))}")
            print(f"Wrote: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
