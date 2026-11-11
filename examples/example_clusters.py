# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""An animated parametric knot, built and ray traced with real OptiX grid clusters.

The surface and renderer are written for this example. No models, textures,
subdivision library, or additional build step are needed. NumPy creates grid
descriptors; Warp evaluates the surface; ClusterAccel owns the OptiX builds.

    python examples/example_clusters.py --live
    python examples/example_clusters.py --samples 64 --output clusters.bmp

Drag to orbit, scroll to dolly, Space to pause, V to inspect clusters/normals,
and +/- to change tessellation. Pausing lets the image accumulate samples.
"""

from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path

import numpy as np
import warp as wp

import warp_optix as wo

TAU = wp.constant(6.283185307179586)


@wp.func
def centerline(t: float, time: float) -> wp.vec3:
    radial = 1.08 + (0.34 + 0.025 * wp.sin(time)) * wp.cos(3.0 * t)
    height = 0.53 + 0.025 * wp.cos(0.7 * time)
    angle = 2.0 * t + 0.16 * wp.sin(0.5 * time)
    return wp.vec3(radial * wp.cos(angle), height * wp.sin(3.0 * t), radial * wp.sin(angle))


@wp.func
def tangent(t: float, time: float) -> wp.vec3:
    width = 0.34 + 0.025 * wp.sin(time)
    radius = 1.08 + width * wp.cos(3.0 * t)
    dr = -3.0 * width * wp.sin(3.0 * t)
    angle = 2.0 * t + 0.16 * wp.sin(0.5 * time)
    return wp.vec3(
        dr * wp.cos(angle) - 2.0 * radius * wp.sin(angle),
        3.0 * (0.53 + 0.025 * wp.cos(0.7 * time)) * wp.cos(3.0 * t),
        dr * wp.sin(angle) + 2.0 * radius * wp.cos(angle),
    )


@wp.func
def surface(t: float, v: float, time: float) -> wp.vec3:
    direction = wp.normalize(tangent(t, time))
    angle = 2.0 * t + 0.16 * wp.sin(0.5 * time)
    radial = wp.vec3(wp.cos(angle), 0.0, wp.sin(angle))
    side = wp.normalize(radial - direction * wp.dot(radial, direction))
    across = wp.cross(direction, side)
    radius = 0.235 + 0.012 * wp.sin(0.8 * time)
    return centerline(t, time) + radius * (wp.cos(v) * side + wp.sin(v) * across)


@wp.kernel(enable_backward=False)
def evaluate_grids(along: int, around: int, edge: int, time: float, points: wp.array[wp.vec3]):
    cluster, vertex = wp.tid()
    row = edge + 1
    # Global integer coordinates make shared edges and both periodic seams agree.
    u = ((cluster // around) * edge + vertex % row) % (along * edge)
    v = ((cluster % around) * edge + vertex // row) % (around * edge)
    t = TAU * float(u) / float(along * edge)
    angle = TAU * float(v) / float(around * edge)
    points[cluster * row * row + vertex] = surface(t, angle, time)


@wp.func
def surface_normal(position: wp.vec3, cluster: int, along: int, around: int, time: float) -> wp.vec3:
    # A cluster identifies a short interval of the centerline. Find the closest
    # point on that interval, then use the circular tube's outward direction.
    # This gives smooth normals without decoding OptiX's triangle index layout.
    lower = TAU * float(cluster // around) / float(along)
    upper = lower + TAU / float(along)
    t = 0.5 * (lower + upper)
    for _ in range(6):
        p = centerline(t, time)
        d = tangent(t, time)
        second = (tangent(t + 0.001, time) - tangent(t - 0.001, time)) / 0.002
        gradient = wp.dot(p - position, d)
        curvature = wp.dot(d, d) + wp.dot(p - position, second)
        t = wp.clamp(t - gradient / wp.max(curvature, 0.01), lower, upper)
    return wp.normalize(position - centerline(t, time))


@wp.func
def studio(direction: wp.vec3) -> wp.vec3:
    blend = wp.clamp(0.5 + 0.5 * direction[1], 0.0, 1.0)
    color = wp.lerp(wp.vec3(0.035, 0.045, 0.065), wp.vec3(0.13, 0.17, 0.23), blend)
    key = wp.pow(wp.max(wp.dot(direction, wp.normalize(wp.vec3(-3.0, 4.0, 3.0))), 0.0), 24.0)
    rim = wp.pow(wp.max(wp.dot(direction, wp.normalize(wp.vec3(3.0, 3.0, -2.0))), 0.0), 40.0)
    return color + key * wp.vec3(3.5, 3.1, 2.6) + rim * wp.vec3(1.4, 2.1, 2.8)


@wp.func
def fresnel(cosine: float, f0: wp.vec3) -> wp.vec3:
    return f0 + (wp.vec3(1.0) - f0) * wp.pow(1.0 - wp.clamp(cosine, 0.0, 1.0), 5.0)


@wp.func
def direct_light(
    scene: wp.uint64,
    position: wp.vec3,
    normal: wp.vec3,
    view: wp.vec3,
    base: wp.vec3,
    roughness: float,
    metallic: float,
    light_point: wp.vec3,
    light_center: wp.vec3,
    energy: wp.vec3,
    area: float,
) -> wp.vec3:
    delta = light_point - position
    distance = wp.length(delta)
    light = delta / distance
    nl = wp.max(wp.dot(normal, light), 0.0)
    nv = wp.max(wp.dot(normal, view), 0.001)
    light_cosine = wp.max(wp.dot(wp.normalize(-light_center), -light), 0.0)
    contribution = wp.vec3()
    if nl > 0.0 and light_cosine > 0.0:
        blocked = wo.accel_query_ray_anyhit(scene, position + 0.001 * normal, light, distance - 0.002, 0.0001)
        if not blocked:
            half = wp.normalize(view + light)
            nh = wp.max(wp.dot(normal, half), 0.0)
            vh = wp.max(wp.dot(view, half), 0.0)
            a2 = roughness * roughness * roughness * roughness
            denominator = nh * nh * (a2 - 1.0) + 1.0
            distribution = a2 / (3.14159265 * denominator * denominator)
            k = (roughness + 1.0) * (roughness + 1.0) / 8.0
            visibility = (nv / (nv * (1.0 - k) + k)) * (nl / (nl * (1.0 - k) + k))
            f = fresnel(vh, wp.lerp(wp.vec3(0.04), base, metallic))
            specular = f * (distribution * visibility / (4.0 * nv * nl + 0.0001))
            diffuse = base * ((1.0 - metallic) / 3.14159265)
            contribution = wp.cw_mul(diffuse + specular, energy) * (nl * light_cosine * area / (distance * distance))
    return contribution


@wp.func
def cluster_color(cluster: int) -> wp.vec3:
    # A stable palette, independent of the render random seed.
    hue = float(cluster) * 2.39996323
    return wp.vec3(0.45 + 0.3 * wp.cos(hue), 0.45 + 0.3 * wp.cos(hue + 2.1), 0.45 + 0.3 * wp.cos(hue + 4.2))


@wp.kernel(module="unique", enable_backward=False)
def render(
    scene: wp.uint64,
    along: int,
    around: int,
    time: float,
    camera: wp.vec3,
    forward: wp.vec3,
    right: wp.vec3,
    up: wp.vec3,
    width: int,
    height: int,
    samples: int,
    seed_frame: int,
    accumulated_frames: int,
    mode: int,
    accumulation: wp.array[wp.vec3],
    image: wp.array[wp.uint32],
    hit_ids: wp.array[int],
):
    y, x = wp.tid()
    pixel = y * width + x
    hit_ids[pixel] = -1
    rng = wp.rand_init(seed_frame * width * height + pixel)
    total = wp.vec3()
    for sample in range(samples):
        sx = (2.0 * (float(x) + wp.randf(rng)) / float(width) - 1.0) * float(width) / float(height)
        sy = 2.0 * (float(y) + wp.randf(rng)) / float(height) - 1.0
        ray = wp.normalize(forward + 0.41421356 * (sx * right + sy * up))
        hit = wo.accel_query_ray(scene, camera, ray, 1.0e6, 0.0001)
        distance = float(1.0e6)
        knot = False
        cid = int(-1)
        if hit.result:
            distance = hit.t
            cid = int(hit.cluster_id)
            knot = True
        floor_distance = float(1.0e6)
        if ray[1] < -0.0001:
            floor_distance = (-0.86 - camera[1]) / ray[1]
        if floor_distance > 0.0 and floor_distance < distance:
            distance = floor_distance
            knot = False
        color = studio(ray) * 0.35
        if distance < 1.0e6:
            position = camera + distance * ray
            normal = wp.vec3(0.0, 1.0, 0.0)
            base = wp.vec3(0.19, 0.21, 0.24)
            roughness = float(0.65)
            metallic = float(0.0)
            if knot:
                normal = surface_normal(position, cid, along, around, time)
                base = wp.vec3(0.025, 0.27, 0.23)
                roughness = 0.25
                metallic = 0.32
                if sample == 0:
                    hit_ids[pixel] = cid
            if wp.dot(normal, ray) > 0.0:
                normal = -normal
            # Two large warm/cool softboxes and a broad overhead fill.
            key_center = wp.vec3(-3.5, 4.5, 3.0)
            key_point = key_center + wp.vec3((wp.randf(rng) - 0.5) * 3.0, 0.0, (wp.randf(rng) - 0.5) * 2.0)
            rim_center = wp.vec3(3.0, 3.0, -2.5)
            rim_point = rim_center + wp.vec3((wp.randf(rng) - 0.5) * 2.0, (wp.randf(rng) - 0.5) * 2.5, 0.0)
            fill_center = wp.vec3(0.0, 5.0, -1.0)
            fill_point = fill_center + wp.vec3((wp.randf(rng) - 0.5) * 3.0, 0.0, (wp.randf(rng) - 0.5) * 3.0)
            color = direct_light(
                scene,
                position,
                normal,
                -ray,
                base,
                roughness,
                metallic,
                key_point,
                key_center,
                wp.vec3(24.0, 20.0, 16.0),
                6.0,
            )
            color += direct_light(
                scene,
                position,
                normal,
                -ray,
                base,
                roughness,
                metallic,
                rim_point,
                rim_center,
                wp.vec3(12.0, 18.0, 26.0),
                5.0,
            )
            color += direct_light(
                scene,
                position,
                normal,
                -ray,
                base,
                roughness,
                metallic,
                fill_point,
                fill_center,
                wp.vec3(5.0, 6.0, 7.0),
                9.0,
            )
            reflected = ray - 2.0 * wp.dot(ray, normal) * normal
            f = fresnel(wp.max(-wp.dot(ray, normal), 0.0), wp.lerp(wp.vec3(0.04), base, metallic))
            color += wp.cw_mul(studio(reflected), f) * 0.45
            # Short visibility rays anchor the sculpture and shade its crossings.
            azimuth = TAU * wp.randf(rng)
            radius = wp.sqrt(wp.randf(rng))
            axis = wp.vec3(0.0, 1.0, 0.0)
            if wp.abs(normal[1]) > 0.95:
                axis = wp.vec3(1.0, 0.0, 0.0)
            basis = wp.normalize(wp.cross(axis, normal))
            ao_ray = radius * wp.cos(azimuth) * basis + radius * wp.sin(azimuth) * wp.cross(normal, basis)
            ao_ray += wp.sqrt(1.0 - radius * radius) * normal
            occluded = wo.accel_query_ray_anyhit(scene, position + 0.001 * normal, ao_ray, 0.65, 0.0001)
            ambient = float(0.14)
            if occluded:
                ambient = 0.025
            color += ambient * base
            if not knot:
                fog = 1.0 - wp.exp(-0.025 * distance)
                color = wp.lerp(color, wp.vec3(0.035, 0.045, 0.065), fog)
            if knot and mode == 1:
                color = cluster_color(cid) * (
                    0.4 + 0.6 * wp.max(wp.dot(normal, wp.normalize(wp.vec3(-1.0, 2.0, 3.0))), 0.0)
                )
            if knot and mode == 2:
                color = 0.5 * (normal + wp.vec3(1.0))
        total += color
    value = total / float(samples)
    if accumulated_frames > 0:
        value = (accumulation[pixel] * float(accumulated_frames) + value) / float(accumulated_frames + 1)
    accumulation[pixel] = value
    # Smooth highlight compression followed by display gamma.
    display = wp.vec3()
    for channel in range(3):
        v = wp.max(value[channel], 0.0)
        mapped = (v * (2.51 * v + 0.03)) / (v * (2.43 * v + 0.59) + 0.14)
        display[channel] = wp.pow(wp.clamp(mapped, 0.0, 1.0), 1.0 / 2.2)
    image[pixel] = (
        wp.uint32(255) << wp.uint32(24)
        | wp.uint32(display[2] * 255.0) << wp.uint32(16)
        | wp.uint32(display[1] * 255.0) << wp.uint32(8)
        | wp.uint32(display[0] * 255.0)
    )


class Example:
    def __init__(self, args):
        self.args = args
        self.width, self.height = args.width, args.height
        self.yaw, self.pitch, self.distance = 0.52, 0.31, args.distance
        self.paused = False
        self.time, self.previous_elapsed = args.time, 0.0
        self.mode = ("shaded", "clusters", "normals").index(args.view)
        self.templates = wo.ClusterTemplates(args.cluster_size, position_truncate_bits=args.truncate)
        self.pipeline = wo.Pipeline(render, clustered_geometry=True)
        self.accel = None
        self.signature = None
        self.accumulated_frames = 0
        self.topology = None

    def draw(self, image, frame, elapsed):
        args = self.args
        if not self.paused:
            self.time += (elapsed - self.previous_elapsed) * args.animate * 0.35
        self.previous_elapsed = elapsed
        topology = (args.along, args.around, args.cluster_size)
        if topology != self.topology:
            count = args.along * args.around
            vertices_per_grid = (args.cluster_size + 1) ** 2
            self.points = wp.empty(count * vertices_per_grid, dtype=wp.vec3)
            self.grid_sizes = wp.array(np.tile([args.cluster_size, args.cluster_size], (count, 1)), dtype=wp.vec2i)
            self.offsets = wp.array(np.arange(count, dtype=np.int32) * vertices_per_grid, dtype=int)
            self.topology = topology
            print(f"{count} clusters, {2 * count * args.cluster_size**2:,} triangles")
        wp.launch(
            evaluate_grids,
            (args.along * args.around, (args.cluster_size + 1) ** 2),
            inputs=[args.along, args.around, args.cluster_size, self.time, self.points],
        )
        if self.accel is None:
            self.accel = wo.ClusterAccel(
                self.points, self.grid_sizes, self.offsets, templates=self.templates, validate=False
            )
        else:
            self.accel.rebuild(
                points=self.points, grid_sizes=self.grid_sizes, vertex_offsets=self.offsets, validate=False
            )
        signature = (topology, self.width, self.height, self.yaw, self.pitch, self.distance, self.time, self.mode)
        if signature != self.signature:
            self.accumulated_frames = 0
            self.signature = signature
        if not hasattr(self, "accumulation") or self.accumulation.size != self.width * self.height:
            self.accumulation = wp.empty(self.width * self.height, dtype=wp.vec3)
            self.hit_ids = wp.empty(self.width * self.height, dtype=int)
        outward = np.array(
            [
                math.sin(self.yaw) * math.cos(self.pitch),
                math.sin(self.pitch),
                math.cos(self.yaw) * math.cos(self.pitch),
            ],
            dtype=np.float32,
        )
        camera = outward * self.distance
        right = np.cross(-outward, [0.0, 1.0, 0.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, -outward)
        wo.launch(
            self.pipeline,
            dim=(self.height, self.width),
            inputs=[
                self.accel,
                args.along,
                args.around,
                self.time,
                wp.vec3(*camera),
                wp.vec3(*(-outward)),
                wp.vec3(*right),
                wp.vec3(*up),
                self.width,
                self.height,
                args.samples,
                frame,
                self.accumulated_frames,
                self.mode,
                self.accumulation,
                image,
                self.hit_ids,
            ],
        )
        self.accumulated_frames += 1

    def close(self):
        wp.synchronize()
        self.pipeline.close()
        if self.accel is not None:
            self.accel.close()
        self.templates.close()


def save_bmp(path, pixels):
    height, width = pixels.shape
    rows = np.stack((pixels >> 16, pixels >> 8, pixels), axis=-1).astype(np.uint8).reshape(height, width * 3)
    rows = np.pad(rows, ((0, 0), (0, -width * 3 % 4)))
    header = struct.pack(
        "<2sIHHIIiiHHIIiiII", b"BM", 54 + rows.size, 0, 0, 54, 40, width, height, 1, 24, 0, rows.size, 0, 0, 0, 0
    )
    path.write_bytes(header + rows.tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--along", type=int, default=32, help="Number of cluster tiles along the knot")
    parser.add_argument("--around", type=int, default=8, help="Number of cluster tiles around the tube")
    parser.add_argument("--cluster-size", type=int, default=8, help="Quads along each grid edge")
    parser.add_argument("--truncate", type=int, default=0, help="Position mantissa bits dropped in CLAS builds")
    parser.add_argument("--animate", type=float, default=1.0, help="Animation speed; 0 enables static accumulation")
    parser.add_argument("--distance", type=float, default=5.8)
    parser.add_argument("--time", type=float, default=0.0)
    parser.add_argument("--time-step", type=float, default=1.0 / 24.0)
    parser.add_argument("--view", choices=["shaded", "clusters", "normals"], default="shaded")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0, help="0: one headless frame or unlimited live frames")
    parser.add_argument("--output", type=Path, default=Path("clusters.bmp"))
    args = parser.parse_args()
    for name in ("width", "height", "samples", "along", "around", "cluster_size"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_frames < 0 or args.distance <= 0:
        parser.error("Invalid frame count or camera distance")
    wp.init()
    with wp.ScopedDevice(args.device):
        app = Example(args)
        try:
            if args.live:
                from warp_optix.gl import GLInteropViewer

                def resize(width, height):
                    app.width, app.height = width, height

                viewer = GLInteropViewer(
                    args.width, args.height, args.device, title="OptiX clusters / parametric knot", on_resize=resize
                )

                def drag(_x, _y, dx, dy, _buttons, _modifiers):
                    app.yaw -= dx * 0.005
                    app.pitch = float(np.clip(app.pitch + dy * 0.005, -0.05, 1.4))

                def scroll(_x, _y, _sx, sy):
                    app.distance = float(np.clip(app.distance * math.exp(-0.1 * sy), 2.5, 15.0))

                def key(symbol, _modifiers):
                    keys = viewer.pyglet.window.key
                    if symbol == keys.SPACE:
                        app.paused = not app.paused
                    elif symbol == keys.V:
                        app.mode = (app.mode + 1) % 3
                    elif symbol in (keys.PLUS, keys.EQUAL):
                        args.along = min(256, args.along + 8)
                    elif symbol == keys.MINUS:
                        args.along = max(8, args.along - 8)

                viewer.window.push_handlers(on_mouse_drag=drag, on_mouse_scroll=scroll, on_key_press=key)
                print("Drag: orbit; scroll: dolly; +/-: tessellation; V: shaded/clusters/normals; Space: pause/refine")
                viewer.run(app.draw, max_frames=args.max_frames)
            else:
                image = wp.empty(args.width * args.height, dtype=wp.uint32)
                for frame in range(args.max_frames or 1):
                    app.draw(image, frame, frame * args.time_step)
                save_bmp(args.output, image.numpy().reshape(args.height, args.width))
                print(f"Wrote {args.output}")
        finally:
            app.close()


if __name__ == "__main__":
    main()
