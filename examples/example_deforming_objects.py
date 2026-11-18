# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Update and query many objects using cluster batches or grouped triangle refits."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from time import perf_counter

import numpy as np
import warp as wp

import warp_optix as wo

PATCHES = wp.constant(8)
EDGE = wp.constant(4)
VERTICES = wp.constant((EDGE + 1) ** 2)
TRIANGLES = 2 * EDGE**2 * PATCHES


@wp.kernel(enable_backward=False)
def update_points(frame: int, points: wp.array[wp.vec3]):
    i = wp.tid()
    body = i // (PATCHES * VERTICES)
    patch = (i // VERTICES) % PATCHES
    vertex = i % VERTICES
    x = float(patch) + float(vertex % (EDGE + 1)) / float(EDGE)
    y = float(vertex // (EDGE + 1)) / float(EDGE)
    # Independent deformation phases. The wave varies across each surface.
    z = 0.15 * wp.sin(0.013 * float(body) + 0.1 * float(frame) + 0.4 * x)
    points[i] = wp.vec3(10.0 * float(body) + x, y, z)


@wp.kernel(enable_backward=False)
def init_instances(records: wp.array[wo.Instance]):
    i = wp.tid()
    records[i] = wo.make_instance(wp.transform_identity(), wp.uint64(0), i)


@wp.func
def ray_origin(body: int) -> wp.vec3:
    return wp.vec3(10.0 * float(body) + 0.41, 0.37, 2.0)


@wp.kernel(module="unique", enable_backward=False)
def trace_scene(scene: wp.uint64, distances: wp.array[float]):
    body = wp.tid()
    hit = wo.accel_query_ray(scene, ray_origin(body), wp.vec3(0.0, 0.0, -1.0), 4.0)
    distances[body] = hit.t if hit.result else -1.0


@wp.kernel(enable_backward=False)
def query_bodies(mesh: wp.uint64, distances: wp.array[float]):
    body = wp.tid()
    root = wp.mesh_get_group_root(mesh, body)
    hit = wp.mesh_query_ray(mesh, ray_origin(body), wp.vec3(0.0, 0.0, -1.0), 4.0, root)
    distances[body] = hit.t if hit.result else -1.0


def triangle_indices(patch_count: int) -> wp.array:
    indices = []
    for y in range(EDGE):
        for x in range(EDGE):
            a = y * (EDGE + 1) + x
            b = a + EDGE + 1
            indices.extend((a, a + 1, b, a + 1, b + 1, b))
    bases = np.arange(patch_count, dtype=np.int32) * VERTICES
    return wp.array((bases[:, None] + np.array(indices, dtype=np.int32)).ravel(), dtype=wp.int32)


def run(mode: str, bodies: int, frames: int) -> np.ndarray:
    """Run on the current CUDA device and return the last frame's hit distances."""
    if not 1 <= bodies <= 0xFFFFFF or frames < 1:
        raise ValueError("Expected a positive frame count and a body count in [1, 16777215]")
    patches = bodies * PATCHES
    flags = wo.BuildFlags.ALLOW_UPDATE | wo.BuildFlags.PREFER_FAST_TRACE
    points = wp.empty(patches * VERTICES, dtype=wp.vec3)
    distances = wp.empty(bodies, dtype=float)
    wp.launch(update_points, points.size, inputs=[0, points])

    with ExitStack() as resources:
        if mode == "clusters":
            templates = resources.enter_context(wo.ClusterTemplates(EDGE))
            grids = wp.full(patches, wp.vec2i(EDGE), dtype=wp.vec2i)
            offsets = wp.array(np.arange(patches, dtype=np.int32) * VERTICES, dtype=wp.int32)
            batch = resources.enter_context(
                wo.ClusterAccelBatch(points, grids, offsets, [PATCHES] * bodies, templates=templates)
            )
            records = wp.empty(bodies, dtype=wo.Instance)
            wp.launch(init_instances, bodies, inputs=[records])
            batch.write_instance_handles(records)
            # Retain scratch for repeated full IAS builds, which accept changed child handles.
            scene = resources.enter_context(wo.Accel([wo.instances(records, [batch])], flags=flags))
        elif mode in ("triangles", "warp"):
            indices = triangle_indices(patches)
            if mode == "triangles":
                scene = resources.enter_context(wo.Accel([wo.triangles(points, indices)], flags=flags))
            else:
                groups = wp.array(np.repeat(np.arange(bodies, dtype=np.int32), TRIANGLES), dtype=wp.int32)
                mesh = wp.Mesh(points, indices, groups=groups)
        else:
            raise ValueError(f"Unknown update mode: {mode}")

        if mode != "warp":
            pipeline = resources.enter_context(wo.Pipeline(trace_scene, clustered_geometry=mode == "clusters"))

        def step(frame):
            wp.launch(update_points, points.size, inputs=[frame, points])
            if mode == "warp":
                mesh.refit()
                wp.launch(query_bodies, bodies, inputs=[mesh.id, distances])
                return
            if mode == "clusters":
                batch.rebuild_async(validate=False)
                batch.write_instance_handles(records)
                scene.rebuild()
            else:
                scene.refit()
            wo.launch(pipeline, bodies, inputs=[scene, distances])

        step(0)  # Compile and warm up before timing.
        wp.synchronize()
        start = perf_counter()
        for frame in range(1, frames + 1):
            step(frame)
        wp.synchronize()
        elapsed = perf_counter() - start
        print(f"{mode}: {bodies:,} bodies, {bodies * TRIANGLES:,} triangles, {elapsed * 1000 / frames:.3f} ms/frame")
        return distances.numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("clusters", "triangles", "warp"), default="clusters")
    parser.add_argument("--bodies", type=int, default=1000)
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    with wp.ScopedDevice(args.device):
        run(args.mode, args.bodies, args.frames)


if __name__ == "__main__":
    main()
