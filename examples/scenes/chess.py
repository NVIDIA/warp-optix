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

"""Procedural chess set with wood textures, generated with NumPy.

Coordinates are in board-square units, with Y up. Components are closed solids;
UV seams and sharp edges intentionally duplicate vertices.
"""

from itertools import pairwise

import numpy as np

CAMERA_POSITION = (10.0, 6.7, 12.5)
CAMERA_TARGET = (0.0, 0.65, 0.0)
CAMERA_FOV = 40.0


def lathe(profile, segments=64, *, round_profile=True):
    """Revolve (radius, height) points, including axis endpoints, about Y.

    Rounded profile corners and interpolated normals give smooth turned forms.
    Axis endpoints produce triangle fans rather than zero-area triangles.
    """
    vertices, normals, uvs, triangles = [], [], [], []
    profile = np.asarray(profile, dtype=np.float64)
    # Corner cutting rounds the authored profile without overshoot or dependencies.
    for _ in range(2 if round_profile else 0):
        rounded = [profile[0]]
        for a, b in pairwise(profile):
            rounded.extend((0.8 * a + 0.2 * b, 0.2 * a + 0.8 * b))
        rounded.append(profile[-1])
        profile = np.asarray(rounded)
    tangent = np.gradient(profile, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    tangent[0] = (np.sign(tangent[0, 0]), 0)
    tangent[-1] = (np.sign(tangent[-1, 0]), 0)
    heights = [p[1] for p in profile]
    height = max(heights) - min(heights)
    for k, ((r0, y0), (r1, y1)) in enumerate(pairwise(profile)):
        for j in range(segments):
            angles = [2 * np.pi * j / segments, 2 * np.pi * (j + 1) / segments]
            start = len(vertices)
            for row, (r, y, angle, u) in enumerate(
                [
                    (r0, y0, angles[0], j / segments),
                    (r1, y1, angles[0], j / segments),
                    (r1, y1, angles[1], (j + 1) / segments),
                    (r0, y0, angles[1], (j + 1) / segments),
                ]
            ):
                dr, dy = tangent[k + (row in (1, 2))]
                vertices.append((r * np.cos(angle), y, r * np.sin(angle)))
                normals.append(
                    (
                        dy * np.cos(angle),
                        -dr,
                        dy * np.sin(angle),
                    )
                )
                uvs.append((u, (y - min(heights)) / height))
            if r1 > 0:
                triangles.append((start, start + 1, start + 2))
            if r0 > 0:
                triangles.append((start, start + 2, start + 3))
    return _arrays(vertices, triangles, normals, uvs)


def extrude(outline, depth):
    """Extrude a simple counterclockwise XY polygon, with flat sides and UVs."""
    points = np.asarray(outline, dtype=np.float64)

    def cross(a, b, c):
        ab, ac = b - a, c - a
        return ab[0] * ac[1] - ab[1] * ac[0]

    remaining = list(range(len(points)))
    faces = []
    while len(remaining) > 3:
        for k, b in enumerate(remaining):
            a, c = remaining[k - 1], remaining[(k + 1) % len(remaining)]
            if cross(points[a], points[b], points[c]) <= 1e-10:
                continue
            if any(
                all(cross(points[i], points[j], points[q]) >= -1e-10 for i, j in [(a, b), (b, c), (c, a)])
                for q in remaining
                if q not in (a, b, c)
            ):
                continue
            faces.append((a, b, c))
            remaining.pop(k)
            break
        else:
            raise ValueError("Outline must be a simple counterclockwise polygon")
    faces.append(tuple(remaining))
    vertices, normals, uvs, triangles = [], [], [], []
    lo, span = points.min(axis=0), np.ptp(points, axis=0)
    for sign in (-1, 1):
        offset = len(vertices)
        for p in points:
            vertices.append((*p, sign * depth / 2))
            normals.append((0, 0, sign))
            uvs.append(tuple((p - lo) / span))
        for a, b, c in faces:
            triangles.append(tuple(offset + i for i in ((c, b, a) if sign < 0 else (a, b, c))))
    perimeter = 0.0
    for a, b in zip(points, np.roll(points, -1, axis=0)):
        d = b - a
        length = np.linalg.norm(d)
        start = len(vertices)
        normal = (d[1] / length, -d[0] / length, 0)
        for p, z, u, v in [
            (a, -depth / 2, perimeter, 0),
            (b, -depth / 2, perimeter + length, 0),
            (b, depth / 2, perimeter + length, depth),
            (a, depth / 2, perimeter, depth),
        ]:
            vertices.append((*p, z))
            normals.append(normal)
            uvs.append((u, v))
        triangles.extend([(start, start + 1, start + 2), (start, start + 2, start + 3)])
        perimeter += length
    return _arrays(vertices, triangles, normals, uvs)


def _arrays(vertices, triangles, normals, uvs):
    return tuple(
        np.asarray(a, dtype=d)
        for a, d in [
            (vertices, np.float32),
            (triangles, np.uint32),
            (normals, np.float32),
            (uvs, np.float32),
        ]
    )


def wood_texture(light, dark, size=512):
    """Generate irregular, seamless growth lines and longitudinal fibers."""
    rng = np.random.default_rng(41)
    v, u = np.mgrid[:size, :size].astype(np.float32) / size

    def noise(rows, columns):
        grid = rng.random((rows, columns), dtype=np.float32)
        x, y = u * columns, v * rows
        ix, iy = x.astype(int), y.astype(int)
        fx, fy = x - ix, y - iy
        fx = fx * fx * (3 - 2 * fx)
        fy = fy * fy * (3 - 2 * fy)
        a = grid[iy % rows, ix % columns] * (1 - fx) + grid[iy % rows, (ix + 1) % columns] * fx
        b = grid[(iy + 1) % rows, ix % columns] * (1 - fx) + grid[(iy + 1) % rows, (ix + 1) % columns] * fx
        return a * (1 - fy) + b * fy

    # Nonuniform ring spacing and slowly varying deviations avoid repeated waves.
    warped = u + 0.032 * (noise(3, 7) - 0.5) + 0.009 * (noise(9, 19) - 0.5)
    gaps = rng.uniform(0.55, 1.6, 24)
    centers = np.cumsum(gaps) / gaps.sum()
    lines = np.zeros_like(u)
    shoulders = np.zeros_like(u)
    for center in centers:
        distance = np.abs((warped - center + 0.5) % 1 - 0.5)
        width = rng.uniform(0.003, 0.006)
        lines = np.maximum(lines, np.exp(-((distance / width) ** 2)) * rng.uniform(0.8, 1))
        shoulders = np.maximum(shoulders, np.exp(-((distance / (width * 3)) ** 2)))
    fibers = noise(5, 140)
    cloud = noise(3, 11)
    strength = 0.75 + 0.25 * noise(9, 29)
    blend = np.clip(
        0.08 + 0.17 * cloud + 0.08 * fibers + 0.20 * shoulders + 0.48 * lines * strength,
        0,
        1,
    )
    rgb = np.asarray(light)[None, None, :] * (1 - blend[..., None])
    rgb += np.asarray(dark)[None, None, :] * blend[..., None]
    return np.concatenate(((rgb * 255).astype(np.uint8), np.full((size, size, 1), 255, np.uint8)), axis=2)


def wood_uvs(positions, grain_axis=1, origin=(0.0, 0.0, 0.0)):
    """Map a cut timber volume at one fixed scale, in board-square units.

    The texture spans two scene units. Growth rings use distance from an
    off-center trunk axis; the second coordinate follows the timber fibers.
    Unlike normalized mesh UVs, narrowing a stem does not compress its grain.
    """
    points = np.asarray(positions, dtype=np.float64) + np.asarray(origin)
    transverse = [axis for axis in range(3) if axis != grain_axis]
    radius = np.hypot(points[:, transverse[0]] + 1.7, points[:, transverse[1]] + 0.8)
    return np.column_stack((radius, points[:, grain_axis])).astype(np.float32) / 2.0


def timber_box(size):
    """Subdivide planar timber faces so curved growth rings interpolate accurately."""
    half = np.asarray(size) / 2
    steps = np.maximum(1, np.ceil(np.asarray(size) / 0.25).astype(int))
    vertices = []
    normals = []
    triangles = []
    for axis in range(3):
        u, v = (axis + 1) % 3, (axis + 2) % 3
        for sign in (-1, 1):
            start = len(vertices)
            normal = np.zeros(3)
            normal[axis] = sign
            for x in np.linspace(-half[u], half[u], steps[u] + 1):
                for y in np.linspace(-half[v], half[v], steps[v] + 1):
                    point = np.zeros(3)
                    point[axis] = sign * half[axis]
                    point[u] = x
                    point[v] = y
                    vertices.append(point)
                    normals.append(normal)
            stride = steps[v] + 1
            for i in range(steps[u]):
                for j in range(steps[v]):
                    a = start + i * stride + j
                    faces = [
                        (a, a + stride, a + stride + 1),
                        (a, a + stride + 1, a + 1),
                    ]
                    triangles.extend(faces if sign > 0 else [face[::-1] for face in faces])
    return _arrays(vertices, triangles, normals, np.zeros((len(vertices), 2)))


def beveled_extrude(outline, depth):
    """Round an extruded silhouette's edges with two shoulder rings."""
    outline = np.asarray(outline, dtype=np.float64)
    center = outline.mean(axis=0)
    inner = center + 0.88 * (outline - center)
    p, t, _, uv = extrude(inner, depth)
    count = len(outline)
    # Keep the two triangulated end caps; replace the vertical walls.
    p = list(p[: 2 * count])
    uv = list(uv[: 2 * count])
    t = list(t[: 2 * (count - 2)])
    for z in (-depth * 0.28, depth * 0.28):
        p.extend((x, y, z) for x, y in outline)
        uv.extend((outline - outline.min(axis=0)) / np.ptp(outline, axis=0))
    for first, second in [(0, 2 * count), (2 * count, 3 * count), (3 * count, count)]:
        for i in range(count):
            j = (i + 1) % count
            t.extend(
                [
                    (first + i, first + j, second + j),
                    (first + i, second + j, second + i),
                ]
            )
    p = np.asarray(p)
    t = np.asarray(t)
    # Cap triangulation must not influence the side normals. In particular,
    # a long ear-clipped triangle otherwise bends the shading across a flat face.
    edges = np.roll(outline, -1, axis=0) - outline
    edge_normals = np.column_stack((edges[:, 1], -edges[:, 0]))
    edge_normals /= np.linalg.norm(edge_normals, axis=1)[:, None]
    outward = edge_normals + np.roll(edge_normals, 1, axis=0)
    outward /= np.linalg.norm(outward, axis=1)[:, None]
    n = np.zeros_like(p)
    n[:count, 2] = -1
    n[count : 2 * count, 2] = 1
    n[2 * count : 3 * count, :2] = outward
    n[3 * count : 4 * count, :2] = outward
    # Duplicate the two cap boundaries for the bevel, retaining planar cap normals.
    bevel_p = np.concatenate((p[:count], p[count : 2 * count]))
    bevel_n = np.concatenate(
        (
            np.column_stack((outward, -np.ones(count))),
            np.column_stack((outward, np.ones(count))),
        )
    ) / np.sqrt(2)
    wall = t[2 * (count - 2) :].copy()
    wall = np.where(wall < 2 * count, wall + 4 * count, wall)
    t = np.concatenate((t[: 2 * (count - 2)], wall))
    return _arrays(
        np.concatenate((p, bevel_p)),
        t,
        np.concatenate((n, bevel_n)),
        np.concatenate((uv, np.asarray(uv)[: 2 * count])),
    )


def _smooth_normals(p, triangles):
    """Average curved-surface normals across duplicated UV seams."""
    _, welded = np.unique(np.round(p, 7), axis=0, return_inverse=True)
    face = np.cross(p[triangles[:, 1]] - p[triangles[:, 0]], p[triangles[:, 2]] - p[triangles[:, 0]])
    normals = np.zeros((welded.max() + 1, 3))
    for col in range(3):
        np.add.at(normals, welded[triangles[:, col]], face)
    normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-20)
    return normals[welded]


def horse_head():
    """Sculpt an original horse from elliptical sections, with a curved neck."""
    # Each section is (height, center X, radius X, radius Z).
    sections = np.array(
        [
            (0.46, -0.055, 0, 0),
            (0.48, -0.055, 0.18, 0.14),
            (0.65, -0.075, 0.24, 0.19),
            (0.85, -0.12, 0.22, 0.18),
            (1.05, -0.16, 0.17, 0.13),
            (1.20, -0.18, 0.14, 0.11),
            (1.25, 0.08, 0.34, 0.13),
            (1.32, 0.08, 0.35, 0.15),
            (1.40, 0.04, 0.34, 0.15),
            (1.47, -0.02, 0.27, 0.15),
            (1.55, -0.10, 0.17, 0.13),
            (1.60, -0.13, 0.10, 0.09),
            (1.63, -0.13, 0, 0),
        ]
    )
    for _ in range(3):
        refined = [sections[0]]
        for a, b in pairwise(sections):
            refined.extend((0.75 * a + 0.25 * b, 0.25 * a + 0.75 * b))
        refined.append(sections[-1])
        sections = np.asarray(refined)
    vertices = []
    uv = []
    triangles = []
    segments = 64
    for k, (y, cx, rx, rz) in enumerate(sections):
        for j in range(segments + 1):
            angle = 2 * np.pi * j / segments
            vertices.append((cx + rx * np.cos(angle), y, rz * np.sin(angle)))
            uv.append((j / segments, k / (len(sections) - 1)))
    for k in range(len(sections) - 1):
        for j in range(segments):
            a = k * (segments + 1) + j
            b = a + segments + 1
            if k < len(sections) - 2:
                triangles.append((a, b, b + 1))
            if k > 0:
                triangles.append((a, b + 1, a + 1))
    p = np.asarray(vertices)
    t = np.asarray(triangles)
    # Let the muzzle slope down from the forehead, with a rounded tip rather
    # than a flat snout. The neck lies behind X=0 and retains its arch.
    p[:, 1] -= 0.30 * np.maximum(p[:, 0], 0) * np.clip((p[:, 1] - 1.0) / 0.2, 0, 1)
    return _arrays(p, t, _smooth_normals(p, t), uv)


def crown():
    """A hollow, eight-point coronet with a rounded, fluted rim."""
    profile = [
        (0, 1.60),
        (0.16, 1.60),
        (0.18, 1.76),
        (0.22, 1.88),
        (0.28, 2.00),
        (0.29, 2.035),
        (0.26, 2.045),
        (0.23, 1.97),
        (0.17, 1.82),
        (0.13, 1.75),
        (0, 1.75),
    ]
    p, t, _, uv = lathe(profile, segments=128)
    angle = np.arctan2(p[:, 2], p[:, 0])
    weight = np.clip((p[:, 1] - 1.78) / 0.25, 0, 1) ** 2
    wave = np.cos(8 * angle)
    p[:, 1] += 0.065 * wave * weight
    p[:, [0, 2]] *= (1 + 0.045 * wave * weight)[:, None]
    return _arrays(p, t, _smooth_normals(p, t), uv)


def _clip_closed_mesh(data, normal, offset):
    """Keep one side of a plane and cap its convex intersection (mitre helper)."""
    p, t, n, uv = data
    normal = np.asarray(normal, dtype=float)
    normal /= np.linalg.norm(normal)
    positions = []
    normals = []
    texcoords = []
    triangles = []
    boundary = {}
    for triangle in t:
        polygon = [(p[i].astype(float), n[i].astype(float), uv[i].astype(float)) for i in triangle]
        clipped = []
        for a, b in zip(polygon, polygon[1:] + polygon[:1]):
            da = np.dot(a[0], normal) - offset
            db = np.dot(b[0], normal) - offset
            if da >= 0:
                clipped.append(a)
            if (da >= 0) != (db >= 0):
                f = da / (da - db)
                point = tuple(x + f * (y - x) for x, y in zip(a, b))
                key = tuple(np.round(point[0], 6))
                canonical = boundary.setdefault(key, point[0])
                clipped.append((canonical, point[1], point[2]))
        if len(clipped) < 3:
            continue
        start = len(positions)
        for v, vn, vt in clipped:
            positions.append(v)
            normals.append(vn / np.linalg.norm(vn))
            texcoords.append(vt)
        for j in range(1, len(clipped) - 1):
            triangles.append((start, start + j, start + j + 1))
    if boundary:
        points = np.asarray(list(boundary.values()))
        center = points.mean(axis=0)
        axis = np.eye(3)[np.argmin(np.abs(normal))]
        u = np.cross(normal, axis)
        u /= np.linalg.norm(u)
        v = np.cross(normal, u)
        relative = points - center
        order = np.argsort(np.arctan2(relative @ v, relative @ u))
        points = points[order]
        start = len(positions)
        positions.append(center)
        normals.append(-normal)
        texcoords.append((0, 0))
        for point in points:
            positions.append(point)
            normals.append(-normal)
            texcoords.append(((point - center) @ u, (point - center) @ v))
        for j in range(len(points)):
            triangles.append((start, start + 1 + (j + 1) % len(points), start + 1 + j))
    return _arrays(positions, triangles, normals, texcoords)


def build_chess_scene(api, pieces="wood"):
    """Populate an initialized, empty PathTracer with a complete chess set.

    ``pieces="glass"`` makes the light pieces clear glass and the dark pieces polished gold.
    """
    textures = [
        wood_texture((0.89, 0.75, 0.52), (0.64, 0.43, 0.22)),
        wood_texture((0.43, 0.26, 0.14), (0.18, 0.075, 0.025)),
        wood_texture((0.91, 0.78, 0.56), (0.67, 0.46, 0.25)),
        wood_texture((0.46, 0.28, 0.15), (0.20, 0.09, 0.035)),
    ]
    api.scene.set_gltf_textures(textures, srgb_texture_indices=set(range(4)))

    glass_and_gold = None
    if pieces == "glass":
        glass_and_gold = (
            api.scene.materials.add_gltf_material(
                base_color=(1, 1, 1, 1),
                metallic=0.0,
                roughness=0.0,
                transmission=1.0,
                ior=1.52,
                thickness=1.0,
                attenuation_color=(0.75, 0.9, 1.0),
                attenuation_distance=6.0,
                double_sided=True,
            ),
            api.scene.materials.add_metal((1.0, 0.77, 0.33), 0.06),
        )
    elif pieces != "wood":
        raise ValueError(f"pieces must be 'wood' or 'glass', not {pieces!r}")

    def wood_material(i, offset=(0, 0)):
        if glass_and_gold is not None and i >= 2:
            return glass_and_gold[i - 2]
        return api.scene.materials.add_gltf_material(
            base_color=(1, 1, 1, 1),
            base_color_texture={"index": i, "transform": {"offset": offset}},
            metallic=0.0,
            roughness=0.34 if i < 2 else 0.30,
            clearcoat=0.12,
            base_color_scale=1.0,
        )

    mats = [wood_material(i) for i in range(4)]
    mesh_materials = {}
    mesh_data = {}
    brass = api.create_pbr_material(color=(0.83, 0.49, 0.10), roughness=0.19, metallic=1.0, base_color_scale=1)
    floor = api.create_pbr_material(color=(0.16, 0.17, 0.18), roughness=0.65, metallic=0, base_color_scale=1)

    def mesh(
        data,
        material,
        *,
        grain_axis=1,
        grain_origin=(0, 0, 0),
        grain_angle=0,
        grain_slope=0,
    ):
        p, t, n, uv = data
        if material in mats:
            timber = p.copy()
            c, s = np.cos(grain_angle), np.sin(grain_angle)
            timber[:, 0] = c * p[:, 0] - s * p[:, 2]
            timber[:, 2] = s * p[:, 0] + c * p[:, 2]
            # A shallow angled cut through cylindrical growth rings produces
            # natural arches along a plank rather than painted sine waves.
            if grain_axis != 1:
                timber[:, 1] += grain_slope * timber[:, grain_axis]
            uv = wood_uvs(timber, grain_axis=grain_axis, origin=grain_origin)
        mid = api.create_mesh(p, t, normals=n, uvs=uv, material_id=material)
        mesh_materials[mid] = material
        mesh_data[mid] = (p, t, n, uv)
        return mid

    def place(mid, position, scale=1.0):
        return api.create_instance_with_transform(mid, position, (0, 0, 0, 1), scale=scale)

    def box(
        x,
        y,
        z,
        material,
        position,
        *,
        grain_axis=2,
        grain_origin=None,
        grain_angle=0,
        grain_slope=0,
    ):
        data = (
            timber_box((x, y, z))
            if material in mats
            else extrude([(-x / 2, -y / 2), (x / 2, -y / 2), (x / 2, y / 2), (-x / 2, y / 2)], z)
        )
        mid = mesh(
            data,
            material,
            grain_axis=grain_axis,
            grain_origin=position if grain_origin is None else grain_origin,
            grain_angle=grain_angle,
            grain_slope=grain_slope,
        )
        place(mid, position)

    box(2000, 0.15, 2000, floor, (0, -0.3, 0))
    box(8, 0.22, 8, mats[1], (0, -0.01, 0))
    # Four timber rails: fibers follow each rail's length, including the end rails.
    for sign in (-1, 1):
        box(0.7, 0.22, 9.4, mats[1], (sign * 4.35, -0.01, 0), grain_axis=2)
        box(8, 0.22, 0.7, mats[1], (0, -0.01, sign * 4.35), grain_axis=0)
    box(8.3, 0.035, 8.3, brass, (0, 0.115, 0))
    cuts = np.random.default_rng(73)
    for row in range(8):
        for col in range(8):
            axis = 0 if (row // 2 + col) % 3 == 0 else 2
            shift, depth, along = (
                cuts.uniform(-0.4, 0.4),
                cuts.uniform(0.25, 0.65),
                cuts.uniform(-2, 2),
            )
            origin = (along, -1.7 + depth, -0.8 + shift) if axis == 0 else (-1.7 + shift, -0.8 + depth, along)
            box(
                0.997,
                0.09,
                0.997,
                mats[(row + col) % 2],
                (col - 3.5, 0.17, row - 3.5),
                grain_axis=axis,
                # Each square is a separate near-tangential cut from round timber.
                grain_origin=origin,
                grain_slope=cuts.uniform(0.035, 0.075),
                grain_angle=np.deg2rad(cuts.uniform(-8, 8)),
            )

    base = [
        (0, 0),
        (0.29, 0),
        (0.345, 0.025),
        (0.36, 0.065),
        (0.355, 0.105),
        (0.31, 0.14),
        (0.28, 0.16),
        (0.325, 0.205),
        (0.31, 0.245),
        (0.245, 0.29),
        (0.21, 0.34),
    ]
    stems = {
        "pawn": [
            (0.15, 0.28),
            (0.115, 0.36),
            (0.085, 0.49),
            (0.085, 0.59),
            (0.12, 0.65),
            (0.18, 0.68),
            (0.18, 0.705),
            (0.13, 0.73),
            (0, 0.73),
        ],
        "rook": [
            (0.25, 0.38),
            (0.23, 0.46),
            (0.22, 0.65),
            (0.215, 0.96),
            (0.23, 1.12),
            (0.29, 1.15),
            (0.30, 1.20),
            (0.30, 1.27),
            (0.24, 1.29),
            (0, 1.29),
        ],
        "knight": [
            (0.24, 0.37),
            (0.28, 0.405),
            (0.28, 0.44),
            (0.22, 0.475),
            (0, 0.475),
        ],
        "bishop": [
            (0.19, 0.39),
            (0.14, 0.50),
            (0.10, 0.73),
            (0.095, 1.0),
            (0.13, 1.12),
            (0.23, 1.17),
            (0.23, 1.205),
            (0.16, 1.25),
            (0.095, 1.29),
            (0.15, 1.35),
            (0.215, 1.45),
            (0.20, 1.55),
            (0.13, 1.65),
            (0.05, 1.76),
            (0, 1.80),
        ],
        "queen": [
            (0.19, 0.43),
            (0.14, 0.63),
            (0.12, 0.94),
            (0.115, 1.29),
            (0.14, 1.45),
            (0.23, 1.50),
            (0.27, 1.53),
            (0.27, 1.57),
            (0.18, 1.62),
            (0.17, 1.69),
            (0.22, 1.84),
            (0.28, 1.99),
            (0.28, 2.035),
            (0.22, 2.02),
            (0.15, 1.80),
            (0, 1.80),
        ],
        "king": [
            (0.19, 0.43),
            (0.15, 0.66),
            (0.125, 1.03),
            (0.12, 1.38),
            (0.14, 1.55),
            (0.24, 1.61),
            (0.28, 1.64),
            (0.28, 1.68),
            (0.19, 1.73),
            (0.17, 1.79),
            (0.25, 2.015),
            (0.25, 2.055),
            (0, 2.055),
        ],
    }

    def ball(radius, center):
        angles = np.linspace(-np.pi / 2, np.pi / 2, 33)
        return [(0 if i in (0, 32) else radius * np.cos(a), center + radius * np.sin(a)) for i, a in enumerate(angles)]

    proportions = {
        "pawn": 0.76,
        "rook": 1.0,
        "knight": 1.0,
        "bishop": 0.94,
        "queen": 1.0,
        "king": 1.05,
    }
    arrangement = [
        "rook",
        "knight",
        "bishop",
        "queen",
        "king",
        "bishop",
        "knight",
        "rook",
    ]
    piece_cuts = np.random.default_rng(119)
    grain_directions = np.random.default_rng(271)
    for side in range(2):
        material = mats[2 + side]
        for kind, profile in stems.items():
            width = proportions[kind]
            foot = [(r * width, y * (0.70 if kind == "pawn" else 1)) for r, y in base]
            ring = mesh(
                lathe(
                    [
                        (0, 0.042),
                        (0.356 * width, 0.042),
                        (0.356 * width, 0.07),
                        (0, 0.07),
                    ]
                ),
                brass,
            )
            body_profile = profile
            if kind == "king":
                body_profile = [
                    (
                        r * (1.20 if y >= 1.73 else 1),
                        y + 0.08 * np.clip((y - 1.73) / 0.325, 0, 1),
                    )
                    for r, y in profile
                ]
            if kind == "bishop":
                body_profile = [point for point in profile if point[1] <= 1.29] + [(0, 1.29)]
            if kind == "queen":
                body_profile = [point for point in profile if point[1] <= 1.62] + [(0, 1.62)]
            components = [(mesh(lathe(foot + body_profile), material), (0, 0, 0))]
            if kind == "queen":
                components.append((mesh(crown(), material), (0, 0, 0)))
            if kind in ("bishop", "queen", "king"):
                y = {"bishop": 1.19, "queen": 1.55, "king": 1.66}[kind]
                r = {"bishop": 0.232, "queen": 0.272, "king": 0.282}[kind]
                components.append(
                    (
                        mesh(
                            lathe(
                                [
                                    (0, y - 0.012),
                                    (r, y - 0.012),
                                    (r, y + 0.012),
                                    (0, y + 0.012),
                                ]
                            ),
                            brass,
                        ),
                        (0, 0, 0),
                    )
                )
            if kind == "pawn":
                components.append(
                    (
                        mesh(lathe(ball(0.18, 0.89), round_profile=False), material),
                        (0, 0, 0),
                    )
                )
            if kind == "knight":
                components.append((mesh(horse_head(), material), (0, 0, 0)))
                # Two separate tapered ears.
                ear = [(-0.20, 1.55), (-0.08, 1.56), (-0.18, 1.77)]
                mane = [
                    (-0.34, 0.75),
                    (-0.29, 0.80),
                    (-0.28, 1.10),
                    (-0.27, 1.30),
                    (-0.20, 1.50),
                    (-0.14, 1.60),
                    (-0.23, 1.64),
                    (-0.33, 1.45),
                    (-0.39, 1.20),
                    (-0.40, 0.95),
                ]
                components.append((mesh(beveled_extrude(mane, 0.10), material), (0, 0, 0)))
                ear_mesh = mesh(beveled_extrude(ear, 0.07), material)
                components.extend((ear_mesh, (0, 0, z)) for z in (-0.105, 0.105))
            if kind == "king":
                cross = [
                    (-0.07, 2.025),
                    (0.07, 2.025),
                    (0.045, 2.19),
                    (0.16, 2.17),
                    (0.16, 2.31),
                    (0.045, 2.29),
                    (0.075, 2.43),
                    (-0.075, 2.43),
                    (-0.045, 2.29),
                    (-0.16, 2.31),
                    (-0.16, 2.17),
                    (-0.045, 2.19),
                ]
                cross = [(x * 1.18, 2.105 + (y - 2.025) * 1.08) for x, y in cross]
                components.append((mesh(beveled_extrude(cross, 0.18), brass), (0, 0, 0)))
            if kind == "bishop":
                head = lathe(
                    [
                        (0, 1.25),
                        (0.095, 1.29),
                        (0.15, 1.35),
                        (0.215, 1.45),
                        (0.20, 1.55),
                        (0.13, 1.65),
                        (0.05, 1.76),
                        (0, 1.80),
                    ]
                )
                bottom = _clip_closed_mesh(head, (0, -1, 0), -1.48)
                upper = _clip_closed_mesh(head, (0, 1, 0), 1.48)
                normal = np.array((0.75, -0.66, 0))
                normal /= np.linalg.norm(normal)
                offset = normal[1] * 1.59
                left = _clip_closed_mesh(upper, normal, offset + 0.013)
                right = _clip_closed_mesh(upper, -normal, -offset + 0.013)
                components.extend((mesh(part, material), (0, 0, 0)) for part in (bottom, left, right))
            if kind == "bishop":
                components.append(
                    (
                        mesh(lathe(ball(0.065, 1.80), round_profile=False), material),
                        (0, 0, 0),
                    )
                )
            if kind == "queen":
                components.append(
                    (
                        mesh(lathe(ball(0.105, 2.04), round_profile=False), brass),
                        (0, 0, 0),
                    )
                )
            if kind == "rook":
                for angle in np.arange(6) * np.pi / 3:
                    angles = np.linspace(angle + 0.10, angle + np.pi / 3 - 0.10, 9)
                    outline = [(0.30 * np.cos(a), 0.30 * np.sin(a)) for a in angles]
                    outline += [(0.205 * np.cos(a), 0.205 * np.sin(a)) for a in angles[::-1]]
                    p, t, n, uv = beveled_extrude(outline, 0.22)
                    # Turn the annular sector into a vertical battlement.
                    p = p[:, [0, 2, 1]]
                    p[:, 1] += 1.36
                    n = n[:, [0, 2, 1]]
                    t = t[:, [0, 2, 1]]
                    components.append((mesh((p, t, n, uv), material), (0, 0, 0)))
            if kind == "queen":
                bead = mesh(lathe(ball(0.033, 2.105), round_profile=False), brass)
                components.extend((bead, (0.28 * np.cos(a), 0, 0.28 * np.sin(a))) for a in np.arange(8) * np.pi / 4)
            squares = (
                [(i, 1 if side == 0 else 6) for i in range(8)]
                if kind == "pawn"
                else [(i, 0 if side == 0 else 7) for i, k in enumerate(arrangement) if k == kind]
            )
            for col, row in squares:
                # Each piece samples a different part of the same timber texture.
                # Rotate the timber around the piece's up axis before computing
                # cylindrical UVs, keeping all wooden components in the same cut.
                # A 2D UV rotation would incorrectly mix radial and axial grain.
                piece_material = wood_material(2 + side, piece_cuts.random(2).tolist())
                grain_angle = grain_directions.uniform(-np.pi, np.pi)
                c, s = np.cos(grain_angle), np.sin(grain_angle)
                origin = np.array((col - 3.5, 0.215, row - 3.5))
                place(ring, origin)
                for mid, offset in components:
                    offset = np.asarray(offset)
                    if mesh_materials[mid] == material:
                        p, t, n, _ = mesh_data[mid]
                        timber = p + offset
                        rotated = timber.copy()
                        rotated[:, 0] = c * timber[:, 0] - s * timber[:, 2]
                        rotated[:, 2] = s * timber[:, 0] + c * timber[:, 2]
                        # UVs belong to mesh vertices, so each wooden component
                        # needs its own mesh for this piece's timber orientation.
                        mid = api.create_mesh(p, t, normals=n, uvs=wood_uvs(rotated), material_id=material)
                        mesh_materials[mid] = material
                    # Rotate the horse as a whole so opposing sets face inward.
                    if kind == "knight":
                        angle = -np.pi / 2 if side == 0 else np.pi / 2
                        x, y, z = offset
                        offset = (
                            np.cos(angle) * x + np.sin(angle) * z,
                            y,
                            -np.sin(angle) * x + np.cos(angle) * z,
                        )
                        instance = api.create_instance_with_transform(
                            mid,
                            origin + offset,
                            (0, np.sin(angle / 2), 0, np.cos(angle / 2)),
                        )
                    else:
                        instance = place(mid, origin + offset)
                    if mesh_materials[mid] == material:
                        api.set_instance_material(instance, piece_material)
    api.set_use_procedural_sky(True)
    api.set_sky_parameters(
        (-0.4, 0.8, 0.3),
        multiplier=0.5,
        haze=0.2,
        saturation=0.6,
        sun_disk_intensity=0.0,
        sun_glow_intensity=0.0,
    )
    api.scene.add_light_sphere(position=(-3, 7, 2), radius=1.8, color=(1, 0.87, 0.70), intensity=35)
    api.scene.add_light_sphere(position=(4, 5, -3), radius=1.2, color=(0.70, 0.85, 1), intensity=15)
