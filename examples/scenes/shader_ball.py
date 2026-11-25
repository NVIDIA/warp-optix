# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Procedural shader ball and pedestal for the material gallery."""

import math
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=3)
def _lathe(profile, planar_uv=False, shell_tiles=0):
    """Revolve a front-to-back (radius, axial position) profile around local Z."""
    profile = np.asarray(profile, dtype=np.float64)
    radius, axial = profile.T
    tangent = np.gradient(profile, axis=0)
    normal = np.stack((-tangent[:, 1], tangent[:, 0]), axis=1)
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    if radius[0] == 0:
        normal[0] = (0, 1)
    if radius[-1] == 0:
        normal[-1] = (0, -1)
    segments = 96
    u = np.linspace(0, 1, segments + 1)
    angle = 2 * math.pi * u
    if shell_tiles:
        angle -= math.pi / 2  # Put the longitudinal seam on the underside.
    radial = np.stack((np.cos(angle), np.sin(angle)), axis=1)

    vertices = np.empty((len(profile), segments + 1, 3))
    vertices[..., :2] = radius[:, None, None] * radial
    vertices[..., 2] = axial[:, None]
    normals = np.empty_like(vertices)
    normals[..., :2] = normal[:, None, :1] * radial
    normals[..., 2] = normal[:, None, 1]
    if planar_uv:
        uvs = 0.5 + vertices[..., :2] / (2 * radius.max())
        uvs[..., 1] = 1 - uvs[..., 1]
    else:
        length = np.r_[0, np.cumsum(np.linalg.norm(np.diff(profile, axis=0), axis=1))]
        if shell_tiles:
            # Match meridian spacing to the equator's texel density. A whole
            # number of tiles around the axis keeps periodic textures aligned.
            v = length * shell_tiles / (2 * math.pi * radius.max())
            u *= shell_tiles
        else:
            v = length / length[-1]
        uvs = np.stack(np.broadcast_arrays(u[None, :], v[:, None]), axis=-1)

    start = np.arange(len(profile) - 1)[:, None] * (segments + 1) + np.arange(segments)
    next_ring = start + segments + 1
    faces = np.stack((start, next_ring, start + 1, start + 1, next_ring, next_ring + 1), axis=-1)
    bands = np.broadcast_to(np.arange(len(profile) - 1)[:, None, None], faces.shape[:-1] + (2,))
    valid = np.ones_like(bands, dtype=bool)
    if radius[0] == 0:
        valid[0, :, 0] = False
    if radius[-1] == 0:
        valid[-1, :, 1] = False
    return (
        vertices.reshape(-1, 3),
        normals.reshape(-1, 3),
        uvs.reshape(-1, 2),
        faces.reshape(-1, 3)[valid.ravel()].astype(np.uint32),
        bands[valid],
    )


# Slightly convex material face, stepped bezel, then a round shell.
_FACE = ((0, .575), (.09, .574), (.20, .568), (.30, .556), (.37, .543), (.405, .531), (.417, .519))
_BEZEL = (
    (.417, .519), (.423, .505), (.425, .474), (.450, .466), (.463, .472), (.466, .517),
    (.476, .529), (.489, .529), (.501, .515), (.504, .476), (.519, .453), (.542, .452),
    (.556, .430), (.560, .401), (.569, .384), (.581, .382), (.598, .363),
)
_SHELL = tuple((.700 * math.sin(angle), .700 * math.cos(angle))
               for angle in np.linspace(math.radians(59), math.pi, 56)[:-1]) + ((0, -.700),)
_PEDESTAL = (
    (0, .27), (.14, .27), (.19, .261), (.22, .234), (.235, .184), (.30, .137),
    (.44, .123), (.545, .118), (.568, .099), (.574, .073), (.587, .068),
    (.593, .050), (.585, .025), (.564, .009), (.536, 0), (0, 0),
)


def add_shader_ball(api, x, z, material, graphite, trim):
    """Create the sample face, shell and a neutral metal plinth."""
    axis = np.array((-.27, .23, 1.0))
    axis /= np.linalg.norm(axis)
    right = np.cross((0, 1, 0), axis)
    right /= np.linalg.norm(right)
    rotation = np.stack((right, np.cross(axis, right), axis))
    base_rotation = np.array(((1, 0, 0), (0, 0, -1), (0, 1, 0)))
    parts = (
        (_FACE, True, 0, rotation, (x, .91, z), (material,) * (len(_FACE) - 1)),
        (_BEZEL + _SHELL, False, 2, rotation, (x, .91, z),
         (material, graphite, graphite, trim, trim, trim, trim, trim, graphite, graphite,
          material, material, graphite, graphite, trim, trim, material) + (material,) * (len(_SHELL) - 1)),
        (_PEDESTAL, False, 0, base_rotation, (x, 0, z),
         (graphite,) * 8 + (trim,) * 3 + (graphite,) * 4),
    )
    for profile, planar, shell_tiles, basis, origin, materials in parts:
        positions, normals, uvs, indices, bands = _lathe(profile, planar, shell_tiles)
        mesh = api.create_mesh(
            positions @ basis + origin, indices, normals=normals @ basis, uvs=uvs,
            material_id=material, material_ids=np.asarray(materials, dtype=np.int32)[bands],
        )
        api.create_instance(mesh)
