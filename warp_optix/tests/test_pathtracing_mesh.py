# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp
from warp_optix.pathtracing import PathTracer, Scene
from warp_optix.pathtracing.scene import Mesh


def test_mesh_generates_normals_and_fallback_tangents_without_authored_attributes():
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    mesh = Mesh(vertices, np.array([[0, 1, 2]], dtype=np.uint32))

    np.testing.assert_allclose(mesh.normals, [[0.0, 0.0, 1.0]] * 3)
    np.testing.assert_array_equal(mesh.tangents, np.array([[1.0, 0.0, 0.0, 1.0]] * 3, dtype=np.float32))


@pytest.mark.parametrize("indices", [[[-1, 1, 2]], [[0, 1, 3]], [[0.0, 1.0, 2.0]]])
def test_mesh_rejects_invalid_indices_before_gpu_upload(indices):
    vertices = np.zeros((3, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="indices"):
        Mesh(vertices, np.asarray(indices))


def test_generated_sphere_winding_matches_outward_normals():
    scene = Scene()
    vertices, indices, normals, _ = scene._create_sphere_geometry((0.0, 0.0, 0.0), 1.0, 16)
    geometric = np.cross(
        vertices[indices[:, 1]] - vertices[indices[:, 0]],
        vertices[indices[:, 2]] - vertices[indices[:, 0]],
    )
    nondegenerate = np.linalg.norm(geometric, axis=1) > 1.0e-8
    alignment = np.einsum("ij,ij->i", geometric, normals[indices[:, 0]])

    assert np.all(alignment[nondegenerate] > 0.0)


@pytest.mark.gpu
@pytest.mark.skipif(not wp.is_cuda_available(), reason="device mesh access requires CUDA")
def test_dynamic_geometry_device_buffers_are_the_shader_buffers():
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    indices = np.array([[0, 1, 2]], dtype=np.uint32)
    scene = Scene()
    api = PathTracer.__new__(PathTracer)
    api._viewer = SimpleNamespace(_scene=scene)
    # Static geometry first, so the dynamic geometry sits at a nonzero offset in each packed buffer.
    api.create_mesh(vertices, indices)
    api.create_curve(vertices, 0.1)
    geometry_id = api.create_mesh(vertices, indices, dynamic=True)
    curve_id = api.create_curve(vertices, 0.2, dynamic=True)
    curve = scene._meshes[curve_id]
    scene._upload_geometry()

    device_vertices = api.get_mesh_vertices_device(geometry_id)
    device_normals = api.get_mesh_normals_device(geometry_id)
    assert device_vertices.dtype == device_normals.dtype == wp.vec3
    assert device_vertices.ptr == scene._meshes[geometry_id].d_vertices.ptr

    # Writes to a geometry's device arrays are what the shaders read; refits copy nothing.
    moved = vertices + 2.0
    device_vertices.assign(moved)
    device_normals.assign(-moved)
    curve.d_vertices.assign(moved.reshape(-1))
    curve.d_widths.fill_(0.5)
    np.testing.assert_array_equal(scene._meshes[geometry_id].d_vertices.numpy().reshape(-1, 3), moved)
    np.testing.assert_array_equal(scene._packed_normals.numpy()[9:], -moved.reshape(-1))
    np.testing.assert_array_equal(scene._packed_positions.numpy()[9:], moved.reshape(-1))
    np.testing.assert_array_equal(scene._packed_radii.numpy()[3:], 0.5)
