# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
Scene management for path tracing viewer.
Handles mesh loading, BLAS/TLAS construction, and instance management.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import numpy as np
import warp as wp

import warp_optix as wo
from warp_optix import native

from .color_utils import srgb_to_linear_u8
from .materials import ALPHA_OPAQUE, MaterialManager

logger = logging.getLogger(__name__)
_MAX_TEXTURE_BUILD_WORKERS = 8
NO_MATERIAL_OVERRIDE = np.uint32(np.iinfo(np.uint32).max)


def _normalize_primitive_material_ids(
    material_ids: np.ndarray | None, primitive_count: int, material_id: int
) -> np.ndarray:
    """Return one contiguous material-table index per geometry primitive."""
    if material_ids is None:
        return np.full(primitive_count, int(material_id), dtype=np.uint32)
    values = np.asarray(material_ids)
    if values.ndim != 1 or values.shape[0] != primitive_count:
        raise ValueError(f"material_ids must have shape ({primitive_count},)")
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError("material_ids must contain integers")
    if np.any(values < 0):
        raise ValueError("material_ids must be non-negative")
    return np.ascontiguousarray(values, dtype=np.uint32)


def _texture_worker_count(item_count: int) -> int:
    """Return bounded concurrency for CPU texture preparation and mip generation."""
    return max(
        1,
        min(
            int(item_count),
            _MAX_TEXTURE_BUILD_WORKERS,
            max(1, int(os.cpu_count() or 1)),
        ),
    )


@wp.kernel
def _update_device_instance_transforms(
    instance_ids: wp.array(dtype=wp.int32),
    xforms: wp.array(dtype=wp.transform),
    scales: wp.array(dtype=wp.vec3),
    transforms: wp.array2d(dtype=wp.float32),
    global_rotation: wp.mat33,
    global_translation: wp.vec3,
):
    batch_index = wp.tid()
    instance_index = instance_ids[batch_index]
    xform = xforms[batch_index]
    position = global_rotation * wp.transform_get_translation(xform) + global_translation
    rotation = global_rotation * wp.quat_to_matrix(wp.transform_get_rotation(xform))
    scale = scales[batch_index]
    for row in range(3):
        for column in range(3):
            transforms[instance_index, row * 4 + column] = rotation[row, column] * scale[column]
        transforms[instance_index, row * 4 + 3] = position[row]


@wp.kernel
def _scatter_usd_local_transforms(
    transform_count: wp.array(dtype=wp.int32),
    node_ids: wp.array(dtype=wp.int32),
    transforms: wp.array(dtype=wp.mat44),
    local_transforms: wp.array(dtype=wp.mat44),
):
    update_index = wp.tid()
    if update_index >= wp.clamp(transform_count[0], 0, node_ids.shape[0]):
        return
    local_transforms[node_ids[update_index]] = transforms[update_index]


@wp.kernel
def _scatter_usd_local_transform_trs(
    transform_count: wp.array(dtype=wp.int32),
    node_ids: wp.array(dtype=wp.int32),
    transforms: wp.array(dtype=wp.transform),
    scales: wp.array(dtype=wp.vec3),
    local_transforms: wp.array(dtype=wp.mat44),
):
    update_index = wp.tid()
    if update_index >= wp.clamp(transform_count[0], 0, node_ids.shape[0]):
        return
    transform = transforms[update_index]
    position = wp.transform_get_translation(transform)
    rotation = wp.quat_to_matrix(wp.transform_get_rotation(transform))
    scale = scales[update_index]
    local_transforms[node_ids[update_index]] = wp.mat44(
        rotation[0, 0] * scale[0],
        rotation[0, 1] * scale[1],
        rotation[0, 2] * scale[2],
        position[0],
        rotation[1, 0] * scale[0],
        rotation[1, 1] * scale[1],
        rotation[1, 2] * scale[2],
        position[1],
        rotation[2, 0] * scale[0],
        rotation[2, 1] * scale[1],
        rotation[2, 2] * scale[2],
        position[2],
        0.0,
        0.0,
        0.0,
        1.0,
    )


@wp.kernel
def _compose_usd_transform_level(
    level_nodes: wp.array(dtype=wp.int32),
    parents: wp.array(dtype=wp.int32),
    local_transforms: wp.array(dtype=wp.mat44),
    world_transforms: wp.array(dtype=wp.mat44),
):
    node_index = level_nodes[wp.tid()]
    parent_index = parents[node_index]
    if parent_index < 0:
        world_transforms[node_index] = local_transforms[node_index]
    else:
        world_transforms[node_index] = world_transforms[parent_index] * local_transforms[node_index]


@wp.kernel
def _write_usd_instance_transforms(
    instance_ids: wp.array(dtype=wp.int32),
    instance_node_ids: wp.array(dtype=wp.int32),
    world_transforms: wp.array(dtype=wp.mat44),
    transforms: wp.array2d(dtype=wp.float32),
):
    mapping_index = wp.tid()
    instance_index = instance_ids[mapping_index]
    matrix = world_transforms[instance_node_ids[mapping_index]]
    for row in range(3):
        for column in range(4):
            transforms[instance_index, row * 4 + column] = matrix[row, column]


@wp.kernel
def _update_device_instance_visibility(
    instance_ids: wp.array(dtype=wp.int32),
    visibility_mask: wp.uint32,
    masks: wp.array(dtype=wp.uint32),
):
    masks[instance_ids[wp.tid()]] = visibility_mask


@wp.kernel
def _write_instance_records(
    transforms: wp.array2d(dtype=wp.float32),
    traversables: wp.array(dtype=wp.uint64),
    sbt_offsets: wp.array(dtype=wp.uint32),
    masks: wp.array(dtype=wp.uint32),
    flags: wp.array(dtype=wp.uint32),
    records: wp.array(dtype=wo.Instance),
):
    i = wp.tid()
    m = transforms
    rotation = wp.mat33(m[i, 0], m[i, 1], m[i, 2], m[i, 4], m[i, 5], m[i, 6], m[i, 8], m[i, 9], m[i, 10])
    translation = wp.vec3(m[i, 3], m[i, 7], m[i, 11])
    records[i] = wo.make_instance(
        rotation, translation, traversables[i], wp.uint32(i), sbt_offsets[i], masks[i], flags[i]
    )


@wp.func
def _srgb_channel_to_linear(value: float):
    if value <= 0.04045:
        return value / 12.92
    return wp.pow((value + 0.055) / 1.055, 2.4)


@wp.kernel
def _update_device_instance_materials(
    material_ids: wp.array(dtype=wp.int32),
    colors: wp.array(dtype=wp.vec3),
    properties: wp.array(dtype=wp.vec4),
    color_count: int,
    property_count: int,
    compact_materials: wp.array2d(dtype=wp.float32),
):
    thread_index = wp.tid()
    color_index = 0 if color_count == 1 else thread_index
    property_index = 0 if property_count == 1 else thread_index
    color = colors[color_index]
    material = properties[property_index]
    material_index = material_ids[thread_index]
    compact_materials[material_index, 0] = _srgb_channel_to_linear(color[0])
    compact_materials[material_index, 1] = _srgb_channel_to_linear(color[1])
    compact_materials[material_index, 2] = _srgb_channel_to_linear(color[2])
    compact_materials[material_index, 6] = wp.clamp(material[0], 0.0, 1.0)
    compact_materials[material_index, 7] = wp.clamp(material[1], 0.0, 1.0)
    compact_materials[material_index, 8] = wp.max(material[2], 0.0)
    compact_materials[material_index, 9] = wp.max(material[3], 0.0)


def _create_vertex_buffers_dtype():
    """Create numpy dtype for VertexBuffers structure (offset-based)."""
    return np.dtype(
        [
            ("positionOffset", np.uint32),
            ("normalOffset", np.uint32),
            ("radiusOffset", np.uint32),
            ("tangentOffset", np.uint32),
            ("texCoord0Offset", np.uint32),
            ("texCoord1Offset", np.uint32),
            ("prevPositionOffset", np.uint32),
            ("hasTexCoord1", np.uint32),
            ("hasPrevPosition", np.uint32),
        ]
    )


def _create_render_primitive_dtype():
    """Create numpy dtype for RenderPrimitive structure."""
    vb_dt = _create_vertex_buffers_dtype()
    return np.dtype(
        [
            ("indexOffset", np.uint32),
            ("materialIdOffset", np.uint32),
            ("vertexBuffer", vb_dt),
            ("numIndices", np.uint32),
            ("numVertices", np.uint32),
        ]
    )


class Mesh:
    """Represents a mesh with vertices, indices, and GPU buffers."""

    primitive_type = "triangles"
    sbt_offset = 0

    def __init__(
        self,
        vertices: np.ndarray,
        indices: np.ndarray,
        normals: np.ndarray = None,
        texcoords: np.ndarray = None,
        texcoords1: np.ndarray = None,
        material_id: int = 0,
        material_ids: np.ndarray | None = None,
        dynamic: bool = False,
    ):
        """
        Create a mesh.

        Args:
            vertices: Nx3 array of vertex positions
            indices: Mx3 array of triangle indices
            normals: Nx3 array of vertex normals (optional, will compute if None)
            texcoords: Nx2 array of texture coordinates (optional)
            material_id: Material index for this mesh
        """
        self.vertices = np.ascontiguousarray(vertices, dtype=np.float32)
        raw_indices = np.asarray(indices)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3 or not np.all(np.isfinite(self.vertices)):
            raise ValueError("vertices must be a finite (N, 3) array")
        if (
            raw_indices.ndim != 2
            or raw_indices.shape[1] != 3
            or not np.issubdtype(raw_indices.dtype, np.integer)
            or np.any(raw_indices < 0)
            or np.any(raw_indices >= len(self.vertices))
        ):
            raise ValueError("indices must be a (M, 3) integer array indexing vertices")
        self.indices = np.ascontiguousarray(raw_indices, dtype=np.uint32)

        if normals is None:
            normals = self._compute_normals(self.vertices, self.indices)
        self.normals = np.ascontiguousarray(normals, dtype=np.float32)
        if self.normals.shape != self.vertices.shape or not np.all(np.isfinite(self.normals)):
            raise ValueError("normals must be a finite (N, 3) array matching vertices")

        has_texcoords = texcoords is not None
        if texcoords is None:
            texcoords = np.zeros((len(vertices), 2), dtype=np.float32)
        self.texcoords = np.ascontiguousarray(texcoords, dtype=np.float32)
        if self.texcoords.shape != (len(self.vertices), 2) or not np.all(np.isfinite(self.texcoords)):
            raise ValueError("texcoords must be a finite (N, 2) array matching vertices")
        if texcoords1 is None:
            texcoords1 = np.zeros((len(vertices), 2), dtype=np.float32)
        self.texcoords1 = np.ascontiguousarray(texcoords1, dtype=np.float32)
        if self.texcoords1.shape != (len(self.vertices), 2) or not np.all(np.isfinite(self.texcoords1)):
            raise ValueError("texcoords1 must be a finite (N, 2) array matching vertices")

        self.material_id = material_id
        self.material_ids = _normalize_primitive_material_ids(material_ids, len(self.indices), self.material_id)
        self.dynamic = bool(dynamic)
        self._packed_normal_offset = 0

        # Tangents only need triangle accumulation when authored UVs exist.
        self.tangents = (
            self._compute_tangents(self.vertices, self.indices, self.normals, self.texcoords)
            if has_texcoords
            else self._fallback_tangents(self.normals)
        )

        # GPU buffers (created on build)
        self.d_vertices = None
        self.d_indices = None
        self.d_normals = None

    @staticmethod
    def _fallback_tangents(normals: np.ndarray) -> np.ndarray:
        """Return deterministic tangents without traversing triangles."""
        tangents = np.zeros((len(normals), 4), dtype=np.float32)
        tangents[:, 0] = 1.0
        use_y = np.abs(normals[:, 0]) > 0.9
        tangents[use_y, 0] = 0.0
        tangents[use_y, 1] = 1.0
        tangents[:, 3] = 1.0
        return tangents

    def _compute_tangents(
        self,
        vertices: np.ndarray,
        indices: np.ndarray,
        normals: np.ndarray,
        texcoords: np.ndarray,
    ) -> np.ndarray:
        """Compute vertex tangents from UVs using vectorized operations."""
        n_verts = len(vertices)
        tangents = np.zeros((n_verts, 4), dtype=np.float32)
        tan1 = np.zeros((n_verts, 3), dtype=np.float32)
        tan2 = np.zeros((n_verts, 3), dtype=np.float32)

        tri = indices.astype(np.int64, copy=False)
        i0 = tri[:, 0]
        i1 = tri[:, 1]
        i2 = tri[:, 2]

        v0 = vertices[i0]
        v1 = vertices[i1]
        v2 = vertices[i2]
        uv0 = texcoords[i0]
        uv1 = texcoords[i1]
        uv2 = texcoords[i2]

        e1 = v1 - v0
        e2 = v2 - v0
        duv1 = uv1 - uv0
        duv2 = uv2 - uv0

        denom = duv1[:, 0] * duv2[:, 1] - duv2[:, 0] * duv1[:, 1]
        valid = np.abs(denom) >= 1.0e-8
        if np.any(valid):
            inv = np.zeros_like(denom, dtype=np.float32)
            inv[valid] = 1.0 / denom[valid]
            inv3 = inv[:, None]
            tri_tangent = (e1 * duv2[:, 1:2] - e2 * duv1[:, 1:2]) * inv3
            tri_bitangent = (e2 * duv1[:, 0:1] - e1 * duv2[:, 0:1]) * inv3
            tri_tangent[~valid] = 0.0
            tri_bitangent[~valid] = 0.0

            np.add.at(tan1, i0, tri_tangent)
            np.add.at(tan1, i1, tri_tangent)
            np.add.at(tan1, i2, tri_tangent)
            np.add.at(tan2, i0, tri_bitangent)
            np.add.at(tan2, i1, tri_bitangent)
            np.add.at(tan2, i2, tri_bitangent)

        n = normals
        tangent = tan1 - n * np.sum(n * tan1, axis=1, keepdims=True)
        t_len = np.linalg.norm(tangent, axis=1, keepdims=True)
        good = t_len[:, 0] > 1.0e-8
        tangent[good] /= t_len[good]

        fallback_x = np.tile(np.array([1.0, 0.0, 0.0], dtype=np.float32), (n_verts, 1))
        fallback_y = np.tile(np.array([0.0, 1.0, 0.0], dtype=np.float32), (n_verts, 1))
        use_y = np.abs(n[:, 0]) > 0.9
        tangent[~good] = np.where(use_y[~good, None], fallback_y[~good], fallback_x[~good])

        cross_nt = np.cross(n, tangent)
        handedness = np.where(np.sum(cross_nt * tan2, axis=1) < 0.0, -1.0, 1.0).astype(np.float32)

        tangents[:, :3] = tangent
        tangents[:, 3] = handedness
        return np.ascontiguousarray(tangents, dtype=np.float32)

    def _compute_normals(self, vertices: np.ndarray, indices: np.ndarray) -> np.ndarray:
        """Compute area-weighted vertex normals in bounded vectorized chunks."""
        normals = np.zeros_like(vertices)
        chunk_size = 1_000_000
        for begin in range(0, len(indices), chunk_size):
            triangles = indices[begin : begin + chunk_size]
            i0 = triangles[:, 0]
            i1 = triangles[:, 1]
            i2 = triangles[:, 2]
            face_normals = np.cross(
                vertices[i1] - vertices[i0],
                vertices[i2] - vertices[i0],
            )
            np.add.at(normals, i0, face_normals)
            np.add.at(normals, i1, face_normals)
            np.add.at(normals, i2, face_normals)

        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        normals /= np.where(lengths > 0.0, lengths, 1.0)
        return normals

    def upload_to_gpu(self):
        """Upload the acceleration-structure inputs and, for dynamic meshes, writable normals."""
        self.d_vertices = wp.array(self.vertices.flatten(), dtype=wp.float32, device="cuda")
        self.d_indices = wp.array(self.indices.flatten(), dtype=wp.uint32, device="cuda")
        self.d_normals = wp.array(self.normals.flatten(), dtype=wp.float32, device="cuda") if self.dynamic else None


class Curve:
    """Native OptiX round linear or cubic Bézier curve geometry.

    Each primitive starts at ``segment_indices[i]`` and consumes the number of
    consecutive control points required by ``basis``. ``radii`` contains one
    swept-surface radius per control point and uses the same interpolation basis.
    """

    _BASIS_SPECS = {
        "linear": ("round_linear", 2, 2, 1),
        "cubic_bezier": ("round_cubic_bezier", 4, 4, 2),
    }

    def __init__(
        self,
        vertices: np.ndarray,
        radii: np.ndarray | float,
        segment_indices: np.ndarray | None = None,
        material_id: int = 0,
        material_ids: np.ndarray | None = None,
        dynamic: bool = False,
        basis: str = "linear",
    ):
        basis = str(basis).lower()
        if basis not in self._BASIS_SPECS:
            raise ValueError("curve basis must be 'linear' or 'cubic_bezier'")
        primitive_type, control_point_count, sbt_offset, geometry_type = self._BASIS_SPECS[basis]
        vertices = np.asarray(vertices, dtype=np.float32)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("curve vertices must have shape (N, 3)")
        if len(vertices) < control_point_count:
            raise ValueError(f"a {basis} curve requires at least {control_point_count} control points")
        if not np.all(np.isfinite(vertices)):
            raise ValueError("curve vertices must be finite")

        radii = np.asarray(radii, dtype=np.float32)
        if radii.ndim == 0:
            radii = np.full(len(vertices), float(radii), dtype=np.float32)
        if radii.shape != (len(vertices),):
            raise ValueError("curve radii must be scalar or have shape (N,)")
        if not np.all(np.isfinite(radii)):
            raise ValueError("curve radii must be finite")
        if np.any(radii < 0.0) or (not dynamic and np.any(radii <= 0.0)):
            qualifier = "non-negative" if dynamic else "positive"
            raise ValueError(f"curve radii must be finite and {qualifier}")

        if segment_indices is None:
            if basis == "linear":
                segment_indices = np.arange(len(vertices) - 1, dtype=np.uint32)
            else:
                if (len(vertices) - 1) % 3:
                    raise ValueError("default cubic_bezier topology requires 3*N + 1 control points")
                segment_indices = np.arange(0, len(vertices) - 3, 3, dtype=np.uint32)
        segment_indices = np.asarray(segment_indices)
        if segment_indices.ndim != 1 or len(segment_indices) == 0:
            raise ValueError("segment_indices must be a non-empty one-dimensional array")
        if not np.issubdtype(segment_indices.dtype, np.integer):
            raise ValueError("segment_indices must contain integers")
        if np.any(segment_indices < 0) or np.any(segment_indices > len(vertices) - control_point_count):
            control_point_label = "two" if control_point_count == 2 else "four"
            raise ValueError(f"each {basis} segment must reference {control_point_label} consecutive control points")

        self.basis = basis
        self.primitive_type = primitive_type
        self.control_point_count = control_point_count
        self.sbt_offset = sbt_offset
        self.geometry_type = geometry_type
        self.vertices = np.ascontiguousarray(vertices, dtype=np.float32)
        self.radii = np.ascontiguousarray(radii, dtype=np.float32)
        self.indices = np.ascontiguousarray(segment_indices, dtype=np.uint32)
        self.material_id = int(material_id)
        self.material_ids = _normalize_primitive_material_ids(material_ids, len(self.indices), self.material_id)
        self.dynamic = bool(dynamic)
        self.d_vertices = None
        self.d_widths = None
        self.d_indices = None
        self._packed_position_offset = 0
        self._packed_radius_offset = 0
        self._packed_material_offset = 0

    def upload_to_gpu(self):
        """Upload control points, radii, and segment starts to CUDA."""
        self.d_vertices = wp.array(self.vertices.reshape(-1), dtype=wp.float32, device="cuda")
        self.d_widths = wp.array(self.radii, dtype=wp.float32, device="cuda")
        self.d_indices = wp.array(self.indices, dtype=wp.uint32, device="cuda")


class Instance:
    """Represents an instance of a mesh with a transform."""

    def __init__(
        self,
        mesh_index: int,
        transform: np.ndarray = None,
        material_id: int | None = None,
        visible: bool = True,
        double_sided: bool = False,
    ):
        """
        Create an instance.

        Args:
            mesh_index: Index of the mesh in the scene
            transform: 4x4 transformation matrix (identity if None)
        """
        self.mesh_index = mesh_index
        self.material_id = material_id
        self.visible = bool(visible)
        self.double_sided = bool(double_sided)
        if transform is None:
            transform = np.eye(4, dtype=np.float32)
        self.transform = np.ascontiguousarray(transform, dtype=np.float32)


class Scene:
    """
    Scene management for path tracing.
    Handles meshes, instances, materials, and acceleration structures.
    """

    def __init__(self, optix_ctx, enable_texture_mipmaps: bool = False):
        """
        Create a scene.

        Args:
            optix_ctx: OptiX device context
        """
        self._optix = None  # Will be set when building
        self._ctx = optix_ctx
        self.enable_texture_mipmaps = bool(enable_texture_mipmaps)
        self.materials = MaterialManager()
        self._meshes = []
        self._instances = []
        self._sphere_lights = []
        self._sphere_light_data = None

        # GPU buffers
        self._render_primitives = None
        self._instance_material_ids = None
        self._compact_materials = None
        self._compact_material_floats = None
        self._instance_render_prim_ids = None
        self._instance_geometry_types = None
        self._texture_data = None
        self._texture_objects = []
        self._packed_indices = None
        self._packed_positions = None
        self._packed_radii = None
        self._packed_normals = None
        self._packed_tangents = None
        self._packed_texcoords0 = None
        self._packed_texcoords1 = None
        self._packed_prev_positions = None
        self._packed_material_ids = None
        self._gltf_textures = []
        # Resolved DomeLight texture discovered while composing the last USD
        # stage. Keeping it here lets PathTracerAPI opt into the environment
        # without reopening (and recomposing) a large stage.
        self.usd_environment_path = None
        self.usd_ambient_light = (0.0, 0.0, 0.0)
        self.usd_scene = None
        self._usd_node_parents = np.empty(0, dtype=np.int32)
        self._usd_local_transforms = np.empty((0, 4, 4), dtype=np.float32)
        self._usd_world_transforms = np.empty((0, 4, 4), dtype=np.float32)
        self._usd_instance_ids = np.empty(0, dtype=np.int32)
        self._usd_instance_node_ids = np.empty(0, dtype=np.int32)
        self._usd_level_nodes = []
        self._d_usd_node_parents = None
        self._d_usd_local_transforms = None
        self._d_usd_world_transforms = None
        self._d_usd_instance_ids = None
        self._d_usd_instance_node_ids = None
        self._d_usd_level_nodes = []

        # Acceleration structures. Instance records are written on the GPU from
        # the per-instance device arrays below before every TLAS build or refit.
        self._gas = []
        self._ias = None
        self._opaque_materials = np.empty(0, dtype=np.bool_)
        self._instance_records = None
        self._instance_traversables = None
        self._instance_sbt_offsets = None
        self._instance_flags = None
        self._instance_masks = None
        self._device_instance_transforms = None
        self._instance_records_dirty = True
        self._instance_transform_cache = np.empty((0, 4, 4), dtype=np.float32)
        self._instance_visibility_cache = np.empty(0, dtype=np.bool_)
        self._instance_cache_capacity = 0
        # Repeated UPDATE operations can make a moving scene's TLAS expensive
        # to traverse. Rebuild occasionally to restore its hierarchy quality.
        self._max_tlas_updates = 16
        self._tlas_updates_since_build = 0

    @property
    def mesh_count(self) -> int:
        """Return the number of triangle-mesh geometry objects."""
        return sum(geometry.primitive_type == "triangles" for geometry in self._meshes)

    @property
    def geometry_count(self) -> int:
        """Return the total number of reusable mesh and curve geometries."""
        return len(self._meshes)

    @property
    def curve_count(self) -> int:
        return sum(geometry.primitive_type != "triangles" for geometry in self._meshes)

    @property
    def instance_count(self) -> int:
        return len(self._instances)

    @property
    def light_count(self) -> int:
        """Return the number of analytic lights."""
        return len(self._sphere_lights)

    def add_light_sphere(
        self,
        position: tuple[float, float, float],
        radius: float,
        color: tuple[float, float, float],
        intensity: float,
    ) -> int:
        """Add a constant-radiance analytic sphere light."""
        radius = float(radius)
        intensity = float(intensity)
        if radius <= 0.0:
            raise ValueError("radius must be positive")
        if intensity < 0.0:
            raise ValueError("intensity must be nonnegative")
        self._sphere_lights.append(
            (
                tuple(float(value) for value in position),
                radius,
                tuple(float(value) for value in color),
                intensity,
            )
        )
        return len(self._sphere_lights) - 1

    @property
    def tlas_handle(self) -> int:
        """Get the TLAS traversable handle."""
        return self._ias.handle if self._ias else 0

    @property
    def instance_material_ids_address(self) -> int:
        """Get device address of per-instance material id buffer."""
        return self._instance_material_ids.ptr if self._instance_material_ids is not None else 0

    @property
    def compact_materials_address(self) -> int:
        """Get device address of compact material table."""
        return self._compact_materials.ptr if self._compact_materials is not None else 0

    @property
    def render_primitives_address(self) -> int:
        """Get device address of render primitive array."""
        return self._render_primitives.ptr if self._render_primitives is not None else 0

    @property
    def instance_render_prim_ids_address(self) -> int:
        """Get device address of per-instance render primitive id buffer."""
        return self._instance_render_prim_ids.ptr if self._instance_render_prim_ids is not None else 0

    @property
    def texture_data_address(self) -> int:
        """Get the device address of the native texture-handle array."""
        return self._texture_data.ptr if self._texture_data is not None else 0

    @property
    def texture_count(self) -> int:
        """Get number of loaded glTF textures."""
        return len(self._gltf_textures)

    @property
    def has_meshes(self) -> bool:
        """Return True when at least one mesh is present."""
        return self.mesh_count > 0

    def get_instance_material_ids_host(self) -> np.ndarray | None:
        """Return per-instance material IDs as a host NumPy array copy."""
        if self._instance_material_ids is None:
            return None
        return self._instance_material_ids.numpy()

    def set_instance_material_ids_host(self, material_ids: np.ndarray):
        """Upload per-instance material IDs from host memory."""
        self._instance_material_ids = wp.array(
            np.asarray(material_ids, dtype=np.uint32), dtype=wp.uint32, device="cuda"
        )

    def set_compact_material_bytes(self, compact_bytes: np.ndarray):
        """Upload compact material table bytes to GPU."""
        self._compact_materials = wp.array(np.asarray(compact_bytes, dtype=np.uint8), dtype=wp.uint8, device="cuda")
        self._compact_material_floats = None

    def set_instance_materials_device(
        self,
        material_ids: wp.array,
        colors: wp.array,
        properties: wp.array,
    ) -> bool:
        """Update instance-owned compact material fields directly on CUDA."""
        if len(material_ids) == 0:
            return True
        if self._compact_materials is None:
            return False
        material_count = self.materials.count
        stride = int(self._compact_materials.capacity) // material_count // 4
        self._compact_material_floats = wp.array(
            ptr=self._compact_materials.ptr,
            shape=(material_count, stride),
            dtype=wp.float32,
            capacity=int(self._compact_materials.capacity),
            device="cuda",
        )
        if len(colors) not in (1, len(material_ids)):
            raise ValueError("Device colors must contain one or one value per instance")
        if len(properties) not in (1, len(material_ids)):
            raise ValueError("Device material properties must contain one or one value per instance")
        wp.launch(
            _update_device_instance_materials,
            dim=len(material_ids),
            inputs=[
                material_ids,
                colors,
                properties,
                len(colors),
                len(properties),
                self._compact_material_floats,
            ],
            device="cuda",
        )
        return True

    def set_gltf_textures(
        self,
        textures: list[np.ndarray],
        srgb_texture_indices: set[int] | None = None,
        append: bool = False,
    ):
        """
        Set glTF texture list as normalized RGBA8 images.

        Args:
            textures: RGBA uint8 textures, or float textures in normalized [0,1] space.
            srgb_texture_indices: Texture indices that represent color data and
                therefore require sRGB->linear decode to match Vulkan hardware
                sampling with ``R8G8B8A8_SRGB``.
        """
        srgb_indices = srgb_texture_indices or set()

        def convert_texture(indexed_texture):
            tex_idx, tex = indexed_texture
            if np.issubdtype(tex.dtype, np.integer):
                tex_u8 = np.ascontiguousarray(tex, dtype=np.uint8)
            else:
                tex_u8 = np.clip(np.asarray(tex) * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
            if tex_idx in srgb_indices:
                tex_u8 = tex_u8.copy()
                tex_u8[..., :3] = srgb_to_linear_u8(tex_u8[..., :3])
            return tex_u8

        workers = _texture_worker_count(len(textures))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            converted = list(executor.map(convert_texture, enumerate(textures)))
        if append:
            self._gltf_textures.extend(converted)
        else:
            self._gltf_textures = converted

    def add_mesh(self, mesh: Mesh) -> int:
        """Add a mesh to the scene and return its index."""
        self._meshes.append(mesh)
        return len(self._meshes) - 1

    def add_curve(self, curve: Curve) -> int:
        """Add native curve geometry and return its reusable geometry index."""
        self._meshes.append(curve)
        return len(self._meshes) - 1

    def _ensure_instance_cache_capacity(self, count: int):
        """Grow contiguous host-side instance state without per-frame allocation."""
        if self._instance_cache_capacity >= count:
            return
        capacity = max(count, 1, self._instance_cache_capacity * 2)
        transforms = np.empty((capacity, 4, 4), dtype=np.float32)
        visibility = np.empty(capacity, dtype=np.bool_)
        if self._instance_cache_capacity:
            transforms[: self._instance_cache_capacity] = self._instance_transform_cache
            visibility[: self._instance_cache_capacity] = self._instance_visibility_cache
        self._instance_transform_cache = transforms
        self._instance_visibility_cache = visibility
        self._instance_cache_capacity = capacity

    def add_instance(
        self,
        mesh_index: int,
        transform: np.ndarray = None,
        material_id: int | None = None,
        visible: bool = True,
        double_sided: bool = False,
    ) -> int:
        """Add an instance of a mesh and return its index."""
        instance = Instance(
            mesh_index,
            transform,
            material_id=material_id,
            visible=visible,
            double_sided=double_sided,
        )
        self._instances.append(instance)
        instance_index = len(self._instances) - 1
        self._ensure_instance_cache_capacity(instance_index + 1)
        self._instance_transform_cache[instance_index] = instance.transform
        self._instance_visibility_cache[instance_index] = instance.visible
        self._instance_records_dirty = True
        return instance_index

    def set_instance_transform(self, instance_index: int, transform: np.ndarray):
        """Update an instance's transform."""
        if 0 <= instance_index < len(self._instances):
            inst = self._instances[instance_index]
            inst.transform = np.ascontiguousarray(transform, dtype=np.float32)
            self._instance_transform_cache[instance_index] = inst.transform
            self._instance_records_dirty = True

    def set_instance_transforms_batch(self, instance_indices, transforms: np.ndarray):
        """Update a batch of instance transforms with one NumPy assignment."""
        indices = np.asarray(instance_indices, dtype=np.intp)
        matrices = np.asarray(transforms, dtype=np.float32).reshape(-1, 4, 4)
        if len(indices) != len(matrices):
            raise ValueError("instance_indices and transforms must have equal length")
        if len(indices) == 0:
            return
        if int(indices.min()) < 0 or int(indices.max()) >= len(self._instances):
            raise IndexError("instance index is out of range")
        self._instance_transform_cache[indices] = matrices
        self._instance_records_dirty = True

    def set_instance_transforms_device(
        self,
        instance_indices: wp.array,
        xforms: wp.array,
        scales: wp.array,
        global_transform: np.ndarray,
    ) -> bool:
        """Update traversal, shading, and motion transforms directly on CUDA."""
        if len(instance_indices) == 0:
            return True
        if self._device_instance_transforms is None:
            return False
        if not instance_indices.device.is_cuda or not xforms.device.is_cuda or not scales.device.is_cuda:
            raise ValueError("Device instance updates require CUDA arrays")
        matrix = np.asarray(global_transform, dtype=np.float32).reshape(4, 4)
        rotation = matrix[:3, :3]
        translation = matrix[:3, 3]
        wp.launch(
            _update_device_instance_transforms,
            dim=len(instance_indices),
            inputs=[
                instance_indices,
                xforms,
                scales,
                self._device_instance_transforms,
                wp.mat33(*rotation.reshape(-1)),
                wp.vec3(*translation),
            ],
            device="cuda",
        )
        return True

    def configure_usd_transform_hierarchy(
        self,
        usd_scene,
        parents: np.ndarray,
        local_transforms: np.ndarray,
        instance_ids: np.ndarray,
        instance_node_ids: np.ndarray,
    ):
        """Retain a composed USD transform hierarchy for dynamic updates."""
        parents = np.asarray(parents, dtype=np.int32).reshape(-1)
        local = np.ascontiguousarray(local_transforms, dtype=np.float32).reshape(-1, 4, 4)
        if len(parents) != len(local):
            raise ValueError("USD hierarchy parents and transforms must have equal length")
        if np.any(parents >= np.arange(len(parents), dtype=np.int32)):
            raise ValueError("USD hierarchy parents must precede their children")

        world = np.empty_like(local)
        depths = np.zeros(len(parents), dtype=np.int32)
        for node_index, parent_index in enumerate(parents):
            if parent_index < 0:
                world[node_index] = local[node_index]
            else:
                world[node_index] = world[parent_index] @ local[node_index]
                depths[node_index] = depths[parent_index] + 1

        self.usd_scene = usd_scene
        self._usd_node_parents = parents
        self._usd_local_transforms = local
        self._usd_world_transforms = world
        self._usd_instance_ids = np.asarray(instance_ids, dtype=np.int32).reshape(-1)
        self._usd_instance_node_ids = np.asarray(instance_node_ids, dtype=np.int32).reshape(-1)
        self._usd_level_nodes = [
            np.flatnonzero(depths == depth).astype(np.int32) for depth in range(int(depths.max(initial=0)) + 1)
        ]
        if len(self._usd_instance_ids) != len(self._usd_instance_node_ids):
            raise ValueError("USD instance and node mappings must have equal length")
        if len(self._usd_instance_ids):
            self.set_instance_transforms_batch(
                self._usd_instance_ids,
                world[self._usd_instance_node_ids],
            )

    def _build_usd_transform_buffers(self):
        if self.usd_scene is None or len(self._usd_node_parents) == 0:
            return
        self._d_usd_node_parents = wp.array(self._usd_node_parents, dtype=wp.int32, device="cuda")
        self._d_usd_local_transforms = wp.array(self._usd_local_transforms, dtype=wp.mat44, device="cuda")
        self._d_usd_world_transforms = wp.array(self._usd_world_transforms, dtype=wp.mat44, device="cuda")
        self._d_usd_instance_ids = wp.array(self._usd_instance_ids, dtype=wp.int32, device="cuda")
        self._d_usd_instance_node_ids = wp.array(self._usd_instance_node_ids, dtype=wp.int32, device="cuda")
        self._d_usd_level_nodes = [wp.array(level, dtype=wp.int32, device="cuda") for level in self._usd_level_nodes]
        self.usd_scene._attach_device_arrays(self._d_usd_local_transforms, self._d_usd_world_transforms)

    def _evaluate_usd_transform_hierarchy_host(self, update_instances: bool = True):
        for node_index, parent_index in enumerate(self._usd_node_parents):
            local = self._usd_local_transforms[node_index]
            if parent_index < 0:
                self._usd_world_transforms[node_index] = local
            else:
                self._usd_world_transforms[node_index] = self._usd_world_transforms[parent_index] @ local
        if update_instances and len(self._usd_instance_ids):
            self.set_instance_transforms_batch(
                self._usd_instance_ids,
                self._usd_world_transforms[self._usd_instance_node_ids],
            )

    def _evaluate_usd_transform_hierarchy(self, stream=None):
        if self._d_usd_local_transforms is None:
            raise RuntimeError("Build the scene before updating USD transforms")
        scope = wp.ScopedStream(stream, sync_enter=False, sync_exit=False) if stream is not None else nullcontext()
        with scope:
            for level_nodes in self._d_usd_level_nodes:
                wp.launch(
                    _compose_usd_transform_level,
                    dim=len(level_nodes),
                    inputs=[
                        level_nodes,
                        self._d_usd_node_parents,
                        self._d_usd_local_transforms,
                        self._d_usd_world_transforms,
                    ],
                    device="cuda",
                )
            if len(self._usd_instance_ids):
                wp.launch(
                    _write_usd_instance_transforms,
                    dim=len(self._usd_instance_ids),
                    inputs=[
                        self._d_usd_instance_ids,
                        self._d_usd_instance_node_ids,
                        self._d_usd_world_transforms,
                        self._device_instance_transforms,
                    ],
                    device="cuda",
                )

    def set_usd_local_transforms_device(
        self,
        transform_count: wp.array,
        node_ids: wp.array,
        transforms: wp.array,
        stream=None,
        rebuild_tlas: bool = True,
    ):
        """Apply CUDA-resident USD local matrices and compose the hierarchy."""
        if self._d_usd_local_transforms is None:
            raise RuntimeError("Build the scene before updating USD transforms")
        if not transform_count.device.is_cuda or not node_ids.device.is_cuda or not transforms.device.is_cuda:
            raise ValueError("USD device transform updates require CUDA arrays")
        if len(transform_count) != 1:
            raise ValueError("transform_count must contain exactly one value")
        if len(node_ids) != len(transforms):
            raise ValueError("node_ids and transforms must have equal length")
        if len(node_ids) == 0:
            return
        if transform_count.dtype != wp.int32 or node_ids.dtype != wp.int32 or transforms.dtype != wp.mat44:
            raise TypeError("USD device batches require an int32 count, int32 IDs, and mat44 matrices")
        scope = wp.ScopedStream(stream, sync_enter=False, sync_exit=False) if stream is not None else nullcontext()
        with scope:
            wp.launch(
                _scatter_usd_local_transforms,
                dim=len(node_ids),
                inputs=[
                    transform_count,
                    node_ids,
                    transforms,
                    self._d_usd_local_transforms,
                ],
                device="cuda",
            )
            self._evaluate_usd_transform_hierarchy()
            if rebuild_tlas:
                self.rebuild_tlas()

    def set_usd_local_transform_trs_device(
        self,
        transform_count: wp.array,
        node_ids: wp.array,
        transforms: wp.array,
        scales: wp.array,
        stream=None,
        rebuild_tlas: bool = True,
    ):
        """Apply CUDA-resident Warp transforms/scales to USD local nodes."""
        if self._d_usd_local_transforms is None:
            raise RuntimeError("Build the scene before updating USD transforms")
        if (
            not transform_count.device.is_cuda
            or not node_ids.device.is_cuda
            or not transforms.device.is_cuda
            or not scales.device.is_cuda
        ):
            raise ValueError("USD device transform updates require CUDA arrays")
        if len(transform_count) != 1:
            raise ValueError("transform_count must contain exactly one value")
        if len(node_ids) != len(transforms) or len(node_ids) != len(scales):
            raise ValueError("node_ids, transforms, and scales must have equal length")
        if len(node_ids) == 0:
            return
        if (
            transform_count.dtype != wp.int32
            or node_ids.dtype != wp.int32
            or transforms.dtype != wp.transform
            or scales.dtype != wp.vec3
        ):
            raise TypeError("USD TRS batches require an int32 count, int32 IDs, transform poses, and vec3 scales")
        scope = wp.ScopedStream(stream, sync_enter=False, sync_exit=False) if stream is not None else nullcontext()
        with scope:
            wp.launch(
                _scatter_usd_local_transform_trs,
                dim=len(node_ids),
                inputs=[
                    transform_count,
                    node_ids,
                    transforms,
                    scales,
                    self._d_usd_local_transforms,
                ],
                device="cuda",
            )
            self._evaluate_usd_transform_hierarchy()
            if rebuild_tlas:
                self.rebuild_tlas()

    def set_usd_local_transforms(self, node_ids, transforms: np.ndarray, stream=None, rebuild_tlas: bool = True):
        """Upload a host batch and apply it through the CUDA hierarchy path."""
        indices = np.asarray(node_ids, dtype=np.int32).reshape(-1)
        matrices = np.ascontiguousarray(transforms, dtype=np.float32).reshape(-1, 4, 4)
        if len(indices) != len(matrices):
            raise ValueError("node_ids and transforms must have equal length")
        if len(indices) == 0:
            return
        if int(indices.min()) < 0 or int(indices.max()) >= len(self._usd_node_parents):
            raise IndexError("USD transform handle is out of range")
        self._usd_local_transforms[indices] = matrices
        self._evaluate_usd_transform_hierarchy_host(update_instances=self._d_usd_local_transforms is None)
        if self._d_usd_local_transforms is None:
            return
        scope = wp.ScopedStream(stream, sync_enter=False, sync_exit=False) if stream is not None else nullcontext()
        with scope:
            device_count = wp.array([len(indices)], dtype=wp.int32, device="cuda")
            device_ids = wp.array(indices, dtype=wp.int32, device="cuda")
            device_matrices = wp.array(matrices, dtype=wp.mat44, device="cuda")
            self.set_usd_local_transforms_device(
                device_count,
                device_ids,
                device_matrices,
                rebuild_tlas=rebuild_tlas,
            )

    def set_instance_material(self, instance_index: int, material_id: int | None):
        """Override the material used by one instance."""
        if 0 <= instance_index < len(self._instances):
            self._instances[instance_index].material_id = material_id

    def set_instance_visible(self, instance_index: int, visible: bool):
        """Control whether one instance participates in traversal."""
        if 0 <= instance_index < len(self._instances):
            value = bool(visible)
            self._instances[instance_index].visible = value
            self._instance_visibility_cache[instance_index] = value
            self._instance_records_dirty = True

    def set_instances_visible_batch(self, instance_indices, visible: bool):
        """Set visibility for a batch with one host-array assignment."""
        indices = np.asarray(instance_indices, dtype=np.intp)
        if len(indices) == 0:
            return
        if int(indices.min()) < 0 or int(indices.max()) >= len(self._instances):
            raise IndexError("instance index is out of range")
        self._instance_visibility_cache[indices] = bool(visible)
        if self._instance_masks is None:
            self._instance_records_dirty = True
            return
        device_indices = wp.array(indices.astype(np.int32), dtype=wp.int32, device="cuda")
        wp.launch(
            _update_device_instance_visibility,
            dim=len(indices),
            inputs=[
                device_indices,
                wp.uint32(0xFF if visible else 0),
                self._instance_masks,
            ],
            device="cuda",
        )

    def create_cornell_box(self):
        """Create a Cornell Box scene."""
        self.clear()

        # Materials
        white = self.materials.add_diffuse((0.8, 0.8, 0.8))
        red = self.materials.add_diffuse((0.8, 0.1, 0.1))
        green = self.materials.add_diffuse((0.1, 0.8, 0.1))
        light = self.materials.add_emissive((1.0, 0.95, 0.85), 15.0)

        # Floor
        self.add_box((-2, -2, -2), (2, -1.9, 2), white)
        # Ceiling
        self.add_box((-2, 1.9, -2), (2, 2, 2), white)
        # Back wall
        self.add_box((-2, -2, -2), (2, 2, -1.9), white)
        # Left wall (red)
        self.add_box((-2, -2, -2), (-1.9, 2, 2), red)
        # Right wall (green)
        self.add_box((1.9, -2, -2), (2, 2, 2), green)
        # Short box
        self.add_box((-0.8, -1.9, -0.8), (0.2, -0.5, 0.2), white)
        # Tall box
        self.add_box((0.3, -1.9, -1.2), (1.2, 0.3, -0.3), white)
        # Light
        self.add_box((-0.5, 1.85, -0.5), (0.5, 1.89, 0.5), light)

    def load_from_gltf(
        self,
        gltf_path: str,
        root_transform: np.ndarray | None = None,
        clear_existing: bool = True,
    ) -> bool:
        """Load a glTF/GLB scene into this scene."""
        from .asset_loaders import load_scene_from_gltf

        if clear_existing:
            self.clear()
        return bool(load_scene_from_gltf(self, gltf_path, root_transform=root_transform))

    def load_from_usd(
        self,
        usd_path: str,
        root_transform: np.ndarray | None = None,
        clear_existing: bool = True,
        apply_stage_units: bool = True,
        convert_up_axis: bool = True,
        max_texture_size: int | None = None,
        max_texture_memory_bytes: int | None = None,
        strict_sidedness: bool = False,
        enable_emissive_materials: bool = True,
        load_usd_lights: bool = False,
        usd_light_radius: float | None = None,
    ) -> bool:
        """Load a composed USD stage into this scene.

        OpenUSD is an optional dependency and is imported only when this method
        is called. Meshes, UVs, material subsets, UsdPreviewSurface materials,
        and common NVIDIA MDL PBR inputs are translated to the internal scene.
        """
        from .usd_loader import load_scene_from_usd

        if clear_existing:
            self.clear()
        return bool(
            load_scene_from_usd(
                self,
                usd_path,
                root_transform=root_transform,
                apply_stage_units=apply_stage_units,
                convert_up_axis=convert_up_axis,
                max_texture_size=max_texture_size,
                max_texture_memory_bytes=max_texture_memory_bytes,
                strict_sidedness=strict_sidedness,
                enable_emissive_materials=enable_emissive_materials,
                load_usd_lights=load_usd_lights,
                usd_light_radius=usd_light_radius,
            )
        )

    def load_from_obj(self, obj_path: str) -> bool:
        """Load an OBJ scene into this scene."""
        from .asset_loaders import load_scene_from_obj

        self.clear()
        return bool(load_scene_from_obj(self, obj_path))

    def add_box(self, min_pt: tuple, max_pt: tuple, material_id: int) -> int:
        """Add a box mesh and return the instance index."""
        vertices, indices = self._create_box_geometry(min_pt, max_pt)
        mesh = Mesh(vertices, indices, material_id=material_id)
        mesh_idx = self.add_mesh(mesh)
        return self.add_instance(mesh_idx)

    def add_sphere(
        self,
        center: tuple,
        radius: float,
        segments: int,
        material_id: int,
    ) -> int:
        """
        Add a UV sphere mesh and return the instance index.

        Args:
            center: Sphere center
            radius: Sphere radius
            segments: Horizontal tessellation (minimum 8)
            material_id: Material index
        """
        vertices, indices, normals, texcoords = self._create_sphere_geometry(
            center=center,
            radius=radius,
            segments=max(8, int(segments)),
        )
        mesh = Mesh(
            vertices=vertices,
            indices=indices,
            normals=normals,
            texcoords=texcoords,
            material_id=material_id,
        )
        mesh_idx = self.add_mesh(mesh)
        return self.add_instance(mesh_idx)

    def _create_box_geometry(self, min_pt: tuple, max_pt: tuple) -> tuple:
        """Create box vertices and indices."""
        mn = np.array(min_pt, dtype=np.float32)
        mx = np.array(max_pt, dtype=np.float32)

        # 8 corners
        c = np.array(
            [
                [mn[0], mn[1], mn[2]],
                [mx[0], mn[1], mn[2]],
                [mx[0], mx[1], mn[2]],
                [mn[0], mx[1], mn[2]],
                [mn[0], mn[1], mx[2]],
                [mx[0], mn[1], mx[2]],
                [mx[0], mx[1], mx[2]],
                [mn[0], mx[1], mx[2]],
            ],
            dtype=np.float32,
        )

        # 6 faces, 4 vertices each (with normals)
        vertices = []
        indices = []

        def add_quad(v0, v1, v2, v3, normal):
            base = len(vertices)
            vertices.extend([v0, v1, v2, v3])
            indices.extend([[base, base + 1, base + 2], [base, base + 2, base + 3]])

        # Front (+Z)
        add_quad(c[4], c[5], c[6], c[7], [0, 0, 1])
        # Back (-Z)
        add_quad(c[1], c[0], c[3], c[2], [0, 0, -1])
        # Left (-X)
        add_quad(c[0], c[4], c[7], c[3], [-1, 0, 0])
        # Right (+X)
        add_quad(c[5], c[1], c[2], c[6], [1, 0, 0])
        # Bottom (-Y)
        add_quad(c[0], c[1], c[5], c[4], [0, -1, 0])
        # Top (+Y)
        add_quad(c[7], c[6], c[2], c[3], [0, 1, 0])

        return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.uint32)

    def _create_sphere_geometry(
        self,
        center: tuple,
        radius: float,
        segments: int,
    ) -> tuple:
        """Create UV sphere vertices, indices, normals, and UVs."""
        cx, cy, cz = center
        lat_segments = max(4, segments // 2)
        lon_segments = max(8, segments)

        vertices = []
        normals = []
        texcoords = []
        indices = []

        for y in range(lat_segments + 1):
            v = y / lat_segments
            theta = v * np.pi
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)

            for x in range(lon_segments + 1):
                u = x / lon_segments
                phi = u * (2.0 * np.pi)
                sin_phi = np.sin(phi)
                cos_phi = np.cos(phi)

                nx = sin_theta * cos_phi
                ny = cos_theta
                nz = sin_theta * sin_phi

                px = cx + radius * nx
                py = cy + radius * ny
                pz = cz + radius * nz

                vertices.append([px, py, pz])
                normals.append([nx, ny, nz])
                texcoords.append([u, 1.0 - v])

        stride = lon_segments + 1
        for y in range(lat_segments):
            for x in range(lon_segments):
                i0 = y * stride + x
                i1 = i0 + 1
                i2 = i0 + stride
                i3 = i2 + 1
                indices.append([i0, i1, i2])
                indices.append([i1, i3, i2])

        return (
            np.array(vertices, dtype=np.float32),
            np.array(indices, dtype=np.uint32),
            np.array(normals, dtype=np.float32),
            np.array(texcoords, dtype=np.float32),
        )

    def clear(self):
        """Clear all meshes and instances."""
        self._meshes.clear()
        self._instances.clear()
        self._sphere_lights.clear()
        self._sphere_light_data = None
        self._gas.clear()
        self._ias = None
        self._instance_records = None
        self._device_instance_transforms = None
        self._instance_records_dirty = True
        self._instance_transform_cache = np.empty((0, 4, 4), dtype=np.float32)
        self._instance_visibility_cache = np.empty(0, dtype=np.bool_)
        self._instance_cache_capacity = 0
        self._tlas_updates_since_build = 0
        self.materials.clear()
        self._instance_material_ids = None
        self._compact_materials = None
        self._compact_material_floats = None
        self._instance_render_prim_ids = None
        self._instance_geometry_types = None
        self._texture_data = None
        self._texture_objects = []
        self._packed_indices = None
        self._packed_positions = None
        self._packed_radii = None
        self._packed_normals = None
        self._packed_tangents = None
        self._packed_texcoords0 = None
        self._packed_texcoords1 = None
        self._packed_prev_positions = None
        self._packed_material_ids = None
        self._gltf_textures = []
        self.usd_environment_path = None
        self.usd_ambient_light = (0.0, 0.0, 0.0)
        self.usd_scene = None
        self._usd_node_parents = np.empty(0, dtype=np.int32)
        self._usd_local_transforms = np.empty((0, 4, 4), dtype=np.float32)
        self._usd_world_transforms = np.empty((0, 4, 4), dtype=np.float32)
        self._usd_instance_ids = np.empty(0, dtype=np.int32)
        self._usd_instance_node_ids = np.empty(0, dtype=np.int32)
        self._usd_level_nodes = []
        self._d_usd_node_parents = None
        self._d_usd_local_transforms = None
        self._d_usd_world_transforms = None
        self._d_usd_instance_ids = None
        self._d_usd_instance_node_ids = None
        self._d_usd_level_nodes = []

    def build(self, optix_module):
        """
        Build the scene: upload meshes, create BLASes, and create TLAS.

        Args:
            optix_module: The imported optix module
        """
        self._optix = optix_module

        if len(self._meshes) == 0:
            logger.info("No meshes to build.")
            return

        # A full build may be requested again after prototypes, instances, or
        # materials change. Rebuild every acceleration structure and the
        # per-instance records so geometry and instance indices stay aligned.
        self._gas.clear()
        self._ias = None
        self._instance_records = None
        self._device_instance_transforms = None
        self._instance_records_dirty = True
        # Any-hit programs only matter for alpha-tested or transmissive materials.
        self._opaque_materials = np.array(
            [m["alphaMode"] == ALPHA_OPAQUE and m["transmissionFactor"] <= 0.0 for m in self.materials._materials],
            dtype=np.bool_,
        )

        logger.info(
            "Building scene with %d geometries and %d instances.",
            len(self._meshes),
            len(self._instances),
        )

        # Upload meshes to GPU
        for mesh in self._meshes:
            mesh.upload_to_gpu()

        # Build BLAS for each mesh
        self._build_blas()

        # Build TLAS
        self._build_tlas()

        # Build scene buffers
        self._build_scene_buffers()

        total_verts = sum(len(m.vertices) for m in self._meshes)
        total_tris = sum(len(m.indices) for m in self._meshes if m.primitive_type == "triangles")
        total_curves = sum(len(m.indices) for m in self._meshes if m.primitive_type != "triangles")
        logger.info(
            "Scene build complete: %d vertices, %d triangles, %d curve segments.",
            total_verts,
            total_tris,
            total_curves,
        )

    def rebuild_tlas(self, stream=None):
        """Rebuild only TLAS after instance transform updates."""
        if len(self._meshes) == 0 or len(self._instances) == 0:
            return
        if self._optix is None:
            return
        if len(self._gas) != len(self._meshes):
            # Fallback if a full build has not been completed yet.
            self.build(self._optix)
            return
        self._build_tlas(stream=stream)

    def _build_blas(self):
        """Build bottom-level acceleration structures for all geometry.

        Geometry whose materials are all opaque disables any-hit programs;
        dynamic geometry keeps them because its per-primitive materials may
        change on the device. Static geometry is not compacted; one
        synchronization after all builds releases the scratch buffers of
        static geometry, which is never refit.
        """
        opaque = self._opaque_materials
        for geometry in self._meshes:
            all_opaque = (
                not geometry.dynamic
                and len(opaque) > 0
                and bool(opaque[np.clip(geometry.material_ids, 0, len(opaque) - 1)].all())
            )
            geometry_flags = native.GEOMETRY_FLAG_DISABLE_ANYHIT if all_opaque else native.GEOMETRY_FLAG_NONE
            if geometry.primitive_type == "triangles":
                build_input = wo.triangles(geometry.d_vertices, geometry.d_indices, flags=geometry_flags)
            else:
                curve_type = (
                    native.PRIMITIVE_TYPE_ROUND_LINEAR
                    if geometry.primitive_type == "round_linear"
                    else native.PRIMITIVE_TYPE_ROUND_CUBIC_BEZIER
                )
                build_input = wo.curves(
                    geometry.d_vertices,
                    geometry.d_widths,
                    geometry.d_indices,
                    curve_type=curve_type,
                    flags=geometry_flags,
                )
            build_flags = native.BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS | (
                native.BUILD_FLAG_ALLOW_UPDATE | native.BUILD_FLAG_PREFER_FAST_BUILD
                if geometry.dynamic
                else native.BUILD_FLAG_PREFER_FAST_TRACE
            )
            self._gas.append(wo.build_accel(self._ctx, [build_input], flags=build_flags, device="cuda"))
        wp.synchronize_stream(wp.get_stream("cuda"))
        self._gas = [
            gas if geometry.dynamic else dataclasses.replace(gas, scratch=None)
            for gas, geometry in zip(self._gas, self._meshes)
        ]

    def _update_dynamic_geometry_accel(
        self,
        geometry_index: int,
        stream,
        rebuild_gas: bool,
        rebuild_tlas: bool,
    ) -> bool:
        self._gas[geometry_index] = wo.update_accel(
            self._ctx, self._gas[geometry_index], rebuild=rebuild_gas, stream=stream
        )
        if rebuild_tlas:
            self.rebuild_tlas(stream=stream)
        return True

    def update_mesh_accel(
        self,
        geometry_index: int,
        stream=None,
        rebuild_gas: bool = False,
        rebuild_tlas: bool = True,
    ) -> bool:
        """Refit or rebuild a dynamic triangle GAS after device-buffer updates."""
        geometry_index = int(geometry_index)
        if geometry_index < 0 or geometry_index >= len(self._meshes):
            raise IndexError("mesh geometry index is out of range")
        geometry = self._meshes[geometry_index]
        if geometry.primitive_type != "triangles" or not geometry.dynamic:
            raise ValueError("geometry must be a dynamic triangle mesh")
        if self._optix is None:
            return False
        if len(self._gas) != len(self._meshes):
            raise RuntimeError("build the scene before refitting dynamic meshes")

        wp.copy(
            self._packed_normals,
            geometry.d_normals,
            dest_offset=geometry._packed_normal_offset,
            count=geometry.normals.size,
            stream=stream,
        )
        return self._update_dynamic_geometry_accel(geometry_index, stream, rebuild_gas, rebuild_tlas)

    def refit_mesh(self, geometry_index: int, stream=None, rebuild_tlas: bool = True):
        """Refit one dynamic triangle GAS after its device buffers were updated."""
        return self.update_mesh_accel(
            geometry_index,
            stream=stream,
            rebuild_gas=False,
            rebuild_tlas=rebuild_tlas,
        )

    def update_curve_accel(
        self,
        geometry_index: int,
        stream=None,
        rebuild_gas: bool = False,
        rebuild_tlas: bool = True,
    ):
        """Refit or rebuild one dynamic curve GAS after updating its buffers."""
        geometry_index = int(geometry_index)
        if geometry_index < 0 or geometry_index >= len(self._meshes):
            raise IndexError("curve geometry index is out of range")
        geometry = self._meshes[geometry_index]
        if geometry.primitive_type == "triangles" or not geometry.dynamic:
            raise ValueError("geometry must be a dynamic curve")
        if self._optix is None:
            return False
        if len(self._gas) != len(self._meshes):
            raise RuntimeError("build the scene before refitting dynamic curves")

        wp.copy(
            self._packed_positions,
            geometry.d_vertices,
            dest_offset=geometry._packed_position_offset,
            count=geometry.vertices.size,
            stream=stream,
        )
        wp.copy(
            self._packed_radii,
            geometry.d_widths,
            dest_offset=geometry._packed_radius_offset,
            count=geometry.radii.size,
            stream=stream,
        )
        return self._update_dynamic_geometry_accel(geometry_index, stream, rebuild_gas, rebuild_tlas)

    def refit_curve(self, geometry_index: int, stream=None, rebuild_tlas: bool = True):
        """Refit one dynamic curve GAS after its retained buffers were updated."""
        return self.update_curve_accel(
            geometry_index,
            stream=stream,
            rebuild_gas=False,
            rebuild_tlas=rebuild_tlas,
        )

    def _build_tlas(self, stream=None):
        """Write instance records on the GPU, then build or refit the TLAS.

        The TLAS is refit in place while the instance count is unchanged and
        rebuilt in place every ``_max_tlas_updates`` refits to restore its
        traversal quality.
        """
        count = len(self._instances)
        if count == 0:
            return
        if self._instance_records is None or len(self._instance_records) != count:
            opaque = self._opaque_materials
            mesh_indices = np.fromiter((inst.mesh_index for inst in self._instances), dtype=np.intp, count=count)
            flags = np.zeros(count, dtype=np.uint32)
            for index, inst in enumerate(self._instances):
                if inst.double_sided:
                    flags[index] |= native.INSTANCE_FLAG_DISABLE_TRIANGLE_FACE_CULLING
                # A material override replaces every per-primitive material of the instance.
                if inst.material_id is not None:
                    override_opaque = 0 <= inst.material_id < len(opaque) and opaque[inst.material_id]
                    flags[index] |= (
                        native.INSTANCE_FLAG_DISABLE_ANYHIT if override_opaque else native.INSTANCE_FLAG_ENFORCE_ANYHIT
                    )
            handles = np.array([gas.handle for gas in self._gas], dtype=np.uint64)
            sbt_offsets = np.array([geometry.sbt_offset for geometry in self._meshes], dtype=np.uint32)
            self._instance_traversables = wp.array(handles[mesh_indices], dtype=wp.uint64, device="cuda")
            self._instance_sbt_offsets = wp.array(sbt_offsets[mesh_indices], dtype=wp.uint32, device="cuda")
            self._instance_flags = wp.array(flags, dtype=wp.uint32, device="cuda")
            self._instance_masks = wp.empty(count, dtype=wp.uint32, device="cuda")
            self._device_instance_transforms = wp.empty((count, 12), dtype=wp.float32, device="cuda")
            self._instance_records = wp.empty(count, dtype=wo.Instance, device="cuda")
            self._instance_records_dirty = True
            self._ias = None

        if self._instance_records_dirty:
            # Motion-vector reconstruction reads the same transforms buffer.
            self._device_instance_transforms.assign(self._instance_transform_cache[:count, :3, :].reshape(count, 12))
            self._instance_masks.assign(np.where(self._instance_visibility_cache[:count], 0xFF, 0).astype(np.uint32))
            self._instance_records_dirty = False

        wp.launch(
            _write_instance_records,
            dim=count,
            inputs=[
                self._device_instance_transforms,
                self._instance_traversables,
                self._instance_sbt_offsets,
                self._instance_masks,
                self._instance_flags,
            ],
            outputs=[self._instance_records],
            device="cuda",
            stream=stream,
        )
        if self._ias is None:
            self._ias = wo.build_accel(
                self._ctx,
                [wo.instances(self._instance_records)],
                flags=native.BUILD_FLAG_ALLOW_UPDATE,
                stream=stream,
                device="cuda",
            )
            self._tlas_updates_since_build = 0
        else:
            rebuild = self._tlas_updates_since_build >= self._max_tlas_updates
            self._ias = wo.update_accel(self._ctx, self._ias, rebuild=rebuild, stream=stream)
            self._tlas_updates_since_build = 0 if rebuild else self._tlas_updates_since_build + 1

    def _build_scene_buffers(self):
        """Build GPU buffers for scene description."""
        # Build RenderPrimitive array
        rp_dtype = _create_render_primitive_dtype()
        render_primitives = np.zeros(len(self._meshes), dtype=rp_dtype)
        packed_indices = []
        packed_positions = []
        packed_radii = []
        packed_normals = []
        packed_tangents = []
        packed_texcoords0 = []
        packed_texcoords1 = []
        packed_material_ids = []

        index_offset = 0
        position_offset = 0
        radius_offset = 0
        normal_offset = 0
        tangent_offset = 0
        tex0_offset = 0
        tex1_offset = 0
        material_offset = 0

        for i, geometry in enumerate(self._meshes):
            rp = render_primitives[i]
            flat_indices = geometry.indices.reshape(-1)
            is_triangle_mesh = geometry.primitive_type == "triangles"
            flat_positions = np.empty(0, dtype=np.float32) if is_triangle_mesh else geometry.vertices.reshape(-1)
            flat_radii = np.empty(0, dtype=np.float32) if is_triangle_mesh else geometry.radii
            flat_normals = geometry.normals.reshape(-1) if is_triangle_mesh else np.empty(0, dtype=np.float32)
            flat_tangents = geometry.tangents.reshape(-1) if is_triangle_mesh else np.empty(0, dtype=np.float32)
            flat_tex0 = geometry.texcoords.reshape(-1) if is_triangle_mesh else np.empty(0, dtype=np.float32)
            flat_tex1 = geometry.texcoords1.reshape(-1) if is_triangle_mesh else np.empty(0, dtype=np.float32)
            primitive_material_ids = geometry.material_ids

            rp["indexOffset"] = np.uint32(index_offset)
            rp["materialIdOffset"] = np.uint32(material_offset)
            rp["vertexBuffer"]["positionOffset"] = np.uint32(position_offset)
            rp["vertexBuffer"]["normalOffset"] = np.uint32(normal_offset)
            rp["vertexBuffer"]["radiusOffset"] = np.uint32(radius_offset)
            rp["vertexBuffer"]["tangentOffset"] = np.uint32(tangent_offset)
            rp["vertexBuffer"]["texCoord0Offset"] = np.uint32(tex0_offset)
            rp["vertexBuffer"]["texCoord1Offset"] = np.uint32(tex1_offset)
            rp["vertexBuffer"]["prevPositionOffset"] = np.uint32(0)
            rp["vertexBuffer"]["hasTexCoord1"] = np.uint32(is_triangle_mesh)
            rp["vertexBuffer"]["hasPrevPosition"] = np.uint32(0)
            rp["numIndices"] = np.uint32(geometry.indices.size)
            geometry._packed_position_offset = position_offset
            geometry._packed_normal_offset = normal_offset
            geometry._packed_radius_offset = radius_offset
            geometry._packed_material_offset = material_offset
            rp["numVertices"] = np.uint32(geometry.vertices.shape[0])

            packed_indices.append(flat_indices.astype(np.uint32, copy=False))
            packed_positions.append(flat_positions.astype(np.float32, copy=False))
            packed_normals.append(flat_normals.astype(np.float32, copy=False))
            packed_tangents.append(flat_tangents.astype(np.float32, copy=False))
            packed_radii.append(flat_radii.astype(np.float32, copy=False))
            packed_texcoords0.append(flat_tex0.astype(np.float32, copy=False))
            packed_texcoords1.append(flat_tex1.astype(np.float32, copy=False))
            packed_material_ids.append(primitive_material_ids)

            index_offset += flat_indices.shape[0]
            position_offset += flat_positions.shape[0]
            normal_offset += flat_normals.shape[0]
            tangent_offset += flat_tangents.shape[0]
            tex0_offset += flat_tex0.shape[0]
            tex1_offset += flat_tex1.shape[0]
            material_offset += primitive_material_ids.shape[0]
            radius_offset += flat_radii.shape[0]

        rp_bytes = render_primitives.view(np.uint8).reshape(-1)
        self._render_primitives = wp.array(rp_bytes, dtype=wp.uint8, device="cuda")
        self._packed_indices = wp.array(np.concatenate(packed_indices), dtype=wp.uint32, device="cuda")

        def packed_float_buffer(parts):
            values = np.concatenate(parts)
            return wp.array(values, dtype=wp.float32, device="cuda") if values.size else None

        self._packed_positions = packed_float_buffer(packed_positions)
        self._packed_radii = packed_float_buffer(packed_radii)
        self._packed_normals = packed_float_buffer(packed_normals)
        self._packed_tangents = packed_float_buffer(packed_tangents)
        self._packed_texcoords0 = packed_float_buffer(packed_texcoords0)
        self._packed_texcoords1 = packed_float_buffer(packed_texcoords1)
        self._packed_prev_positions = None
        self._packed_material_ids = wp.array(np.concatenate(packed_material_ids), dtype=wp.uint32, device="cuda")

        self._build_usd_transform_buffers()

        light_data = np.zeros((len(self._sphere_lights), 8), dtype=np.float32)
        for index, (position, radius, color, intensity) in enumerate(self._sphere_lights):
            light_data[index, :3] = position
            light_data[index, 3] = radius
            light_data[index, 4:7] = np.asarray(color) * intensity
        self._sphere_light_data = wp.array(light_data, dtype=wp.float32, device="cuda") if len(light_data) else None

        # Build per-instance material ID lookup buffer.
        instance_material_ids = np.full(len(self._instances), NO_MATERIAL_OVERRIDE, dtype=np.uint32)
        instance_render_prim_ids = np.zeros(len(self._instances), dtype=np.uint32)
        instance_geometry_types = np.zeros(len(self._instances), dtype=np.uint32)
        for i, inst in enumerate(self._instances):
            if inst.material_id is not None:
                instance_material_ids[i] = np.uint32(inst.material_id)
            instance_render_prim_ids[i] = np.uint32(inst.mesh_index)
            instance_geometry_types[i] = np.uint32(getattr(self._meshes[inst.mesh_index], "geometry_type", 0))
        self._instance_material_ids = wp.array(instance_material_ids, dtype=wp.uint32, device="cuda")
        self._instance_render_prim_ids = wp.array(instance_render_prim_ids, dtype=wp.uint32, device="cuda")
        self._instance_geometry_types = wp.array(instance_geometry_types, dtype=wp.uint32, device="cuda")

        # Build compact material table for robust device-side lookup.
        compact_dt = np.dtype(
            [
                ("baseColor", np.float32, (3,)),
                ("emissive", np.float32, (3,)),
                ("roughness", np.float32),
                ("metallic", np.float32),
                ("uSubdiv", np.float32),
                ("vSubdiv", np.float32),
                ("baseColorScale", np.float32),
                ("baseColorAdd", np.float32),
                ("baseColorDesaturation", np.float32),
                ("alphaMode", np.int32),
                ("alphaCutoff", np.float32),
                ("transmission", np.float32),
                ("transmissionColor", np.float32, (3,)),
                ("textureSize", np.float32),
                ("ior", np.float32),
                ("specularColor", np.float32, (3,)),
                ("specular", np.float32),
                ("clearcoat", np.float32),
                ("clearcoatRoughness", np.float32),
                ("sheenRoughness", np.float32),
                ("occlusion", np.float32),
                ("occlusionTexIndex", np.int32),
                ("occlusionTexCoord", np.int32),
                ("sheenColor", np.float32, (3,)),
                ("diffuseTransmissionFactor", np.float32),
                ("diffuseTransmissionColor", np.float32, (3,)),
                ("isThinWalled", np.int32),
                ("clearcoatNormalTexIndex", np.int32),
                ("clearcoatNormalTexCoord", np.int32),
                ("opacity", np.float32),
                ("opacityFresnelLow", np.float32),
                ("opacityFresnelHigh", np.float32),
                ("opacityFresnelFalloff", np.float32),
                ("baseColorTexIndex", np.int32),
                ("baseColorTexCoord", np.int32),
                ("metallicRoughnessTexIndex", np.int32),
                ("metallicRoughnessTexCoord", np.int32),
                ("normalTexIndex", np.int32),
                ("normalTexCoord", np.int32),
                ("emissiveTexIndex", np.int32),
                ("emissiveTexCoord", np.int32),
                ("normalScale", np.float32, (2,)),
                ("baseColorUvTransform", np.float32, (6,)),
                ("metallicRoughnessUvTransform", np.float32, (6,)),
                ("normalUvTransform", np.float32, (6,)),
                ("emissiveUvTransform", np.float32, (6,)),
                ("occlusionUvTransform", np.float32, (6,)),
                ("clearcoatNormalUvTransform", np.float32, (6,)),
            ],
            align=True,
        )
        compact = np.zeros(self.materials.count, dtype=compact_dt)
        for i, mat in enumerate(self.materials._materials):
            compact[i]["baseColor"] = mat["pbrBaseColorFactor"][:3]
            compact[i]["emissive"] = mat["emissiveFactor"]
            compact[i]["roughness"] = mat["pbrRoughnessFactor"]
            compact[i]["metallic"] = mat["pbrMetallicFactor"]
            compact[i]["uSubdiv"] = mat["uSubdiv"]
            compact[i]["vSubdiv"] = mat["vSubdiv"]
            compact[i]["baseColorScale"] = mat["baseColorScale"]
            compact[i]["baseColorAdd"] = mat["pbrDiffuseFactor"][0]
            compact[i]["baseColorDesaturation"] = mat["pbrDiffuseFactor"][1]
            compact[i]["alphaMode"] = mat["alphaMode"]
            compact[i]["alphaCutoff"] = mat["alphaCutoff"]
            compact[i]["transmission"] = mat["transmissionFactor"]
            compact[i]["transmissionColor"] = mat["transmissionColorFactor"]
            texture_size = 1
            for texture_field in (
                "pbrBaseColorTexture",
                "normalTexture",
                "pbrMetallicRoughnessTexture",
                "emissiveTexture",
                "occlusionTexture",
                "clearcoatNormalTexture",
            ):
                texture_index = int(mat[texture_field]["index"])
                if 0 <= texture_index < len(self._gltf_textures):
                    texture = self._gltf_textures[texture_index]
                    texture_size = max(texture_size, texture.shape[0], texture.shape[1])
            compact[i]["textureSize"] = float(texture_size)
            compact[i]["ior"] = mat["ior"]
            compact[i]["specularColor"] = mat["specularColorFactor"]
            compact[i]["specular"] = mat["specularFactor"]
            compact[i]["clearcoat"] = mat["clearcoatFactor"]
            compact[i]["clearcoatRoughness"] = mat["clearcoatRoughness"]
            compact[i]["sheenRoughness"] = mat["sheenRoughnessFactor"]
            compact[i]["occlusion"] = mat["occlusionStrength"]
            compact[i]["occlusionTexIndex"] = mat["occlusionTexture"]["index"]
            compact[i]["occlusionTexCoord"] = mat["occlusionTexture"]["texCoord"]
            compact[i]["sheenColor"] = mat["sheenColorFactor"]
            compact[i]["diffuseTransmissionFactor"] = mat["diffuseTransmissionFactor"]
            compact[i]["diffuseTransmissionColor"] = mat["diffuseTransmissionColor"]
            compact[i]["isThinWalled"] = 1 if mat["thicknessFactor"] == 0.0 else 0
            compact[i]["clearcoatNormalTexIndex"] = mat["clearcoatNormalTexture"]["index"]
            compact[i]["clearcoatNormalTexCoord"] = mat["clearcoatNormalTexture"]["texCoord"]
            compact[i]["opacity"] = mat["pbrBaseColorFactor"][3]
            compact[i]["opacityFresnelLow"] = mat["opacityFresnelLow"]
            compact[i]["opacityFresnelHigh"] = mat["opacityFresnelHigh"]
            compact[i]["opacityFresnelFalloff"] = mat["opacityFresnelFalloff"]
            compact[i]["baseColorTexIndex"] = mat["pbrBaseColorTexture"]["index"]
            compact[i]["baseColorTexCoord"] = mat["pbrBaseColorTexture"]["texCoord"]
            compact[i]["metallicRoughnessTexIndex"] = mat["pbrMetallicRoughnessTexture"]["index"]
            compact[i]["metallicRoughnessTexCoord"] = mat["pbrMetallicRoughnessTexture"]["texCoord"]
            compact[i]["normalTexIndex"] = mat["normalTexture"]["index"]
            compact[i]["normalTexCoord"] = mat["normalTexture"]["texCoord"]
            compact[i]["emissiveTexIndex"] = mat["emissiveTexture"]["index"]
            compact[i]["emissiveTexCoord"] = mat["emissiveTexture"]["texCoord"]
            compact[i]["normalScale"] = mat["normalTextureScale"]
            compact[i]["baseColorUvTransform"] = (
                mat["pbrBaseColorTexture"]["uvTransform00"],
                mat["pbrBaseColorTexture"]["uvTransform01"],
                mat["pbrBaseColorTexture"]["uvTransform02"],
                mat["pbrBaseColorTexture"]["uvTransform10"],
                mat["pbrBaseColorTexture"]["uvTransform11"],
                mat["pbrBaseColorTexture"]["uvTransform12"],
            )
            compact[i]["metallicRoughnessUvTransform"] = (
                mat["pbrMetallicRoughnessTexture"]["uvTransform00"],
                mat["pbrMetallicRoughnessTexture"]["uvTransform01"],
                mat["pbrMetallicRoughnessTexture"]["uvTransform02"],
                mat["pbrMetallicRoughnessTexture"]["uvTransform10"],
                mat["pbrMetallicRoughnessTexture"]["uvTransform11"],
                mat["pbrMetallicRoughnessTexture"]["uvTransform12"],
            )
            compact[i]["normalUvTransform"] = (
                mat["normalTexture"]["uvTransform00"],
                mat["normalTexture"]["uvTransform01"],
                mat["normalTexture"]["uvTransform02"],
                mat["normalTexture"]["uvTransform10"],
                mat["normalTexture"]["uvTransform11"],
                mat["normalTexture"]["uvTransform12"],
            )
            compact[i]["emissiveUvTransform"] = (
                mat["emissiveTexture"]["uvTransform00"],
                mat["emissiveTexture"]["uvTransform01"],
                mat["emissiveTexture"]["uvTransform02"],
                mat["emissiveTexture"]["uvTransform10"],
                mat["emissiveTexture"]["uvTransform11"],
                mat["emissiveTexture"]["uvTransform12"],
            )
            compact[i]["occlusionUvTransform"] = (
                mat["occlusionTexture"]["uvTransform00"],
                mat["occlusionTexture"]["uvTransform01"],
                mat["occlusionTexture"]["uvTransform02"],
                mat["occlusionTexture"]["uvTransform10"],
                mat["occlusionTexture"]["uvTransform11"],
                mat["occlusionTexture"]["uvTransform12"],
            )
            compact[i]["clearcoatNormalUvTransform"] = (
                mat["clearcoatNormalTexture"]["uvTransform00"],
                mat["clearcoatNormalTexture"]["uvTransform01"],
                mat["clearcoatNormalTexture"]["uvTransform02"],
                mat["clearcoatNormalTexture"]["uvTransform10"],
                mat["clearcoatNormalTexture"]["uvTransform11"],
                mat["clearcoatNormalTexture"]["uvTransform12"],
            )
        compact_bytes = compact.view(np.uint8).reshape(-1)
        self._compact_materials = wp.array(compact_bytes, dtype=wp.uint8, device="cuda")
        self._compact_material_floats = None

        # Keep one CUDA texture per source image. The compact device array stores
        # only texture handles, so aggregate texture memory is not constrained by
        # Warp's signed 32-bit limit for a single array dimension.
        if self._gltf_textures:

            def create_texture(tex):
                return wp.Texture2D(
                    tex,
                    num_mip_levels=0 if self.enable_texture_mipmaps else 1,
                    filter_mode=wp.TextureFilterMode.LINEAR,
                    address_mode=wp.TextureAddressMode.WRAP,
                    normalized_coords=True,
                    device="cuda",
                )

            workers = _texture_worker_count(len(self._gltf_textures))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                self._texture_objects = list(executor.map(create_texture, self._gltf_textures))
            self._texture_data = wp.array(
                self._texture_objects,
                dtype=wp.Texture2D,
                device="cuda",
            )
        else:
            self._texture_data = None
            self._texture_objects = []
