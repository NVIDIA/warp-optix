# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Owned cluster builds. Descriptor packing and memory management stay internal."""

import ctypes
import sys

import numpy as np
import warp as wp

from warp_optix import native as n
from warp_optix._src.context import get_context, resolve
from warp_optix._src.enums import GeometryFlags


@wp.struct
class _TemplateArgs:
    cluster_id: wp.uint32
    sbt_offset: wp.uint32
    cluster_template: wp.uint64
    vertices: wp.uint64
    stride: wp.uint32
    reserved: wp.uint32


@wp.struct
class _GridArgs:
    cluster_id: wp.uint32
    flags: wp.uint32
    primitive_info: wp.uint32
    truncate: wp.uint32
    dimensions: wp.uint32


@wp.struct
class _GasArgs:
    count: wp.uint32
    stride: wp.uint32
    handles: wp.uint64


for _struct_type, _expected_size in ((_GridArgs, 20), (_TemplateArgs, 32), (_GasArgs, 16)):
    # Warp's struct decorator installs this ABI descriptor dynamically.
    assert ctypes.sizeof(_struct_type.ctype) == _expected_size  # pyright: ignore[reportAttributeAccessIssue]


@wp.kernel(enable_backward=False)
def _grid_args(size: int, truncate: int, primitive_flags: int, args: wp.array[_GridArgs]):
    i = wp.tid()
    a = _GridArgs()
    a.primitive_info = wp.uint32(primitive_flags) << wp.uint32(29)
    a.truncate = wp.uint32(truncate)
    a.dimensions = wp.uint32(i // size + 1) | (wp.uint32(i % size + 1) << wp.uint32(8))
    args[i] = a


@wp.kernel(enable_backward=False)
def _gas_args(count: int, handles: wp.uint64, args: wp.array[_GasArgs]):
    a = _GasArgs()
    a.count = wp.uint32(count)
    a.handles = handles
    args[0] = a


@wp.kernel(enable_backward=False)
def _pack_args(
    points: wp.uint64,
    grids: wp.array[wp.vec2i],
    offsets: wp.array[int],
    ids: wp.array[wp.uint32],
    sbt_offsets: wp.array[wp.uint32],
    templates: wp.array[wp.uint64],
    size: int,
    args: wp.array[_TemplateArgs],
):
    i = wp.tid()
    c = grids[i]
    a = _TemplateArgs()
    a.cluster_id = wp.uint32(i)
    if ids.shape[0] != 0:
        a.cluster_id = ids[i]
    if sbt_offsets.shape[0] != 0:
        a.sbt_offset = sbt_offsets[i]
    a.cluster_template = templates[(c[0] - 1) * size + c[1] - 1]
    a.vertices = points + wp.uint64(12) * wp.uint64(offsets[i])
    a.stride = wp.uint32(12)
    args[i] = a


def _build_input(kind, count, size=8, truncate=0, max_sbt_index=0):
    result = n.ClusterAccelBuildInput()
    result.type = kind
    if kind == n.CLUSTER_ACCEL_BUILD_TYPE_TEMPLATES_FROM_GRIDS:
        desc = n.ClusterAccelBuildInputGrids()
        desc.flags = n.CLUSTER_ACCEL_BUILD_FLAG_NONE
        desc.maxArgCount = count
        desc.vertexFormat = n.VERTEX_FORMAT_FLOAT3
        desc.maxWidth = size
        desc.maxHeight = size
        desc.maxSbtIndexValue = max_sbt_index
        result.grids = desc
    elif kind == n.CLUSTER_ACCEL_BUILD_TYPE_GASES_FROM_CLUSTERS:
        desc = n.ClusterAccelBuildInputClusters()
        desc.flags = n.CLUSTER_ACCEL_BUILD_FLAG_NONE
        desc.maxArgCount = 1
        desc.maxTotalClusterCount = count
        desc.maxClusterCountPerArg = count
        result.clusters = desc
    else:
        desc = n.ClusterAccelBuildInputTriangles()
        desc.flags = n.CLUSTER_ACCEL_BUILD_FLAG_NONE
        desc.maxArgCount = count
        desc.vertexFormat = n.VERTEX_FORMAT_FLOAT3
        desc.maxUniqueSbtIndexCountPerArg = 1
        desc.maxTriangleCountPerArg = 2 * size * size
        desc.maxVertexCountPerArg = (size + 1) ** 2
        desc.minPositionTruncateBitCount = truncate
        desc.maxSbtIndexValue = max_sbt_index
        result.triangles = desc
    return result


def _primitive_flags(geometry_flags):
    """Translate OptixGeometryFlags into OptixClusterAccelPrimitiveFlags, whose bits differ."""
    flags = int(geometry_flags)
    result = 0
    for geometry, primitive in (
        (n.GEOMETRY_FLAG_DISABLE_ANYHIT, n.CLUSTER_ACCEL_PRIMITIVE_FLAG_DISABLE_ANYHIT),
        (n.GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL, n.CLUSTER_ACCEL_PRIMITIVE_FLAG_REQUIRE_SINGLE_ANYHIT_CALL),
        (n.GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_CULLING, n.CLUSTER_ACCEL_PRIMITIVE_FLAG_DISABLE_TRIANGLE_FACE_CULLING),
    ):
        if flags & int(geometry):
            flags &= ~int(geometry)
            result |= int(primitive)
    if flags:
        raise ValueError("Unsupported geometry_flags for cluster templates")
    return result


class _ImplicitBuild:
    """An implicit-destination build whose buffers fit ``capacity`` arguments of ``desc``.

    OptiX sizes the output for the worst case, so rebuilds with fewer arguments
    reuse every buffer, and neither sizes nor offsets are read back to the host.
    """

    def __init__(self, context, desc, capacity):
        self.context = context
        self.desc = desc
        output, scratch = n.clusterAccelComputeMemoryUsage(
            context.native, n.CLUSTER_ACCEL_BUILD_MODE_IMPLICIT_DESTINATIONS, desc
        )
        self.output = wp.empty(output, dtype=wp.uint8)
        self.scratch = wp.empty(scratch, dtype=wp.uint8)
        self.handles = wp.empty(capacity, dtype=wp.uint64)

    def build(self, args, count=None):
        """Build from device ``args``; ``count`` is an optional device uint32 array overriding ``maxArgCount``."""
        dest = n.ClusterAccelBuildModeDescImplicitDest()
        dest.outputBuffer = self.output.ptr
        dest.outputBufferSizeInBytes = self.output.size
        dest.tempBuffer = self.scratch.ptr
        dest.tempBufferSizeInBytes = self.scratch.size
        dest.outputHandlesBuffer = self.handles.ptr
        mode = n.ClusterAccelBuildModeDesc()
        mode.mode = n.CLUSTER_ACCEL_BUILD_MODE_IMPLICIT_DESTINATIONS
        mode.implicitDest = dest
        n.clusterAccelBuild(
            self.context.native,
            wp.get_stream().cuda_stream,
            mode,
            self.desc,
            args.ptr,
            0 if count is None else count.ptr,
        )


class _TemplateState:
    """Device state of a :class:`ClusterTemplates`, shared by the structures instantiating it."""

    def __init__(self, size, truncate, max_sbt_index, primitive_flags):
        context = get_context()
        props = n.getClusterProperties(context.native)
        if not props["flags"]:
            raise RuntimeError("The OptiX driver reports no Cluster API support on this device")
        if (
            size > props["max_grid_resolution"]
            or (size + 1) ** 2 > props["max_vertices"]
            or 2 * size**2 > props["max_triangles"]
        ):
            raise ValueError(f"Cluster edge size {size} exceeds device limits: {props}")
        if max_sbt_index > props["max_sbt_index"]:
            raise ValueError("max_sbt_index exceeds the device limit")
        self.capabilities = props
        self.size = size
        self.truncate = truncate
        self.max_sbt_index = max_sbt_index
        # One template per grid resolution: index (width - 1) * size + (height - 1).
        count = size * size
        build = _ImplicitBuild(
            context, _build_input(n.CLUSTER_ACCEL_BUILD_TYPE_TEMPLATES_FROM_GRIDS, count, size, 0, max_sbt_index), count
        )
        grid_args = wp.empty(count, dtype=_GridArgs)
        wp.launch(_grid_args, count, inputs=[size, truncate, primitive_flags, grid_args])
        build.build(grid_args)
        # The arguments and scratch are freed in stream order after the build.
        self.output = build.output
        self.handles = build.handles


class _ClusterBuild:
    """CLAS and GAS buffers for up to ``capacity`` clusters."""

    def __init__(self, context, templates, capacity):
        self.capacity = capacity
        self.clas = _ImplicitBuild(
            context,
            _build_input(
                n.CLUSTER_ACCEL_BUILD_TYPE_CLUSTERS_FROM_TEMPLATES,
                capacity,
                templates.size,
                templates.truncate,
                templates.max_sbt_index,
            ),
            capacity,
        )
        self.gas = _ImplicitBuild(context, _build_input(n.CLUSTER_ACCEL_BUILD_TYPE_GASES_FROM_CLUSTERS, capacity), 1)
        self.args = wp.empty(capacity, dtype=_TemplateArgs)
        self.count = wp.empty(1, dtype=wp.uint32)
        self.gas_args = wp.empty(1, dtype=_GasArgs)


def cluster_capabilities(device=None):
    """Return Cluster API support and device limits, without allocating geometry."""
    context, stream = resolve(device)
    with wp.ScopedDevice(context.device), wp.ScopedStream(stream, sync_enter=False):
        return n.getClusterProperties(context.native)


class ClusterTemplates:
    """Reusable grid templates for :class:`ClusterAccel`.

    ``max_edge_segments`` bounds each grid dimension, in quads. Templates are
    created automatically by ``ClusterAccel``; create this object explicitly
    to share templates across structures. ``position_truncate_bits`` controls
    position mantissa truncation during CLAS construction. ``max_sbt_index``
    bounds per-cluster SBT offsets. Unsupported hardware raises an error.
    """

    def __init__(
        self,
        max_edge_segments=8,
        *,
        position_truncate_bits=0,
        max_sbt_index=0,
        geometry_flags=GeometryFlags.DISABLE_ANYHIT,
        device=None,
        stream=None,
    ):
        self._closed = True
        self._context, stream = resolve(device, stream)
        if not isinstance(max_edge_segments, int) or max_edge_segments < 1:
            raise ValueError("max_edge_segments must be a positive integer")
        if not isinstance(position_truncate_bits, int) or not 0 <= position_truncate_bits <= 23:
            raise ValueError("position_truncate_bits must lie in [0, 23]")
        if not isinstance(max_sbt_index, int) or not 0 <= max_sbt_index <= 0xFFFFFF:
            raise ValueError("max_sbt_index must fit in 24 bits")
        primitive_flags = _primitive_flags(geometry_flags)
        self.max_edge_segments = max_edge_segments
        self.position_truncate_bits = position_truncate_bits
        self.max_sbt_index = max_sbt_index
        with wp.ScopedDevice(self.device), wp.ScopedStream(stream, sync_enter=False):
            self._state = _TemplateState(max_edge_segments, position_truncate_bits, max_sbt_index, primitive_flags)
            self._ready = stream.record_event()
        self.capabilities = self._state.capabilities
        self._closed = False

    @property
    def device(self):
        return self._context.device

    @property
    def closed(self):
        return self._closed

    def close(self):
        """Release this owner's references. Existing structures retain their templates."""
        self._closed = True
        self._state = None
        self._ready = None

    def __enter__(self):
        if self.closed:
            raise RuntimeError("ClusterTemplates is closed")
        return self

    def __exit__(self, *exc):
        self.close()


class ClusterAccel:
    """An owned GAS over dynamic grid clusters, built from Warp device arrays.

    ``points`` is a contiguous CUDA ``vec3`` array. ``grid_sizes`` contains one
    ``vec2i(width, height)`` per cluster, measured in quads. ``vertex_offsets``
    contains the first vertex of each grid in ``points``. Vertices within each
    grid are row-major with width varying fastest. The default cluster ID is
    its array index; optional ``cluster_ids`` and ``sbt_offsets`` are uint32 arrays.
    Nonzero SBT offsets require shared templates with a sufficient ``max_sbt_index``.
    ``max_edge_segments``, ``position_truncate_bits`` and ``geometry_flags``
    configure the templates created when ``templates`` is omitted; combining
    them with ``templates`` is an error.

    Construction builds the structure. Mutate the arrays and call ``rebuild()``;
    pass replacement arrays to ``rebuild`` when grid counts change. Pass this
    object directly to ``wo.launch`` or use ``accel.id``, like ``wo.Accel``.
    Pipelines enable ``clustered_geometry=True``. All geometry, templates and
    acceleration buffers are retained automatically. This API always uses the
    Cluster API, and never falls back to ordinary triangle geometry.

    Buffers are sized for the worst case of the largest cluster count built so
    far and are reused by rebuilds. A rebuild with more clusters than that
    allocates new buffers and may change ``id``: rewrite instance records that
    reference it and rebuild the instance structure, and set it again on recorded
    launches (``record_cmd=True``). Instance structures and recorded launches keep
    every buffer they may reference alive, also after this object is closed.

    ``validate=False`` skips host bounds checks for trusted GPU-generated grids;
    an out-of-range grid size or vertex offset then faults on the GPU.
    Each rebuild synchronizes the stream once, to read the traversable handle;
    validation adds further copies. Builds and traversals on different streams
    need explicit ordering, as with ``wo.Accel``. Cluster structures support
    rebuilds, not refits or compaction.
    """

    def __init__(
        self,
        points,
        grid_sizes,
        vertex_offsets,
        *,
        templates=None,
        max_edge_segments=None,
        position_truncate_bits=None,
        geometry_flags=None,
        cluster_ids=None,
        sbt_offsets=None,
        validate=True,
        stream=None,
    ):
        self._closed = True
        if not isinstance(points, wp.array):
            raise TypeError("points must be a Warp array")
        if templates is not None and (
            max_edge_segments is not None or position_truncate_bits is not None or geometry_flags is not None
        ):
            raise ValueError(
                "max_edge_segments, position_truncate_bits and geometry_flags configure new templates; "
                "set them on the ClusterTemplates instead"
            )
        self._context, stream = resolve(points.device, stream)
        with wp.ScopedDevice(self.device), wp.ScopedStream(stream, sync_enter=False):
            if templates is None:
                templates = ClusterTemplates(
                    8 if max_edge_segments is None else max_edge_segments,
                    position_truncate_bits=0 if position_truncate_bits is None else position_truncate_bits,
                    geometry_flags=GeometryFlags.DISABLE_ANYHIT if geometry_flags is None else geometry_flags,
                    device=self.device,
                    stream=stream,
                )
            if not isinstance(templates, ClusterTemplates) or templates.closed:
                raise ValueError("templates must be an open ClusterTemplates object")
            if templates.device != self.device:
                raise ValueError("Templates and points must use the same CUDA device")
            stream.wait_event(templates._ready)
            self._templates = templates._state
            self._build = None
            # Shared with instance structures, which retain it rather than a snapshot. It only
            # grows when buffers are reallocated, so any handle ever returned stays valid.
            self._memory = [self._templates.output]
            self._empty = wp.empty(0, dtype=wp.uint32)
            self.points, self.grid_sizes, self.vertex_offsets = points, grid_sizes, vertex_offsets
            self.cluster_ids = self._empty if cluster_ids is None else cluster_ids
            self.sbt_offsets = self._empty if sbt_offsets is None else sbt_offsets
            self._closed = False
            try:
                self.rebuild(validate=validate, stream=stream)
            except BaseException:
                self.close()
                raise

    @property
    def device(self):
        return self._context.device

    @property
    def closed(self):
        return self._closed

    @property
    def id(self):
        self._check_open()
        return self._id

    def _check_open(self):
        if self.closed:
            raise RuntimeError("ClusterAccel is closed")

    def rebuild(
        self,
        *,
        points=None,
        grid_sizes=None,
        vertex_offsets=None,
        cluster_ids=None,
        sbt_offsets=None,
        validate=True,
        stream=None,
    ):
        """Rebuild current grids, optionally replacing arrays when topology changes."""
        self._check_open()
        _, stream = resolve(self.device, stream)
        points = self.points if points is None else points
        grids = self.grid_sizes if grid_sizes is None else grid_sizes
        offsets = self.vertex_offsets if vertex_offsets is None else vertex_offsets
        ids = self.cluster_ids if cluster_ids is None else cluster_ids
        sbt = self.sbt_offsets if sbt_offsets is None else sbt_offsets
        if not isinstance(grids, wp.array):
            raise TypeError("grid_sizes must be a Warp array")
        count = grids.size
        if count == 0:
            raise ValueError("At least one cluster grid is required")
        limit = self._templates.capabilities["max_clusters_per_gas"]
        if count > limit:
            raise ValueError("Cluster count exceeds the device's per-GAS limit")
        for name, value, dtype, optional in (
            ("points", points, wp.vec3, False),
            ("grid_sizes", grids, wp.vec2i, False),
            ("vertex_offsets", offsets, wp.int32, False),
            ("cluster_ids", ids, wp.uint32, True),
            ("sbt_offsets", sbt, wp.uint32, True),
        ):
            if not isinstance(value, wp.array) or value.dtype != dtype or value.ndim != 1:
                raise TypeError(f"{name} must be a one-dimensional Warp array of {dtype}")
            if value.device != self.device or not value.is_contiguous:
                raise ValueError(f"{name} must be contiguous and on {self.device}")
            if name != "points" and value.size not in ((0, count) if optional else (count,)):
                raise ValueError(f"{name} must have one entry per cluster")
        templates = self._templates
        with wp.ScopedDevice(self.device), wp.ScopedStream(stream, sync_enter=False):
            if validate:
                host_grids = grids.numpy().astype(np.int64)
                host_offsets = offsets.numpy().astype(np.int64)
                if np.any(host_grids < 1) or np.any(host_grids > templates.size):
                    raise ValueError(f"Grid edge sizes must lie in [1, {templates.size}]")
                ends = host_offsets + (host_grids[:, 0] + 1) * (host_grids[:, 1] + 1)
                if np.any(host_offsets < 0) or np.any(ends > points.size):
                    raise ValueError("Grid vertices extend outside points")
                if ids.size and np.any(ids.numpy() == 0xFFFFFFFF):
                    raise ValueError("0xFFFFFFFF is reserved as the invalid cluster ID")
                if sbt.size and np.any(sbt.numpy() > templates.max_sbt_index):
                    raise ValueError("SBT offsets exceed the templates' max_sbt_index")
            build = self._build
            if build is None or count > build.capacity:
                # Grow geometrically, so that incremental growth reallocates rarely.
                capacity = count if build is None else min(max(count, build.capacity * 3 // 2), limit)
                build = _ClusterBuild(self._context, templates, capacity)
                # Nothing else retains the buffers when no instance structure holds this list.
                if sys.getrefcount(self._memory) <= 2:
                    self._memory = [templates.output]
                self._memory.extend((build.clas.output, build.gas.output))
                self._build = build
            wp.launch(
                _pack_args,
                count,
                inputs=[wp.uint64(points.ptr), grids, offsets, ids, sbt, templates.handles, templates.size, build.args],
            )
            build.count.fill_(count)
            build.clas.build(build.args, build.count)
            wp.launch(_gas_args, 1, inputs=[count, wp.uint64(build.clas.handles.ptr), build.gas_args])
            build.gas.build(build.gas_args)
            self._id = int(build.gas.handles.numpy()[0])
            self.points, self.grid_sizes, self.vertex_offsets = points, grids, offsets
            self.cluster_ids, self.sbt_offsets = ids, sbt

    def close(self):
        if self.closed:
            return
        self._closed = True
        self._memory = []
        self._build = self._templates = self._empty = None
        self.points = self.grid_sizes = self.vertex_offsets = self.cluster_ids = self.sbt_offsets = None

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, *exc):
        self.close()
