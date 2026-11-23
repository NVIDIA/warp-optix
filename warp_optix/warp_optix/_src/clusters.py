# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Owned cluster builds. Descriptor packing and memory management stay internal."""

from __future__ import annotations

import ctypes
import sys
from collections.abc import Sequence
from itertools import accumulate
from operator import index

import numpy as np
import warp as wp

from warp_optix import native as n
from warp_optix._src.context import get_context, resolve
from warp_optix._src.enums import GeometryFlags


@wp.struct
class TemplateArgs:
    cluster_id: wp.uint32
    sbt_offset: wp.uint32
    cluster_template: wp.uint64
    vertices: wp.uint64
    stride: wp.uint32
    reserved: wp.uint32


@wp.struct
class GridArgs:
    cluster_id: wp.uint32
    flags: wp.uint32
    primitive_info: wp.uint32
    truncate: wp.uint32
    dimensions: wp.uint32


@wp.struct
class GasArgs:
    count: wp.uint32
    stride: wp.uint32
    handles: wp.uint64


for _struct_type, _expected_size in ((GridArgs, 20), (TemplateArgs, 32), (GasArgs, 16)):
    # Warp's struct decorator installs this ABI descriptor dynamically.
    assert ctypes.sizeof(_struct_type.ctype) == _expected_size  # pyright: ignore[reportAttributeAccessIssue]


@wp.kernel(enable_backward=False)
def _grid_args(size: int, truncate: int, primitive_flags: int, args: wp.array[GridArgs]):
    i = wp.tid()
    a = GridArgs()
    a.primitive_info = wp.uint32(primitive_flags) << wp.uint32(29)
    a.truncate = wp.uint32(truncate)
    a.dimensions = wp.uint32(i // size + 1) | (wp.uint32(i % size + 1) << wp.uint32(8))
    args[i] = a


@wp.kernel(enable_backward=False)
def _gas_args(count: int, handles: wp.uint64, args: wp.array[GasArgs]):
    a = GasArgs()
    a.count = wp.uint32(count)
    a.handles = handles
    args[0] = a


@wp.kernel(enable_backward=False)
def _batch_gas_args(offsets: wp.array[wp.uint32], handles: wp.uint64, args: wp.array[GasArgs]):
    i = wp.tid()
    start = offsets[i]
    arg = GasArgs()
    arg.count = offsets[i + 1] - start
    arg.handles = handles + wp.uint64(8) * wp.uint64(start)
    args[i] = arg


@wp.kernel(enable_backward=False)
def _pack_args(
    points: wp.uint64,
    grids: wp.array[wp.vec2i],
    offsets: wp.array[int],
    ids: wp.array[wp.uint32],
    sbt_offsets: wp.array[wp.uint32],
    templates: wp.array[wp.uint64],
    size: int,
    args: wp.array[TemplateArgs],
):
    i = wp.tid()
    c = grids[i]
    a = TemplateArgs()
    a.cluster_id = wp.uint32(i)
    if ids.shape[0] != 0:
        a.cluster_id = ids[i]
    if sbt_offsets.shape[0] != 0:
        a.sbt_offset = sbt_offsets[i]
    a.cluster_template = templates[(c[0] - 1) * size + c[1] - 1]
    a.vertices = points + wp.uint64(12) * wp.uint64(offsets[i])
    a.stride = wp.uint32(12)
    args[i] = a


def _build_input(kind, count, size=8, truncate=0, max_sbt_index=0, *, gas_count=1, max_clusters_per_gas=None):
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
        desc.maxArgCount = gas_count
        desc.maxTotalClusterCount = count
        desc.maxClusterCountPerArg = count if max_clusters_per_gas is None else max_clusters_per_gas
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


def _primitive_flags(geometry_flags: GeometryFlags | int) -> int:
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
        raise ValueError("Unsupported geometry flags for cluster templates")
    return result


def _check_cluster_arrays(device, count, points, grids, offsets, ids, sbt):
    for name, value, dtype, optional in (
        ("points", points, wp.vec3, False),
        ("grid_sizes", grids, wp.vec2i, False),
        ("vertex_offsets", offsets, wp.int32, False),
        ("cluster_ids", ids, wp.uint32, True),
        ("sbt_offsets", sbt, wp.uint32, True),
    ):
        if not isinstance(value, wp.array) or value.dtype != dtype or value.ndim != 1:
            raise TypeError(f"{name} must be a one-dimensional Warp array of {dtype}")
        if value.device != device or not value.is_contiguous:
            raise ValueError(f"{name} must be contiguous and on {device}")
        if name != "points" and value.size not in ((0, count) if optional else (count,)):
            raise ValueError(f"{name} must have one entry per cluster")


def _validate_cluster_values(templates, points, grids, offsets, ids, sbt):
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


class ImplicitBuild:
    """An implicit-destination build whose buffers fit ``capacity`` arguments of ``desc``.

    OptiX sizes the output for the worst case, so rebuilds with fewer arguments
    reuse every buffer, and neither sizes nor offsets are read back to the host.
    """

    def __init__(self, context, desc, capacity, *, handles=None):
        self.context = context
        self.desc = desc
        output, scratch = n.clusterAccelComputeMemoryUsage(
            context.native, n.CLUSTER_ACCEL_BUILD_MODE_IMPLICIT_DESTINATIONS, desc
        )
        self.output = wp.empty(output, dtype=wp.uint8)
        self.scratch = wp.empty(scratch, dtype=wp.uint8)
        self.handles = wp.empty(capacity, dtype=wp.uint64) if handles is None else handles

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


class TemplateState:
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
        build = ImplicitBuild(
            context, _build_input(n.CLUSTER_ACCEL_BUILD_TYPE_TEMPLATES_FROM_GRIDS, count, size, 0, max_sbt_index), count
        )
        grid_args = wp.empty(count, dtype=GridArgs)
        wp.launch(_grid_args, count, inputs=[size, truncate, primitive_flags, grid_args])
        build.build(grid_args)
        # Warp frees on the legacy default stream, which may not wait for the build stream.
        context.release(lambda buffers=(grid_args, build.scratch): None, (wp.get_stream(),))
        self.output = build.output
        self.handles = build.handles


class ClusterBuild:
    """CLAS and GAS buffers for up to ``capacity`` clusters."""

    def __init__(self, context, templates, capacity, gas_handles):
        self.capacity = capacity
        self.clas = ImplicitBuild(
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
        self.gas = ImplicitBuild(
            context, _build_input(n.CLUSTER_ACCEL_BUILD_TYPE_GASES_FROM_CLUSTERS, capacity), 1, handles=gas_handles
        )
        self.args = wp.empty(capacity, dtype=TemplateArgs)
        self.count = wp.empty(1, dtype=wp.uint32)
        self.gas_args = wp.empty(1, dtype=GasArgs)


def cluster_capabilities(device: wp.DeviceLike = None) -> dict[str, int]:
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
    bounds per-cluster SBT offsets. ``flags`` are the geometry flags of every
    cluster, as in :func:`triangles`. Templates hold no input arrays, so
    ``device``/``stream`` follow the usual contract, like :class:`Accel`.
    Unsupported hardware raises an error.
    """

    def __init__(
        self,
        max_edge_segments: int = 8,
        *,
        position_truncate_bits: int = 0,
        max_sbt_index: int = 0,
        flags: GeometryFlags = GeometryFlags.DISABLE_ANYHIT,
        device: wp.DeviceLike = None,
        stream: wp.Stream | None = None,
    ):
        self._closed = True
        self._context, stream = resolve(device, stream)
        if not isinstance(max_edge_segments, int) or max_edge_segments < 1:
            raise ValueError("max_edge_segments must be a positive integer")
        if not isinstance(position_truncate_bits, int) or not 0 <= position_truncate_bits <= 23:
            raise ValueError("position_truncate_bits must lie in [0, 23]")
        if not isinstance(max_sbt_index, int) or not 0 <= max_sbt_index <= 0xFFFFFF:
            raise ValueError("max_sbt_index must fit in 24 bits")
        primitive_flags = _primitive_flags(flags)
        self.max_edge_segments = max_edge_segments
        self.position_truncate_bits = position_truncate_bits
        self.max_sbt_index = max_sbt_index
        with wp.ScopedDevice(self.device), wp.ScopedStream(stream, sync_enter=False):
            self._state = TemplateState(max_edge_segments, position_truncate_bits, max_sbt_index, primitive_flags)
            self._ready = stream.record_event()
        self.capabilities = self._state.capabilities
        self._closed = False

    @property
    def device(self) -> wp.Device:
        return self._context.device

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        """Release this owner's references. Existing structures retain their templates."""
        self._closed = True
        self._state = None
        self._ready = None

    def __enter__(self) -> ClusterTemplates:
        if self.closed:
            raise RuntimeError("ClusterTemplates is closed")
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class ClusterAccel:
    """An owned GAS over dynamic grid clusters, built from Warp device arrays.

    ``points`` is a contiguous CUDA ``vec3`` array. ``grid_sizes`` contains one
    ``vec2i(width, height)`` per cluster, measured in quads. ``vertex_offsets``
    contains the first vertex of each grid in ``points``. Vertices within each
    grid are row-major with width varying fastest. The default cluster ID is
    its array index; optional ``cluster_ids`` and ``sbt_offsets`` are uint32 arrays.
    Nonzero SBT offsets require shared templates with a sufficient ``max_sbt_index``.
    ``max_edge_segments``, ``position_truncate_bits`` and ``flags`` configure
    the templates created when ``templates`` is omitted; combining them with
    ``templates`` is an error.

    Construction builds the structure. Mutate the arrays and call ``rebuild()``;
    pass replacement arrays to ``rebuild`` when grid counts change. Pass this
    object directly to ``wo.launch`` or use ``accel.id``, like ``wo.Accel``.
    Pipelines enable ``clustered_geometry=True``. All geometry, templates and
    acceleration buffers are retained automatically. This API always uses the
    Cluster API, and never falls back to ordinary triangle geometry.

    Unlike ``wo.Accel``, ``id`` can change on ``rebuild()``. Buffers are sized
    for the worst case of :attr:`capacity` clusters and are reused by rebuilds
    with at most that many clusters. A rebuild with more clusters grows the
    capacity and allocates new buffers. Any rebuild can change ``id`` and
    returns ``True`` when it does: rewrite instance records that reference it,
    rebuild the instance structure, and set it again on recorded launches
    (``record_cmd=True``).
    Pass ``capacity`` to reserve room for the largest expected cluster count up
    front to avoid reallocations. Instance structures and recorded launches
    keep every buffer they may reference alive, also after this object is closed.

    ``validate=True`` reads the grids back to check bounds on the host, which
    synchronizes the stream. ``validate=False`` skips these checks for trusted
    GPU-generated grids; an out-of-range grid size or vertex offset then faults
    on the GPU. Use :meth:`rebuild` when the CPU needs the new handle or a
    change flag, and :meth:`rebuild_async` with :attr:`device_handle` for
    per-frame rendering that can consume the handle on the GPU. Order each
    rebuild after prior traversals and before the next traversals, using the
    same stream or explicit dependencies across streams.
    Cluster structures support rebuilds, not refits or compaction.
    """

    def __init__(
        self,
        points: wp.array[wp.vec3],
        grid_sizes: wp.array[wp.vec2i],
        vertex_offsets: wp.array[wp.int32],
        *,
        templates: ClusterTemplates | None = None,
        max_edge_segments: int | None = None,
        position_truncate_bits: int | None = None,
        flags: GeometryFlags | None = None,
        cluster_ids: wp.array[wp.uint32] | None = None,
        sbt_offsets: wp.array[wp.uint32] | None = None,
        capacity: int | None = None,
        validate: bool = True,
        stream: wp.Stream | None = None,
    ):
        self._closed = True
        if not isinstance(points, wp.array):
            raise TypeError("points must be a Warp array")
        if templates is not None and (
            max_edge_segments is not None or position_truncate_bits is not None or flags is not None
        ):
            raise ValueError(
                "max_edge_segments, position_truncate_bits and flags configure new templates; "
                "set them on the ClusterTemplates instead"
            )
        if capacity is not None and (not isinstance(capacity, int) or capacity < 1):
            raise ValueError("capacity must be a positive integer")
        self._context, stream = resolve(points.device, stream)
        with wp.ScopedDevice(self.device), wp.ScopedStream(stream, sync_enter=False):
            if templates is None:
                templates = ClusterTemplates(
                    8 if max_edge_segments is None else max_edge_segments,
                    position_truncate_bits=0 if position_truncate_bits is None else position_truncate_bits,
                    flags=GeometryFlags.DISABLE_ANYHIT if flags is None else flags,
                    device=self.device,
                    stream=stream,
                )
            if not isinstance(templates, ClusterTemplates) or templates.closed:
                raise ValueError("templates must be an open ClusterTemplates object")
            if templates.device != self.device:
                raise ValueError("Templates and points must use the same CUDA device")
            stream.wait_event(templates._ready)
            self._templates = templates._state
            if capacity is not None and capacity > self._templates.capabilities["max_clusters_per_gas"]:
                raise ValueError("capacity exceeds the device's per-GAS cluster limit")
            self._build = None
            self._reserve = capacity
            # Shared with instance structures, which retain it rather than a snapshot. It only
            # grows when buffers are reallocated, so referenced output buffers stay alive.
            self._memory = [self._templates.output]
            self._empty = wp.empty(0, dtype=wp.uint32)
            self._device_handle = wp.empty(1, dtype=wp.uint64)
            self._host_handle = wp.empty(1, dtype=wp.uint64, device="cpu", pinned=True)
            self._handle_stream: wp.Stream | None = None
            self._id: int | None = None
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
    def device(self) -> wp.Device:
        return self._context.device

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def id(self) -> int:
        """Latest traversable handle as a Python integer; waits after an async rebuild."""
        self._check_open()
        if self._handle_stream is not None:
            wp.copy(self._host_handle, self._device_handle, stream=self._handle_stream)
            wp.synchronize_stream(self._handle_stream)
            self._id = int(self._host_handle.numpy()[0])
            self._handle_stream = None
        assert self._id is not None
        return self._id

    @property
    def device_handle(self) -> wp.array[wp.uint64]:
        """One-element CUDA array holding the latest handle. Treat it as read-only.

        The array stays the same across rebuilds, including capacity growth.
        Read element zero in GPU code and keep this owner open while submitting
        work that uses it. The array alone does not own the geometry.
        """
        self._check_open()
        return self._device_handle

    @property
    def capacity(self) -> int:
        """Number of clusters the current buffers hold without reallocating."""
        self._check_open()
        return self._build.capacity

    def _check_open(self) -> None:
        if self.closed:
            raise RuntimeError("ClusterAccel is closed")

    def rebuild(
        self,
        *,
        points: wp.array[wp.vec3] | None = None,
        grid_sizes: wp.array[wp.vec2i] | None = None,
        vertex_offsets: wp.array[wp.int32] | None = None,
        cluster_ids: wp.array[wp.uint32] | None = None,
        sbt_offsets: wp.array[wp.uint32] | None = None,
        validate: bool = True,
        stream: wp.Stream | None = None,
    ) -> bool:
        """Rebuild current grids, optionally replacing arrays when topology changes.

        Synchronizes the stream to read :attr:`id`, even with ``validate=False``.
        Returns ``True`` when :attr:`id` changed; update instance records and
        recorded launch parameters that reference the previous handle.
        """
        previous = self.id if self._build is not None else None
        self.rebuild_async(
            points=points,
            grid_sizes=grid_sizes,
            vertex_offsets=vertex_offsets,
            cluster_ids=cluster_ids,
            sbt_offsets=sbt_offsets,
            validate=validate,
            stream=stream,
        )
        handle = self.id
        return previous is not None and handle != previous

    def rebuild_async(
        self,
        *,
        points: wp.array[wp.vec3] | None = None,
        grid_sizes: wp.array[wp.vec2i] | None = None,
        vertex_offsets: wp.array[wp.int32] | None = None,
        cluster_ids: wp.array[wp.uint32] | None = None,
        sbt_offsets: wp.array[wp.uint32] | None = None,
        validate: bool = True,
        stream: wp.Stream | None = None,
    ) -> None:
        """Enqueue a rebuild with the same input options as :meth:`rebuild`.

        Read :attr:`device_handle` in subsequent GPU work on the same stream
        or use a stream dependency. There is no handle readback or change flag.
        ``validate=True`` still waits for input checks; use ``validate=False``
        for trusted GPU inputs. Reserve capacity to avoid allocations in the
        frame loop. Reading :attr:`id` later waits for the latest handle. If
        you switch back to host handles, refresh any cached handles before use.
        """
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
        _check_cluster_arrays(self.device, count, points, grids, offsets, ids, sbt)
        templates = self._templates
        with wp.ScopedDevice(self.device), wp.ScopedStream(stream, sync_enter=False):
            if validate:
                _validate_cluster_values(templates, points, grids, offsets, ids, sbt)
            build = self._build
            reallocated = build is None or count > build.capacity
            if reallocated:
                if build is None:
                    capacity = max(count, self._reserve or 0)
                else:
                    # Grow geometrically, so that incremental growth reallocates rarely.
                    capacity = min(max(count, build.capacity * 3 // 2), limit)
                build = ClusterBuild(self._context, templates, capacity, self._device_handle)
            wp.launch(
                _pack_args,
                count,
                inputs=[wp.uint64(points.ptr), grids, offsets, ids, sbt, templates.handles, templates.size, build.args],
            )
            build.count.fill_(count)
            build.clas.build(build.args, build.count)
            wp.launch(_gas_args, 1, inputs=[count, wp.uint64(build.clas.handles.ptr), build.gas_args])
            build.gas.build(build.gas_args)
            if reallocated:
                # Nothing else retains the buffers when no instance structure holds this list.
                if sys.getrefcount(self._memory) <= 2:
                    self._memory = [templates.output]
                self._memory.extend((build.clas.output, build.gas.output))
                self._build = build
            self._handle_stream = stream
            self.points, self.grid_sizes, self.vertex_offsets = points, grids, offsets
            self.cluster_ids, self.sbt_offsets = ids, sbt

    def close(self) -> None:
        if self.closed:
            return
        self._closed = True
        self._memory = []
        self._device_handle = self._host_handle = self._handle_stream = None
        self._build = self._templates = self._empty = None
        self.points = self.grid_sizes = self.vertex_offsets = self.cluster_ids = self.sbt_offsets = None

    def __enter__(self) -> ClusterAccel:
        self._check_open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class ClusterAccelBatch:
    """Owned multi-GAS build over packed Warp arrays, with one GAS per object.

    ``cluster_counts`` contains the positive number of consecutive clusters in
    each GAS. The counts and array shapes stay fixed for the lifetime of the
    batch. Grid sizes, vertex offsets, IDs and SBT offsets use the same layout
    as :class:`ClusterAccel`; default cluster IDs are indices in the packed arrays.
    Mutate the arrays in place, then rebuild. Share :class:`ClusterTemplates`
    to set grid limits or material options across batches.

    Construction and :meth:`rebuild_async` enqueue a CLAS build and a multi-GAS
    build. With trusted inputs and ``validate=False``, they skip host readback.
    Use :attr:`device_handles` on the GPU or :attr:`ids` for a synchronized host
    read. Order input writes, builds and traversal on one stream or with explicit
    dependencies. Use separate batches for objects with different update rates.
    """

    def __init__(
        self,
        points: wp.array[wp.vec3],
        grid_sizes: wp.array[wp.vec2i],
        vertex_offsets: wp.array[wp.int32],
        cluster_counts: Sequence[int],
        *,
        templates: ClusterTemplates | None = None,
        cluster_ids: wp.array[wp.uint32] | None = None,
        sbt_offsets: wp.array[wp.uint32] | None = None,
        validate: bool = True,
        stream: wp.Stream | None = None,
    ):
        self._closed = True
        if not isinstance(points, wp.array):
            raise TypeError("points must be a Warp array")
        counts = tuple(map(index, cluster_counts))
        if not counts or min(counts) <= 0:
            raise ValueError("cluster_counts must contain a positive count for each GAS")
        count = sum(counts)
        self._context, stream = resolve(points.device, stream)
        with wp.ScopedDevice(self.device), wp.ScopedStream(stream, sync_enter=False):
            empty = wp.empty(0, dtype=wp.uint32)
            ids = empty if cluster_ids is None else cluster_ids
            sbt = empty if sbt_offsets is None else sbt_offsets
            _check_cluster_arrays(self.device, count, points, grid_sizes, vertex_offsets, ids, sbt)
            if templates is None:
                templates = ClusterTemplates(device=self.device, stream=stream)
            if not isinstance(templates, ClusterTemplates) or templates.closed:
                raise ValueError("templates must be an open ClusterTemplates object")
            if templates.device != self.device:
                raise ValueError("Templates and points must use the same CUDA device")
            state = templates._state
            if max(counts) > state.capabilities["max_clusters_per_gas"]:
                raise ValueError("Cluster count exceeds the device's per-GAS limit")
            if count * max(2 * state.size**2, (state.size + 1) ** 2) > 0xFFFFFFFF:
                raise ValueError("Batch exceeds the OptiX aggregate build limits; split it into smaller batches")
            if validate:
                _validate_cluster_values(state, points, grid_sizes, vertex_offsets, ids, sbt)
            stream.wait_event(templates._ready)
            self._cluster_counts = counts
            self._inputs: tuple[wp.array, ...] = (points, grid_sizes, vertex_offsets, ids, sbt)
            self._templates = state
            self._args = wp.empty(count, dtype=TemplateArgs)
            self._gas_args = wp.empty(len(counts), dtype=GasArgs)
            self._clas = ImplicitBuild(
                self._context,
                _build_input(
                    n.CLUSTER_ACCEL_BUILD_TYPE_CLUSTERS_FROM_TEMPLATES,
                    count,
                    state.size,
                    state.truncate,
                    state.max_sbt_index,
                ),
                count,
            )
            self._gas = ImplicitBuild(
                self._context,
                _build_input(
                    n.CLUSTER_ACCEL_BUILD_TYPE_GASES_FROM_CLUSTERS,
                    count,
                    gas_count=len(counts),
                    max_clusters_per_gas=max(counts),
                ),
                len(counts),
            )
            offsets = wp.array(list(accumulate(counts, initial=0)), dtype=wp.uint32)
            wp.launch(_batch_gas_args, len(counts), inputs=[offsets, wp.uint64(self._clas.handles.ptr), self._gas_args])
            self._context.release(lambda buffer=offsets: None, (stream,))
            self._memory = (state.output, self._clas.output, self._gas.output)
            self._host_handles = None
            self._ids = ()
            self._handle_stream = None
            self._closed = False
            try:
                self.rebuild_async(validate=False, stream=stream)
            except BaseException:
                self.close()
                raise

    @property
    def device(self) -> wp.Device:
        return self._context.device

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def cluster_counts(self) -> tuple[int, ...]:
        """Number of clusters in each GAS, in build order."""
        return self._cluster_counts

    @property
    def device_handles(self) -> wp.array[wp.uint64]:
        """Read-only device array of GAS handles, in ``cluster_counts`` order.

        The array persists across rebuilds. Keep the batch alive while submitting
        work with these handles; the array alone does not own the geometry.
        """
        self._check_open()
        return self._gas.handles

    @property
    def ids(self) -> tuple[int, ...]:
        """GAS handles in build order, waiting for a pending device readback."""
        self._check_open()
        if self._handle_stream is not None:
            if self._host_handles is None:
                self._host_handles = wp.empty(len(self.cluster_counts), dtype=wp.uint64, device="cpu", pinned=True)
            wp.copy(self._host_handles, self.device_handles, stream=self._handle_stream)
            wp.synchronize_stream(self._handle_stream)
            self._ids = tuple(map(int, self._host_handles.numpy()))
            self._handle_stream = None
        return self._ids

    def _check_open(self) -> None:
        if self.closed:
            raise RuntimeError("ClusterAccelBatch is closed")

    def rebuild(self, *, validate: bool = True, stream: wp.Stream | None = None) -> bool:
        """Rebuild and wait for the handles. Return whether any handle changed."""
        previous = self.ids
        self.rebuild_async(validate=validate, stream=stream)
        return self.ids != previous

    def rebuild_async(self, *, validate: bool = True, stream: wp.Stream | None = None) -> None:
        """Rebuild all GASes from the current arrays, reusing device buffers.

        ``validate=True`` reads descriptors back for bounds checks. With trusted
        inputs, ``validate=False`` skips that wait. Rewrite instance records from
        :attr:`device_handles` and rebuild the IAS before tracing it again.
        """
        self._check_open()
        _, stream = resolve(self.device, stream)
        points, grids, offsets, ids, sbt = self._inputs
        state = self._templates
        with wp.ScopedDevice(self.device), wp.ScopedStream(stream, sync_enter=False):
            if validate:
                _validate_cluster_values(state, points, grids, offsets, ids, sbt)
            wp.launch(
                _pack_args,
                self._args.size,
                inputs=[wp.uint64(points.ptr), grids, offsets, ids, sbt, state.handles, state.size, self._args],
            )
            self._clas.build(self._args)
            self._gas.build(self._gas_args)
            self._handle_stream = stream

    def write_instance_handles(self, records: wp.array, *, stream: wp.Stream | None = None) -> None:
        """Write one handle per ``Instance`` record in batch order, preserving other fields."""
        from warp_optix._src.accel import Instance, _write_instance_handles

        handles = self.device_handles
        if (
            not isinstance(records, wp.array)
            or records.dtype is not Instance
            or records.ndim != 1
            or not records.is_contiguous
            or records.device != self.device
            or records.size != handles.size
        ):
            raise ValueError("Expected one contiguous Instance record per GAS on the batch device")
        _, stream = resolve(self.device, stream)
        wp.launch(_write_instance_handles, records.size, inputs=[handles, records], stream=stream)

    def close(self) -> None:
        """Release the batch. Existing instance structures retain its geometry buffers."""
        if self.closed:
            return
        self._closed = True
        self._memory = self._inputs = ()
        self._clas = self._gas = self._args = self._gas_args = self._templates = None
        self._handle_stream = self._host_handles = None

    def __enter__(self) -> ClusterAccelBatch:
        self._check_open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()


@wp.struct
class TriangleArgs:
    """``OptixClusterAccelBuildInputTrianglesArgs``, with its bitfields packed into words."""

    cluster_id: wp.uint32
    cluster_flags: wp.uint32
    counts: wp.uint32  # triangleCount:9, vertexCount:9, positionTruncateBitCount:6, indexFormat:4, ommIndexFormat:4
    primitive_info: wp.uint32  # sbtIndex:24, reserved:5, primitiveFlags:3
    strides: wp.vec2ui  # 0: natural strides
    indices: wp.uint64
    vertices: wp.uint64
    primitive_info_buffer: wp.uint64
    opacity_micromap_array: wp.uint64
    opacity_micromap_indices: wp.uint64
    bounding_box_limit: wp.uint64


assert ctypes.sizeof(TriangleArgs.ctype) == 72  # pyright: ignore[reportAttributeAccessIssue]


@wp.kernel(enable_backward=False)
def _triangle_args(
    points: wp.uint64,
    indices: wp.uint64,
    clusters: wp.array[wp.vec4i],
    ids: wp.array[wp.uint32],
    primitive_flags: int,
    args: wp.array[TriangleArgs],
):
    i = wp.tid()
    c = clusters[i]
    a = TriangleArgs()
    a.cluster_id = wp.uint32(i)
    if ids.shape[0] != 0:
        a.cluster_id = ids[i]
    a.counts = wp.uint32(c[3]) | (wp.uint32(c[1]) << wp.uint32(9)) | (wp.uint32(4) << wp.uint32(24))
    a.primitive_info = wp.uint32(primitive_flags) << wp.uint32(29)
    a.indices = indices + wp.uint64(12) * wp.uint64(c[2])
    a.vertices = points + wp.uint64(12) * wp.uint64(c[0])
    args[i] = a


class TriangleClusterAccel:
    """An owned GAS over clusters of arbitrary triangles, rebuilt from Warp device arrays.

    ``points`` is a contiguous one-dimensional ``vec3`` array. ``clusters`` is a contiguous
    one-dimensional array holding one ``vec4i(first_vertex, vertex_count, first_triangle,
    triangle_count)`` per cluster. ``indices`` holds packed int32 or uint32 vertex triplets
    local to their cluster:
    triangle ``t`` of cluster ``c`` uses ``points[c[0] + indices[3 * (c[2] + t) + k]]``. Each
    cluster holds at most :func:`cluster_capabilities` ``max_vertices`` and ``max_triangles``.

    Every :meth:`rebuild` builds all clusters and their GAS from scratch in two batched OptiX
    calls, which suits deforming meshes: it is typically faster than refitting an equivalent
    triangle GAS and keeps full trace quality. Mutate ``points`` in place between rebuilds; the
    topology stays fixed. The GAS is built into a fixed destination, so :attr:`id` never changes
    and instance structures over it can be refit after a rebuild. Rebuilds reuse device buffers
    and run asynchronously. Order point writes, rebuilds, instance refits and traversal on one
    stream or with explicit stream dependencies.

    Hits report ``cluster_id`` (``cluster_ids[c]``, or ``c`` by default) and the triangle within
    the cluster as ``primitive_index``. ``flags`` are the geometry flags of every triangle, as in
    :func:`triangles`; ``build_flags`` is ``BuildFlags.PREFER_FAST_BUILD`` or ``PREFER_FAST_TRACE``.
    ``validate=True`` checks the topology on the host once. Pipelines tracing the structure need
    ``clustered_geometry=True``.
    """

    def __init__(
        self,
        points: wp.array[wp.vec3],
        indices: wp.array,
        clusters: wp.array[wp.vec4i],
        *,
        cluster_ids: wp.array[wp.uint32] | None = None,
        flags: GeometryFlags = GeometryFlags.DISABLE_ANYHIT,
        build_flags: int | None = None,
        validate: bool = True,
        stream: wp.Stream | None = None,
    ):
        from warp_optix._src.enums import BuildFlags

        self._closed = True
        for name, array, dtypes in (
            ("points", points, (wp.vec3,)),
            ("indices", indices, (wp.int32, wp.uint32)),
            ("clusters", clusters, (wp.vec4i,)),
            ("cluster_ids", cluster_ids, (wp.uint32,)),
        ):
            if name == "cluster_ids" and array is None:
                continue
            if not isinstance(array, wp.array) or array.dtype not in dtypes:
                raise TypeError(f"{name} must be a Warp array with dtype {' or '.join(str(dtype) for dtype in dtypes)}")
            if name != "indices" and array.ndim != 1:
                raise ValueError(f"{name} must be one-dimensional")
            if not array.is_contiguous:
                raise ValueError(f"{name} must be contiguous")
        if indices.size % 3:
            raise ValueError("indices must contain complete vertex triplets")
        self._context, stream = resolve(points.device, stream)
        build_flags = BuildFlags.PREFER_FAST_BUILD if build_flags is None else build_flags
        if build_flags not in (BuildFlags.PREFER_FAST_BUILD, BuildFlags.PREFER_FAST_TRACE):
            raise ValueError("build_flags must be BuildFlags.PREFER_FAST_BUILD or PREFER_FAST_TRACE")
        cluster_flags = (
            n.CLUSTER_ACCEL_BUILD_FLAG_PREFER_FAST_BUILD
            if build_flags == BuildFlags.PREFER_FAST_BUILD
            else n.CLUSTER_ACCEL_BUILD_FLAG_PREFER_FAST_TRACE
        )
        ids = wp.empty(0, dtype=wp.uint32, device=self.device) if cluster_ids is None else cluster_ids
        if ids.size not in (0, clusters.size):
            raise ValueError("cluster_ids must have one entry per cluster")
        self._context.check_arrays((points, indices, clusters, ids))
        count = clusters.size
        props = n.getClusterProperties(self._context.native)
        if not props["flags"]:
            raise RuntimeError("The OptiX driver reports no Cluster API support on this device")
        # Size the build from the topology. Wait for writes on the selected stream
        # before numpy() reads the arrays back on the CUDA default stream.
        wp.synchronize_stream(stream)
        first_vertex, vertex_count, first_triangle, triangle_count = clusters.numpy().astype(np.int64).T
        if not 0 < count <= props["max_clusters_per_gas"] or min(vertex_count.min(), triangle_count.min()) < 1:
            raise ValueError("Expected at least one cluster, each with vertices and triangles")
        if vertex_count.max() > props["max_vertices"] or triangle_count.max() > props["max_triangles"]:
            raise ValueError(
                f"A cluster exceeds the device limits of {props['max_vertices']} vertices and "
                f"{props['max_triangles']} triangles"
            )
        if validate:
            if (
                first_vertex.min() < 0
                or (first_vertex + vertex_count).max() > points.size
                or first_triangle.min() < 0
                or (first_triangle + triangle_count).max() > indices.size // 3
            ):
                raise ValueError("Cluster vertices or triangles lie outside points or indices")
            if ids.size and np.any(ids.numpy() == 0xFFFFFFFF):
                raise ValueError("0xFFFFFFFF is reserved as the invalid cluster ID")
            local = indices.numpy().reshape(-1, 3)
            # Bound the temporary gathers independently of the mesh size.
            for start in range(0, count, 4096):
                batch = slice(start, start + 4096)
                counts = triangle_count[batch]
                starts = first_triangle[batch] - (np.cumsum(counts) - counts)
                rows = np.arange(counts.sum()) + np.repeat(starts, counts)
                vertices = local[rows]
                limits = np.repeat(vertex_count[batch].astype(vertices.dtype), counts)
                if vertices.min() < 0 or np.any(vertices >= limits[:, None]):
                    raise ValueError("Cluster-local indices lie outside their cluster's vertices")

        with wp.ScopedDevice(self.device), wp.ScopedStream(stream, sync_enter=False):
            clas = n.ClusterAccelBuildInputTriangles()
            clas.flags = cluster_flags
            clas.maxArgCount = count
            clas.vertexFormat = n.VERTEX_FORMAT_FLOAT3
            clas.maxUniqueSbtIndexCountPerArg = 1
            clas.maxTriangleCountPerArg = int(triangle_count.max())
            clas.maxVertexCountPerArg = int(vertex_count.max())
            clas.maxTotalTriangleCount = int(triangle_count.sum())
            clas.maxTotalVertexCount = int(vertex_count.sum())
            clas_input = n.ClusterAccelBuildInput()
            clas_input.type = n.CLUSTER_ACCEL_BUILD_TYPE_CLUSTERS_FROM_TRIANGLES
            clas_input.triangles = clas
            self._clas = ImplicitBuild(self._context, clas_input, count)
            self._args = wp.empty(count, dtype=TriangleArgs)
            wp.launch(
                _triangle_args,
                count,
                inputs=[wp.uint64(points.ptr), wp.uint64(indices.ptr), clusters, ids, _primitive_flags(flags)],
                outputs=[self._args],
            )
            gas = n.ClusterAccelBuildInputClusters()
            gas.flags = cluster_flags
            gas.maxArgCount = 1
            gas.maxTotalClusterCount = count
            gas.maxClusterCountPerArg = count
            self._gas_input = n.ClusterAccelBuildInput()
            self._gas_input.type = n.CLUSTER_ACCEL_BUILD_TYPE_GASES_FROM_CLUSTERS
            self._gas_input.clusters = gas
            # An explicit destination keeps the GAS, and so its handle, at one address.
            output, scratch = n.clusterAccelComputeMemoryUsage(
                self._context.native, n.CLUSTER_ACCEL_BUILD_MODE_EXPLICIT_DESTINATIONS, self._gas_input
            )
            self._gas_output = wp.empty(output, dtype=wp.uint8)
            self._gas_scratch = wp.empty(scratch, dtype=wp.uint8)
            self._gas_destination = wp.array([self._gas_output.ptr], dtype=wp.uint64)
            self._gas_handle = wp.empty(1, dtype=wp.uint64)
            self._gas_args = wp.empty(1, dtype=GasArgs)
            wp.launch(_gas_args, 1, inputs=[count, wp.uint64(self._clas.handles.ptr), self._gas_args])
            # Instance structures retain this tuple, so the geometry outlives this owner.
            self._memory = (points, indices, self._clas.output, self._gas_output)
            self._closed = False
            self.rebuild(stream)
            host_handle = wp.empty(1, dtype=wp.uint64, device="cpu", pinned=True)
            wp.copy(host_handle, self._gas_handle, stream=stream)
            wp.synchronize_stream(stream)
            self._id = int(host_handle.numpy()[0])

    @property
    def device(self) -> wp.Device:
        return self._context.device

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def id(self) -> int:
        """Traversable handle of the GAS; it stays the same across rebuilds."""
        self._check_open()
        return self._id

    def _check_open(self) -> None:
        if self.closed:
            raise RuntimeError("TriangleClusterAccel is closed")

    def rebuild(self, stream: wp.Stream | None = None) -> None:
        """Rebuild every cluster and the GAS from the current ``points``."""
        self._check_open()
        _, stream = resolve(self.device, stream)
        with wp.ScopedDevice(self.device), wp.ScopedStream(stream, sync_enter=False):
            self._clas.build(self._args)
            dest = n.ClusterAccelBuildModeDescExplicitDest()
            dest.tempBuffer = self._gas_scratch.ptr
            dest.tempBufferSizeInBytes = self._gas_scratch.size
            dest.destAddressesBuffer = self._gas_destination.ptr
            dest.outputHandlesBuffer = self._gas_handle.ptr
            mode = n.ClusterAccelBuildModeDesc()
            mode.mode = n.CLUSTER_ACCEL_BUILD_MODE_EXPLICIT_DESTINATIONS
            mode.explicitDest = dest
            n.clusterAccelBuild(self._context.native, stream.cuda_stream, mode, self._gas_input, self._gas_args.ptr)

    def close(self) -> None:
        """Release this owner. Instance structures over it retain the geometry."""
        self._closed = True
        self._memory = ()
        self._clas = self._args = self._gas_args = self._gas_output = self._gas_scratch = None
        self._gas_destination = self._gas_handle = self._gas_input = None

    def __enter__(self) -> TriangleClusterAccel:
        self._check_open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()
