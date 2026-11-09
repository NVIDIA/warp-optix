# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OptiX acceleration structures over Warp device arrays.

:class:`Accel` mirrors ``wp.Mesh``/``wp.Bvh``: construct it from build inputs,
pass ``accel.id`` to kernels, and update it with :meth:`Accel.refit` or
:meth:`Accel.rebuild`. Build inputs retain their device arrays. Instance
structures keep the children listed in :func:`instances` alive and never read
their GPU-written records back (unless ``config.verify_instances``). Only
compaction synchronizes (once per :func:`compact` batch). See
:mod:`warp_optix.context` for the lifetime contract.
"""

from __future__ import annotations

import ctypes
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import warp as wp
from warp._src import context as _warp_context

from warp_optix import config, native
from warp_optix.context import CAPTURE_ERROR, resolve
from warp_optix.enums import BuildFlags, GeometryFlags, PrimitiveType

# ---------------------------------------------------------------------------
# Instances
# ---------------------------------------------------------------------------

_mat34 = wp.types.matrix(shape=(3, 4), dtype=wp.float32)


@wp.struct
class Instance:
    """Device layout of ``OptixInstance`` (80 bytes); build records with :func:`make_instance`."""

    transform: _mat34
    instance_id: wp.uint32
    sbt_offset: wp.uint32
    visibility_mask: wp.uint32
    flags: wp.uint32
    traversable: wp.uint64
    _pad: wp.vec2ui


assert ctypes.sizeof(Instance.ctype) == 80, "Instance must match OptixInstance"


@wp.func
def _instance(
    transform: _mat34,
    accel_id: wp.uint64,
    instance_id: wp.Int,
    sbt_offset: wp.Int,
    mask: wp.Int,
    flags: wp.Int,
) -> Instance:
    instance = Instance()
    instance.transform = transform
    instance.instance_id = wp.uint32(instance_id)
    instance.sbt_offset = wp.uint32(sbt_offset)
    instance.visibility_mask = wp.uint32(mask)
    instance.flags = wp.uint32(flags)
    instance.traversable = accel_id
    return instance


@wp.func
def make_instance(
    xform: wp.transform,
    accel_id: wp.uint64,
    instance_id: wp.Int = 0,
    sbt_offset: wp.Int = 0,
    mask: wp.Int = 0xFF,
    flags: wp.Int = 0,
) -> Instance:
    """Return an instance record placing ``accel_id`` at the rigid pose ``xform``.

    Overloads take a full affine ``wp.mat44`` (scale and shear allowed) or a
    ``rotation: wp.mat33`` plus ``translation: wp.vec3``. ``flags`` takes
    :class:`InstanceFlags`; ``mask`` is the 8-bit visibility mask. In kernels with
    backward passes enabled, pass the integer arguments positionally (Warp 1.18
    miscompiles keyword arguments to generic integer parameters in adjoint code).
    """
    r = wp.quat_to_matrix(wp.transform_get_rotation(xform))
    p = wp.transform_get_translation(xform)
    m = _mat34(r[0, 0], r[0, 1], r[0, 2], p[0], r[1, 0], r[1, 1], r[1, 2], p[1], r[2, 0], r[2, 1], r[2, 2], p[2])
    return _instance(m, accel_id, instance_id, sbt_offset, mask, flags)


@wp.func
def make_instance(
    xform: wp.mat44,
    accel_id: wp.uint64,
    instance_id: wp.Int = 0,
    sbt_offset: wp.Int = 0,
    mask: wp.Int = 0xFF,
    flags: wp.Int = 0,
) -> Instance:
    """Return an instance record placing ``accel_id`` at the affine object-to-world matrix ``xform``."""
    m = _mat34(
        xform[0, 0],
        xform[0, 1],
        xform[0, 2],
        xform[0, 3],
        xform[1, 0],
        xform[1, 1],
        xform[1, 2],
        xform[1, 3],
        xform[2, 0],
        xform[2, 1],
        xform[2, 2],
        xform[2, 3],
    )
    return _instance(m, accel_id, instance_id, sbt_offset, mask, flags)


@wp.func
def make_instance(
    rotation: wp.mat33,
    translation: wp.vec3,
    accel_id: wp.uint64,
    instance_id: wp.Int = 0,
    sbt_offset: wp.Int = 0,
    mask: wp.Int = 0xFF,
    flags: wp.Int = 0,
) -> Instance:
    """Return an instance record with linear part ``rotation`` (scale/shear allowed) and ``translation``."""
    r, p = rotation, translation
    m = _mat34(r[0, 0], r[0, 1], r[0, 2], p[0], r[1, 0], r[1, 1], r[1, 2], p[1], r[2, 0], r[2, 1], r[2, 2], p[2])
    return _instance(m, accel_id, instance_id, sbt_offset, mask, flags)


# ---------------------------------------------------------------------------
# Build inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BuildInput:
    """A native descriptor together with the device arrays it borrows.

    ``prepare(stream)`` (if set) runs before every build and refit with the
    build stream, e.g. to pack separate bounds arrays into the layout OptiX reads.
    """

    native: object
    arrays: tuple
    instance_records: wp.array | None = None
    instance_children: tuple[Accel, ...] | None = None
    prepare: Callable[[wp.Stream], None] | None = None

    @property
    def device(self) -> wp.Device:
        return self.arrays[0].device


def _vertex_count(keys):
    if not keys:
        raise ValueError("points must contain at least one motion key")
    count = None
    for key in keys:
        if not key.device.is_cuda or not key.is_contiguous:
            raise ValueError("points must be contiguous CUDA arrays")
        if key.dtype == wp.vec3:
            current = key.size
        elif key.dtype == wp.float32 and key.size % 3 == 0:
            current = key.size // 3
        else:
            raise TypeError("points must contain float32 triplets or vec3 values")
        if count is not None and current != count:
            raise ValueError("point motion keys must have equal sizes")
        if key.device != keys[0].device:
            raise ValueError("point motion keys must use the same device")
        count = current
    return count


def _validate_indices(indices, limit, device, *, triplets=False, bounds=True):
    if indices.dtype not in (wp.int32, wp.uint32):
        raise TypeError("indices must contain int32 or uint32 values")
    if indices.device != device or not indices.is_contiguous:
        raise ValueError("indices must be contiguous and on the points device")
    if triplets and indices.size % 3:
        raise ValueError("triangle indices must contain complete triplets")
    if not bounds:
        return
    if wp.get_stream(device).is_capturing:
        raise RuntimeError(CAPTURE_ERROR)
    # Validate once when creating the input. Point-only refits do not repeat this host readback.
    host = indices.numpy()
    if host.size and (host.min() < 0 or host.max() >= limit):
        raise ValueError(f"indices must lie in [0, {limit})")


def triangles(
    points: wp.array | Sequence[wp.array],
    indices: wp.array,
    *,
    flags: GeometryFlags = GeometryFlags.DISABLE_ANYHIT,
    validate: bool = True,
) -> BuildInput:
    """Triangle build input from ``vec3``/``float32[N, 3]`` points and int32/uint32 index triplets.

    A sequence of point arrays gives one motion key per array. Creating the
    input reads indices back once to validate bounds; pass ``validate=False``
    to skip that synchronization for indices already known to be in range. Keep
    indices unchanged while reusing the input for builds and refits.
    """
    keys = [points] if isinstance(points, wp.array) else list(points)
    num_vertices = _vertex_count(keys)
    _validate_indices(indices, num_vertices, keys[0].device, triplets=True, bounds=validate)
    build_input = native.BuildInputTriangleArray()
    build_input.vertexFormat = native.VERTEX_FORMAT_FLOAT3
    build_input.vertexStrideInBytes = 12
    # OptiX requires null buffers for empty geometry, which builds a structure nothing intersects.
    empty = indices.size == 0
    build_input.numVertices = 0 if empty else num_vertices
    build_input.vertexBuffers = [] if empty else [key.ptr for key in keys]
    build_input.indexFormat = native.INDICES_FORMAT_UNSIGNED_INT3
    build_input.indexStrideInBytes = 12
    build_input.numIndexTriplets = indices.size * wp.types.type_size_in_bytes(indices.dtype) // 12
    build_input.indexBuffer = 0 if empty else indices.ptr
    build_input.flags = [int(flags)]
    build_input.numSbtRecords = 1
    return BuildInput(build_input, (*keys, indices))


@wp.kernel(module="unique", enable_backward=False)
def _pack_bounds(lowers: wp.array[wp.vec3], uppers: wp.array[wp.vec3], bounds: wp.array2d[float]):
    i = wp.tid()
    lo, hi = lowers[i], uppers[i]
    bounds[i, 0] = lo[0]
    bounds[i, 1] = lo[1]
    bounds[i, 2] = lo[2]
    bounds[i, 3] = hi[0]
    bounds[i, 4] = hi[1]
    bounds[i, 5] = hi[2]


def aabbs(
    lowers: wp.array | None = None,
    uppers: wp.array | None = None,
    *,
    bounds: wp.array | None = None,
    flags: GeometryFlags | Sequence[GeometryFlags] = GeometryFlags.DISABLE_ANYHIT,
    sbt_indices: wp.array | None = None,
) -> BuildInput:
    """Custom-primitive build input, like ``wp.Bvh(lowers, uppers)``.

    Give ``lowers`` and ``uppers`` (``vec3`` per primitive), or packed ``bounds``
    (``float32[N, 6]`` as ``(min, max)``) to avoid a packing copy. Separate arrays
    are packed into a retained buffer before every build and refit. With
    ``sbt_indices`` (uint32 per primitive), primitive ``i`` uses hit record
    ``sbt_indices[i]`` and ``flags`` holds one geometry flag per record.
    """
    prepare = None
    sources: tuple = ()
    if bounds is None:
        if lowers is None or uppers is None:
            raise ValueError("aabbs() needs lowers and uppers, or packed bounds")
        if lowers.dtype != wp.vec3 or uppers.dtype != wp.vec3 or lowers.shape != uppers.shape or lowers.ndim != 1:
            raise ValueError("lowers and uppers must be 1-D vec3 arrays of equal size")
        if lowers.device != uppers.device or not lowers.device.is_cuda:
            raise ValueError("lowers and uppers must be CUDA arrays on the same device")
        bounds = wp.empty((lowers.size, 6), dtype=wp.float32, device=lowers.device)
        sources = (lowers, uppers)
        if lowers.size:
            allocated = wp.get_stream(bounds.device).record_event()
            pack = wp.launch(
                _pack_bounds, lowers.size, inputs=[lowers, uppers, bounds], device=bounds.device, record_cmd=True
            )

            def prepare(stream):
                nonlocal allocated
                # The private bounds allocation may belong to a different stream.
                # Wait once for the allocation, without waiting for later work there.
                if allocated is not None:
                    stream.wait_event(allocated)
                    allocated = None
                pack.launch(stream)

    elif lowers is not None or uppers is not None:
        raise ValueError("pass either lowers and uppers, or bounds, not both")
    if bounds.dtype != wp.float32 or bounds.size % 6 or not bounds.device.is_cuda or not bounds.is_contiguous:
        raise ValueError("bounds must be a contiguous CUDA float32 array of AABB sextuplets")
    flags = [int(f) for f in flags] if isinstance(flags, Sequence) else [int(flags)]
    if not flags:
        raise ValueError("flags must contain at least one SBT record flag")
    if sbt_indices is not None:
        if sbt_indices.dtype != wp.uint32 or sbt_indices.size != bounds.size // 6:
            raise ValueError("sbt_indices must contain one uint32 value per AABB")
        _validate_indices(sbt_indices, len(flags), bounds.device)
    descriptor = native.BuildInputCustomPrimitiveArray(
        aabbBuffers=[bounds.ptr] if bounds.size else [],
        numPrimitives=bounds.size // 6,
        strideInBytes=24,
        flags=flags,
        numSbtRecords=len(flags),
        sbtIndexOffsetBuffer=0 if sbt_indices is None else sbt_indices.ptr,
        sbtIndexOffsetSizeInBytes=0 if sbt_indices is None else 4,
        sbtIndexOffsetStrideInBytes=0 if sbt_indices is None else 4,
    )
    arrays = (bounds, *sources) if sbt_indices is None else (bounds, *sources, sbt_indices)
    return BuildInput(descriptor, arrays, prepare=prepare)


# Control points per curve segment, minus one.
_CURVE_DEGREES = {
    PrimitiveType.ROUND_LINEAR: 1,
    PrimitiveType.ROUND_QUADRATIC_BSPLINE: 2,
    PrimitiveType.ROUND_QUADRATIC_BSPLINE_ROCAPS: 2,
    PrimitiveType.FLAT_QUADRATIC_BSPLINE: 2,
    PrimitiveType.ROUND_CUBIC_BSPLINE: 3,
    PrimitiveType.ROUND_CUBIC_BSPLINE_ROCAPS: 3,
    PrimitiveType.ROUND_CATMULLROM: 3,
    PrimitiveType.ROUND_CATMULLROM_ROCAPS: 3,
    PrimitiveType.ROUND_CUBIC_BEZIER: 3,
    PrimitiveType.ROUND_CUBIC_BEZIER_ROCAPS: 3,
}


def curves(
    points: wp.array | Sequence[wp.array],
    widths: wp.array | Sequence[wp.array],
    indices: wp.array,
    *,
    curve_type: PrimitiveType = PrimitiveType.ROUND_LINEAR,
    flags: GeometryFlags = GeometryFlags.DISABLE_ANYHIT,
    validate: bool = True,
) -> BuildInput:
    """Curve build input; ``indices`` holds the first control point of each segment.

    Creating the input reads indices back once to validate bounds, unless
    ``validate=False``. Keep indices unchanged while reusing the input for
    builds and refits.
    """
    point_keys = [points] if isinstance(points, wp.array) else list(points)
    width_keys = [widths] * len(point_keys) if isinstance(widths, wp.array) else list(widths)
    num_vertices = _vertex_count(point_keys)
    degree = _CURVE_DEGREES.get(int(curve_type))
    if degree is None:
        raise ValueError(f"curve_type must be a curve PrimitiveType, got {curve_type!r}")
    _validate_indices(indices, num_vertices - degree, point_keys[0].device, bounds=validate)
    if len(width_keys) != len(point_keys) or any(
        key.dtype != wp.float32
        or key.size != num_vertices
        or not key.is_contiguous
        or key.device != point_keys[0].device
        for key in width_keys
    ):
        raise ValueError("widths must provide one float32 value per point on the points device")
    if indices.size:
        buffers = {
            "vertexBuffers": [key.ptr for key in point_keys],
            "numVertices": num_vertices,
            "vertexStrideInBytes": 12,
            "widthBuffers": [key.ptr for key in width_keys],
            "widthStrideInBytes": 4,
            "normalBuffers": [0] * len(point_keys),
            "indexBuffer": indices.ptr,
            "indexStrideInBytes": 4,
        }
    else:
        # OptiX requires null buffers and zero strides for empty geometry.
        buffers = {}
    descriptor = native.BuildInputCurveArray(
        curveType=native.PrimitiveType(int(curve_type)), numPrimitives=indices.size, flag=int(flags), **buffers
    )
    return BuildInput(descriptor, (*point_keys, *width_keys, indices))


def instances(records: wp.array, children: Sequence[Accel]) -> BuildInput:
    """Instance input over ``records`` that reference structures in ``children``.

    The instance structure keeps ``children`` alive, so they may be closed or
    dropped while it is in use; builds never read the GPU-written records back
    (set ``config.verify_instances`` to check them). Child ids must stay unchanged
    on refits and rebuilds.
    """
    if records.dtype is not Instance or not records.device.is_cuda or not records.is_contiguous:
        raise ValueError("records must be a contiguous CUDA array of Instance")
    children = tuple(children)
    for child in children:
        if not isinstance(child, Accel):
            raise TypeError("children must be warp_optix.Accel objects")
        if child.device != records.device:
            raise ValueError("children must be on the device of the records")
    return BuildInput(
        native.BuildInputInstanceArray(instances=records.ptr if records.size else 0, numInstances=records.size),
        (records,),
        instance_records=records,
        instance_children=children,
    )


# ---------------------------------------------------------------------------
# Acceleration structures
# ---------------------------------------------------------------------------


class Accel:
    """An OptiX acceleration structure; pass :attr:`id` to kernels.

    Like ``wp.Mesh``/``wp.Bvh``, construction builds the structure. ``inputs`` come
    from :func:`triangles`, :func:`curves`, :func:`aabbs`, or :func:`instances`.
    ``device``/``stream`` follow the usual contract; with neither given, the
    device of the inputs is used. Structures built with
    ``BuildFlags.ALLOW_UPDATE`` keep their scratch buffer for :meth:`refit`;
    others release it once the build has finished and allocate a temporary one
    per :meth:`rebuild`. ``compact=True`` synchronizes ``stream`` once to shrink a
    static structure; to compact many structures with a single synchronization,
    build them with ``BuildFlags.ALLOW_COMPACTION`` (they keep their scratch buffer
    until then) and pass them to :func:`compact`. Compacted structures cannot be
    refit or rebuilt.

    Release is automatic when the last reference is dropped, or explicit with
    :meth:`close` or ``with``; memory is freed in stream order, like ``wp.array``.
    """

    def __init__(
        self,
        inputs: Sequence[BuildInput],
        *,
        flags: BuildFlags = BuildFlags.PREFER_FAST_TRACE,
        compact: bool = False,
        motion: native.MotionOptions | None = None,
        device: wp.DeviceLike = None,
        stream: wp.Stream | None = None,
    ):
        self._closed = True
        inputs = tuple(inputs)
        for item in inputs:
            if not isinstance(item, BuildInput):
                raise TypeError("inputs must be created with triangles(), curves(), aabbs(), or instances()")
        if device is None and stream is None and inputs:
            device = inputs[0].device
        context, stream = resolve(device, stream)
        if not inputs:
            raise ValueError("Accel needs at least one build input")
        context.check_arrays([item.arrays for item in inputs])
        flags = int(flags) | (BuildFlags.ALLOW_COMPACTION if compact else 0)
        if flags & BuildFlags.ALLOW_COMPACTION and flags & BuildFlags.ALLOW_UPDATE:
            raise ValueError("compacted acceleration structures cannot be refit")

        def build_options(operation):
            options = native.AccelBuildOptions(buildFlags=flags, operation=operation)
            if motion is not None:
                options.motionOptions = motion
            return [options] * len(inputs)

        options = build_options(native.BUILD_OPERATION_BUILD)
        native_inputs = [item.native for item in inputs]
        with wp.ScopedDevice(context.device), wp.ScopedStream(stream, sync_enter=False):
            children = {}
            for item in inputs:
                if item.instance_children is not None:
                    for child in item.instance_children:
                        child._check_open()
                        children[child._id] = child._memory
                    if config.verify_instances:
                        wp.synchronize_stream(stream)
                        child_ids = np.unique(item.instance_records.numpy()["traversable"]).tolist()
                        if any(child_id and child_id not in children for child_id in child_ids):
                            raise ValueError("Instance records reference a structure missing from children")
                if item.prepare is not None:
                    item.prepare(stream)
            sizes = context.native.accelComputeMemoryUsage(options, native_inputs)
            scratch_size = (max(sizes.tempSizeInBytes, sizes.tempUpdateSizeInBytes) + 7) // 8 * 8
            compacting = bool(flags & BuildFlags.ALLOW_COMPACTION)
            # The compacted size is emitted into 8 bytes after the scratch space, saving an allocation.
            scratch = wp.empty(scratch_size + 8 * compacting, dtype=wp.uint8, device=context.device)
            buffer = wp.empty(sizes.outputSizeInBytes, dtype=wp.uint8, device=context.device)
            compacted_size = scratch.ptr + scratch_size if compacting else None
            emitted = [native.AccelEmitDesc(compacted_size, native.PROPERTY_TYPE_COMPACTED_SIZE)] if compacting else []
            handle = context.native.accelBuild(
                stream.cuda_stream,
                options,
                native_inputs,
                scratch.ptr,
                scratch_size,
                buffer.ptr,
                buffer.size,
                emitted,
            )

        self._context = context
        self._id = int(handle)
        self._buffer = buffer
        # Dropping a scratch buffer the build still reads is safe: Warp frees device memory in stream order.
        # Structures awaiting compaction keep it until compact(), which synchronizes anyway: each free
        # orders all blocking streams, so dropping it here would serialize builds spread over streams.
        self._scratch = scratch if flags & (BuildFlags.ALLOW_UPDATE | BuildFlags.ALLOW_COMPACTION) else None
        self._scratch_size = scratch_size
        self._compacted = False
        self._compacted_size = compacted_size
        self._inputs = inputs
        self._native_inputs = native_inputs
        self._flags = flags
        self._rebuild_options = options
        self._refit_options = build_options(native.BUILD_OPERATION_UPDATE) if flags & BuildFlags.ALLOW_UPDATE else None
        # Retain every level of instance memory even when child objects are closed.
        self._memory = (buffer, *children.values())
        self._closed = False
        if compact:
            _compact([self], context, stream)

    @property
    def id(self) -> int:
        """The traversable handle passed to kernels (``wp.uint64``)."""
        self._check_open()
        return self._id

    @property
    def device(self) -> wp.Device:
        return self._context.device

    @property
    def flags(self) -> int:
        """The :class:`BuildFlags` the structure was built with."""
        return self._flags

    @property
    def closed(self) -> bool:
        return self._closed

    def refit(self, stream: wp.Stream | None = None) -> None:
        """Refit to the inputs' current contents; requires ``BuildFlags.ALLOW_UPDATE``.

        Inputs must keep their primitive counts, indices, and instance child ids.
        Neither allocates nor synchronizes the host.
        """
        self._update(rebuild=False, stream=stream)

    def rebuild(self, stream: wp.Stream | None = None) -> None:
        """Rebuild in place, restoring trace quality after many refits."""
        self._update(rebuild=True, stream=stream)

    def close(self) -> None:
        """Make the structure unusable and release its memory (stream-ordered, like ``wp.array``)."""
        if self._closed:
            return
        self._closed = True
        self._buffer = self._scratch = self._compacted_size = None
        self._inputs = self._memory = ()

    def __enter__(self) -> Accel:
        self._check_open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self):
        if not getattr(self, "_closed", True):
            try:
                self.close()
            except Exception:
                pass

    def __repr__(self):
        state = "closed" if self._closed else f"id={self._id:#x}"
        return f"Accel({state}, device={self.device.alias!r})"

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("The acceleration structure is closed")

    def _update(self, *, rebuild: bool, stream: wp.Stream | None) -> None:
        self._check_open()
        if not rebuild and self._refit_options is None:
            raise ValueError("refit() requires an acceleration structure built with BuildFlags.ALLOW_UPDATE")
        if self._compacted:
            raise ValueError("compacted acceleration structures cannot be refit or rebuilt")
        context, stream = resolve(self.device, stream)
        options = self._rebuild_options if rebuild else self._refit_options
        with wp.ScopedDevice(context.device):
            scratch, scratch_size = self._scratch, self._scratch_size
            if scratch is None:
                sizes = context.native.accelComputeMemoryUsage(options, self._native_inputs)
                with wp.ScopedStream(stream, sync_enter=False):
                    scratch = wp.empty(sizes.tempSizeInBytes, dtype=wp.uint8, device=context.device)
                scratch_size = scratch.size
            for item in self._inputs:
                if item.prepare is not None:
                    item.prepare(stream)
            handle = int(
                context.native.accelBuild(
                    stream.cuda_stream,
                    options,
                    self._native_inputs,
                    scratch.ptr,
                    scratch_size,
                    self._buffer.ptr,
                    self._buffer.size,
                    []
                    if self._compacted_size is None
                    else [native.AccelEmitDesc(self._compacted_size, native.PROPERTY_TYPE_COMPACTED_SIZE)],
                )
            )
        self._id = int(handle)


def compact(accels: Sequence[Accel], stream: wp.Stream | None = None) -> None:
    """Compact structures built with ``BuildFlags.ALLOW_COMPACTION``, synchronizing ``stream`` once.

    Compaction changes :attr:`Accel.id`, so compact children before building
    instance structures over them. Compacted structures cannot be refit or rebuilt.
    """
    accels = list(accels)
    if not accels:
        return
    for accel in accels:
        if not isinstance(accel, Accel):
            raise TypeError("compact() takes Accel objects")
        accel._check_open()
        if accel._compacted_size is None:
            raise ValueError("compact() requires structures built with BuildFlags.ALLOW_COMPACTION")
        if accel.device != accels[0].device:
            raise ValueError("compact() requires structures on the same device")
    context, stream = resolve(accels[0].device, stream)
    _compact(accels, context, stream)


def _compact(accels: list[Accel], context, stream: wp.Stream) -> None:
    if stream.is_capturing:
        raise RuntimeError(CAPTURE_ERROR)
    device = context.device
    host_sizes = wp.empty(len(accels), dtype=wp.uint64, device="cpu", pinned=True)
    core = _warp_context.runtime.core
    for i, accel in enumerate(accels):
        core.wp_memcpy_d2h(device.context, host_sizes.ptr + 8 * i, accel._compacted_size, 8, stream.cuda_stream)
    wp.synchronize_stream(stream)
    with wp.ScopedDevice(device), wp.ScopedStream(stream, sync_enter=False):
        for accel, size in zip(accels, host_sizes.numpy().tolist(), strict=True):
            if size < accel._buffer.size:
                output = wp.empty(size, dtype=wp.uint8, device=device)
                accel._id = int(context.native.accelCompact(stream.cuda_stream, accel._id, output.ptr, size))
                accel._buffer = output
                accel._memory = (output, *accel._memory[1:])
            accel._compacted = True
            accel._compacted_size = accel._scratch = None
