# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OptiX acceleration structures over Warp device arrays.

Build inputs retain their device arrays. Managed contexts retain acceleration
structures until closure. Initial instance builds validate child handles with
a host readback; compaction also synchronizes. Transform-only updates remain
asynchronous. OptiX work cannot be captured into CUDA graphs.
"""

from __future__ import annotations

import ctypes
import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass

import warp as wp

from warp_optix import native
from warp_optix.context import Context, require_context
from warp_optix.enums import BuildFlags, GeometryFlags, PrimitiveType

# ---------------------------------------------------------------------------
# Instances
# ---------------------------------------------------------------------------

# OptiX enqueues builds and launches outside CUDA stream capture: a captured graph
# would silently omit the work. Every entry point that enqueues OptiX work refuses capture.
CAPTURE_ERROR = "OptiX work cannot be recorded into CUDA graphs; call it outside wp.ScopedCapture"

_mat34 = wp.types.matrix(shape=(3, 4), dtype=wp.float32)


@wp.struct
class Instance:
    """Device layout of ``OptixInstance`` (80 bytes)."""

    transform: _mat34
    instance_id: wp.uint32
    sbt_offset: wp.uint32
    visibility_mask: wp.uint32
    flags: wp.uint32
    traversable: wp.uint64
    pad: wp.vec2ui


assert ctypes.sizeof(Instance.ctype) == 80, "Instance must match OptixInstance"


@wp.func
def make_instance(
    rotation: wp.mat33,
    translation: wp.vec3,
    traversable: wp.uint64,
    instance_id: wp.uint32,
    sbt_offset: wp.uint32,
    visibility_mask: wp.uint32,
    flags: wp.uint32,
) -> Instance:
    """Return an instance record placing ``traversable`` at ``rotation`` and ``translation``.

    ``rotation`` may include scale or shear; it is the linear part of the object-to-world map.
    """
    r, p = rotation, translation
    instance = Instance()
    instance.transform = _mat34(
        r[0, 0], r[0, 1], r[0, 2], p[0], r[1, 0], r[1, 1], r[1, 2], p[1], r[2, 0], r[2, 1], r[2, 2], p[2]
    )
    instance.instance_id = instance_id
    instance.sbt_offset = sbt_offset
    instance.visibility_mask = visibility_mask
    instance.flags = flags
    instance.traversable = traversable
    return instance


# ---------------------------------------------------------------------------
# Build inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BuildInput:
    """A native descriptor together with the device arrays it borrows."""

    native: object
    arrays: tuple
    instance_records: wp.array | None = None


def _vertex_count(keys):
    if not keys:
        raise ValueError("vertices must contain at least one motion key")
    count = None
    for key in keys:
        if not key.device.is_cuda or not key.is_contiguous:
            raise ValueError("vertices must be contiguous CUDA arrays")
        if key.dtype == wp.vec3:
            current = key.size
        elif key.dtype == wp.float32 and key.size % 3 == 0:
            current = key.size // 3
        else:
            raise TypeError("vertices must contain float32 triplets or vec3 values")
        if count is not None and current != count:
            raise ValueError("vertex motion keys must have equal sizes")
        if key.device != keys[0].device:
            raise ValueError("vertex motion keys must use the same device")
        count = current
    return count


def _validate_indices(indices, limit, device, *, triplets=False):
    if indices.dtype not in (wp.int32, wp.uint32):
        raise TypeError("indices must contain int32 or uint32 values")
    if indices.device != device or not indices.is_contiguous:
        raise ValueError("indices must be contiguous and on the vertex device")
    if triplets and indices.size % 3:
        raise ValueError("triangle indices must contain complete triplets")
    if wp.get_stream(device).is_capturing:
        raise RuntimeError(CAPTURE_ERROR)
    # Validate once when creating the input, retaining the previous API's bounds
    # protection. Vertex-only refits do not repeat this host readback.
    host = indices.numpy()
    if host.size and (host.min() < 0 or host.max() >= limit):
        raise ValueError(f"indices must lie in [0, {limit})")


def triangles(
    vertices: wp.array | Sequence[wp.array],
    indices: wp.array,
    *,
    flags: int = GeometryFlags.DISABLE_ANYHIT,
):
    """Triangle build input from ``vec3``/``float32[N, 3]`` vertices and int32/uint32 index triplets.

    A sequence of vertex arrays gives one motion key per array. Creating the
    input reads indices back once to validate bounds; keep indices unchanged
    while reusing the input for builds and updates.
    """
    keys = [vertices] if isinstance(vertices, wp.array) else list(vertices)
    num_vertices = _vertex_count(keys)
    _validate_indices(indices, num_vertices, keys[0].device, triplets=True)
    build_input = native.BuildInputTriangleArray()
    build_input.vertexFormat = native.VERTEX_FORMAT_FLOAT3
    build_input.vertexStrideInBytes = 12
    build_input.numVertices = num_vertices
    build_input.vertexBuffers = [key.ptr for key in keys]
    build_input.indexFormat = native.INDICES_FORMAT_UNSIGNED_INT3
    build_input.indexStrideInBytes = 12
    build_input.numIndexTriplets = indices.size * wp.types.type_size_in_bytes(indices.dtype) // 12
    build_input.indexBuffer = indices.ptr
    build_input.flags = [int(flags)]
    build_input.numSbtRecords = 1
    return BuildInput(build_input, (*keys, indices))


def aabbs(
    bounds: wp.array,
    *,
    flags: int | Sequence[int] = GeometryFlags.DISABLE_ANYHIT,
    sbt_indices: wp.array | None = None,
):
    """Custom-primitive build input from ``float32[N, 6]`` ``(min, max)`` bounds.

    With ``sbt_indices`` (uint32 per primitive), primitive ``i`` uses hit record
    ``sbt_indices[i]`` and ``flags`` holds one geometry flag per record.
    """
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
        aabbBuffers=[bounds.ptr],
        numPrimitives=bounds.size * wp.types.type_size_in_bytes(bounds.dtype) // 24,
        strideInBytes=24,
        flags=flags,
        numSbtRecords=len(flags),
        sbtIndexOffsetBuffer=0 if sbt_indices is None else sbt_indices.ptr,
        sbtIndexOffsetSizeInBytes=0 if sbt_indices is None else 4,
        sbtIndexOffsetStrideInBytes=0 if sbt_indices is None else 4,
    )
    return BuildInput(descriptor, (bounds,) if sbt_indices is None else (bounds, sbt_indices))


def curves(
    vertices: wp.array | Sequence[wp.array],
    widths: wp.array | Sequence[wp.array],
    indices: wp.array,
    *,
    curve_type: int = PrimitiveType.ROUND_LINEAR,
    flags: int = GeometryFlags.DISABLE_ANYHIT,
):
    """Curve build input; ``indices`` holds the first control vertex of each segment.

    Creating the input reads indices back once to validate bounds. Keep indices
    unchanged while reusing the input for builds and updates.
    """
    vertex_keys = [vertices] if isinstance(vertices, wp.array) else list(vertices)
    width_keys = [widths] * len(vertex_keys) if isinstance(widths, wp.array) else list(widths)
    num_vertices = _vertex_count(vertex_keys)
    degree = 3
    for name, value in (("ROUND_LINEAR", 1), ("ROUND_QUADRATIC_BSPLINE", 2),
                        ("ROUND_QUADRATIC_BSPLINE_ROCAPS", 2), ("FLAT_QUADRATIC_BSPLINE", 2)):
        primitive = getattr(PrimitiveType, name, None)
        if primitive is not None and int(curve_type) == int(primitive):
            degree = value
            break
    _validate_indices(indices, num_vertices - degree, vertex_keys[0].device)
    if len(width_keys) != len(vertex_keys) or any(
        key.dtype != wp.float32 or key.size != num_vertices or not key.is_contiguous
        or key.device != vertex_keys[0].device for key in width_keys
    ):
        raise ValueError("widths must provide one float32 value per vertex on the vertex device")
    descriptor = native.BuildInputCurveArray(
        curveType=native.PrimitiveType(int(curve_type)),
        numPrimitives=indices.size,
        vertexBuffers=[key.ptr for key in vertex_keys],
        numVertices=num_vertices,
        vertexStrideInBytes=12,
        widthBuffers=[key.ptr for key in width_keys],
        widthStrideInBytes=4,
        normalBuffers=[0] * len(vertex_keys),
        indexBuffer=indices.ptr,
        indexStrideInBytes=4,
        flag=int(flags),
    )
    return BuildInput(descriptor, (*vertex_keys, *width_keys, indices))


def instances(records: wp.array):
    """Instance input retaining its records; child handles must stay unchanged on updates."""
    if records.dtype is not Instance or not records.device.is_cuda or not records.is_contiguous:
        raise ValueError("records must be a contiguous CUDA array of Instance")
    return BuildInput(native.BuildInputInstanceArray(instances=records.ptr, numInstances=records.size),
                      (records,), instance_records=records)


# ---------------------------------------------------------------------------
# Build and update
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Accel:
    """An acceleration structure and the buffers needed to update it in place."""

    handle: int
    buffer: wp.array
    scratch: wp.array | None
    inputs: tuple
    flags: int
    motion: object | None
    device: wp.Device
    context: Context
    dependencies: tuple = ()


def build_accel(
    ctx,
    inputs: Sequence,
    *,
    flags: int = BuildFlags.PREFER_FAST_TRACE,
    compact: bool = False,
    motion=None,
    stream: wp.Stream | None = None,
    device=None,
) -> Accel:
    """Build an acceleration structure over ``inputs`` on ``stream``.

    The result keeps its scratch buffer, sized for :func:`update_accel` refits
    and in-place rebuilds, because freeing it could race the asynchronous build.
    ``compact=True`` synchronizes ``stream`` once to shrink a static structure to
    its final size and release that memory.
    """
    ctx = require_context(ctx)
    device = ctx._device(device)
    stream = ctx._stream(stream)
    if stream.is_capturing:
        raise RuntimeError(CAPTURE_ERROR)
    flags = int(flags) | (BuildFlags.ALLOW_COMPACTION if compact else 0)
    if compact and flags & BuildFlags.ALLOW_UPDATE:
        raise ValueError("compacted acceleration structures cannot be updated")

    with ctx._transaction(), wp.ScopedStream(stream):
        options = native.AccelBuildOptions(buildFlags=flags, operation=native.BUILD_OPERATION_BUILD)
        if motion is not None:
            options.motionOptions = motion
        inputs = tuple(inputs)
        dependencies = []
        for item in inputs:
            if not isinstance(item, BuildInput):
                raise TypeError("inputs must be created with triangles(), curves(), aabbs(), or instances()")
            ctx._check_arrays(item.arrays)
            if item.instance_records is not None:
                # Validate GPU-written child handles once. Transform-only updates
                # retain this topology and do not need another host readback.
                wp.synchronize_stream(stream)
                for handle in item.instance_records.numpy()["traversable"].reshape(-1):
                    if int(handle) == 0:
                        continue
                    dependency = ctx._accels.get(int(handle))
                    if dependency is None:
                        raise ValueError("Instance geometry must belong to this context")
                    dependencies.append(dependency)
        native_inputs = [item.native for item in inputs]
        sizes = ctx.accelComputeMemoryUsage([options] * len(inputs), native_inputs)
        scratch = wp.empty(max(sizes.tempSizeInBytes, sizes.tempUpdateSizeInBytes), dtype=wp.uint8, device=device)
        buffer = wp.empty(sizes.outputSizeInBytes, dtype=wp.uint8, device=device)
        compacted_size = wp.empty(1, dtype=wp.uint64, device=device) if compact else None
        emitted = [native.AccelEmitDesc(compacted_size.ptr, native.PROPERTY_TYPE_COMPACTED_SIZE)] if compact else []
        handle = ctx.accelBuild(
            stream.cuda_stream,
            [options] * len(inputs),
            native_inputs,
            scratch.ptr,
            scratch.size,
            buffer.ptr,
            buffer.size,
            emitted,
        )
        if compact:
            size = int(compacted_size.numpy()[0])
            if size < buffer.size:
                output = wp.empty(size, dtype=wp.uint8, device=device)
                handle = ctx.accelCompact(stream.cuda_stream, handle, output.ptr, size)
                wp.synchronize_stream(stream)
                buffer = output
            scratch = None

        result = Accel(
            handle=int(handle),
            buffer=buffer,
            scratch=scratch,
            inputs=inputs,
            flags=flags,
            motion=motion,
            device=device,
            context=ctx,
            dependencies=tuple(dependencies),
        )
        ctx._accels[result.handle] = result
        return result


def update_accel(ctx, accel: Accel, *, rebuild: bool = False, stream: wp.Stream | None = None) -> Accel:
    """Refit ``accel`` to its inputs' current contents, or rebuild it in place.

    Inputs must keep their primitive counts, indices, and instance child handles.
    Rebuilding restores trace quality after many refits. Neither allocates nor
    synchronizes the host on success.
    """
    ctx = require_context(ctx)
    if accel.context is not ctx:
        raise ValueError("Acceleration structure must belong to this context")
    if not rebuild and not accel.flags & BuildFlags.ALLOW_UPDATE:
        raise ValueError("refitting requires an acceleration structure built with BuildFlags.ALLOW_UPDATE")
    if accel.scratch is None:
        raise ValueError("compacted acceleration structures cannot be updated")
    stream = ctx._stream(stream)
    if stream.is_capturing:
        raise RuntimeError(CAPTURE_ERROR)
    with ctx._transaction(), wp.ScopedStream(stream):
        options = native.AccelBuildOptions(
            buildFlags=accel.flags,
            operation=native.BUILD_OPERATION_BUILD if rebuild else native.BUILD_OPERATION_UPDATE,
        )
        if accel.motion is not None:
            options.motionOptions = accel.motion
        handle = ctx.accelBuild(
            stream.cuda_stream,
            [options] * len(accel.inputs),
            [item.native for item in accel.inputs],
            accel.scratch.ptr,
            accel.scratch.size,
            accel.buffer.ptr,
            accel.buffer.size,
            [],
        )
        result = dataclasses.replace(accel, handle=int(handle))
        ctx._accels[result.handle] = result
        return result
