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

"""Minimal reusable runtime helpers for Warp + otk-pyoptix integration."""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass

import numpy as np
import warp as wp

from warp_optix._addon import get_module_build_options
from warp_optix._compat import (
    compile_module_to_ptx,
    has_public_addon_hooks,
    validate_module_tag,
)


def require_optix():
    try:
        from warp_optix._backend import get_backend
        optix = get_backend()
    except ImportError as e:
        raise RuntimeError(
            "Install warp-optix and its PyOptiX dependency: pip install ."
        ) from e
    return optix


def _optix_version_at_least(optix, major: int, minor: int) -> bool:
    version = tuple(optix.version())
    return version >= (major, minor)


class OptixLogger:
    def __init__(self):
        self.num_messages = 0

    def __call__(self, level, tag, message):
        print(f"[{level:>2}][{tag:>12}]: {message}")
        self.num_messages += 1


def create_optix_context(
    optix, cuda_context, log_level: int = 4, *, enable_validation: bool = False
):
    logger = OptixLogger()
    options = optix.DeviceContextOptions(
        logCallbackFunction=logger, logCallbackLevel=log_level
    )
    if _optix_version_at_least(optix, 7, 2):
        options.validationMode = (
            optix.DEVICE_CONTEXT_VALIDATION_MODE_ALL
            if enable_validation
            else optix.DEVICE_CONTEXT_VALIDATION_MODE_OFF
        )
    return optix.deviceContextCreate(cuda_context, options), logger


@dataclass
class LaunchParamsBuffer:
    struct_type: type
    struct_ctype: type[ctypes.Structure]
    bytes: wp.array
    writer: wp.Kernel
    nbytes: int
    device: str


def _is_warp_struct_type(value) -> bool:
    return hasattr(value, "ctype") and hasattr(value, "instance_type")


def _get_struct_type_info(params_struct_type):
    if _is_warp_struct_type(params_struct_type):
        wp_struct = params_struct_type
    else:
        wp_struct = getattr(params_struct_type, "_cls", None)
        if wp_struct is None and hasattr(params_struct_type, "__class__"):
            wp_struct = getattr(params_struct_type.__class__, "_cls", None)

    if not _is_warp_struct_type(wp_struct):
        raise TypeError(
            "params_struct_type must be a @wp.struct definition (example: create_launch_params_buffer(MyLaunchParams, ...))"
        )
    return wp_struct, wp_struct.ctype


def _create_launch_params_writer(wp_struct):
    def write_params(params, output):
        output[0] = params

    write_params.__annotations__ = {
        "params": wp_struct,
        "output": wp.array(dtype=wp_struct),
    }
    return wp.Kernel(
        func=write_params,
        key=f"write_launch_params_{wp_struct.key}",
    )


def create_launch_params_buffer(
    params_struct_type: type, device: str = "cuda"
) -> LaunchParamsBuffer:
    wp_struct, struct_ctype = _get_struct_type_info(params_struct_type)
    nbytes = ctypes.sizeof(struct_ctype)
    storage = wp.empty(1, dtype=wp_struct, device=device)
    return LaunchParamsBuffer(
        struct_type=wp_struct.instance_type,
        struct_ctype=struct_ctype,
        bytes=storage,
        writer=_create_launch_params_writer(wp_struct),
        nbytes=nbytes,
        device=device,
    )


def write_launch_params(
    buffer: LaunchParamsBuffer, params_struct_instance, stream=None
) -> None:
    if not isinstance(buffer, LaunchParamsBuffer):
        raise TypeError(
            "buffer must be a LaunchParamsBuffer created by create_launch_params_buffer()"
        )
    if not hasattr(params_struct_instance, "_ctype"):
        raise TypeError("params_struct_instance must be a Warp struct instance")
    if not isinstance(params_struct_instance, buffer.struct_type):
        raise TypeError(
            f"params struct type mismatch: expected {buffer.struct_type.__name__}, got {type(params_struct_instance).__name__}"
        )

    wp.launch(
        buffer.writer,
        dim=1,
        inputs=[params_struct_instance],
        outputs=[buffer.bytes],
        device=buffer.device,
        stream=stream,
    )


def launch(
    optix,
    pipeline,
    sbt,
    width: int,
    height: int,
    params_buffer: LaunchParamsBuffer,
    stream: int = 0,
) -> None:
    if not isinstance(params_buffer, LaunchParamsBuffer):
        raise TypeError("params_buffer must be a LaunchParamsBuffer")
    optix.launch(
        pipeline,
        stream,
        params_buffer.bytes.ptr,
        params_buffer.nbytes,
        sbt,
        width,
        height,
        1,
    )


def get_optix_entry_name(kernel_or_entry, expected_kernel_type=None) -> str:
    if isinstance(kernel_or_entry, str):
        return kernel_or_entry

    if not isinstance(kernel_or_entry, wp.Kernel):
        raise TypeError("expected a Warp kernel object or explicit entry name string")

    kernel_type = kernel_or_entry.options.get("optix_kernel_type")
    if kernel_type is None:
        raise TypeError(
            f"kernel '{kernel_or_entry.key}' is not an OptiX kernel "
            "(decorate it with @warp_optix.optix_kernel(...) instead of @wp.kernel)"
        )
    if expected_kernel_type is not None and kernel_type is not expected_kernel_type:
        raise TypeError(
            f"kernel '{kernel_or_entry.key}' has type {kernel_type}, expected {expected_kernel_type}"
        )

    return kernel_or_entry.get_mangled_name()


def _prepend_cuda_preamble(
    build_options: wp.ModuleBuildOptions, preamble: str
) -> wp.ModuleBuildOptions:
    if not preamble:
        return build_options

    if not preamble.endswith("\n"):
        preamble += "\n"

    return wp.ModuleBuildOptions(
        extra_cuda_include_dirs=build_options.extra_cuda_include_dirs,
        extra_cpu_include_dirs=build_options.extra_cpu_include_dirs,
        extra_cuda_preamble=preamble + build_options.extra_cuda_preamble,
        extra_cpu_preamble=build_options.extra_cpu_preamble,
        extra_build_dependencies=build_options.extra_build_dependencies,
    )


def compile_warp_module_to_ptx(
    module: wp.Module,
    launch_preamble: str,
    module_tag: str,
    script_dir: str,
    device: str = "cuda",
) -> bytes:
    del script_dir  # Preserved for backward-compatible call sites.
    validate_module_tag(module_tag)
    # Device-only cooperative-vector wrappers deliberately fail when compiled
    # as an ordinary Warp CUDA kernel. Mark both OptiX compilation paths before
    # the addon header is included so those wrappers can be instantiated.
    launch_preamble = "#define WP_OPTIX_PROGRAM 1\n" + launch_preamble

    if not has_public_addon_hooks(wp):
        return compile_module_to_ptx(wp, module, launch_preamble, module_tag, device)

    old_build_options = wp.get_module_options(module).get("extra_build_options")
    build_options = get_module_build_options(wp, old_build_options)
    wp.set_module_options(
        {"extra_build_options": _prepend_cuda_preamble(build_options, launch_preamble)},
        module=module,
    )
    try:
        wp.get_device(device)
        module_dir = os.path.join(wp.config.kernel_cache_dir, "optix", module_tag)
        artifacts = wp.compile_aot_module(
            module, device=device, module_dir=module_dir, use_ptx=True
        )
        return artifacts[0].read_bytes()
    finally:
        wp.set_module_options({"extra_build_options": old_build_options}, module=module)


class AccelResources(dict):
    """Device buffers and retained build state for one acceleration structure."""

    def __init__(
        self,
        buffers,
        *,
        handle,
        build_inputs,
        build_flags,
        motion_options,
        output_buffer,
        output_size,
        uncompacted_size,
        update_temp_size,
        compacted,
        device,
    ):
        super().__init__(buffers)
        self.handle = int(handle)
        self.build_inputs = list(build_inputs)
        self.build_flags = int(build_flags)
        self.motion_options = motion_options
        self.output_buffer = output_buffer
        self.output_size_in_bytes = int(output_size)
        self.uncompacted_size_in_bytes = int(uncompacted_size)
        self.update_temp_size_in_bytes = int(update_temp_size)
        self.compacted = bool(compacted)
        self.device = device


def _build_acceleration_structure(
    optix,
    ctx,
    build_inputs,
    buffers,
    output_key: str,
    device: str,
    *,
    build_flags,
    compact: bool,
    motion_options=None,
):
    build_flags = int(build_flags)
    if compact and build_flags & int(optix.BUILD_FLAG_ALLOW_UPDATE):
        raise ValueError("compact and BUILD_FLAG_ALLOW_UPDATE cannot be combined")
    if compact:
        build_flags |= int(optix.BUILD_FLAG_ALLOW_COMPACTION)

    options_kwargs = {
        "buildFlags": build_flags,
        "operation": optix.BUILD_OPERATION_BUILD,
    }
    if motion_options is not None:
        options_kwargs["motionOptions"] = motion_options
    options = optix.AccelBuildOptions(**options_kwargs)
    sizes = ctx.accelComputeMemoryUsage([options], build_inputs)
    d_temp = wp.empty(sizes.tempSizeInBytes, dtype=wp.uint8, device=device)
    d_output = wp.empty(sizes.outputSizeInBytes, dtype=wp.uint8, device=device)

    emitted = []
    d_compacted_size = None
    if compact:
        d_compacted_size = wp.zeros(1, dtype=wp.uint64, device=device)
        emitted.append(
            optix.AccelEmitDesc(
                d_compacted_size.ptr, optix.PROPERTY_TYPE_COMPACTED_SIZE
            )
        )

    handle = ctx.accelBuild(
        0,
        [options],
        build_inputs,
        d_temp.ptr,
        sizes.tempSizeInBytes,
        d_output.ptr,
        sizes.outputSizeInBytes,
        emitted,
    )
    wp.synchronize_device(device)

    compacted = False
    output_size = int(sizes.outputSizeInBytes)
    if d_compacted_size is not None:
        compacted_size = int(d_compacted_size.numpy()[0])
        if 0 < compacted_size < output_size:
            compacted_output = wp.empty(compacted_size, dtype=wp.uint8, device=device)
            handle = ctx.accelCompact(0, handle, compacted_output.ptr, compacted_size)
            wp.synchronize_device(device)
            d_output = compacted_output
            output_size = compacted_size
            compacted = True

    buffers[output_key] = d_output
    resources = AccelResources(
        buffers,
        handle=handle,
        build_inputs=build_inputs,
        build_flags=build_flags,
        motion_options=getattr(options, "motionOptions", None),
        output_buffer=d_output,
        output_size=output_size,
        uncompacted_size=sizes.outputSizeInBytes,
        update_temp_size=getattr(sizes, "tempUpdateSizeInBytes", 0),
        compacted=compacted,
        device=device,
    )
    return resources.handle, resources


def refit_acceleration_structure(
    optix, ctx, resources: AccelResources, *, stream: int = 0
) -> int:
    """Refit an acceleration structure after updating its retained device buffers."""
    if not isinstance(resources, AccelResources):
        raise TypeError(
            "resources must be returned by a warp_optix acceleration-structure helper"
        )
    if not resources.build_flags & int(optix.BUILD_FLAG_ALLOW_UPDATE):
        raise ValueError(
            "the acceleration structure was not built with BUILD_FLAG_ALLOW_UPDATE"
        )
    if resources.compacted:
        raise ValueError(
            "compacted acceleration structures cannot be refit by this helper"
        )

    options = optix.AccelBuildOptions(
        buildFlags=resources.build_flags,
        operation=optix.BUILD_OPERATION_UPDATE,
        motionOptions=resources.motion_options,
    )
    d_temp = wp.empty(
        resources.update_temp_size_in_bytes, dtype=wp.uint8, device=resources.device
    )
    resources.handle = int(
        ctx.accelBuild(
            stream,
            [options],
            resources.build_inputs,
            d_temp.ptr,
            resources.update_temp_size_in_bytes,
            resources.output_buffer.ptr,
            resources.uncompacted_size_in_bytes,
            [],
        )
    )
    wp.synchronize_device(resources.device)
    return resources.handle


def _motion_options_for_keys(optix, num_keys: int, motion_time_range):
    if num_keys == 1:
        return None
    if num_keys < 2:
        raise ValueError("motion geometry requires at least two keys")
    time_begin, time_end = (
        (0.0, 1.0) if motion_time_range is None else motion_time_range
    )
    if not float(time_begin) < float(time_end):
        raise ValueError("motion_time_range must have an increasing begin and end")
    return optix.MotionOptions(
        numKeys=num_keys,
        flags=optix.MOTION_FLAG_NONE,
        timeBegin=float(time_begin),
        timeEnd=float(time_end),
    )


def _checked_indices(indices, limit: int, name: str) -> np.ndarray:
    """Convert indices to uint32 after checking that every entry is in ``[0, limit)``.

    OptiX dereferences these indices on the device without bounds checks, so an
    out-of-range value would read past the end of the vertex buffers.
    """
    host_indices = np.asarray(indices)
    if host_indices.size == 0:
        return host_indices.astype(np.uint32)
    if not np.issubdtype(host_indices.dtype, np.integer):
        raise TypeError(f"{name} must be an integer array, got {host_indices.dtype}")
    lowest = int(host_indices.min())
    highest = int(host_indices.max())
    if lowest < 0 or highest >= limit:
        raise ValueError(
            f"{name} must lie in [0, {limit}), got values in [{lowest}, {highest}]"
        )
    return host_indices.astype(np.uint32, copy=False)


def _curve_degree(optix, curve_type) -> int:
    """Number of control points following a curve segment's start index."""
    degrees = {
        "ROUND_LINEAR": 1,
        "ROUND_QUADRATIC_BSPLINE": 2,
        "ROUND_QUADRATIC_BSPLINE_ROCAPS": 2,
        "FLAT_QUADRATIC_BSPLINE": 2,
    }
    for name, degree in degrees.items():
        value = getattr(optix, f"PRIMITIVE_TYPE_{name}", None)
        if value is not None and int(value) == int(curve_type):
            return degree
    return 3  # Cubic B-spline, Catmull-Rom, and Bezier segments.


def create_triangle_gas(
    optix,
    ctx,
    vertices: np.ndarray,
    indices: np.ndarray,
    device: str,
    *,
    build_flags=None,
    compact: bool = False,
    motion_time_range=None,
):
    host_vertices = np.asarray(vertices, dtype=np.float32)
    if host_vertices.ndim == 2 and host_vertices.shape[1] == 3:
        host_vertex_keys = [np.ascontiguousarray(host_vertices)]
    elif host_vertices.ndim == 3 and host_vertices.shape[2] == 3:
        host_vertex_keys = [np.ascontiguousarray(key) for key in host_vertices]
    else:
        raise ValueError("vertices must have shape (N, 3) or (K, N, 3)")
    d_vertex_buffers = [
        wp.array(key, dtype=wp.float32, device=device) for key in host_vertex_keys
    ]

    host_indices = _checked_indices(indices, host_vertex_keys[0].shape[0], "indices")
    if host_indices.ndim != 2 or host_indices.shape[1] != 3:
        raise ValueError("indices must have shape (M, 3)")
    d_indices = wp.array(
        np.ascontiguousarray(host_indices), dtype=wp.uint32, device=device
    )
    motion_options = _motion_options_for_keys(
        optix, len(d_vertex_buffers), motion_time_range
    )

    tri = optix.BuildInputTriangleArray()
    tri.vertexFormat = optix.VERTEX_FORMAT_FLOAT3
    tri.numVertices = host_vertex_keys[0].shape[0]
    tri.vertexStrideInBytes = 12
    tri.vertexBuffers = [buffer.ptr for buffer in d_vertex_buffers]
    tri.indexFormat = optix.INDICES_FORMAT_UNSIGNED_INT3
    tri.numIndexTriplets = host_indices.shape[0]
    tri.indexStrideInBytes = 12
    tri.indexBuffer = d_indices.ptr
    tri.flags = [optix.GEOMETRY_FLAG_NONE]
    tri.numSbtRecords = 1

    if build_flags is None:
        build_flags = optix.BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS
    return _build_acceleration_structure(
        optix,
        ctx,
        [tri],
        {
            "d_vertices": d_vertex_buffers[0],
            "d_vertex_buffers": d_vertex_buffers,
            "d_indices": d_indices,
        },
        "d_gas",
        device,
        build_flags=build_flags,
        compact=compact,
        motion_options=motion_options,
    )


def create_custom_primitive_gas(
    optix,
    ctx,
    aabbs: np.ndarray,
    device: str,
    *,
    geometry_flags=None,
    build_flags=None,
    sbt_index_offsets: np.ndarray | None = None,
    num_sbt_records: int = 1,
    primitive_index_offset: int = 0,
    compact: bool = False,
):
    """Build a GAS for custom primitives from object-space AABBs.

    ``aabbs`` must contain one ``(min_x, min_y, min_z, max_x, max_y,
    max_z)`` record per primitive. It may be shaped ``(N, 6)`` or
    ``(N, 2, 3)``. Optional SBT index offsets select a hit-group record per
    primitive, following the normal OptiX custom-primitive build semantics.
    """
    host_aabbs = np.asarray(aabbs, dtype=np.float32)
    if host_aabbs.ndim == 3 and host_aabbs.shape[1:] == (2, 3):
        host_aabbs = host_aabbs.reshape((-1, 6))
    if host_aabbs.ndim != 2 or host_aabbs.shape[1] != 6:
        raise ValueError("aabbs must have shape (N, 6) or (N, 2, 3)")
    if host_aabbs.shape[0] == 0:
        raise ValueError("aabbs must contain at least one primitive")
    if not np.all(np.isfinite(host_aabbs)):
        raise ValueError("aabbs must contain only finite values")
    if np.any(host_aabbs[:, :3] > host_aabbs[:, 3:]):
        raise ValueError("each AABB minimum must be less than or equal to its maximum")
    host_aabbs = np.ascontiguousarray(host_aabbs)

    num_sbt_records = int(num_sbt_records)
    if num_sbt_records < 1:
        raise ValueError("num_sbt_records must be at least 1")

    if geometry_flags is None:
        flags = [int(optix.GEOMETRY_FLAG_NONE)] * num_sbt_records
    elif isinstance(geometry_flags, (int, np.integer)):
        flags = [int(geometry_flags)] * num_sbt_records
    else:
        flags = [int(flag) for flag in geometry_flags]
        if len(flags) != num_sbt_records:
            raise ValueError("geometry_flags must contain one flag per SBT record")

    d_aabbs = wp.array(host_aabbs, dtype=wp.float32, device=device)
    d_sbt_indices = None
    if sbt_index_offsets is not None:
        host_sbt_indices = np.asarray(sbt_index_offsets, dtype=np.uint32)
        if host_sbt_indices.shape != (host_aabbs.shape[0],):
            raise ValueError(
                "sbt_index_offsets must have shape (N,), matching the AABB count"
            )
        if np.any(host_sbt_indices >= num_sbt_records):
            raise ValueError(
                "sbt_index_offsets values must be less than num_sbt_records"
            )
        d_sbt_indices = wp.array(
            np.ascontiguousarray(host_sbt_indices), dtype=wp.uint32, device=device
        )

    custom = optix.BuildInputCustomPrimitiveArray(
        aabbBuffers=[d_aabbs.ptr],
        numPrimitives=host_aabbs.shape[0],
        strideInBytes=6 * np.dtype(np.float32).itemsize,
        flags=flags,
        numSbtRecords=num_sbt_records,
        sbtIndexOffsetBuffer=0 if d_sbt_indices is None else d_sbt_indices.ptr,
        sbtIndexOffsetSizeInBytes=0
        if d_sbt_indices is None
        else np.dtype(np.uint32).itemsize,
        sbtIndexOffsetStrideInBytes=0
        if d_sbt_indices is None
        else np.dtype(np.uint32).itemsize,
        primitiveIndexOffset=int(primitive_index_offset),
    )

    return _build_acceleration_structure(
        optix,
        ctx,
        [custom],
        {"d_aabbs": d_aabbs, "d_sbt_indices": d_sbt_indices},
        "d_gas",
        device,
        build_flags=optix.BUILD_FLAG_NONE if build_flags is None else build_flags,
        compact=compact,
    )


def create_curve_gas(
    optix,
    ctx,
    vertices: np.ndarray,
    widths: np.ndarray,
    segment_indices: np.ndarray,
    device: str,
    *,
    curve_type=None,
    geometry_flag=None,
    build_flags=None,
    compact: bool = False,
    motion_time_range=None,
):
    """Build a GAS of round linear, B-spline, Catmull-Rom, or Bezier curves."""
    if curve_type is None:
        curve_type = optix.PRIMITIVE_TYPE_ROUND_LINEAR

    host_vertices = np.asarray(vertices, dtype=np.float32)
    if host_vertices.ndim == 2 and host_vertices.shape[1] == 3:
        host_vertex_keys = [np.ascontiguousarray(host_vertices)]
    elif host_vertices.ndim == 3 and host_vertices.shape[2] == 3:
        host_vertex_keys = [np.ascontiguousarray(key) for key in host_vertices]
    else:
        raise ValueError("vertices must have shape (N, 3) or (K, N, 3)")
    num_keys = len(host_vertex_keys)

    host_widths = np.asarray(widths, dtype=np.float32)
    if host_widths.ndim == 1:
        if host_widths.shape[0] != host_vertex_keys[0].shape[0]:
            raise ValueError("widths must contain one value per curve vertex")
        host_width_keys = [np.ascontiguousarray(host_widths)] * num_keys
    elif host_widths.ndim == 2 and host_widths.shape == (
        num_keys,
        host_vertex_keys[0].shape[0],
    ):
        host_width_keys = [np.ascontiguousarray(key) for key in host_widths]
    else:
        raise ValueError("widths must have shape (N,) or (K, N), matching vertices")

    # Segment i reads control points i..i+degree, so the last valid start index
    # leaves room for the remaining control points of the segment.
    num_vertices = host_vertex_keys[0].shape[0]
    host_indices = _checked_indices(
        segment_indices,
        num_vertices - _curve_degree(optix, curve_type),
        "segment_indices",
    )
    if host_indices.ndim != 1 or host_indices.size == 0:
        raise ValueError("segment_indices must be a non-empty one-dimensional array")
    host_indices = np.ascontiguousarray(host_indices)

    d_vertex_buffers = [
        wp.array(key, dtype=wp.float32, device=device) for key in host_vertex_keys
    ]
    if len({id(key) for key in host_width_keys}) == 1:
        d_width = wp.array(host_width_keys[0], dtype=wp.float32, device=device)
        d_width_buffers = [d_width] * num_keys
    else:
        d_width_buffers = [
            wp.array(key, dtype=wp.float32, device=device) for key in host_width_keys
        ]
    d_indices = wp.array(host_indices, dtype=wp.uint32, device=device)
    motion_options = _motion_options_for_keys(optix, num_keys, motion_time_range)

    curve = optix.BuildInputCurveArray(
        curveType=curve_type,
        numPrimitives=host_indices.size,
        vertexBuffers=[buffer.ptr for buffer in d_vertex_buffers],
        numVertices=host_vertex_keys[0].shape[0],
        vertexStrideInBytes=3 * np.dtype(np.float32).itemsize,
        widthBuffers=[buffer.ptr for buffer in d_width_buffers],
        widthStrideInBytes=np.dtype(np.float32).itemsize,
        normalBuffers=[0] * num_keys,
        normalStrideInBytes=0,
        indexBuffer=d_indices.ptr,
        indexStrideInBytes=np.dtype(np.uint32).itemsize,
        flag=optix.GEOMETRY_FLAG_NONE if geometry_flag is None else geometry_flag,
        primitiveIndexOffset=0,
    )
    return _build_acceleration_structure(
        optix,
        ctx,
        [curve],
        {
            "d_vertex_buffers": d_vertex_buffers,
            "d_width_buffers": d_width_buffers,
            "d_indices": d_indices,
        },
        "d_gas",
        device,
        build_flags=optix.BUILD_FLAG_NONE if build_flags is None else build_flags,
        compact=compact,
        motion_options=motion_options,
    )


def create_instance_acceleration_structure(
    optix, ctx, instances, device: str, *, build_flags=None, compact: bool = False
):
    """Build an IAS from a non-empty sequence of ``optix.Instance`` objects."""
    instances = list(instances)
    if not instances:
        raise ValueError("instances must contain at least one optix.Instance")

    host_bytes = np.frombuffer(optix.getDeviceRepresentation(instances), dtype=np.uint8)
    d_instances = wp.array(host_bytes, dtype=wp.uint8, device=device)
    instance_input = optix.BuildInputInstanceArray(
        instances=d_instances.ptr, numInstances=len(instances)
    )
    return _build_acceleration_structure(
        optix,
        ctx,
        [instance_input],
        {"d_instances": d_instances},
        "d_ias",
        device,
        build_flags=optix.BUILD_FLAG_NONE if build_flags is None else build_flags,
        compact=compact,
    )


def _set_pipeline_stack_size(
    optix,
    pipeline,
    program_groups,
    max_trace_depth: int,
    max_traversable_depth: int,
    max_continuation_callable_depth: int = 0,
    max_direct_callable_depth: int = 0,
):
    if hasattr(optix, "util") and hasattr(optix, "StackSizes"):
        stack_sizes = optix.StackSizes()
        for group in program_groups:
            if _optix_version_at_least(optix, 7, 7):
                optix.util.accumulateStackSizes(group, stack_sizes, pipeline)
            else:
                optix.util.accumulateStackSizes(group, stack_sizes)
        direct_traversal, direct_state, continuation = optix.util.computeStackSizes(
            stack_sizes,
            max_trace_depth,
            max_continuation_callable_depth,
            max_direct_callable_depth,
        )
        pipeline.setStackSize(
            direct_traversal, direct_state, continuation, max_traversable_depth
        )
    else:
        pipeline.setStackSize(2 * 1024, 2 * 1024, 2 * 1024, max_traversable_depth)


def _primitive_type_flag(optix, primitive_type) -> int:
    names = (
        "ROUND_QUADRATIC_BSPLINE",
        "ROUND_CUBIC_BSPLINE",
        "ROUND_LINEAR",
        "ROUND_CATMULLROM",
        "FLAT_QUADRATIC_BSPLINE",
        "ROUND_CUBIC_BEZIER",
    )
    for name in names:
        if primitive_type == getattr(optix, f"PRIMITIVE_TYPE_{name}", None):
            return int(getattr(optix, f"PRIMITIVE_TYPE_FLAGS_{name}"))
    raise ValueError(
        f"Unsupported built-in intersection primitive type: {primitive_type}"
    )


def create_pipeline_and_sbt(
    optix,
    ctx,
    ptx: bytes,
    raygen_entry: wp.Kernel | str,
    miss_entry: wp.Kernel | str,
    closest_hit_entry: wp.Kernel | str | None,
    num_payload_values: int,
    num_attribute_values: int,
    device: str,
    intersection_entry: wp.Kernel | str | None = None,
    any_hit_entry: wp.Kernel | str | None = None,
    primitive_type_flags: int | None = None,
    hit_groups=None,
    traversable_graph_flags: int | None = None,
    max_traversable_depth: int | None = None,
    uses_motion_blur: bool = False,
    exception_entry: wp.Kernel | str | None = None,
    direct_callable_entries=None,
    continuation_callable_entries=None,
    exception_flags: int | None = None,
):
    from warp_optix._codegen import OptixKernelType  # noqa: PLC0415
    from warp_optix._runtime.hit_kernels import HitKernel  # noqa: PLC0415
    from warp_optix._runtime.sbt import SbtKernelManager  # noqa: PLC0415

    raygen_name = get_optix_entry_name(
        raygen_entry, expected_kernel_type=OptixKernelType.RAYGEN
    )
    miss_name = get_optix_entry_name(
        miss_entry, expected_kernel_type=OptixKernelType.MISS
    )

    if hit_groups is not None:
        if any(
            entry is not None
            for entry in (closest_hit_entry, intersection_entry, any_hit_entry)
        ):
            raise ValueError(
                "hit_groups cannot be combined with the individual hit entry arguments"
            )
        requested_hit_groups = list(hit_groups)
        if not all(isinstance(group, HitKernel) for group in requested_hit_groups):
            raise TypeError("hit_groups must contain HitKernel objects")
    else:
        requested_hit_groups = []
        if any(
            entry is not None
            for entry in (closest_hit_entry, intersection_entry, any_hit_entry)
        ):
            requested_hit_groups.append(
                HitKernel(closest_hit_entry, any_hit_entry, intersection_entry)
            )

    resolved_hit_groups = []
    for group in requested_hit_groups:
        closest_hit_name = (
            get_optix_entry_name(
                group.closest_hit, expected_kernel_type=OptixKernelType.CLOSEST_HIT
            )
            if group.closest_hit is not None
            else None
        )
        any_hit_name = (
            get_optix_entry_name(
                group.any_hit, expected_kernel_type=OptixKernelType.ANY_HIT
            )
            if group.any_hit is not None
            else None
        )
        intersection_name = (
            get_optix_entry_name(
                group.intersection, expected_kernel_type=OptixKernelType.INTERSECTION
            )
            if group.intersection is not None
            else None
        )
        if (
            intersection_name is not None
            and group.builtin_intersection_type is not None
        ):
            raise ValueError(
                "a hit group cannot combine custom and built-in intersection programs"
            )
        if not (
            closest_hit_name
            or any_hit_name
            or intersection_name
            or group.builtin_intersection_type is not None
        ):
            raise ValueError("each hit group must define at least one program")
        resolved_hit_groups.append(
            HitKernel(
                closest_hit_name,
                any_hit_name,
                intersection_name,
                builtin_intersection_type=group.builtin_intersection_type,
            )
        )

    if traversable_graph_flags is None:
        traversable_graph_flags = optix.TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS
    if max_traversable_depth is None:
        max_traversable_depth = (
            1
            if int(traversable_graph_flags)
            == int(optix.TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS)
            else 2
        )

    if exception_flags is None:
        exception_flags = (
            optix.EXCEPTION_FLAG_USER
            if exception_entry is not None
            else optix.EXCEPTION_FLAG_NONE
        )

    kwargs = {
        "usesMotionBlur": bool(uses_motion_blur),
        "traversableGraphFlags": int(traversable_graph_flags),
        "numPayloadValues": num_payload_values,
        "numAttributeValues": num_attribute_values,
        "exceptionFlags": int(exception_flags),
        "pipelineLaunchParamsVariableName": "params",
    }
    if _optix_version_at_least(optix, 7, 2):
        if primitive_type_flags is None:
            primitive_type_flags = 0
            for group in resolved_hit_groups:
                if group.builtin_intersection_type is not None:
                    primitive_type_flags |= _primitive_type_flag(
                        optix, group.builtin_intersection_type
                    )
                elif group.intersection is not None:
                    primitive_type_flags |= int(optix.PRIMITIVE_TYPE_FLAGS_CUSTOM)
                else:
                    primitive_type_flags |= int(optix.PRIMITIVE_TYPE_FLAGS_TRIANGLE)
            if not resolved_hit_groups:
                primitive_type_flags = optix.PRIMITIVE_TYPE_FLAGS_TRIANGLE
        kwargs["usesPrimitiveTypeFlags"] = int(primitive_type_flags)
    pipeline_options = optix.PipelineCompileOptions(**kwargs)
    module_options = optix.ModuleCompileOptions(
        maxRegisterCount=optix.COMPILE_DEFAULT_MAX_REGISTER_COUNT,
        optLevel=optix.COMPILE_OPTIMIZATION_DEFAULT,
        debugLevel=optix.COMPILE_DEBUG_LEVEL_DEFAULT,
    )
    module, log = ctx.moduleCreate(module_options, pipeline_options, ptx)
    if log:
        print(f"Module create log:\n{log}")

    sbt_manager = SbtKernelManager(optix, ctx, module)
    sbt_manager.set_raygen_kernel(raygen_name)
    if exception_entry is not None:
        sbt_manager.set_exception_kernel(exception_entry)
    sbt_manager.add_miss_kernels([miss_name])
    hit_group_handles = []
    builtin_modules = []
    for group in resolved_hit_groups:
        intersection_module = None
        if group.builtin_intersection_type is not None:
            builtin_options = optix.BuiltinISOptions(
                builtinISModuleType=group.builtin_intersection_type,
                usesMotionBlur=bool(uses_motion_blur),
            )
            intersection_module = ctx.builtinISModuleGet(
                module_options, pipeline_options, builtin_options
            )
            builtin_modules.append(intersection_module)
        hit_group_handles.append(
            sbt_manager.register_hit_shader_type(
                HitKernel(
                    group.closest_hit,
                    group.any_hit,
                    group.intersection,
                    builtin_intersection_type=group.builtin_intersection_type,
                    intersection_module=intersection_module,
                )
            )
        )

    direct_callable_handles = [
        sbt_manager.add_callable_kernel(entry)
        for entry in (direct_callable_entries or [])
    ]
    continuation_callable_handles = [
        sbt_manager.add_callable_kernel(entry, continuation=True)
        for entry in (continuation_callable_entries or [])
    ]

    groups = sbt_manager.get_all_program_groups()

    link_options = optix.PipelineLinkOptions()
    link_options.maxTraceDepth = 1
    pipeline = ctx.pipelineCreate(pipeline_options, link_options, groups, "")
    _set_pipeline_stack_size(
        optix,
        pipeline,
        groups,
        link_options.maxTraceDepth,
        int(max_traversable_depth),
        max_continuation_callable_depth=1 if continuation_callable_handles else 0,
        max_direct_callable_depth=1 if direct_callable_handles else 0,
    )

    sbt_resources = sbt_manager.build_sbt(device=device)
    keepalive = {
        "module": module,
        "program_groups": groups,
        "sbt": sbt_resources.keepalive,
        "sbt_manager": sbt_manager,
        "hit_group_handles": hit_group_handles,
        "builtin_modules": builtin_modules,
        "direct_callable_handles": direct_callable_handles,
        "continuation_callable_handles": continuation_callable_handles,
    }
    return pipeline, sbt_resources.sbt, keepalive
