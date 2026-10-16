# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OptiX contexts, pipelines, and launches."""

from __future__ import annotations

import ctypes
import itertools
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import warp as wp

from warp_optix import native
from warp_optix.accel import CAPTURE_ERROR
from warp_optix.addon import build_options
from warp_optix.enums import TraversableGraphFlags
from warp_optix.program import ProgramKind, as_program, params_struct
from warp_optix.sbt import SbtBuilder, SbtLayout

_log = logging.getLogger(__name__)
_PIPELINE_IDS = itertools.count()
_LOG_LEVELS = {1: logging.CRITICAL, 2: logging.ERROR, 3: logging.WARNING, 4: logging.INFO}
_BUILTIN_PRIMITIVE_FLAGS = {
    int(native.PRIMITIVE_TYPE_ROUND_LINEAR): int(native.PRIMITIVE_TYPE_FLAGS_ROUND_LINEAR),
    int(native.PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE): int(native.PRIMITIVE_TYPE_FLAGS_ROUND_QUADRATIC_BSPLINE),
    int(native.PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE): int(native.PRIMITIVE_TYPE_FLAGS_ROUND_CUBIC_BSPLINE),
    int(native.PRIMITIVE_TYPE_ROUND_CATMULLROM): int(native.PRIMITIVE_TYPE_FLAGS_ROUND_CATMULLROM),
    int(native.PRIMITIVE_TYPE_ROUND_CUBIC_BEZIER): int(native.PRIMITIVE_TYPE_FLAGS_ROUND_CUBIC_BEZIER),
    int(native.PRIMITIVE_TYPE_FLAT_QUADRATIC_BSPLINE): int(native.PRIMITIVE_TYPE_FLAGS_FLAT_QUADRATIC_BSPLINE),
}


def _log_callback(level: int, tag: str, message: str) -> None:
    _log.log(_LOG_LEVELS.get(level, logging.DEBUG), "[%s] %s", tag, message)


def create_context(device=None, *, validation: bool = False):
    """Create an OptiX device context on a Warp CUDA device; messages go to ``logging``."""
    options = native.DeviceContextOptions(
        logCallbackFunction=_log_callback,
        logCallbackLevel=4,
        validationMode=(
            native.DEVICE_CONTEXT_VALIDATION_MODE_ALL if validation else native.DEVICE_CONTEXT_VALIDATION_MODE_OFF
        ),
    )
    return native.deviceContextCreate(wp.get_device(device).context, options)


def destroy_context(ctx, device=None) -> None:
    """Wait for ``device`` and destroy ``ctx`` with every pipeline and module created from it.

    Keep acceleration structures and pipelines referenced until this returns; their
    device buffers may be freed afterwards.
    """
    wp.synchronize_device(device)
    ctx.destroy()


@dataclass(frozen=True)
class HitGroup:
    """Programs run for one geometry type and ray type.

    ``builtin`` selects an OptiX built-in intersection (a :class:`PrimitiveType`)
    for curves. A hit group without programs still occupies its SBT record, which
    is what pipelines that only use ``optix_traverse`` hit objects need.
    """

    closest_hit: wp.Kernel | None = None
    any_hit: wp.Kernel | None = None
    intersection: wp.Kernel | None = None
    builtin: int | None = None


@dataclass(frozen=True)
class Pipeline:
    """A linked OptiX pipeline, its shader binding table, and its launch parameters."""

    pipeline: object
    sbt: object
    params: wp.array
    params_size: int
    writer: wp.Kernel
    device: wp.Device
    records: wp.array
    groups: tuple
    modules: tuple
    sbt_layout: SbtLayout | None = None


def create_pipeline(
    ctx,
    raygen: wp.Kernel,
    *,
    miss: Sequence[wp.Kernel | None] | None = None,
    hit_groups: Sequence[HitGroup] | None = None,
    ray_types: int | None = None,
    sbt: SbtBuilder | SbtLayout | None = None,
    callables: Sequence[wp.Kernel] = (),
    continuation_callables: Sequence[wp.Kernel] = (),
    exception: wp.Kernel | None = None,
    payload_words: int = 0,
    attribute_words: int = 2,
    max_trace_depth: int = 1,
    max_traversable_depth: int = 2,
    graph_flags: int = TraversableGraphFlags.ALLOW_ANY,
    motion_blur: bool = False,
    max_registers: int = native.COMPILE_DEFAULT_MAX_REGISTER_COUNT,
    device=None,
) -> Pipeline:
    """Compile Warp kernels into an OptiX pipeline.

    ``raygen`` is an ordinary Warp kernel; its arguments become the launch
    parameters passed to :func:`launch`, and ``wp.tid()`` is the launch index.
    Every other program is a Warp kernel whose arguments are a subset of the
    raygen arguments, bound by name. ``None`` in ``miss`` is a miss record
    without a program.

    The hit record of geometry ``g`` and ray type ``r`` is
    ``hit_groups[g * ray_types + r]``; trace with ``sbt_stride=ray_types``.

    Pass ``sbt=SbtBuilder(...)`` or a snapshot for named ray types and hit
    groups. This cannot be combined with ``miss``, ``hit_groups``, or
    ``ray_types``. The pipeline retains the immutable snapshot as ``sbt_layout``;
    ``sbt`` on the returned pipeline remains the native shader binding table.
    """
    layout = None
    if sbt is not None:
        if any(value is not None for value in (miss, hit_groups, ray_types)):
            raise ValueError("sbt cannot be combined with miss, hit_groups, or ray_types")
        if not isinstance(sbt, (SbtBuilder, SbtLayout)):
            raise TypeError("sbt must be an SbtBuilder or SbtLayout")
        layout = sbt.snapshot() if isinstance(sbt, SbtBuilder) else sbt
        miss = layout._miss_entries
        hit_groups = tuple(group for row in layout._hit_kernels for group in row)
        ray_types = layout.trace_stride
    else:
        miss = (None,) if miss is None else tuple(miss)
        hit_groups = (HitGroup(),) if hit_groups is None else tuple(hit_groups)
        ray_types = 1 if ray_types is None else ray_types
    if not isinstance(ray_types, int) or isinstance(ray_types, bool) or not 1 <= ray_types <= 15:
        raise ValueError("ray_types must be an integer between 1 and 15")
    if len(hit_groups) % ray_types:
        raise ValueError("hit_groups must contain one group per geometry type and ray type")
    device = wp.get_device(device)
    if wp.get_stream(device).is_capturing:
        raise RuntimeError(CAPTURE_ERROR)

    # All programs compile into one Warp module that inherits the raygen module's options.
    params = params_struct(raygen)
    module_name = f"warp_optix_pipeline_{next(_PIPELINE_IDS)}"

    def program(kernel, kind):
        return None if kernel is None else as_program(kernel, kind, params, module_name)

    raygen_program = program(raygen, ProgramKind.RAYGEN)
    miss_programs = [program(k, ProgramKind.MISS) for k in miss]
    hit_programs = [
        (
            program(g.closest_hit, ProgramKind.CLOSEST_HIT),
            program(g.any_hit, ProgramKind.ANY_HIT),
            program(g.intersection, ProgramKind.INTERSECTION),
        )
        for g in hit_groups
    ]
    callable_programs = [program(k, ProgramKind.DIRECT_CALLABLE) for k in callables]
    continuation_programs = [program(k, ProgramKind.CONTINUATION_CALLABLE) for k in continuation_callables]
    exception_program = program(exception, ProgramKind.EXCEPTION)

    module = wp.get_module(module_name)
    wp.set_module_options(
        {**raygen.module.options, "enable_backward": False, "extra_build_options": build_options()}, module=module
    )
    module_dir = os.path.join(wp.config.kernel_cache_dir, "warp_optix", module_name)
    ptx = wp.compile_aot_module(module, device=device, module_dir=module_dir, use_ptx=True)[0].read_bytes()

    primitive_flags = 0
    for group in hit_groups:
        if group.builtin is not None:
            primitive_flags |= _BUILTIN_PRIMITIVE_FLAGS[int(group.builtin)]
        elif group.intersection is not None:
            primitive_flags |= int(native.PRIMITIVE_TYPE_FLAGS_CUSTOM)
        else:
            primitive_flags |= int(native.PRIMITIVE_TYPE_FLAGS_TRIANGLE)

    pipeline_options = native.PipelineCompileOptions(
        usesMotionBlur=motion_blur,
        traversableGraphFlags=int(graph_flags),
        numPayloadValues=payload_words,
        numAttributeValues=attribute_words,
        exceptionFlags=int(native.EXCEPTION_FLAG_USER if exception else native.EXCEPTION_FLAG_NONE),
        pipelineLaunchParamsVariableName="params",
        usesPrimitiveTypeFlags=primitive_flags or int(native.PRIMITIVE_TYPE_FLAGS_TRIANGLE),
    )
    module_options = native.ModuleCompileOptions(
        maxRegisterCount=max_registers,
        optLevel=native.COMPILE_OPTIMIZATION_DEFAULT,
        debugLevel=native.COMPILE_DEBUG_LEVEL_DEFAULT,
    )
    optix_module, log = ctx.moduleCreate(module_options, pipeline_options, ptx)
    if log:
        _log.debug("module %s: %s", module_name, log)
    builtin_modules = {
        group.builtin: ctx.builtinISModuleGet(
            module_options,
            pipeline_options,
            native.BuiltinISOptions(
                builtinISModuleType=native.PrimitiveType(int(group.builtin)), usesMotionBlur=motion_blur
            ),
        )
        for group in hit_groups
        if group.builtin is not None
    }

    # Keyword arguments of one ProgramGroupDesc slot. OptiX treats an empty entry
    # name without a module as "no program"; built-in intersections have a module only.
    def slot(prefix: str, suffix: str, kernel, module=optix_module) -> dict:
        if kernel is not None:
            return {f"{prefix}Module{suffix}": module, f"{prefix}EntryFunctionName{suffix}": kernel.get_mangled_name()}
        if module is not optix_module:
            return {f"{prefix}Module{suffix}": module}
        return {f"{prefix}EntryFunctionName{suffix}": ""}

    raygen_desc = native.ProgramGroupDesc(**slot("raygen", "", raygen_program))
    miss_descs = [native.ProgramGroupDesc(**slot("miss", "", k)) for k in miss_programs]
    hit_descs = [
        native.ProgramGroupDesc(
            **slot("hitgroup", "CH", ch),
            **slot("hitgroup", "AH", ah),
            **(
                slot("hitgroup", "IS", None, builtin_modules[group.builtin])
                if group.builtin is not None
                else slot("hitgroup", "IS", is_)
            ),
        )
        for group, (ch, ah, is_) in zip(hit_groups, hit_programs)
    ]
    callable_descs = [native.ProgramGroupDesc(**slot("callables", "DC", k)) for k in callable_programs]
    callable_descs += [native.ProgramGroupDesc(**slot("callables", "CC", k)) for k in continuation_programs]
    exception_descs = [native.ProgramGroupDesc(**slot("exception", "", exception_program))] if exception_program else []
    descs = [raygen_desc, *exception_descs, *miss_descs, *hit_descs, *callable_descs]
    groups, log = ctx.programGroupCreate(descs, native.ProgramGroupOptions())
    if log:
        _log.debug("program groups %s: %s", module_name, log)

    link_options = native.PipelineLinkOptions()
    link_options.maxTraceDepth = max_trace_depth
    pipeline = ctx.pipelineCreate(pipeline_options, link_options, groups, "")
    stack_sizes = native.StackSizes()
    for group in groups:
        native.util.accumulateStackSizes(group, stack_sizes, pipeline)
    pipeline.setStackSize(
        *native.util.computeStackSizes(
            stack_sizes, max_trace_depth, 1 if continuation_callables else 0, 1 if callables else 0
        ),
        max_traversable_depth,
    )

    # One device buffer holds every header-only SBT record, in descriptor order.
    stride = -(-native.SBT_RECORD_HEADER_SIZE // native.SBT_RECORD_ALIGNMENT) * native.SBT_RECORD_ALIGNMENT
    host_records = np.zeros((len(groups), stride), dtype=np.uint8)
    for group, record in zip(groups, host_records):
        native.sbtRecordPackHeader(group, record)
    records = wp.array(host_records.reshape(-1), dtype=wp.uint8, device=device)

    first_miss = records.ptr + (1 + len(exception_descs)) * stride
    first_hit = first_miss + len(miss_descs) * stride
    first_callable = first_hit + len(hit_descs) * stride
    sbt = native.ShaderBindingTable(
        raygenRecord=records.ptr,
        exceptionRecord=records.ptr + stride if exception_descs else 0,
        missRecordBase=first_miss,
        missRecordStrideInBytes=stride,
        missRecordCount=len(miss_descs),
        hitgroupRecordBase=first_hit if hit_descs else 0,
        hitgroupRecordStrideInBytes=stride,
        hitgroupRecordCount=len(hit_descs),
        callablesRecordBase=first_callable if callable_descs else 0,
        callablesRecordStrideInBytes=stride,
        callablesRecordCount=len(callable_descs),
    )

    # launch() stores the raygen arguments into the device params block with a
    # generated kernel, so Warp checks and packs them exactly as for wp.launch.
    names = [arg.label for arg in raygen.adj.args]
    arguments = ", ".join(f"{n}: T{i}" for i, n in enumerate(names))
    stores = "".join(f"    p.{n} = {n}\n" for n in names)
    source = f"def write_params({arguments}, _wo_params: wp.array(dtype=Params)):\n    p = Params()\n{stores}    _wo_params[0] = p\n"
    scope = {"wp": wp, "Params": params} | {f"T{i}": arg.type for i, arg in enumerate(raygen.adj.args)}
    namespace: dict = {}
    exec(compile(source, f"<warp_optix params {raygen.key}>", "exec"), scope, namespace)  # noqa: S102
    writer = wp.Kernel(
        namespace["write_params"], key="write_params", module=wp.get_module(f"{module_name}_params"), source=source
    )

    return Pipeline(
        pipeline=pipeline,
        sbt=sbt,
        params=wp.empty(1, dtype=params, device=device),
        params_size=ctypes.sizeof(params.ctype),
        writer=writer,
        device=device,
        records=records,
        groups=tuple(groups),
        modules=(optix_module, *builtin_modules.values()),
        sbt_layout=layout,
    )


def launch(pipeline: Pipeline, dim: Sequence[int], inputs: Sequence, stream: wp.Stream | None = None) -> None:
    """Launch ``pipeline`` like ``wp.launch(raygen, dim, inputs)``, on ``stream``.

    Parameters are written on the stream before the launch, so consecutive launches
    may reuse the same pipeline without synchronizing.
    """
    stream = stream or wp.get_stream(pipeline.device)
    if stream.is_capturing:
        raise RuntimeError(CAPTURE_ERROR)
    wp.launch(pipeline.writer, dim=1, inputs=[*inputs, pipeline.params], device=pipeline.device, stream=stream)
    width, height, depth = [*reversed(dim), 1, 1][:3]
    native.launch(
        pipeline.pipeline,
        stream.cuda_stream,
        pipeline.params.ptr,
        pipeline.params_size,
        pipeline.sbt,
        width,
        height,
        depth,
    )
