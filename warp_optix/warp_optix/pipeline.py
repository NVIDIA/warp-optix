# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OptiX pipelines, launch parameters, and launches."""

from __future__ import annotations

import ctypes
import hashlib
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import warp as wp

from warp_optix import native
from warp_optix.accel import Accel
from warp_optix.addon import build_options
from warp_optix.context import CAPTURE_ERROR, native_objects, resolve
from warp_optix.enums import CompileDebugLevel, CompileOptimizationLevel, ExceptionFlags, TraversableGraphFlags
from warp_optix.program import ProgramKind, as_program, params_struct
from warp_optix.sbt import SbtBuilder, SbtLayout

_log = logging.getLogger(__name__)
# Warp module options every pipeline sets on its programs' module.
_PIPELINE_MODULE_OPTIONS = ("enable_backward", "extra_build_options")
_BUILTIN_PRIMITIVE_FLAGS = {
    int(native.PRIMITIVE_TYPE_ROUND_LINEAR): int(native.PRIMITIVE_TYPE_FLAGS_ROUND_LINEAR),
    int(native.PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE): int(native.PRIMITIVE_TYPE_FLAGS_ROUND_QUADRATIC_BSPLINE),
    int(native.PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE): int(native.PRIMITIVE_TYPE_FLAGS_ROUND_CUBIC_BSPLINE),
    int(native.PRIMITIVE_TYPE_ROUND_CATMULLROM): int(native.PRIMITIVE_TYPE_FLAGS_ROUND_CATMULLROM),
    int(native.PRIMITIVE_TYPE_ROUND_CUBIC_BEZIER): int(native.PRIMITIVE_TYPE_FLAGS_ROUND_CUBIC_BEZIER),
    int(native.PRIMITIVE_TYPE_FLAT_QUADRATIC_BSPLINE): int(native.PRIMITIVE_TYPE_FLAGS_FLAT_QUADRATIC_BSPLINE),
}


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


class Pipeline:
    """Warp kernels compiled into a linked OptiX pipeline with its shader binding table.

    ``raygen`` is an ordinary Warp kernel; its arguments become the launch
    parameters passed to :func:`launch`, and ``wp.tid()`` is the launch index.
    Every other program is a Warp kernel whose arguments are a subset of the
    raygen arguments, bound by name. With ``params_type=MyStruct``, all stages
    instead bind any subset of that struct's fields, and launch takes
    ``inputs=[my_struct_instance]``. ``None`` in ``miss`` is a miss record
    without a program.

    The hit record of geometry ``g`` and ray type ``r`` is
    ``hit_groups[g * ray_types + r]``; trace with ``sbt_stride=ray_types``.

    Pass ``sbt=SbtBuilder(...)`` or a snapshot for named ray types and hit
    groups. This cannot be combined with ``miss``, ``hit_groups``, or
    ``ray_types``; the pipeline keeps the snapshot as :attr:`sbt_layout`.

    All programs compile into one Warp module. ``module_options`` sets its Warp
    module options; any option not given there must be equal in the Warp
    modules of all the program kernels.

    Release is automatic when the last reference is dropped, or explicit with
    :meth:`close` or ``with``. Device memory is freed in stream order, like
    ``wp.array``; the native OptiX objects are destroyed once the launches
    submitted before the release have finished.
    """

    def __init__(
        self,
        raygen: wp.Kernel,
        *,
        miss: Sequence[wp.Kernel | None] | None = None,
        hit_groups: Sequence[HitGroup] | None = None,
        ray_types: int | None = None,
        sbt: SbtBuilder | SbtLayout | None = None,
        callables: Sequence[wp.Kernel] = (),
        continuation_callables: Sequence[wp.Kernel] = (),
        exception: wp.Kernel | None = None,
        exception_flags: int | None = None,
        params_type: type | None = None,
        payload_words: int = 0,
        attribute_words: int = 2,
        max_trace_depth: int = 1,
        max_traversable_depth: int = 2,
        max_callable_depth: int | None = None,
        graph_flags: int = TraversableGraphFlags.ALLOW_ANY,
        motion_blur: bool = False,
        max_registers: int = native.COMPILE_DEFAULT_MAX_REGISTER_COUNT,
        optimization_level: int = CompileOptimizationLevel.DEFAULT,
        debug_level: int = CompileDebugLevel.DEFAULT,
        module_options: dict | None = None,
        device: wp.DeviceLike = None,
    ):
        self._closed = True
        layout, miss, hit_groups, ray_types = _sbt_entries(sbt, miss, hit_groups, ray_types)
        if params_type is not None and not all(hasattr(params_type, name) for name in ("vars", "ctype")):
            raise TypeError("params_type must be a Warp struct type")
        if max_callable_depth is None:
            max_callable_depth = 1 if callables or continuation_callables else 0
        context, _ = resolve(device)
        device = context.device

        params = params_struct(raygen) if params_type is None else params_type
        slots = [
            (raygen, ProgramKind.RAYGEN),
            *((k, ProgramKind.MISS) for k in miss),
            *(
                slot
                for g in hit_groups
                for slot in (
                    (g.closest_hit, ProgramKind.CLOSEST_HIT),
                    (g.any_hit, ProgramKind.ANY_HIT),
                    (g.intersection, ProgramKind.INTERSECTION),
                )
            ),
            *((k, ProgramKind.DIRECT_CALLABLE) for k in callables),
            *((k, ProgramKind.CONTINUATION_CALLABLE) for k in continuation_callables),
            (exception, ProgramKind.EXCEPTION),
        ]
        slots = [(kernel, kind) for kernel, kind in slots if kernel is not None]
        module_name = _module_name(params, raygen, miss, hit_groups, callables, continuation_callables, exception)
        options = _warp_module_options([kernel for kernel, _ in slots], module_options)
        ptx, entries = _compile_ptx(slots, params, module_name, options, device)

        with native_objects(context) as owned:
            pipeline_options = native.PipelineCompileOptions(
                usesMotionBlur=motion_blur,
                traversableGraphFlags=int(graph_flags),
                numPayloadValues=payload_words,
                numAttributeValues=attribute_words,
                exceptionFlags=int(
                    (ExceptionFlags.USER if exception else ExceptionFlags.NONE)
                    if exception_flags is None
                    else exception_flags
                ),
                pipelineLaunchParamsVariableName="params",
                usesPrimitiveTypeFlags=_primitive_flags(hit_groups),
            )
            compile_options = native.ModuleCompileOptions(
                maxRegisterCount=max_registers,
                optLevel=native.CompileOptimizationLevel(int(optimization_level)),
                debugLevel=native.CompileDebugLevel(int(debug_level)),
            )
            optix_module, log = context.native.moduleCreate(compile_options, pipeline_options, ptx)
            owned.append(optix_module)
            if log:
                _log.debug("module %s: %s", module_name, log)
            # Built-in intersection modules belong to the device context; they are not destroyed with the pipeline.
            builtin_modules = {
                group.builtin: context.native.builtinISModuleGet(
                    compile_options,
                    pipeline_options,
                    native.BuiltinISOptions(
                        builtinISModuleType=native.PrimitiveType(int(group.builtin)), usesMotionBlur=motion_blur
                    ),
                )
                for group in hit_groups
                if group.builtin is not None
            }

            descs = _program_group_descs(
                optix_module,
                builtin_modules,
                entries,
                raygen,
                miss,
                hit_groups,
                callables,
                continuation_callables,
                exception,
            )
            groups, log = context.native.programGroupCreate(descs, native.ProgramGroupOptions())
            owned.extend(groups)
            if log:
                _log.debug("program groups %s: %s", module_name, log)

            link_options = native.PipelineLinkOptions()
            link_options.maxTraceDepth = max_trace_depth
            pipeline = context.native.pipelineCreate(pipeline_options, link_options, groups, "")
            owned.append(pipeline)
            stack_sizes = native.StackSizes()
            for group in groups:
                native.util.accumulateStackSizes(group, stack_sizes, pipeline)
            pipeline.setStackSize(
                *native.util.computeStackSizes(
                    stack_sizes,
                    max_trace_depth,
                    max_callable_depth if continuation_callables else 0,
                    max_callable_depth if callables else 0,
                ),
                max_traversable_depth,
            )

            records, sbt_table = _shader_binding_table(
                groups,
                device,
                exception=exception is not None,
                miss_count=len(miss),
                hit_count=len(hit_groups),
            )

        self._context = context
        self._native = pipeline
        self._owned = tuple(owned)
        self._sbt = sbt_table
        self._records = records
        self._params_size = ctypes.sizeof(params.ctype)
        self._param_names = tuple(arg.label for arg in raygen.adj.args)
        self._params_type = params
        self._explicit_params = params_type is not None
        self._sbt_layout = layout
        self._buffers: dict[int, _ParamsStorage] = {}  # automatic parameter storage per CUDA stream
        self._streams: dict[int, wp.Stream] = {}  # streams launched on, by CUDA stream handle
        self._closed = False

    @property
    def device(self) -> wp.Device:
        return self._context.device

    @property
    def params_type(self):
        """The Warp struct type of the launch-parameter block."""
        return self._params_type

    @property
    def sbt_layout(self) -> SbtLayout | None:
        """The SBT snapshot the pipeline was built from, if ``sbt=`` was given."""
        return self._sbt_layout

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        """Make the pipeline unusable and release it; native objects outlive launches already submitted."""
        if self._closed:
            return
        self._closed = True
        owned = self._owned
        streams = tuple(self._streams.values())
        self._buffers = {}
        self._streams = {}
        self._owned = ()
        self._records = None

        def destroy():
            for obj in reversed(owned):
                obj.destroy()

        self._context.release(destroy, streams)

    def __enter__(self) -> Pipeline:
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
        return f"Pipeline({'closed, ' if self._closed else ''}device={self.device.alias!r})"

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("The pipeline is closed")


class _ParamsStorage:
    """A device launch-parameter block and the inputs written into it.

    Holds no reference to its pipeline, so a pipeline's automatic storage does
    not form a reference cycle and ``del pipeline`` releases it at once.
    """

    __slots__ = ("data", "host", "initialized", "inputs", "staging", "stream")

    def __init__(self, params_type, device: wp.Device):
        self.data = wp.empty(1, dtype=params_type, device=device)
        # Parameters are packed on the host and copied into ``data`` in stream order.
        self.host = params_type()
        self.staging = wp.array(ptr=ctypes.addressof(self.host._ctype), dtype=params_type, shape=(1,), device="cpu")
        self.inputs: tuple = ()
        self.initialized = False
        # The allocation above is ordered on the current stream; the first use on another stream waits for it.
        self.stream: wp.Stream = wp.get_stream(device)

    def use(self, stream: wp.Stream) -> None:
        """Order a use on ``stream`` after the previous use when that was on another stream."""
        if self.stream.cuda_stream != stream.cuda_stream:
            stream.wait_stream(self.stream)
        self.stream = stream

    def write(self, pipeline: Pipeline, context, stream: wp.Stream, values) -> None:
        if pipeline._explicit_params:
            if getattr(values, "_cls", None) is not pipeline.params_type:
                raise TypeError("values must be an instance of the pipeline params_type")
            context.check_arrays((values,))
            ctypes.memmove(ctypes.addressof(self.host._ctype), ctypes.addressof(values._ctype), pipeline._params_size)
            inputs = (values,)
        else:
            inputs = tuple(values)
            names = pipeline._param_names
            if len(inputs) != len(names):
                raise TypeError(f"The raygen program takes {len(names)} arguments, but {len(inputs)} were given")
            inputs, accels = _replace_accels(inputs)
            context.check_arrays(inputs)
            for name, value in zip(names, inputs):
                setattr(self.host, name, value)
            inputs = (*inputs, *accels)
        self.use(stream)
        wp.copy(self.data, self.staging, stream=stream)
        self.inputs = inputs
        self.initialized = True


class LaunchParams:
    """Launch-parameter storage for one pipeline, written with :meth:`assign`.

    Pass it as ``launch(pipeline, dim, params=...)`` to launch with preset
    parameters. The device comes from the pipeline. Uses on different streams
    are ordered on the GPU; use separate objects for independent concurrent
    launches. Writing ``data`` directly bypasses that ordering.

    Release is automatic when the last reference is dropped, or explicit with
    :meth:`close` or ``with``; memory is freed in stream order, like ``wp.array``.
    """

    def __init__(self, pipeline: Pipeline):
        if not isinstance(pipeline, Pipeline):
            raise TypeError("LaunchParams needs a warp_optix.Pipeline")
        pipeline._check_open()
        if wp.get_stream(pipeline.device).is_capturing:
            raise RuntimeError(CAPTURE_ERROR)
        self.pipeline = pipeline
        self._storage: _ParamsStorage | None = _ParamsStorage(pipeline.params_type, pipeline.device)

    @property
    def data(self) -> wp.array:
        """The device parameter block (one ``params_type`` element)."""
        return self._open_storage().data

    @property
    def closed(self) -> bool:
        return self._storage is None

    def assign(self, values, stream: wp.Stream | None = None) -> None:
        """Write a params-struct instance, or the raygen arguments as a sequence.

        Waits on prior uses of this object on other streams on the GPU, never on
        the host. Arrays and acceleration structures in ``values`` stay referenced
        until replaced or until this object is released.
        """
        storage = self._open_storage()
        pipeline = self.pipeline
        pipeline._check_open()
        context, stream = resolve(pipeline.device, stream)
        storage.write(pipeline, context, stream, values)

    def close(self) -> None:
        """Make the parameters unusable and release their memory and referenced inputs."""
        self._storage = None

    def __enter__(self) -> LaunchParams:
        self._open_storage()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self):
        return f"LaunchParams({'closed, ' if self.closed else ''}device={self.pipeline.device.alias!r})"

    def _open_storage(self) -> _ParamsStorage:
        if self._storage is None:
            raise RuntimeError("The launch parameters are closed")
        return self._storage


def launch(
    pipeline: Pipeline,
    dim: int | Sequence[int],
    inputs: Sequence | None = None,
    device: wp.DeviceLike = None,
    stream: wp.Stream | None = None,
    *,
    params: LaunchParams | None = None,
) -> None:
    """Launch an OptiX pipeline, mirroring ``wp.launch``.

    Pass the raygen arguments as ``inputs`` (or ``inputs=[struct]`` with an
    explicit ``params_type``), or preset :class:`LaunchParams` as ``params``.
    ``wo.Accel`` objects in ``inputs`` are passed as their ``id``. As with
    ``wp.launch``, keep resources alive while submitted work may use them.
    ``device`` defaults to the pipeline's device; a different device is
    rejected. Automatic parameter storage is separate for each CUDA stream.
    Calls from multiple host threads require external synchronization.
    """
    if not isinstance(pipeline, Pipeline):
        raise TypeError("pipeline must be a warp_optix.Pipeline")
    pipeline._check_open()
    if device is not None and wp.get_device(device) != pipeline.device:
        raise ValueError("Device and pipeline must be the same CUDA device")
    context, stream = resolve(pipeline.device, stream)
    dim = (dim,) if isinstance(dim, int) else tuple(dim)
    if not 1 <= len(dim) <= 3 or any(not isinstance(n, int) or n <= 0 for n in dim):
        raise ValueError("dim must contain one to three positive integers")
    if (inputs is None) == (params is None):
        raise ValueError("Provide exactly one of inputs or params")
    key = stream.cuda_stream
    if params is not None:
        if not isinstance(params, LaunchParams) or params.pipeline is not pipeline:
            raise ValueError("params must be LaunchParams of this pipeline")
        storage = params._open_storage()
        if not storage.initialized:
            raise ValueError("Assign launch parameters before launching with them")
    else:
        inputs = tuple(inputs)
        storage = pipeline._buffers.get(key)
        if storage is None:
            storage = pipeline._buffers[key] = _ParamsStorage(pipeline.params_type, context.device)
        if pipeline._explicit_params:
            if len(inputs) != 1:
                raise ValueError("Explicit params_type requires inputs=[struct_instance]")
            storage.write(pipeline, context, stream, inputs[0])
        else:
            storage.write(pipeline, context, stream, inputs)
    storage.use(stream)
    width, height, depth = [*reversed(dim), 1, 1][:3]
    with wp.ScopedDevice(context.device):
        native.launch(
            pipeline._native,
            stream.cuda_stream,
            storage.data.ptr,
            pipeline._params_size,
            pipeline._sbt,
            width,
            height,
            depth,
        )
    # Remembered so close() can wait for these launches before destroying the native pipeline.
    pipeline._streams[key] = stream


def _sbt_entries(sbt, miss, hit_groups, ray_types) -> tuple[SbtLayout | None, tuple, tuple, int]:
    """Return the SBT snapshot, miss kernels, hit groups, and ray-type count of a pipeline."""
    layout = None
    if sbt is not None:
        if any(value is not None for value in (miss, hit_groups, ray_types)):
            raise ValueError("sbt cannot be combined with miss, hit_groups, or ray_types")
        if not isinstance(sbt, (SbtBuilder, SbtLayout)):
            raise TypeError("sbt must be an SbtBuilder or SbtLayout")
        layout = sbt.snapshot() if isinstance(sbt, SbtBuilder) else sbt
        miss = tuple(layout._miss_entries)
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
    return layout, miss, hit_groups, ray_types


def _warp_module_options(kernels: Sequence[wp.Kernel], overrides: dict | None) -> dict:
    """Return the Warp module options for compiling ``kernels`` into one pipeline module.

    Options in ``overrides`` win; every other option must agree across the
    kernels' Warp modules.
    """
    overrides = dict(overrides or {})
    defaults = kernels[0].module.options
    unknown = sorted(set(overrides) - set(defaults))
    if unknown:
        raise ValueError(f"Unknown Warp module options: {', '.join(unknown)}")
    reserved = sorted(set(overrides) & set(_PIPELINE_MODULE_OPTIONS))
    if reserved:
        raise ValueError(f"Pipelines set the Warp module options {', '.join(reserved)} themselves")
    modules = {id(kernel.module): kernel for kernel in kernels}
    for name, value in defaults.items():
        if name in overrides or name in _PIPELINE_MODULE_OPTIONS:
            continue
        for kernel in modules.values():
            other = kernel.module.options.get(name)
            if other != value:
                raise ValueError(
                    f"Warp module option {name!r} is {value!r} for kernel {kernels[0].key!r} but {other!r} for "
                    f"kernel {kernel.key!r}; choose one with Pipeline(module_options={{{name!r}: ...}})"
                )
    return {**defaults, **overrides, "enable_backward": False, "extra_build_options": build_options()}


def _compile_ptx(slots, params, module_name: str, options: dict, device: wp.Device) -> tuple[bytes, dict]:
    """Compile each ``(kernel, kind)`` slot as an OptiX program into one PTX module.

    Returns the PTX and the entry-function name of each slot. The Warp module
    exists only for this compile: it is never registered with Warp, and its
    dependency links are removed afterwards, so nothing retains it.
    """
    module = wp.Module(module_name)
    try:
        programs = {}
        for kernel, kind in slots:
            if (kernel, kind) not in programs:
                programs[kernel, kind] = as_program(kernel, kind, params, module)
        wp.set_module_options(options, module=module)
        module_dir = os.path.join(wp.config.kernel_cache_dir, "warp_optix", module_name)
        ptx = wp.compile_aot_module(module, device=device, module_dir=module_dir, use_ptx=True)[0].read_bytes()
        return ptx, {slot: program.get_mangled_name() for slot, program in programs.items()}
    finally:
        module._detach_references()


def _primitive_flags(hit_groups) -> int:
    flags = 0
    for group in hit_groups:
        if group.builtin is not None:
            flags |= _BUILTIN_PRIMITIVE_FLAGS[int(group.builtin)]
        elif group.intersection is not None:
            flags |= int(native.PRIMITIVE_TYPE_FLAGS_CUSTOM)
        else:
            flags |= int(native.PRIMITIVE_TYPE_FLAGS_TRIANGLE)
    return flags or int(native.PRIMITIVE_TYPE_FLAGS_TRIANGLE)


def _program_group_descs(
    optix_module, builtin_modules, entries, raygen, miss, hit_groups, callables, continuation_callables, exception
) -> list:
    """Return the program group descriptors in SBT order: raygen, exception, miss, hit groups, callables."""

    # Keyword arguments of one ProgramGroupDesc slot. OptiX treats an empty entry
    # name without a module as "no program"; built-in intersections have a module only.
    def slot(prefix: str, suffix: str, kernel, kind, module=optix_module) -> dict:
        if kernel is not None:
            return {f"{prefix}Module{suffix}": module, f"{prefix}EntryFunctionName{suffix}": entries[kernel, kind]}
        if module is not optix_module:
            return {f"{prefix}Module{suffix}": module}
        return {f"{prefix}EntryFunctionName{suffix}": ""}

    descs = [native.ProgramGroupDesc(**slot("raygen", "", raygen, ProgramKind.RAYGEN))]
    if exception is not None:
        descs.append(native.ProgramGroupDesc(**slot("exception", "", exception, ProgramKind.EXCEPTION)))
    descs += [native.ProgramGroupDesc(**slot("miss", "", k, ProgramKind.MISS)) for k in miss]
    descs += [
        native.ProgramGroupDesc(
            **slot("hitgroup", "CH", g.closest_hit, ProgramKind.CLOSEST_HIT),
            **slot("hitgroup", "AH", g.any_hit, ProgramKind.ANY_HIT),
            **(
                slot("hitgroup", "IS", None, None, builtin_modules[g.builtin])
                if g.builtin is not None
                else slot("hitgroup", "IS", g.intersection, ProgramKind.INTERSECTION)
            ),
        )
        for g in hit_groups
    ]
    descs += [native.ProgramGroupDesc(**slot("callables", "DC", k, ProgramKind.DIRECT_CALLABLE)) for k in callables]
    descs += [
        native.ProgramGroupDesc(**slot("callables", "CC", k, ProgramKind.CONTINUATION_CALLABLE))
        for k in continuation_callables
    ]
    return descs


def _shader_binding_table(groups, device: wp.Device, *, exception: bool, miss_count: int, hit_count: int):
    """Upload one header-only SBT record per program group, in descriptor order; return it and the table."""
    stride = -(-native.SBT_RECORD_HEADER_SIZE // native.SBT_RECORD_ALIGNMENT) * native.SBT_RECORD_ALIGNMENT
    host_records = np.zeros((len(groups), stride), dtype=np.uint8)
    for group, record in zip(groups, host_records):
        native.sbtRecordPackHeader(group, record)
    records = wp.array(host_records.reshape(-1), dtype=wp.uint8, device=device)
    # Launches on other streams do not wait for this upload.
    wp.synchronize_stream(wp.get_stream(device))

    first_miss = records.ptr + (2 if exception else 1) * stride
    first_hit = first_miss + miss_count * stride
    first_callable = first_hit + hit_count * stride
    callable_count = len(groups) - (first_callable - records.ptr) // stride
    table = native.ShaderBindingTable(
        raygenRecord=records.ptr,
        exceptionRecord=records.ptr + stride if exception else 0,
        missRecordBase=first_miss,
        missRecordStrideInBytes=stride,
        missRecordCount=miss_count,
        hitgroupRecordBase=first_hit if hit_count else 0,
        hitgroupRecordStrideInBytes=stride,
        hitgroupRecordCount=hit_count,
        callablesRecordBase=first_callable if callable_count else 0,
        callablesRecordStrideInBytes=stride,
        callablesRecordCount=callable_count,
    )
    return records, table


def _module_name(params, raygen, miss, hit_groups, callables, continuation_callables, exception) -> str:
    """Name the Warp module of a pipeline after its programs and parameters.

    Equal pipelines get equal names, and so equal generated code, which lets
    Warp's kernel cache and OptiX's module cache serve repeated pipelines.
    """

    def identity(kernel):
        return None if kernel is None else (kernel.module.name, kernel.key)

    key = (
        params.native_name,
        identity(raygen),
        [identity(k) for k in miss],
        [(identity(g.closest_hit), identity(g.any_hit), identity(g.intersection)) for g in hit_groups],
        [identity(k) for k in callables],
        [identity(k) for k in continuation_callables],
        identity(exception),
    )
    return f"warp_optix_{hashlib.sha256(repr(key).encode()).hexdigest()[:16]}"


def _replace_accels(values: tuple) -> tuple[tuple, tuple]:
    """Replace top-level ``Accel`` objects by their ids; return the values and the objects."""
    accels = tuple(value for value in values if isinstance(value, Accel))
    if not accels:
        return values, ()
    return tuple(wp.uint64(value.id) if isinstance(value, Accel) else value for value in values), accels
