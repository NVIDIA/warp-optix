# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OptiX pipelines, launch parameters, and launches."""

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
from warp_optix.accel import Accel
from warp_optix.addon import build_options
from warp_optix.context import CAPTURE_ERROR, native_objects, resolve
from warp_optix.enums import ExceptionFlags, TraversableGraphFlags
from warp_optix.program import ProgramKind, as_program, params_struct
from warp_optix.sbt import SbtBuilder, SbtLayout

_log = logging.getLogger(__name__)
_PIPELINE_IDS = itertools.count()
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

    Release is automatic (``__del__``) or explicit with :meth:`close` or ``with``;
    either way it is deferred until submitted OptiX work has finished.
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
        graph_flags: int = TraversableGraphFlags.ALLOW_ANY,
        motion_blur: bool = False,
        max_registers: int = native.COMPILE_DEFAULT_MAX_REGISTER_COUNT,
        device: wp.DeviceLike = None,
    ):
        self._closed = True
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
        if params_type is not None and not all(hasattr(params_type, name) for name in ("vars", "ctype")):
            raise TypeError("params_type must be a Warp struct type")
        context, _ = resolve(device)
        device = context.device

        with native_objects(context) as owned:
            # All programs compile into one Warp module that inherits the raygen module's options.
            params = params_struct(raygen) if params_type is None else params_type
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
                {**raygen.module.options, "enable_backward": False, "extra_build_options": build_options()},
                module=module,
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
                exceptionFlags=int(
                    (ExceptionFlags.USER if exception else ExceptionFlags.NONE)
                    if exception_flags is None
                    else exception_flags
                ),
                pipelineLaunchParamsVariableName="params",
                usesPrimitiveTypeFlags=primitive_flags or int(native.PRIMITIVE_TYPE_FLAGS_TRIANGLE),
            )
            module_options = native.ModuleCompileOptions(
                maxRegisterCount=max_registers,
                optLevel=native.COMPILE_OPTIMIZATION_DEFAULT,
                debugLevel=native.COMPILE_DEBUG_LEVEL_DEFAULT,
            )
            optix_module, log = context.native.moduleCreate(module_options, pipeline_options, ptx)
            owned.append(optix_module)
            if log:
                _log.debug("module %s: %s", module_name, log)
            # Built-in intersection modules belong to the device context; they are not destroyed with the pipeline.
            builtin_modules = {
                group.builtin: context.native.builtinISModuleGet(
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
                    return {
                        f"{prefix}Module{suffix}": module,
                        f"{prefix}EntryFunctionName{suffix}": kernel.get_mangled_name(),
                    }
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
            exception_descs = (
                [native.ProgramGroupDesc(**slot("exception", "", exception_program))] if exception_program else []
            )
            descs = [raygen_desc, *exception_descs, *miss_descs, *hit_descs, *callable_descs]
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
            sbt_table = native.ShaderBindingTable(
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

            # LaunchParams.assign() stores the raygen arguments into the device params block with a
            # generated kernel, so Warp checks and packs them exactly as for wp.launch.
            names = [arg.label for arg in raygen.adj.args]
            arguments = ", ".join(f"{n}: T{i}" for i, n in enumerate(names))
            stores = "".join(f"    p.{n} = {n}\n" for n in names)
            arguments = f"{arguments}, " if arguments else ""
            source = f"def write_params({arguments}_wo_params: wp.array(dtype=Params)):\n    p = Params()\n{stores}    _wo_params[0] = p\n"
            if params_type is not None:
                source = (
                    "def write_params(value: Params, _wo_params: wp.array(dtype=Params)):\n    _wo_params[0] = value\n"
                )
            scope = {"wp": wp, "Params": params} | {f"T{i}": arg.type for i, arg in enumerate(raygen.adj.args)}
            namespace: dict = {}
            exec(compile(source, f"<warp_optix params {raygen.key}>", "exec"), scope, namespace)  # noqa: S102
            writer = wp.Kernel(
                namespace["write_params"],
                key="write_params",
                module=wp.get_module(f"{module_name}_params"),
                source=source,
            )

        self._context = context
        self._native = pipeline
        self._owned = tuple(owned)
        self._sbt = sbt_table
        self._records = records
        self._params_size = ctypes.sizeof(params.ctype)
        self._writer = writer
        self._params_type = params
        self._explicit_params = params_type is not None
        self._sbt_layout = layout
        self._buffers: dict[int, LaunchParams] = {}  # automatic parameter storage per CUDA stream
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
        """Make the pipeline unusable and destroy it once submitted work has finished."""
        if self._closed:
            return
        self._closed = True
        owned = self._owned
        references = (self._records, tuple(self._buffers.values()))
        self._buffers = {}
        self._owned = ()

        def destroy():
            for obj in reversed(owned):
                obj.destroy()

        self._context.release(destroy, references)

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


class LaunchParams:
    """Launch-parameter storage for one pipeline, written with :meth:`assign`.

    Pass it as ``launch(pipeline, dim, params=...)`` to launch with preset
    parameters. The device comes from the pipeline. Reuse across streams is
    ordered with CUDA events; use separate objects for independent concurrent
    launches. Writing ``data`` directly bypasses that ordering.
    """

    def __init__(self, pipeline: Pipeline):
        if not isinstance(pipeline, Pipeline):
            raise TypeError("LaunchParams needs a warp_optix.Pipeline")
        pipeline._check_open()
        if wp.get_stream(pipeline.device).is_capturing:
            raise RuntimeError(CAPTURE_ERROR)
        self.pipeline = pipeline
        self.data = wp.empty(1, dtype=pipeline.params_type, device=pipeline.device)
        self._event = None
        self._inputs: tuple = ()
        self._initialized = False

    def assign(self, values, stream: wp.Stream | None = None) -> None:
        """Write a params-struct instance, or the raygen arguments as a sequence.

        Waits on prior uses of this object on the GPU, never on the host. Arrays and
        acceleration structures in ``values`` stay referenced until replaced.
        """
        pipeline = self.pipeline
        pipeline._check_open()
        context, stream = resolve(pipeline.device, stream)
        if pipeline._explicit_params:
            if getattr(values, "_cls", None) is not pipeline.params_type:
                raise TypeError("values must be an instance of the pipeline params_type")
            inputs = (values,)
        else:
            inputs = tuple(values)
        inputs, accels = _replace_accels(inputs)
        context.check_arrays(inputs)
        with wp.ScopedDevice(context.device), wp.ScopedStream(stream):
            if self._event is not None:
                stream.wait_event(self._event)
            wp.launch(pipeline._writer, dim=1, inputs=[*inputs, self.data], device=pipeline.device, stream=stream)
            self._inputs = (*inputs, *accels)
            self._initialized = True
            self._event = stream.record_event()
        context.retain(stream, (self.data, self._inputs))


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
    ``wo.Accel`` objects in ``inputs`` are passed as their ``id`` and kept alive
    for the launch. ``device`` defaults to the pipeline's device; a different
    device is rejected. Automatic parameter storage is separate for each CUDA
    stream. Calls from multiple host threads require external synchronization.
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
    if params is not None:
        if not isinstance(params, LaunchParams) or params.pipeline is not pipeline:
            raise ValueError("params must be LaunchParams of this pipeline")
        if not params._initialized:
            raise ValueError("Assign launch parameters before launching with them")
        buffer = params
    else:
        inputs = tuple(inputs)
        key = stream.cuda_stream
        buffer = pipeline._buffers.get(key)
        if buffer is None:
            buffer = pipeline._buffers[key] = LaunchParams(pipeline)
        if pipeline._explicit_params:
            if len(inputs) != 1:
                raise ValueError("Explicit params_type requires inputs=[struct_instance]")
            buffer.assign(inputs[0], stream)
        else:
            buffer.assign(inputs, stream)
    with wp.ScopedDevice(context.device), wp.ScopedStream(stream):
        if buffer._event is not None:
            stream.wait_event(buffer._event)
        width, height, depth = [*reversed(dim), 1, 1][:3]
        native.launch(
            pipeline._native,
            stream.cuda_stream,
            buffer.data.ptr,
            pipeline._params_size,
            pipeline._sbt,
            width,
            height,
            depth,
        )
        buffer._event = stream.record_event()
    context.retain(stream, (pipeline, buffer, buffer.data, buffer._inputs))


def _replace_accels(values: tuple) -> tuple[tuple, tuple]:
    """Replace top-level ``Accel`` objects by their ids; return the values and the objects."""
    accels = tuple(value for value in values if isinstance(value, Accel))
    if not accels:
        return values, ()
    return tuple(wp.uint64(value.id) if isinstance(value, Accel) else value for value in values), accels
