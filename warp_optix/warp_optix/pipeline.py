# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OptiX pipelines, launch parameters, and launches."""

from __future__ import annotations

import ctypes
import hashlib
import logging
import operator
import os
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import warp as wp
from warp._src import context as _warp_context

from warp_optix import native
from warp_optix.accel import Accel
from warp_optix.addon import build_options
from warp_optix.context import CAPTURE_ERROR, native_objects, resolve
from warp_optix.enums import BuildFlags, ExceptionFlags, PrimitiveType, TraversableGraphFlags
from warp_optix.program import ProgramKind, as_program, params_struct
from warp_optix.sbt import SbtBuilder, SbtLayout

_log = logging.getLogger(__name__)
_BUILTIN_PRIMITIVE_FLAGS = {
    int(native.PRIMITIVE_TYPE_ROUND_LINEAR): int(native.PRIMITIVE_TYPE_FLAGS_ROUND_LINEAR),
    int(native.PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE): int(native.PRIMITIVE_TYPE_FLAGS_ROUND_QUADRATIC_BSPLINE),
    int(native.PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE): int(native.PRIMITIVE_TYPE_FLAGS_ROUND_CUBIC_BSPLINE),
    int(native.PRIMITIVE_TYPE_ROUND_CATMULLROM): int(native.PRIMITIVE_TYPE_FLAGS_ROUND_CATMULLROM),
    int(native.PRIMITIVE_TYPE_ROUND_CUBIC_BEZIER): int(native.PRIMITIVE_TYPE_FLAGS_ROUND_CUBIC_BEZIER),
    int(native.PRIMITIVE_TYPE_FLAT_QUADRATIC_BSPLINE): int(native.PRIMITIVE_TYPE_FLAGS_FLAT_QUADRATIC_BSPLINE),
    # Not exposed by PyOptiX 9.1; flag values from optix_types.h.
    PrimitiveType.ROUND_QUADRATIC_BSPLINE_ROCAPS: 1 << 8,
    PrimitiveType.ROUND_CUBIC_BSPLINE_ROCAPS: 1 << 9,
    PrimitiveType.ROUND_CATMULLROM_ROCAPS: 1 << 10,
    PrimitiveType.ROUND_CUBIC_BEZIER_ROCAPS: 1 << 11,
}


@dataclass(frozen=True)
class HitGroup:
    """Programs run for one geometry type and ray type.

    ``builtin`` selects an OptiX built-in intersection (a :class:`PrimitiveType`)
    for curves. OptiX specializes it on the curve structures' build flags, so
    ``build_flags`` must match the :class:`BuildFlags` those structures were built
    with; curves built with different flags need separate hit groups. A hit group
    without programs still occupies its SBT record, which is what pipelines that
    only use ``optix_traverse`` hit objects need.
    """

    closest_hit: wp.Kernel | None = None
    any_hit: wp.Kernel | None = None
    intersection: wp.Kernel | None = None
    builtin: int | None = None
    build_flags: int = BuildFlags.PREFER_FAST_TRACE

    def __post_init__(self):
        if self.builtin is None:
            return
        if self.intersection is not None:
            raise ValueError("HitGroup intersection and builtin cannot be combined")
        if int(self.builtin) not in _BUILTIN_PRIMITIVE_FLAGS:
            raise ValueError(f"HitGroup.builtin must be a curve PrimitiveType, got {self.builtin!r}")


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
    Direct and continuation callables share one SBT section: ``callables[k]``
    is index ``k`` and ``continuation_callables[k]`` is index ``len(callables) + k``.

    Pass ``sbt=SbtBuilder(...)`` or a snapshot for named ray types and hit
    groups. This cannot be combined with ``miss``, ``hit_groups``, or
    ``ray_types``; the pipeline keeps the snapshot as :attr:`sbt_layout`.

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
        if not miss:
            raise ValueError("miss must contain at least one entry (None for a record without a program)")
        if params_type is not None and not all(hasattr(params_type, name) for name in ("vars", "ctype")):
            raise TypeError("params_type must be a Warp struct type")
        context, _ = resolve(device)
        device = context.device

        with native_objects(context) as owned:
            # All programs compile into one Warp module that inherits the raygen module's options.
            params = params_struct(raygen) if params_type is None else params_type
            module_name = _module_name(params, raygen, miss, hit_groups, callables, continuation_callables, exception)

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
                (group.builtin, group.build_flags): context.native.builtinISModuleGet(
                    module_options,
                    pipeline_options,
                    native.BuiltinISOptions(
                        builtinISModuleType=native.PrimitiveType(int(group.builtin)),
                        usesMotionBlur=motion_blur,
                        buildFlags=int(group.build_flags),
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
                        slot("hitgroup", "IS", None, builtin_modules[group.builtin, group.build_flags])
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
            # Launches on other streams do not wait for this upload.
            wp.synchronize_stream(wp.get_stream(device))

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

        self._context = context
        self._native = pipeline
        self._owned = tuple(owned)
        self._sbt = sbt_table
        self._records = records
        self._params_size = ctypes.sizeof(params.ctype)
        self._param_names = tuple(arg.label for arg in raygen.adj.args)
        self._tid_ndim = raygen_program._wo_tid_ndim
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

    def _forget_idle_streams(self) -> None:
        """Drop the streams, and their parameter blocks, whose launches have all finished.

        Runs when a launch first uses a stream, so launching on a new stream each
        frame keeps only the streams with work in flight.
        """
        if self.device.is_capturing:
            return
        for key, stream in list(self._streams.items()):
            if stream.is_complete:
                del self._streams[key]
                self._buffers.pop(key, None)


class _ParamsStorage:
    """A device launch-parameter block, its host copy, and the inputs referenced by it.

    Parameters are packed into the host copy and copied to ``data`` in stream
    order. Holds no reference to its pipeline, so a pipeline's automatic storage
    does not form a reference cycle and ``del pipeline`` releases it at once.
    """

    __slots__ = ("allocated", "data", "dirty", "host", "inputs", "reset_host", "stream")

    def __init__(self, params_type, device: wp.Device):
        self.data = wp.empty(1, dtype=params_type, device=device)
        self.host = params_type()
        self.inputs: dict = {}
        self.reset_host = False
        self.dirty = False
        # The allocation is ordered on the current stream: the first use waits for it, not for later work there.
        self.allocated: wp.Event | None = wp.get_stream(device).record_event()
        self.stream: wp.Stream | None = None

    def pack(self, pipeline: Pipeline, context, values) -> None:
        """Pack ``values`` (the raygen arguments, or one ``params_type`` instance) into the host copy."""
        if pipeline._explicit_params:
            values = tuple(values) if isinstance(values, Sequence) else (values,)
            if len(values) != 1 or getattr(values[0], "_cls", None) is not pipeline.params_type:
                raise TypeError("A pipeline with params_type takes one instance of it as its parameters")
            context.check_arrays(values)
            # Field setters retain Python references in the host struct as well.
            # Discard those when replacing the entire parameter block.
            if self.reset_host:
                self.host = pipeline.params_type()
                self.reset_host = False
            ctypes.memmove(
                ctypes.addressof(self.host._ctype), ctypes.addressof(values[0]._ctype), pipeline._params_size
            )
            self.inputs = {name: getattr(values[0], name) for name in pipeline.params_type.vars}
        else:
            names = pipeline._param_names
            values = tuple(values)
            if len(values) != len(names):
                raise TypeError(f"The raygen program takes {len(names)} arguments, but {len(values)} were given")
            self.inputs = {}
            for name, value in zip(names, values):
                self.set(context, name, value)
        self.dirty = True

    def set(self, context, name: str, value) -> None:
        """Set one parameter; an ``Accel`` is passed as its id and retained."""
        retained = value
        if isinstance(value, Accel):
            value = wp.uint64(value.id)
        elif isinstance(value, wp.array):
            if value.device != context.device:
                raise ValueError("Array and OptiX resource must use the same CUDA device")
        elif wp.types.is_struct(value):
            context.check_arrays((value,))
            field = getattr(type(self.host._ctype), name)
            if value._cls is self.host._cls.vars[name].type:
                # Copy the bytes: assigning through the struct re-validates every nested field.
                ctypes.memmove(
                    ctypes.addressof(self.host._ctype) + field.offset, ctypes.addressof(value._ctype), field.size
                )
                self.inputs[name] = retained
                self.dirty = True
                return
        setattr(self.host, name, value)
        self.reset_host = True
        self.inputs[name] = retained
        self.dirty = True

    def use(self, stream: wp.Stream) -> None:
        """Order a use on ``stream`` after the previous use, and upload the host copy if it changed."""
        if self.stream is None:
            stream.wait_event(self.allocated)
            self.allocated = None
            self.stream = stream
        elif self.stream.cuda_stream != stream.cuda_stream:
            stream.wait_stream(self.stream)
            self.stream = stream
        if self.dirty:
            device = self.data.device
            # Copying from pageable memory returns once the bytes are staged, so the host copy can be reused.
            _warp_context.runtime.core.wp_memcpy_h2d(
                device.context,
                self.data.ptr,
                ctypes.addressof(self.host._ctype),
                self.data.capacity,
                stream.cuda_stream,
            )
            self.dirty = False


class Launch:
    """A recorded launch that replays quickly, like ``wp.launch(..., record_cmd=True)``.

    Created by ``launch(..., record_cmd=True)``. The parameters are checked and
    packed once and kept in this object's own parameter block; :meth:`launch`
    copies them to the device only after they change, then launches. Replays on
    different streams are ordered on the GPU; use separate ``Launch`` objects for
    independent concurrent launches. Like ``wp.array``, the parameter block is
    freed in stream order when the last reference is dropped. Set them
    with :meth:`set_param_at_index`, :meth:`set_param_by_name`, or
    :meth:`set_params`; names and indices are the launch-parameter fields (the
    raygen arguments, or the fields of ``params_type``).
    """

    def __init__(self, pipeline: Pipeline, dim, inputs, context):
        self.pipeline = pipeline
        self._context = context
        self._storage = _ParamsStorage(pipeline.params_type, context.device)
        self._fields = tuple(pipeline.params_type.vars)
        self.set_dim(dim)
        self.set_params(inputs)

    def set_dim(self, dim: int | Sequence[int]) -> None:
        """Set the launch dimensions, as in :func:`launch`."""
        self.dim = _launch_dim(dim, self.pipeline)
        self._size = [*reversed(self.dim), 1, 1][:3]

    def set_params(self, values: Sequence) -> None:
        """Set all parameters: the ``inputs`` of :func:`launch`."""
        self._storage.pack(self.pipeline, self._context, values)

    def set_param_at_index(self, index: int, value) -> None:
        """Set the parameter at ``index`` of the launch-parameter fields."""
        self.set_param_by_name(self._fields[index], value)

    def set_param_by_name(self, name: str, value) -> None:
        """Set the launch-parameter field ``name``."""
        if name not in self._fields:
            raise KeyError(f"The launch parameters have no field {name!r}")
        self._storage.set(self._context, name, value)

    def launch(self, stream: wp.Stream | None = None) -> None:
        """Launch on ``stream`` (default: the device's current stream)."""
        _launch(self.pipeline, self._context, self._storage, self._size, stream)


def launch(
    pipeline: Pipeline,
    dim: int | Sequence[int],
    inputs: Sequence | None = None,
    outputs: Sequence | None = None,
    *,
    device: wp.DeviceLike = None,
    stream: wp.Stream | None = None,
    record_cmd: bool = False,
) -> Launch | None:
    """Launch an OptiX pipeline, mirroring ``wp.launch``.

    Pass raygen arguments as ``inputs`` and ``outputs``, concatenated as in
    ``wp.launch``, or ``inputs=[struct]`` with an
    explicit ``params_type``. ``wo.Accel`` objects in ``inputs`` are passed as
    their ``id``. As with ``wp.launch``, keep resources alive while submitted
    work may use them. ``device`` defaults to the pipeline's device; a different
    device is rejected. Parameter storage is separate for each CUDA stream.
    With ``record_cmd=True``, nothing is launched; a :class:`Launch` is returned
    for fast replays. Calls from multiple host threads require external synchronization.
    """
    if not isinstance(pipeline, Pipeline):
        raise TypeError("pipeline must be a warp_optix.Pipeline")
    pipeline._check_open()
    if device is not None and wp.get_device(device) != pipeline.device:
        raise ValueError("Device and pipeline must be the same CUDA device")
    context = pipeline._context
    inputs = (*(inputs or ()), *(outputs or ()))
    if record_cmd:
        return Launch(pipeline, dim, inputs, context)
    size = [*reversed(_launch_dim(dim, pipeline)), 1, 1][:3]
    stream = _stream(context, stream)
    if stream.is_capturing:
        raise RuntimeError(CAPTURE_ERROR)
    if stream.cuda_stream not in pipeline._streams:
        pipeline._forget_idle_streams()
    storage = pipeline._buffers.get(stream.cuda_stream)
    if storage is None:
        storage = pipeline._buffers[stream.cuda_stream] = _ParamsStorage(pipeline.params_type, context.device)
    storage.pack(pipeline, context, inputs)
    _launch(pipeline, context, storage, size, stream)


def _launch_dim(dim, pipeline: Pipeline) -> tuple[int, ...]:
    try:
        dim = (operator.index(dim),) if not isinstance(dim, Sequence) else tuple(operator.index(n) for n in dim)
    except TypeError:
        raise ValueError("dim must contain one to three positive integers") from None
    if not 1 <= len(dim) <= 3 or any(n <= 0 for n in dim):
        raise ValueError("dim must contain one to three positive integers")
    if pipeline._tid_ndim and pipeline._tid_ndim != len(dim):
        raise ValueError(
            f"The raygen program unpacks {pipeline._tid_ndim} wp.tid() indices, but dim has {len(dim)} entries"
        )
    return dim


def _stream(context, stream: wp.Stream | None) -> wp.Stream:
    if stream is None:
        return context.device.stream
    if not isinstance(stream, wp.Stream):
        raise TypeError("stream must be a Warp Stream")
    if stream.device != context.device:
        raise ValueError("Stream and pipeline must be the same CUDA device")
    return stream


def _launch(pipeline: Pipeline, context, storage: _ParamsStorage, size, stream: wp.Stream | None) -> None:
    """Upload changed parameters and launch: the path every launch shares."""
    pipeline._check_open()
    stream = _stream(context, stream)
    if stream.is_capturing:
        raise RuntimeError(CAPTURE_ERROR)
    if context._releases:
        context.collect()
    if stream.cuda_stream not in pipeline._streams:
        pipeline._forget_idle_streams()
    storage.use(stream)
    args = (pipeline._native, stream.cuda_stream, storage.data.ptr, pipeline._params_size, pipeline._sbt, *size)
    device = context.device
    # OptiX launches into the current CUDA context; switch only when another one is current.
    if _warp_context.runtime.core.wp_cuda_context_get_current() == device.context:
        native.launch(*args)
    else:
        with wp.ScopedDevice(device):
            native.launch(*args)
    # Remembered so close() can wait for these launches before destroying the native pipeline.
    pipeline._streams[stream.cuda_stream] = stream


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
