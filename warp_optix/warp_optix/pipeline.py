# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OptiX pipelines, launch parameters, and launches."""

from __future__ import annotations

import ctypes
import itertools
import logging
import os
from collections.abc import Sequence

import numpy as np
import warp as wp

from warp_optix import native
from warp_optix.accel import Accel
from warp_optix.addon import build_options
from warp_optix.context import CAPTURE_ERROR, _arrays, native_objects, resolve
from warp_optix.enums import ExceptionFlags, TraversableGraphFlags
from warp_optix.program import ProgramKind, as_program, params_struct
from warp_optix.sbt import Record, SbtLayout

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


class Pipeline:
    """Warp kernels compiled into a linked OptiX pipeline with its shader binding table.

    ``raygen`` is an ordinary Warp kernel; its arguments become the launch
    parameters passed to :func:`launch`, and ``wp.tid()`` is the launch index.
    Every other program is a Warp kernel whose arguments are a subset of the
    raygen arguments, bound by name. With ``params_type=MyStruct``, all stages
    instead bind any subset of that struct's fields, and launch takes
    ``inputs=[my_struct_instance]``.

    ``sbt`` is an :class:`SbtLayout` describing the ray types, miss records, and
    hit groups; the pipeline keeps it as :attr:`sbt_layout`.
    ``wo.Record(kernel, data=...)`` in the miss records or callables, and
    ``HitGroup(data=...)``, store a Warp struct instance in the record;
    :meth:`update_sbt_data` uploads later changes to those instances.

    Release is automatic when the last reference is dropped, or explicit with
    :meth:`close` or ``with``. Device memory is freed in stream order, like
    ``wp.array``; the native OptiX objects are destroyed once the launches
    submitted before the release have finished.
    """

    def __init__(
        self,
        raygen: wp.Kernel,
        *,
        sbt: SbtLayout | None = None,
        callables: Sequence[wp.Kernel | Record] = (),
        continuation_callables: Sequence[wp.Kernel | Record] = (),
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
        layout = SbtLayout() if sbt is None else sbt
        if not isinstance(layout, SbtLayout):
            raise TypeError("sbt must be a warp_optix.SbtLayout")
        miss = list(layout.miss.values())
        hit_groups = [group for per_ray in layout.hit_groups.values() for group in per_ray.values()]
        if params_type is not None and not all(hasattr(params_type, name) for name in ("vars", "ctype")):
            raise TypeError("params_type must be a Warp struct type")
        miss, miss_data = _split_records(miss)
        callables, callable_data = _split_records(callables)
        continuation_callables, continuation_data = _split_records(continuation_callables)
        hit_data = [group.data for group in hit_groups]
        record_data = [*miss_data, *hit_data, *callable_data, *continuation_data]
        if any(data is not None and not _is_struct(data) for data in record_data):
            raise TypeError("SBT record data must be a Warp struct instance")
        context, _ = resolve(device)
        device = context.device
        context.check_arrays(record_data)

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

            # One device buffer holds every SBT record in descriptor order: the header, then
            # the record data. Each section's stride fits its largest record.
            header = native.SBT_RECORD_HEADER_SIZE
            sections = [[None], [None] * len(exception_descs), miss_data, hit_data, callable_data + continuation_data]
            strides = [_record_stride(section) for section in sections]
            bases = list(itertools.accumulate((len(s) * n for s, n in zip(sections, strides)), initial=0))
            offsets = [base + i * n for s, n, base in zip(sections, strides, bases) for i in range(len(s))]
            host_records = np.zeros(bases[-1], dtype=np.uint8)
            for group, offset in zip(groups, offsets):
                native.sbtRecordPackHeader(group, host_records[offset : offset + header])
            data_offsets = [
                (offset + header, data)
                for offset, data in zip(offsets, [data for section in sections for data in section])
                if data is not None
            ]
            _pack_record_data(host_records, data_offsets)
            records = wp.array(host_records, dtype=wp.uint8, device=device)

            _, exception_base, miss_base, hit_base, callable_base, _ = (records.ptr + base for base in bases)
            sbt_table = native.ShaderBindingTable(
                raygenRecord=records.ptr,
                exceptionRecord=exception_base if exception_descs else 0,
                missRecordBase=miss_base,
                missRecordStrideInBytes=strides[2],
                missRecordCount=len(miss_descs),
                hitgroupRecordBase=hit_base if hit_descs else 0,
                hitgroupRecordStrideInBytes=strides[3],
                hitgroupRecordCount=len(hit_descs),
                callablesRecordBase=callable_base if callable_descs else 0,
                callablesRecordStrideInBytes=strides[4],
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
        self._host_records = host_records
        self._data_offsets = data_offsets
        self._data_arrays = tuple(_arrays(record_data))
        self._params_size = ctypes.sizeof(params.ctype)
        self._writer = writer
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
    def sbt_layout(self) -> SbtLayout:
        """The SBT layout the pipeline was built from."""
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
        self._data_arrays = ()

        def destroy():
            for obj in reversed(owned):
                obj.destroy()

        self._context.release(destroy, streams)

    def update_sbt_data(self, stream: wp.Stream | None = None) -> None:
        """Upload the current field values of every record's ``data`` struct instance.

        The copy runs on ``stream`` like ``wp.copy`` from host memory, ordered with
        launches on that stream. Arrays referenced by the data stay alive with the pipeline.
        """
        self._check_open()
        context, stream = resolve(self.device, stream)
        data = [data for _, data in self._data_offsets]
        context.check_arrays(data)
        _pack_record_data(self._host_records, self._data_offsets)
        wp.copy(self._records, wp.array(self._host_records, dtype=wp.uint8, device="cpu"), stream=stream)
        self._data_arrays = tuple(_arrays(data))

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

    __slots__ = ("data", "initialized", "inputs", "stream")

    def __init__(self, params_type, device: wp.Device):
        self.data = wp.empty(1, dtype=params_type, device=device)
        self.inputs: tuple = ()
        self.initialized = False
        self.stream: wp.Stream | None = None

    def use(self, stream: wp.Stream) -> None:
        """Order a use on ``stream`` after the previous use when that was on another stream."""
        previous = self.stream
        if previous is not None and previous.cuda_stream != stream.cuda_stream:
            stream.wait_stream(previous)
        self.stream = stream

    def write(self, pipeline: Pipeline, context, stream: wp.Stream, values) -> None:
        if pipeline._explicit_params:
            if getattr(values, "_cls", None) is not pipeline.params_type:
                raise TypeError("values must be an instance of the pipeline params_type")
            inputs = (values,)
        else:
            inputs = tuple(values)
        inputs, accels = _replace_accels(inputs)
        context.check_arrays(inputs)
        with wp.ScopedDevice(context.device), wp.ScopedStream(stream):
            self.use(stream)
            wp.launch(pipeline._writer, dim=1, inputs=[*inputs, self.data], device=context.device, stream=stream)
        self.inputs = (*inputs, *accels)
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
    with wp.ScopedDevice(context.device), wp.ScopedStream(stream):
        storage.use(stream)
        width, height, depth = [*reversed(dim), 1, 1][:3]
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


def _split_records(entries) -> tuple[list, list]:
    """Split miss or callable entries into their programs and record data."""
    pairs = [(e.program, e.data) if isinstance(e, Record) else (e, None) for e in entries]
    return [program for program, _ in pairs], [data for _, data in pairs]


def _is_struct(value) -> bool:
    return hasattr(getattr(value, "_cls", None), "vars") and hasattr(value, "__ctype__")


def _record_stride(section: list) -> int:
    size = native.SBT_RECORD_HEADER_SIZE + max(
        (ctypes.sizeof(d._cls.ctype) for d in section if d is not None), default=0
    )
    return -(-size // native.SBT_RECORD_ALIGNMENT) * native.SBT_RECORD_ALIGNMENT


def _pack_record_data(host_records: np.ndarray, data_offsets) -> None:
    for offset, data in data_offsets:
        packed = bytes(data.__ctype__())
        host_records[offset : offset + len(packed)] = np.frombuffer(packed, dtype=np.uint8)


def _replace_accels(values: tuple) -> tuple[tuple, tuple]:
    """Replace top-level ``Accel`` objects by their ids; return the values and the objects."""
    accels = tuple(value for value in values if isinstance(value, Accel))
    if not accels:
        return values, ()
    return tuple(wp.uint64(value.id) if isinstance(value, Accel) else value for value in values), accels
