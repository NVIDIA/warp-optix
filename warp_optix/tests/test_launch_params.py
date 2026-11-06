# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit parameter schemas, buffer ordering, and stage declarations."""

import inspect
import weakref
from typing import Any

import numpy as np
import pytest
import warp as wp

import warp_optix as wo
from warp_optix import pipeline as pipeline_module

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA unavailable")]


@wp.struct
class Params:
    value: wp.uint32
    output: wp.array[wp.uint32]


@wp.struct
class Payload:
    value: wp.uint32


@wp.kernel(module="unique")
def write(value: wp.uint32, output: wp.array[wp.uint32]):
    i = wp.tid()
    output[i] = value


@wo.kernel(kind=wo.ProgramKind.RAYGEN, module="unique")
def trace(output: wp.array[wp.uint32]):
    i = wp.tid()
    payload = Payload()
    wp.optix_trace(
        wp.uint64(0),
        wp.vec3(0.0),
        wp.vec3(0.0, 0.0, -1.0),
        1.0,
        payload,
        0.0,
        0.0,
        wp.uint32(255),
        wp.uint32(0),
        wp.uint32(0),
        wp.uint32(1),
        wp.uint32(0),
    )
    output[i] = payload.value


@wo.kernel(kind=wo.ProgramKind.MISS)
@wp.kernel(module="unique")
def miss(value: wp.uint32):
    wp.optix_set_payload(0, value)


def params(value, output):
    result = Params()
    result.value = wp.uint32(value)
    result.output = output
    return result


def test_schema_independent_of_raygen_and_automatic_explicit_equivalence():
    sbt = wo.SbtBuilder(["radiance"])
    sbt.set_miss("radiance", miss)
    pipeline = wo.Pipeline(trace, sbt=sbt, params_type=Params, payload_words=1)
    assert pipeline.params_type is Params
    first = wp.empty(64, dtype=wp.uint32, device=pipeline.device)
    second = wp.empty_like(first)
    wo.launch(pipeline, (64,), [params(37, first)])
    launch_params = wo.LaunchParams(pipeline)
    assert launch_params.pipeline is pipeline and launch_params.data.device == pipeline.device
    launch_params.assign(params(37, second))
    wo.launch(pipeline, (64,), params=launch_params)
    np.testing.assert_array_equal(first.numpy(), np.full(64, 37))
    np.testing.assert_array_equal(first.numpy(), second.numpy())


def test_automatic_storage_is_independent_between_streams():
    pipeline = wo.Pipeline(write)
    device = pipeline.device
    streams = [wp.Stream(device), wp.Stream(device)]
    outputs = []
    for value in range(20):
        stream = streams[value % 2]
        output = wp.empty(4096, dtype=wp.uint32, device=device)
        wo.launch(pipeline, (4096,), [value, output], stream=stream)
        outputs.append(output)
    assert len({b.data.ptr for b in pipeline._buffers.values()}) == 2
    wp.synchronize_device(device)
    for value, output in enumerate(outputs):
        np.testing.assert_array_equal(output.numpy(), np.full(4096, value))


def test_explicit_params_update_and_launch_across_streams():
    pipeline = wo.Pipeline(write)
    device = pipeline.device
    launch_params = wo.LaunchParams(pipeline)
    first, second = wp.Stream(device), wp.Stream(device)
    outputs = []
    for value in range(12):
        output = wp.empty(4096, dtype=wp.uint32, device=device)
        launch_params.assign([value, output], stream=first)
        wo.launch(pipeline, (4096,), params=launch_params, stream=second)
        outputs.append(output)
    wp.synchronize_device(device)
    for value, output in enumerate(outputs):
        np.testing.assert_array_equal(output.numpy(), np.full(4096, value))


def test_invalid_params_schemas_and_stage_placements():
    with pytest.raises(TypeError, match="kind must be a ProgramKind"):
        wo.kernel(write, kind="miss")
    with pytest.raises(TypeError, match="declares MISS.*RAYGEN"):
        wo.Pipeline(miss)
    with pytest.raises(TypeError, match="declares RAYGEN.*MISS"):
        wo.Pipeline(write, miss=[trace])
    with pytest.raises(TypeError, match="Warp struct"):
        wo.Pipeline(write, params_type=int)
    pipeline = wo.Pipeline(write, params_type=Params)
    other = wo.Pipeline(write)
    launch_params = wo.LaunchParams(pipeline)
    with pytest.raises(TypeError, match="Pipeline"):
        wo.LaunchParams(object())
    with pytest.raises(TypeError, match="Pipeline"):
        wo.launch(object(), (1,), [])
    with pytest.raises(ValueError, match="Assign launch parameters"):
        wo.launch(pipeline, (1,), params=launch_params)
    with pytest.raises(ValueError, match="exactly one"):
        wo.launch(pipeline, (1,), [], params=launch_params)
    with pytest.raises(ValueError, match="exactly one"):
        wo.launch(pipeline, (1,))
    with pytest.raises(ValueError, match="LaunchParams of this pipeline"):
        wo.launch(other, (1,), params=launch_params)
    with pytest.raises(ValueError, match="dim"):
        wo.launch(other, (0,), [1, wp.empty(1, dtype=wp.uint32, device="cuda")])
    with pytest.raises(ValueError, match="dim"):
        wo.launch(other, (1, 1, 1, 1), [1, wp.empty(1, dtype=wp.uint32, device="cuda")])
    with pytest.raises(ValueError, match="requires inputs"):
        wo.launch(pipeline, (1,), [])
    with pytest.raises(TypeError, match="instance"):
        launch_params.assign([])
    with pytest.raises(ValueError, match="same CUDA device"):
        launch_params.assign(params(1, wp.empty(1, dtype=wp.uint32, device="cpu")))
    pipeline.close()
    with pytest.raises(RuntimeError, match="closed"):
        launch_params.assign(Params())
    with pytest.raises(RuntimeError, match="closed"):
        wo.launch(pipeline, (1,), params=launch_params)


@wp.kernel(module="unique")
def spin(iterations: int, sink: wp.array[float]):
    x = float(0.0)
    for _ in range(iterations):
        x = wp.sin(x + 1.0)
    sink[0] = x


def test_dropping_launch_params_frees_storage_in_stream_order():
    pipeline = wo.Pipeline(write, params_type=Params)
    device = pipeline.device
    stream = wp.Stream(device)
    sink = wp.empty(1, dtype=float, device=device)
    # Warm up every kernel first: loading a CUDA module synchronizes the device.
    warm = wo.LaunchParams(pipeline)
    warm.assign(params(1, wp.empty(1, dtype=wp.uint32, device=device)), stream=stream)
    wo.launch(pipeline, (1,), params=warm, stream=stream)
    wp.launch(spin, 1, inputs=[1, sink], stream=stream)
    wp.synchronize_device(device)

    # Keep the stream busy (about one second) so the launch below is still pending.
    wp.launch(spin, 1, inputs=[4_000_000, sink], stream=stream)
    busy = stream.record_event()
    output = wp.empty(128, dtype=wp.uint32, device=device)
    launch_params = wo.LaunchParams(pipeline)
    storage_ref = weakref.ref(launch_params.data)
    launch_params.assign(params(53, output), stream=stream)
    wo.launch(pipeline, (128,), params=launch_params, stream=stream)
    assert not busy.is_complete, "busy kernel finished too early for this test"
    # Like a wp.array, the storage goes away with its last reference (no garbage collection
    # needed); Warp frees the device memory in stream order, so the pending launch still reads it.
    del launch_params
    assert storage_ref() is None
    wp.synchronize_stream(stream)
    np.testing.assert_array_equal(output.numpy(), np.full(128, 53))


def test_launch_params_close_and_with():
    pipeline = wo.Pipeline(write, params_type=Params)
    output = wp.zeros(16, dtype=wp.uint32, device=pipeline.device)
    with wo.LaunchParams(pipeline) as launch_params:
        storage_ref = weakref.ref(launch_params.data)
        launch_params.assign(params(7, output))
        wo.launch(pipeline, (16,), params=launch_params)
    assert launch_params.closed and "closed" in repr(launch_params)
    assert storage_ref() is None
    np.testing.assert_array_equal(output.numpy(), np.full(16, 7))
    launch_params.close()  # Idempotent.
    for use in (
        lambda: launch_params.data,
        lambda: launch_params.assign(params(1, output)),
        lambda: wo.launch(pipeline, (16,), params=launch_params),
    ):
        with pytest.raises(RuntimeError, match="closed"):
            use()
    with pytest.raises(RuntimeError, match="closed"):
        with launch_params:
            pass


def test_launches_record_no_events(monkeypatch):
    pipeline = wo.Pipeline(write)
    device = pipeline.device
    output = wp.empty(64, dtype=wp.uint32, device=device)
    explicit = wo.LaunchParams(pipeline)
    wo.launch(pipeline, (64,), [1, output])  # Warm-up.
    recorded = []
    original = wp.Stream.record_event

    def record_event(self, *args, **kwargs):
        recorded.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(wp.Stream, "record_event", record_event)
    for value in range(8):
        wo.launch(pipeline, (64,), [value, output])
        explicit.assign([value, output])
        wo.launch(pipeline, (64,), params=explicit)
    assert recorded == []
    np.testing.assert_array_equal(output.numpy(), np.full(64, 7))


def test_work_on_another_stream_does_not_wait_for_the_current_stream():
    pipeline = wo.Pipeline(write)
    device = pipeline.device
    side = wp.Stream(device)
    sink = wp.empty(1, dtype=float, device=device)
    output = wp.empty(1, dtype=wp.uint32, device=device)
    points = wp.zeros(3, dtype=wp.vec3, device=device)
    accel = wo.Accel(
        [wo.triangles(points, wp.array([0, 1, 2], dtype=wp.int32, device=device))], flags=wo.BuildFlags.ALLOW_UPDATE
    )
    launch_params = wo.LaunchParams(pipeline)
    wo.launch(pipeline, (1,), [1, output], stream=side)
    launch_params.assign([1, output], stream=side)
    wp.launch(spin, 1, inputs=[1, sink])
    wp.synchronize_device(device)

    # As with wp.launch(stream=...), OptiX work on a stream is not ordered after the current stream.
    wp.launch(spin, 1, inputs=[4_000_000, sink])
    busy = wp.get_stream(device).record_event()
    wo.launch(pipeline, (1,), [2, output], stream=side)
    launch_params.assign([3, output], stream=side)
    accel.refit(stream=side)
    wp.synchronize_stream(side)
    assert not busy.is_complete, "busy kernel finished too early for this test"
    wp.synchronize_device(device)


@wp.kernel(module="unique")
def write_reserved_names(p: wp.uint32, Params: wp.array[wp.uint32]):
    i = wp.tid()
    Params[i] = p


def test_raygen_arguments_may_use_any_name():
    output = wp.zeros(4, dtype=wp.uint32, device="cuda")
    wo.launch(wo.Pipeline(write_reserved_names), (4,), [9, output])
    np.testing.assert_array_equal(output.numpy(), np.full(4, 9))


def test_raygen_arguments_are_checked():
    pipeline = wo.Pipeline(write)
    output = wp.empty(4, dtype=wp.uint32, device="cuda")
    with pytest.raises(TypeError, match="takes 2 arguments, but 1 were given"):
        wo.launch(pipeline, (4,), [output])
    with pytest.raises(TypeError, match="expects dtype uint32"):
        wo.launch(pipeline, (4,), [1, wp.empty(4, dtype=float, device="cuda")])
    with pytest.raises(TypeError):
        wo.launch(pipeline, (4,), [1.5, output])


@wp.kernel(module="unique")
def generic_write(value: Any, output: wp.array2d[Any]):
    i, j = wp.tid()
    output[i, j] = value * type(value)(i * 10 + j)


@wp.kernel(module="unique")
def generic_miss(value: Any):
    wp.optix_set_payload(0, value + value)


@wp.kernel(module="unique")
def generic_array_miss(value: wp.array[Any]):
    value[0] = value[0]


def test_generic_programs_take_launch_parameter_types():
    output = wp.zeros(4, dtype=wp.uint32, device="cuda")
    wo.launch(wo.Pipeline(trace, miss=[generic_miss], params_type=Params, payload_words=1), (4,), [params(3, output)])
    np.testing.assert_array_equal(output.numpy(), np.full(4, 6))

    values = wp.zeros((2, 3), dtype=wp.float64, device="cuda")
    wo.launch(
        wo.Pipeline(wp.overload(generic_write, [wp.float64, wp.array2d[wp.float64]])), values.shape, [2.0, values]
    )
    np.testing.assert_array_equal(values.numpy(), [[0.0, 2.0, 4.0], [20.0, 22.0, 24.0]])

    with pytest.raises(TypeError, match="type mismatch"):
        wo.Pipeline(trace, miss=[generic_array_miss], params_type=Params)


@wp.kernel(module="unique")
def undefined_symbol(output: wp.array[wp.uint32]):
    i = wp.tid()
    output[i] = missing_symbol  # noqa: F821


def test_program_errors_point_at_kernel_source():
    lines, start = inspect.getsourcelines(undefined_symbol.func)
    line = start + next(n for n, text in enumerate(lines) if "missing_symbol" in text)
    with pytest.raises(wp.WarpCodegenKeyError, match="missing_symbol") as error:
        wo.Pipeline(undefined_symbol)
    assert f"{__file__}:{line}:" in str(error.value)


def test_equal_pipelines_share_one_module_name():
    @wp.kernel(module="unique")
    def other(value: wp.uint32, output: wp.array[wp.uint32]):
        i = wp.tid()
        output[i] = value + wp.uint32(1)

    def name(raygen, miss=(None,)):
        return pipeline_module._module_name(
            pipeline_module.params_struct(raygen), raygen, miss, [wo.HitGroup()], (), (), None
        )

    assert name(write) == name(write)
    assert name(write) != name(other)
    assert name(write) != name(write, miss=(miss,))
