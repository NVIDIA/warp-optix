# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit parameter schemas, buffer ordering, and stage declarations."""

import weakref

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


def test_schema_independent_of_raygen_and_recorded_equivalence():
    sbt = wo.SbtBuilder(["radiance"])
    sbt.set_miss("radiance", miss)
    pipeline = wo.Pipeline(trace, sbt=sbt, params_type=Params, payload_words=1)
    assert pipeline.params_type is Params
    first = wp.empty(64, dtype=wp.uint32, device=pipeline.device)
    second = wp.empty_like(first)
    wo.launch(pipeline, (64,), [params(37, first)])
    wo.launch(pipeline, (64,), [params(37, second)], record_cmd=True).launch()
    np.testing.assert_array_equal(first.numpy(), np.full(64, 37))
    np.testing.assert_array_equal(first.numpy(), second.numpy())


def test_launches_across_streams_use_their_own_parameters():
    pipeline = wo.Pipeline(write)
    device = pipeline.device
    streams = [wp.Stream(device), wp.Stream(device)]
    recorded = wo.launch(pipeline, (4096,), [0, wp.empty(1, dtype=wp.uint32, device=device)], record_cmd=True)
    outputs = []
    for value in range(12):
        stream = streams[value % 2]
        automatic, replayed = (wp.empty(4096, dtype=wp.uint32, device=device) for _ in range(2))
        # Automatic storage is separate per stream; a recorded launch orders its updates across streams.
        wo.launch(pipeline, (4096,), [value, automatic], stream=stream)
        recorded.set_params([value, replayed])
        recorded.launch(stream=stream)
        outputs += [(value, automatic), (value, replayed)]
    assert set(pipeline._buffers) <= {stream.cuda_stream for stream in streams}
    wp.synchronize_device(device)
    for value, output in outputs:
        np.testing.assert_array_equal(output.numpy(), np.full(4096, value))


def test_explicit_parameter_replacements_release_previous_arrays():
    pipeline = wo.Pipeline(write, params_type=Params)
    initial = wp.empty(8, dtype=wp.uint32, device=pipeline.device)
    initial_ref = weakref.ref(initial)
    recorded = wo.launch(pipeline, 8, [params(1, initial)], record_cmd=True)
    del initial
    assert initial_ref() is not None

    # Updating another field must preserve the output's keepalive.
    recorded.set_param_by_name("value", 2)
    assert initial_ref() is not None
    replacement = wp.empty(8, dtype=wp.uint32, device=pipeline.device)
    replacement_ref = weakref.ref(replacement)
    recorded.set_param_by_name("output", replacement)
    del replacement
    assert initial_ref() is None
    assert replacement_ref() is not None

    output = wp.empty(8, dtype=wp.uint32, device=pipeline.device)
    recorded.set_params([params(3, output)])
    assert replacement_ref() is None
    recorded.launch()
    np.testing.assert_array_equal(output.numpy(), np.full(8, 3))


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
    with pytest.raises(TypeError, match="Pipeline"):
        wo.launch(object(), (1,), [])
    with pytest.raises(ValueError, match="dim"):
        wo.launch(other, (0,), [1, wp.empty(1, dtype=wp.uint32, device="cuda")])
    with pytest.raises(ValueError, match="dim"):
        wo.launch(other, (1, 1, 1, 1), [1, wp.empty(1, dtype=wp.uint32, device="cuda")])
    with pytest.raises(TypeError, match="takes one instance"):
        wo.launch(pipeline, (1,))
    with pytest.raises(ValueError, match="same CUDA device"):
        wo.launch(pipeline, (1,), [params(1, wp.empty(1, dtype=wp.uint32, device="cpu"))])
    recorded = wo.launch(pipeline, (1,), [params(1, wp.empty(1, dtype=wp.uint32, device="cuda"))], record_cmd=True)
    pipeline.close()
    with pytest.raises(RuntimeError, match="closed"):
        recorded.launch()


def test_dropping_a_recorded_launch_frees_its_parameters_in_stream_order(gate):
    pipeline = wo.Pipeline(write, params_type=Params)
    device = pipeline.device
    stream = wp.Stream(device)
    output = wp.empty(128, dtype=wp.uint32, device=device)
    # Warm-up: loading a CUDA module synchronizes the device.
    wo.launch(pipeline, (128,), [params(1, output)], stream=stream)
    wp.synchronize_device(device)

    pending = gate.block(stream)
    recorded = wo.launch(pipeline, (128,), [params(53, output)], record_cmd=True)
    storage_ref = weakref.ref(recorded._storage.data)
    recorded.launch(stream=stream)
    # Like a wp.array, the parameters go away with their last reference (no garbage collection
    # needed); Warp frees the device memory in stream order, so the pending launch still reads it.
    assert not pending.is_complete, "The gate expired before launch parameter release"
    del recorded
    assert storage_ref() is None
    gate.open()
    wp.synchronize_stream(stream)
    np.testing.assert_array_equal(output.numpy(), np.full(128, 53))


def test_launches_record_no_events(monkeypatch):
    pipeline = wo.Pipeline(write)
    device = pipeline.device
    output = wp.empty(64, dtype=wp.uint32, device=device)
    wo.launch(pipeline, (64,), [1, output])  # Warm-up.
    recorded = []
    original = wp.Stream.record_event

    def record_event(self, *args, **kwargs):
        recorded.append(self)
        return original(self, *args, **kwargs)

    recorded_launch = wo.launch(pipeline, (64,), [0, output], record_cmd=True)
    monkeypatch.setattr(wp.Stream, "record_event", record_event)
    for value in range(8):
        recorded_launch.set_param_at_index(0, value)
        recorded_launch.launch()
        wo.launch(pipeline, (64,), [value, output])
    assert recorded == []
    np.testing.assert_array_equal(output.numpy(), np.full(64, 7))


def test_work_on_another_stream_does_not_wait_for_the_current_stream(gate):
    pipeline = wo.Pipeline(write)
    device = pipeline.device
    side = wp.Stream(device)
    output = wp.empty(1, dtype=wp.uint32, device=device)
    points = wp.zeros(3, dtype=wp.vec3, device=device)
    accel = wo.Accel(
        [wo.triangles(points, wp.array([0, 1, 2], dtype=wp.int32, device=device))], flags=wo.BuildFlags.ALLOW_UPDATE
    )
    recorded = wo.launch(pipeline, (1,), [1, output], record_cmd=True)
    # Warm-up: loading a CUDA module synchronizes the device.
    wo.launch(pipeline, (1,), [1, output], stream=side)
    wp.synchronize_device(device)

    # As with wp.launch(stream=...), OptiX work on a stream is not ordered after the current stream.
    held = gate.block(wp.get_stream(device))
    wo.launch(pipeline, (1,), [2, output], stream=side)
    recorded.set_param_at_index(0, 3)
    recorded.launch(stream=side)
    accel.refit(stream=side)
    wp.synchronize_stream(side)
    assert not held.is_complete
    gate.open()


@wp.kernel(module="unique")
def write_reserved_names(p: wp.uint32, Params: wp.array[wp.uint32]):
    i = wp.tid()
    Params[i] = p


def test_raygen_arguments_may_use_any_name_and_are_checked():
    # Names used by the old generated writer kernel (p, Params) are ordinary argument names.
    pipeline = wo.Pipeline(write_reserved_names)
    output = wp.empty(4, dtype=wp.uint32, device="cuda")
    wo.launch(pipeline, (4,), [9, output])
    np.testing.assert_array_equal(output.numpy(), np.full(4, 9))
    with pytest.raises(TypeError, match="takes 2 arguments, but 1 were given"):
        wo.launch(pipeline, (4,), [output])
    with pytest.raises(TypeError, match="expects dtype uint32"):
        wo.launch(pipeline, (4,), [1, wp.empty(4, dtype=float, device="cuda")])
    with pytest.raises(TypeError):
        wo.launch(pipeline, (4,), [1.5, output])


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


def test_recorded_launch_replays_with_updated_parameters():
    pipeline = wo.Pipeline(write)
    device = pipeline.device
    output = wp.zeros(8, dtype=wp.uint32, device=device)
    launch = wo.launch(pipeline, (4,), [1, output], record_cmd=True)
    assert isinstance(launch, wo.Launch)
    assert not output.numpy().any(), "recording does not launch"
    launch.launch()
    launch.set_param_at_index(0, 2)
    launch.set_dim(8)
    launch.launch(stream=wp.Stream(device))
    wp.synchronize_device(device)
    np.testing.assert_array_equal(output.numpy(), np.full(8, 2))
    other = wp.empty(8, dtype=wp.uint32, device=device)
    launch.set_param_by_name("output", other)
    launch.launch()
    launch.set_params([3, output])
    launch.launch()
    np.testing.assert_array_equal(other.numpy(), np.full(8, 2))
    np.testing.assert_array_equal(output.numpy(), np.full(8, 3))
    with pytest.raises(KeyError, match="typo"):
        launch.set_param_by_name("typo", 1)
    with pytest.raises(ValueError, match="dim"):
        launch.set_dim(0)
    with pytest.raises(ValueError, match="same CUDA device"):
        launch.set_param_by_name("output", wp.empty(8, dtype=wp.uint32, device="cpu"))

    # With params_type, the names are the fields of that struct.
    explicit = wo.launch(wo.Pipeline(write, params_type=Params), (8,), [params(4, output)], record_cmd=True)
    explicit.set_param_by_name("value", 5)
    explicit.launch()
    np.testing.assert_array_equal(output.numpy(), np.full(8, 5))


@wp.kernel(module="unique")
def write_index_2d(output: wp.array2d[int]):
    i, j = wp.tid()
    output[i, j] = i * 100 + j


@wp.kernel(module="unique")
def no_arguments():
    pass


def test_launch_matches_wp_launch_conventions():
    pipeline = wo.Pipeline(write_index_2d)
    expected = wp.empty((2, 3), dtype=int, device="cuda")
    wp.launch(write_index_2d, (2, 3), inputs=[expected])
    # outputs are appended to inputs, and numpy integers are accepted in dim.
    output = wp.empty((2, 3), dtype=int, device="cuda")
    wo.launch(pipeline, (np.int64(2), np.int32(3)), outputs=[output])
    np.testing.assert_array_equal(output.numpy(), expected.numpy())

    # wp.tid() indices must match the launch dimensions, where OptiX and wp.launch could disagree.
    with pytest.raises(ValueError, match="unpacks 2 wp.tid"):
        wo.launch(pipeline, 6, [output])
    with pytest.raises(ValueError, match="positive integers"):
        wo.launch(pipeline, (2, 3.0), [output])
    recorded = wo.launch(pipeline, (np.int64(2), np.int32(3)), outputs=[output], record_cmd=True)
    recorded.launch()
    with pytest.raises(ValueError, match="unpacks 2 wp.tid"):
        recorded.set_dim(6)

    # A raygen that ignores wp.tid() accepts any dimensions, and one without arguments needs no inputs.
    wo.launch(wo.Pipeline(no_arguments), (2, 2))
