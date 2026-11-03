# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit parameter schemas, buffer ordering, and stage declarations."""

import gc
import weakref

import numpy as np
import pytest
import warp as wp

import warp_optix as wo

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


def test_submitted_launch_retains_explicit_storage_and_nested_inputs():
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
    output_ref = weakref.ref(output)
    launch_params.assign(params(53, output), stream=stream)
    wo.launch(pipeline, (128,), params=launch_params, stream=stream)
    assert not busy.is_complete, "busy kernel finished too early for this test"
    del launch_params, output
    gc.collect()
    # Pending operations retain storage even if the caller drops the parameter object.
    assert storage_ref() is not None and output_ref() is not None
    wp.synchronize_stream(stream)
    np.testing.assert_array_equal(output_ref().numpy(), np.full(128, 53))
    # Once the work has completed, nothing holds them.
    wo.get_context(device).collect()
    gc.collect()
    assert storage_ref() is None and output_ref() is None
