# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resource lifetime, native-object release, failed construction, and device/stream boundaries."""

import gc
import subprocess
import sys
import weakref

import numpy as np
import pytest
import warp as wp

import warp_optix as wo
from warp_optix import native

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA unavailable")]

# Iterations of the single-thread busy kernel used to keep a stream occupied (about one second on an RTX 3080).
SPIN_ITERATIONS = 4_000_000


@wp.kernel(module="unique")
def spin(iterations: int, sink: wp.array[float]):
    x = float(0.0)
    for _ in range(iterations):
        x = wp.sin(x + 1.0)
    sink[0] = x


@wp.kernel(module="unique")
def write_value(output: wp.array[float]):
    i = wp.tid()
    output[i] = 42.0


@wp.struct
class Params:
    output: wp.array[float]


@wp.kernel(module="unique")
def write_struct(params: Params):
    i = wp.tid()
    params.output[i] = 42.0


@wp.kernel(module="unique")
def write_instance(handle: wp.uint64, records: wp.array[wo.Instance]):
    records[0] = wo.make_instance(wp.identity(3, float), wp.vec3(0.0), handle)


@wp.kernel(module="unique")
def depth_raygen(scene: wp.uint64, depth: wp.array[float]):
    query = wo.accel_query_ray(scene, wp.vec3(0.25, 0.25, 5.0), wp.vec3(0.0, 0.0, -1.0), 100.0)
    depth[0] = -1.0
    if query.result:
        depth[0] = query.t


def geometry():
    vertices = wp.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=wp.vec3, device="cuda")
    indices = wp.array([0, 1, 2], dtype=wp.int32, device="cuda")
    return vertices, indices


def settle():
    """Wait for all GPU work, then run every native release whose work has completed."""
    wp.synchronize()
    wo.get_context("cuda").collect()
    gc.collect()


def occupy(stream):
    """Queue a long-running kernel on ``stream`` and return an event marking its end.

    Callers warm up every kernel they launch next: loading a CUDA module synchronizes the device.
    """
    sink = wp.empty(1, dtype=float, device=stream.device)
    wp.launch(spin, 1, inputs=[SPIN_ITERATIONS, sink], stream=stream)
    return stream.record_event()


@pytest.fixture
def destroyed(monkeypatch):
    """Record every native Module/ProgramGroup/Pipeline destroyed during the test."""
    records = []
    for cls in (type_ for type_ in (native.Module, native.ProgramGroup, native.Pipeline)):
        original = cls.destroy

        def destroy(self, original=original):
            records.append(self)
            return original(self)

        monkeypatch.setattr(cls, "destroy", destroy)
    return records


def test_use_after_close_raises_and_with_closes_on_exception():
    output = wp.zeros(1, dtype=float, device="cuda")
    with pytest.raises(ValueError, match="application error"):
        with wo.Pipeline(write_value) as pipeline:
            wo.launch(pipeline, 1, [output])
            raise ValueError("application error")
    assert pipeline.closed
    np.testing.assert_array_equal(output.numpy(), [42.0])
    pipeline.close()  # Idempotent.
    with pytest.raises(RuntimeError, match="closed"):
        wo.launch(pipeline, 1, [output])
    with pytest.raises(RuntimeError, match="closed"):
        wo.LaunchParams(pipeline)
    with pytest.raises(RuntimeError, match="closed"):
        with pipeline:
            pass

    live = wo.Pipeline(write_value)
    params = wo.LaunchParams(live)
    live.close()
    with pytest.raises(RuntimeError, match="closed"):
        params.assign([output])

    vertices, indices = geometry()
    with wo.Accel([wo.triangles(vertices, indices)], flags=wo.BuildFlags.ALLOW_UPDATE) as accel:
        assert not accel.closed
    assert accel.closed
    accel.close()  # Idempotent.
    for use in (lambda: accel.id, accel.refit, accel.rebuild):
        with pytest.raises(RuntimeError, match="closed"):
            use()
    assert "closed" in repr(accel) and "closed" in repr(pipeline)


def test_pipeline_native_objects_outlive_submitted_launches(destroyed):
    settle()
    context = wo.get_context("cuda")
    stream = wp.Stream("cuda")
    pipeline = wo.Pipeline(write_value)
    owned = list(pipeline._owned)
    output = wp.zeros(1, dtype=float, device="cuda")
    wo.launch(pipeline, 1, [wp.zeros(1, dtype=float, device="cuda")], stream=stream)
    occupy(stream)
    settle()
    busy = occupy(stream)
    wo.launch(pipeline, 1, [output], stream=stream)
    assert not busy.is_complete, "busy kernel finished too early for this test"

    # close() returns at once; the native objects survive the pending launch.
    pipeline.close()
    assert pipeline.closed
    assert destroyed == []
    assert len(context._releases) == 1
    with pytest.raises(RuntimeError, match="closed"):
        wo.launch(pipeline, 1, [wp.zeros(1, dtype=float, device="cuda")])

    settle()
    assert context._releases == []
    assert {id(obj) for obj in destroyed} == {id(obj) for obj in owned}
    np.testing.assert_array_equal(output.numpy(), [42.0])


def test_dropping_a_launched_pipeline_releases_it_without_garbage_collection(destroyed):
    output = wp.zeros(1, dtype=float, device="cuda")
    pipeline = wo.Pipeline(write_value)
    owned = list(pipeline._owned)
    pipeline_ref = weakref.ref(pipeline)
    records_ref = weakref.ref(pipeline._records)
    wo.launch(pipeline, 1, [output])
    wp.synchronize()
    gc.disable()
    try:
        # Automatic parameter storage does not point back at the pipeline, so refcounting frees it.
        del pipeline
        assert pipeline_ref() is None and records_ref() is None
        assert {id(obj) for obj in destroyed} == {id(obj) for obj in owned}
    finally:
        gc.enable()
    np.testing.assert_array_equal(output.numpy(), [42.0])


def test_accel_close_releases_memory_at_once_in_stream_order():
    settle()
    stream = wp.Stream("cuda")
    vertices, indices = geometry()
    vertex_ref, index_ref = weakref.ref(vertices), weakref.ref(indices)
    gas = wo.Accel([wo.triangles(vertices, indices)], stream=stream)
    buffer_ref = weakref.ref(gas._buffer)
    del vertices, indices
    gc.collect()
    # The structure retains its build inputs while open.
    assert vertex_ref() is not None and index_ref() is not None

    depth = wp.zeros(1, dtype=float, device="cuda")
    pipeline = wo.Pipeline(depth_raygen)
    wo.launch(pipeline, 1, [wp.uint64(0), depth], stream=stream)
    occupy(stream)
    settle()
    busy = occupy(stream)
    wo.launch(pipeline, 1, [gas.id, depth], stream=stream)
    assert not busy.is_complete, "busy kernel finished too early for this test"
    # Like wp.Mesh: memory is released right away; Warp's stream-ordered free keeps
    # it valid for the launch already submitted.
    gas.close()
    assert gas.closed and gas._buffer is None
    assert buffer_ref() is None and vertex_ref() is None and index_ref() is None

    wp.synchronize_stream(stream)
    np.testing.assert_allclose(depth.numpy(), [5.0])


def test_native_release_runs_under_the_resource_device(destroyed):
    devices = wp.get_cuda_devices()
    if len(devices) < 2:
        pytest.skip("requires two CUDA devices")
    pipeline = wo.Pipeline(write_value, device=devices[1])
    output = wp.zeros(1, dtype=float, device=devices[1])
    wo.launch(pipeline, 1, [output])
    wp.synchronize()
    seen = []
    original = native.Pipeline.destroy

    def destroy(self):
        seen.append(wp.get_device())
        return original(self)

    native.Pipeline.destroy = destroy
    try:
        with wp.ScopedDevice(devices[0]):
            pipeline.close()
    finally:
        native.Pipeline.destroy = original
    assert seen == [devices[1]]


def test_native_release_waits_for_graph_capture_to_end(destroyed):
    pipeline = wo.Pipeline(write_value)
    owned = list(pipeline._owned)
    output = wp.zeros(1, dtype=float, device="cuda")
    wo.launch(pipeline, 1, [output])
    wp.synchronize()
    context = wo.get_context("cuda")
    with wp.ScopedCapture(device="cuda"):
        pipeline.close()
        assert destroyed == []
        assert len(context._releases) == 1
    context.collect()
    assert context._releases == []
    assert {id(obj) for obj in destroyed} == {id(obj) for obj in owned}


def test_instance_accel_keeps_closed_child_alive_and_traces():
    settle()
    vertices, indices = geometry()
    gas = wo.Accel([wo.triangles(vertices, indices)])
    child_buffer = weakref.ref(gas._buffer)
    records = wp.empty(1, dtype=wo.Instance, device="cuda")
    wp.launch(write_instance, 1, inputs=[gas.id, records], device="cuda")
    ias = wo.Accel([wo.instances(records)])
    assert [child for child, _, _ in ias._dependencies] == [gas]

    gas.close()
    del vertices, indices
    settle()
    assert child_buffer() is not None
    depth = wp.zeros(1, dtype=float, device="cuda")
    wo.launch(wo.Pipeline(depth_raygen), 1, [ias, depth])
    np.testing.assert_allclose(depth.numpy(), [5.0])

    # The closed child can no longer be instanced.
    with pytest.raises(ValueError, match="live acceleration structures"):
        wo.Accel([wo.instances(records)])

    ias.close()
    settle()
    assert child_buffer() is None


def test_nested_instances_keep_every_level_alive():
    settle()
    vertices, indices = geometry()
    gas = wo.Accel([wo.triangles(vertices, indices)])
    gas_buffer = weakref.ref(gas._buffer)
    inner_records = wp.empty(1, dtype=wo.Instance, device="cuda")
    wp.launch(write_instance, 1, inputs=[gas.id, inner_records], device="cuda")
    inner = wo.Accel([wo.instances(inner_records)])
    outer_records = wp.empty(1, dtype=wo.Instance, device="cuda")
    wp.launch(write_instance, 1, inputs=[inner.id, outer_records], device="cuda")
    outer = wo.Accel([wo.instances(outer_records)])

    gas.close()
    inner.close()
    del vertices, indices
    settle()
    assert gas_buffer() is not None
    depth = wp.zeros(1, dtype=float, device="cuda")
    wo.launch(wo.Pipeline(depth_raygen, max_traversable_depth=3), 1, [outer, depth])
    np.testing.assert_allclose(depth.numpy(), [5.0])

    outer.close()
    settle()
    assert gas_buffer() is None


def test_release_callbacks_run_once_when_collect_reenters():
    settle()
    context = wo.get_context("cuda")
    calls = []

    def inner():
        calls.append("inner")

    def outer():
        calls.append("outer")
        # Re-enters collect(), like a pipeline finalized by the garbage collector during collect().
        context.release(inner)

    context.release(outer)
    settle()
    assert calls == ["outer", "inner"]
    assert context._releases == []


def test_pipeline_close_keeps_launch_buffers_until_launches_finish():
    settle()
    stream = wp.Stream("cuda")
    pipeline = wo.Pipeline(write_value)
    output = wp.zeros(1, dtype=float, device="cuda")
    wo.launch(pipeline, 1, [output], stream=stream)
    occupy(stream)
    settle()
    records = weakref.ref(pipeline._records)
    params = weakref.ref(pipeline._buffers[stream.cuda_stream].data)
    busy = occupy(stream)
    wo.launch(pipeline, 1, [output], stream=stream)
    assert not busy.is_complete, "busy kernel finished too early for this test"

    pipeline.close()
    gc.collect()
    assert records() is not None and params() is not None
    settle()
    assert records() is None and params() is None
    np.testing.assert_array_equal(output.numpy(), [42.0])


def test_pipeline_forgets_streams_whose_launches_finished():
    pipeline = wo.Pipeline(write_value)
    output = wp.zeros(1, dtype=float, device="cuda")
    for _ in range(8):
        wo.launch(pipeline, 1, [output], stream=wp.Stream("cuda"))
        wp.synchronize()
    assert len(pipeline._streams) == len(pipeline._buffers) == 1

    busy_stream = wp.Stream("cuda")
    wo.launch(pipeline, 1, [output], stream=busy_stream)
    occupy(busy_stream)
    wp.synchronize_stream(busy_stream)
    busy = occupy(busy_stream)
    wo.launch(pipeline, 1, [output], stream=wp.Stream("cuda"))
    assert not busy.is_complete, "busy kernel finished too early for this test"
    assert busy_stream.cuda_stream in pipeline._streams
    settle()


def test_instance_build_rejects_unknown_children():
    records = wp.empty(1, dtype=wo.Instance, device="cuda")
    wp.launch(write_instance, 1, inputs=[wp.uint64(0x123456789A0), records], device="cuda")
    with pytest.raises(ValueError, match="Instance children must be live acceleration structures on this device"):
        wo.Accel([wo.instances(records)])


def test_accel_objects_in_launch_inputs_are_converted_and_retained():
    settle()
    stream = wp.Stream("cuda")
    vertices, indices = geometry()
    gas = wo.Accel([wo.triangles(vertices, indices)])
    gas_ref = weakref.ref(gas)
    pipeline = wo.Pipeline(depth_raygen)
    depth = wp.zeros(1, dtype=float, device="cuda")
    wo.launch(pipeline, 1, [wp.uint64(0), depth], stream=stream)
    occupy(stream)
    settle()
    busy = occupy(stream)
    wo.launch(pipeline, 1, [gas, depth], stream=stream)
    assert not busy.is_complete, "busy kernel finished too early for this test"
    del gas
    gc.collect()
    assert gas_ref() is not None
    settle()
    np.testing.assert_allclose(depth.numpy(), [5.0])

    # Once the parameters are replaced and the work has finished, nothing holds the structure.
    wo.launch(pipeline, 1, [wp.uint64(0), depth], stream=stream)
    settle()
    assert gas_ref() is None


def test_wrong_devices_and_streams_are_rejected_before_native_work():
    with pytest.raises(ValueError, match="OptiX requires a CUDA device"):
        wo.get_context("cpu")
    with pytest.raises(ValueError, match="OptiX requires a CUDA device"):
        wo.Pipeline(write_value, device="cpu")
    with pytest.raises(ValueError, match="OptiX requires a CUDA device"):
        wo.Accel([], device="cpu")
    pipeline = wo.Pipeline(write_value)
    assert pipeline.device == wp.get_device("cuda")
    cpu = wp.zeros(1, dtype=float, device="cpu")
    with pytest.raises(ValueError, match="Array and OptiX resource must use the same CUDA device"):
        wo.launch(pipeline, 1, [cpu])
    params = Params()
    params.output = cpu
    struct_pipeline = wo.Pipeline(write_struct)
    with pytest.raises(ValueError, match="Array and OptiX resource must use the same CUDA device"):
        wo.launch(struct_pipeline, 1, [params])
    output = wp.zeros(1, dtype=float, device="cuda")
    with pytest.raises(TypeError, match="stream must be a Warp Stream"):
        wo.launch(pipeline, 1, [output], stream=123)
    with pytest.raises(ValueError, match="Device and pipeline must be the same CUDA device"):
        wo.launch(pipeline, 1, [output], device="cpu")
    with pytest.raises(ValueError, match="Stream and device must be the same CUDA device"):
        wo.Accel([], device="cpu", stream=wp.Stream("cuda"))
    # Matching device and stream are accepted.
    wo.launch(pipeline, 1, [output], device="cuda", stream=wp.get_stream("cuda"))
    np.testing.assert_array_equal(output.numpy(), [42.0])


def test_stream_from_another_device_is_rejected():
    devices = wp.get_cuda_devices()
    if len(devices) < 2:
        pytest.skip("requires two CUDA devices")
    vertices, indices = geometry()
    with pytest.raises(ValueError, match="Stream and device must be the same CUDA device"):
        wo.Accel([wo.triangles(vertices, indices)], device=devices[0], stream=wp.Stream(devices[1]))
    other = wp.zeros(1, dtype=float, device=devices[1])
    with pytest.raises(ValueError, match="Array and OptiX resource must use the same CUDA device"):
        wo.launch(wo.Pipeline(write_value, device=devices[0]), 1, [other])


def test_failed_pipeline_construction_destroys_only_its_own_objects(monkeypatch, destroyed):
    good = wo.Pipeline(write_value)
    created = []
    for name in ("moduleCreate", "programGroupCreate", "pipelineCreate"):
        original = getattr(native.DeviceContext, name)

        def record(self, *args, original=original):
            result = original(self, *args)
            objects = result[0] if isinstance(result, tuple) else result
            created.extend(objects if isinstance(objects, list) else [objects])
            return result

        monkeypatch.setattr(native.DeviceContext, name, record)

    def fail(*args):
        raise RuntimeError("injected SBT failure")

    original_pack = native.sbtRecordPackHeader
    monkeypatch.setattr(native, "sbtRecordPackHeader", fail)
    with pytest.raises(RuntimeError, match="injected SBT failure"):
        wo.Pipeline(write_value)
    # Module, raygen + default miss + default hit group, and the pipeline.
    assert len(created) == 5
    assert {id(obj) for obj in destroyed} == {id(obj) for obj in created}
    assert not any(obj is own for obj in destroyed for own in good._owned)

    monkeypatch.setattr(native, "sbtRecordPackHeader", original_pack)
    replacement = wo.Pipeline(write_value)
    output = wp.zeros(2, dtype=float, device="cuda")
    wo.launch(good, 2, [output])
    wo.launch(replacement, 2, [output])
    np.testing.assert_array_equal(output.numpy(), [42.0, 42.0])


def test_exception_debug_flags_reach_native_pipeline(monkeypatch):
    flags = wo.ExceptionFlags.STACK_OVERFLOW | wo.ExceptionFlags.TRACE_DEPTH
    original = native.DeviceContext.pipelineCreate
    seen = []

    def capture(self, options, *args):
        seen.append(options.exceptionFlags)
        return original(self, options, *args)

    monkeypatch.setattr(native.DeviceContext, "pipelineCreate", capture)
    pipeline = wo.Pipeline(write_value, exception_flags=flags)
    assert seen == [int(flags)]
    output = wp.zeros(1, dtype=float, device="cuda")
    wo.launch(pipeline, 1, [output])
    np.testing.assert_array_equal(output.numpy(), [42.0])


def test_config_is_read_when_the_context_is_first_created():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import warp as wp
import warp_optix as wo
from warp_optix import context as context_module

assert context_module._contexts == {}
wo.config.validation = True
wo.config.log_level = 3
ctx = wo.get_context("cuda")
assert ctx.validation is True and ctx.log_level == 3
assert ctx.device == wp.get_device("cuda") and ctx.native is not None

# Later changes do not affect the existing context.
wo.config.validation = False
wo.config.log_level = 1
again = wo.get_context("cuda")
assert again is ctx and again.validation is True and again.log_level == 3
""",
        ],
        check=True,
    )
