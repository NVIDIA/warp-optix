# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resource lifetime, native-object release, failed construction, and device/stream boundaries."""

import gc
import weakref

import numpy as np
import pytest
import warp as wp

import warp_optix as wo
from warp_optix import context as context_module
from warp_optix import native

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA unavailable")]


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


def instance_records(child):
    records = wp.empty(1, dtype=wo.Instance, device="cuda")
    wp.launch(write_instance, 1, inputs=[child.id, records], device="cuda")
    return records


def settle():
    """Wait for all GPU work, then run every native release whose work has completed."""
    wp.synchronize()
    wo.get_context("cuda").collect()
    gc.collect()


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
    output = wp.empty(1, dtype=float, device="cuda")
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


def test_pipeline_close_defers_native_objects_and_buffers_until_launches_finish(destroyed, gate):
    settle()
    context = wo.get_context("cuda")
    stream = wp.Stream("cuda")
    pipeline = wo.Pipeline(write_value)
    owned = list(pipeline._owned)
    # Warm-up: loading a CUDA module synchronizes the device.
    wo.launch(pipeline, 1, [wp.empty(1, dtype=float, device="cuda")], stream=stream)
    settle()
    records = weakref.ref(pipeline._records)
    params = weakref.ref(pipeline._buffers[stream.cuda_stream].data)
    output = wp.empty(1, dtype=float, device="cuda")
    gate.block(stream)
    wo.launch(pipeline, 1, [output], stream=stream)

    # close() returns at once; native objects and launch buffers survive the pending launch.
    pipeline.close()
    gc.collect()
    assert pipeline.closed and destroyed == [] and len(context._releases) == 1
    assert records() is not None and params() is not None
    with pytest.raises(RuntimeError, match="closed"):
        wo.launch(pipeline, 1, [output])

    gate.open()
    settle()
    assert context._releases == []
    assert {id(obj) for obj in destroyed} == {id(obj) for obj in owned}
    assert records() is None and params() is None
    np.testing.assert_array_equal(output.numpy(), [42.0])


def test_dropping_a_launched_pipeline_releases_it_without_garbage_collection(destroyed):
    output = wp.empty(1, dtype=float, device="cuda")
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


def test_accels_are_released_in_stream_order(gate):
    settle()
    stream = wp.Stream("cuda")
    closed_inputs, dropped_inputs = geometry(), geometry()
    input_refs = [weakref.ref(a) for a in (*closed_inputs, *dropped_inputs)]
    closed = wo.Accel([wo.triangles(*closed_inputs)], stream=stream)
    closed_buffer = weakref.ref(closed._buffer)
    dropped = wo.Accel([wo.triangles(*dropped_inputs)], stream=stream)
    dropped_ref = weakref.ref(dropped)
    del closed_inputs, dropped_inputs
    gc.collect()
    # Structures retain their build inputs while open.
    assert all(ref() is not None for ref in input_refs)

    pipeline = wo.Pipeline(depth_raygen)
    by_id = wp.empty(1, dtype=float, device="cuda")
    by_object = wp.empty(1, dtype=float, device="cuda")
    wo.launch(pipeline, 1, [wp.uint64(0), by_id], stream=stream)  # Warm-up.
    settle()
    gate.block(stream)
    wo.launch(pipeline, 1, [closed.id, by_id], stream=stream)
    wo.launch(pipeline, 1, [dropped, by_object], stream=stream)
    # Like wp.Mesh, close() frees memory at once; Warp's stream-ordered free keeps
    # it valid for the launch already submitted.
    closed.close()
    assert closed.closed and closed._buffer is None
    assert closed_buffer() is None and all(ref() is None for ref in input_refs[:2])
    # An Accel passed in launch inputs is retained by the launch parameters.
    del dropped
    gc.collect()
    assert dropped_ref() is not None

    gate.open()
    settle()
    np.testing.assert_allclose(by_id.numpy(), [5.0])
    np.testing.assert_allclose(by_object.numpy(), [5.0])
    # Once the parameters are replaced and the work has finished, nothing holds the structure.
    wo.launch(pipeline, 1, [wp.uint64(0), by_object], stream=stream)
    settle()
    assert dropped_ref() is None and all(ref() is None for ref in input_refs)


def test_native_release_runs_under_the_resource_device(destroyed):
    devices = wp.get_cuda_devices()
    if len(devices) < 2:
        pytest.skip("requires two CUDA devices")
    pipeline = wo.Pipeline(write_value, device=devices[1])
    output = wp.empty(1, dtype=float, device=devices[1])
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
    output = wp.empty(1, dtype=float, device="cuda")
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


def test_instances_keep_closed_children_alive_at_every_level():
    settle()
    vertices, indices = geometry()
    gas = wo.Accel([wo.triangles(vertices, indices)])
    gas_buffer = weakref.ref(gas._buffer)
    inner_records = instance_records(gas)
    inner = wo.Accel([wo.instances(inner_records)])
    assert [child for child, _, _ in inner._dependencies] == [gas]
    outer = wo.Accel([wo.instances(instance_records(inner))])

    gas.close()
    inner.close()
    del vertices, indices
    settle()
    assert gas_buffer() is not None
    depth = wp.empty(1, dtype=float, device="cuda")
    wo.launch(wo.Pipeline(depth_raygen, max_traversable_depth=3), 1, [outer, depth])
    np.testing.assert_allclose(depth.numpy(), [5.0])

    # A closed child can no longer be instanced.
    with pytest.raises(ValueError, match="live acceleration structures"):
        wo.Accel([wo.instances(inner_records)])

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


def test_pipeline_forgets_streams_whose_launches_finished(gate):
    pipeline = wo.Pipeline(write_value)
    output = wp.empty(1, dtype=float, device="cuda")
    for _ in range(8):
        wo.launch(pipeline, 1, [output], stream=wp.Stream("cuda"))
        wp.synchronize()
    assert len(pipeline._streams) == len(pipeline._buffers) == 1

    busy_stream = wp.Stream("cuda")
    gate.block(busy_stream)
    wo.launch(pipeline, 1, [output], stream=busy_stream)
    wo.launch(pipeline, 1, [output], stream=wp.Stream("cuda"))
    assert busy_stream.cuda_stream in pipeline._streams
    gate.open()
    settle()


def test_instance_build_rejects_unknown_children():
    records = wp.empty(1, dtype=wo.Instance, device="cuda")
    wp.launch(write_instance, 1, inputs=[wp.uint64(0x123456789A0), records], device="cuda")
    with pytest.raises(ValueError, match="Instance children must be live acceleration structures on this device"):
        wo.Accel([wo.instances(records)])


def test_wrong_devices_and_streams_are_rejected_before_native_work():
    with pytest.raises(ValueError, match="OptiX requires a CUDA device"):
        wo.get_context("cpu")
    with pytest.raises(ValueError, match="OptiX requires a CUDA device"):
        wo.Pipeline(write_value, device="cpu")
    with pytest.raises(ValueError, match="OptiX requires a CUDA device"):
        wo.Accel([], device="cpu")
    pipeline = wo.Pipeline(write_value)
    assert pipeline.device == wp.get_device("cuda")
    cpu = wp.empty(1, dtype=float, device="cpu")
    with pytest.raises(ValueError, match="Array and OptiX resource must use the same CUDA device"):
        wo.launch(pipeline, 1, [cpu])
    params = Params()
    params.output = cpu
    struct_pipeline = wo.Pipeline(write_struct)
    with pytest.raises(ValueError, match="Array and OptiX resource must use the same CUDA device"):
        wo.launch(struct_pipeline, 1, [params])
    output = wp.empty(1, dtype=float, device="cuda")
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
    other = wp.empty(1, dtype=float, device=devices[1])
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
    output = wp.empty(2, dtype=float, device="cuda")
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
    output = wp.empty(1, dtype=float, device="cuda")
    wo.launch(pipeline, 1, [output])
    np.testing.assert_array_equal(output.numpy(), [42.0])


def test_config_is_read_when_the_context_is_first_created(monkeypatch):
    monkeypatch.setattr(context_module, "_contexts", {})
    monkeypatch.setattr(wo.config, "validation", True)
    monkeypatch.setattr(wo.config, "log_level", 3)
    ctx = wo.get_context("cuda")
    try:
        assert ctx.validation is True and ctx.log_level == 3
        assert ctx.device == wp.get_device("cuda") and ctx.native is not None
        # Later changes do not affect the existing context.
        monkeypatch.setattr(wo.config, "validation", False)
        monkeypatch.setattr(wo.config, "log_level", 1)
        again = wo.get_context("cuda")
        assert again is ctx and again.validation is True and again.log_level == 3
    finally:
        ctx._shutdown()
