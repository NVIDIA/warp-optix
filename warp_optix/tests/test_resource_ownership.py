# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resource lifetime, failed construction, and device/context boundaries."""

import gc
import weakref

import numpy as np
import pytest
import warp as wp

import warp_optix as wo
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


@wp.kernel
def write_instance(handle: wp.uint64, records: wp.array[wo.Instance]):
    records[0] = wo.make_instance(
        wp.identity(3, float), wp.vec3(0.0), handle, wp.uint32(0), wp.uint32(0), wp.uint32(255), wp.uint32(0)
    )


def geometry():
    vertices = wp.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=wp.vec3, device="cuda")
    indices = wp.array([0, 1, 2], dtype=wp.int32, device="cuda")
    return vertices, indices


def test_context_manager_closes_on_exception_and_rejects_use_after_close():
    with pytest.raises(ValueError, match="application error"):
        with wo.create_context("cuda") as ctx:
            pipeline = wo.create_pipeline(ctx, write_value)
            output = wp.zeros(1, dtype=float, device="cuda")
            wo.launch(pipeline, (1,), [output])
            raise ValueError("application error")
    assert ctx.closed
    np.testing.assert_array_equal(output.numpy(), [42.0])
    wo.destroy_context(ctx)  # Idempotent.
    with pytest.raises(RuntimeError, match="closed"):
        wo.launch(pipeline, (1,), [output])
    with pytest.raises(RuntimeError, match="closed"):
        wo.create_pipeline(ctx, write_value)
    with pytest.raises(RuntimeError, match="closed"):
        wo.build_accel(ctx, [])


def test_context_keeps_geometry_and_source_arrays_alive_until_close():
    ctx = wo.create_context("cuda")
    vertices, indices = geometry()
    vertex_ref, index_ref = weakref.ref(vertices), weakref.ref(indices)
    build_input = wo.triangles(vertices, indices)
    gas = wo.build_accel(ctx, [build_input])
    gas_ref = weakref.ref(gas)
    del vertices, indices, build_input, gas
    gc.collect()
    assert gas_ref() is not None and vertex_ref() is not None and index_ref() is not None
    ctx.close()
    gc.collect()
    assert gas_ref() is None and vertex_ref() is None and index_ref() is None


def test_accel_and_instance_geometry_cannot_cross_contexts():
    with wo.create_context("cuda") as first, wo.create_context("cuda") as second:
        vertices, indices = geometry()
        gas = wo.build_accel(first, [wo.triangles(vertices, indices)], flags=wo.BuildFlags.ALLOW_UPDATE)
        with pytest.raises(ValueError, match="belong to this context"):
            wo.update_accel(second, gas)
        records = wp.empty(1, dtype=wo.Instance, device="cuda")
        wp.launch(write_instance, 1, inputs=[gas.handle, records], device="cuda")
        with pytest.raises(ValueError, match="belong to this context"):
            wo.build_accel(second, [wo.instances(records)])
        ias = wo.build_accel(first, [wo.instances(records)])
        assert ias.dependencies == (gas,)


def test_wrong_devices_rejected_before_native_work():
    with pytest.raises(ValueError, match="CUDA device"):
        wo.create_context("cpu")
    with wo.create_context("cuda") as ctx:
        with pytest.raises(ValueError, match="same CUDA device"):
            wo.create_pipeline(ctx, write_value, device="cpu")
        with pytest.raises(ValueError, match="same CUDA device"):
            wo.build_accel(ctx, [], device="cpu")
        pipeline = wo.create_pipeline(ctx, write_value)
        cpu = wp.zeros(1, dtype=float, device="cpu")
        with pytest.raises(ValueError, match="same CUDA device"):
            wo.launch(pipeline, (1,), [cpu])
        params = Params()
        params.output = cpu
        struct_pipeline = wo.create_pipeline(ctx, write_struct)
        with pytest.raises(ValueError, match="same CUDA device"):
            wo.launch(struct_pipeline, (1,), [params])
        with pytest.raises(TypeError, match="Warp Stream"):
            wo.launch(pipeline, (1,), [wp.zeros(1, device="cuda")], stream=123)


def test_stream_from_another_device_is_rejected():
    devices = wp.get_cuda_devices()
    if len(devices) < 2:
        pytest.skip("requires two CUDA devices")
    with wo.create_context(devices[0]) as ctx:
        with pytest.raises(ValueError, match="same CUDA device"):
            wo.build_accel(ctx, [], stream=wp.Stream(devices[1]))


def test_failed_pipeline_construction_releases_only_its_own_objects(monkeypatch):
    with wo.create_context("cuda") as ctx:
        good = wo.create_pipeline(ctx, write_value)
        before = tuple(ctx._objects)
        original = native.sbtRecordPackHeader

        def fail(*args):
            raise RuntimeError("injected SBT failure")

        monkeypatch.setattr(native, "sbtRecordPackHeader", fail)
        with pytest.raises(RuntimeError, match="injected SBT failure"):
            wo.create_pipeline(ctx, write_value)
        assert tuple(ctx._objects) == before
        monkeypatch.setattr(native, "sbtRecordPackHeader", original)
        replacement = wo.create_pipeline(ctx, write_value)
        output = wp.zeros(2, dtype=float, device="cuda")
        wo.launch(good, (2,), [output])
        wo.launch(replacement, (2,), [output])
        np.testing.assert_array_equal(output.numpy(), [42.0, 42.0])


def test_exception_debug_flags_reach_native_pipeline(monkeypatch):
    with wo.create_context("cuda", log_level=1) as ctx:
        flags = wo.ExceptionFlags.STACK_OVERFLOW | wo.ExceptionFlags.TRACE_DEPTH
        original = ctx.pipelineCreate
        seen = []

        def capture(options, *args):
            seen.append(options.exceptionFlags)
            return original(options, *args)

        monkeypatch.setattr(ctx, "pipelineCreate", capture)
        pipeline = wo.create_pipeline(ctx, write_value, exception_flags=flags)
        assert seen == [int(flags)]
        output = wp.zeros(1, dtype=float, device="cuda")
        wo.launch(pipeline, (1,), [output])
        np.testing.assert_array_equal(output.numpy(), [42.0])
