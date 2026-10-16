# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit parameter schemas, buffer ordering, and stage declarations."""

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


@wo.optix_kernel(wo.OptixKernelType.RAYGEN, module="unique")
def trace(output: wp.array[wp.uint32]):
    i = wp.tid()
    payload = Payload()
    wp.optix_trace(wp.uint64(0), wp.vec3(0.0), wp.vec3(0.0, 0.0, -1.0),
                   0.0, 1.0, 0.0, wp.uint32(255), wp.uint32(0),
                   wp.uint32(0), wp.uint32(1), wp.uint32(0), payload)
    output[i] = payload.value


@wo.optix_kernel(wo.OptixKernelType.MISS)
@wp.kernel(module="unique")
def miss(value: wp.uint32):
    wp.optix_set_payload_0(value)


def params(value, output):
    result = Params()
    result.value = wp.uint32(value)
    result.output = output
    return result


def test_schema_independent_of_raygen_and_automatic_explicit_equivalence():
    with wo.create_context("cuda") as ctx:
        sbt = wo.SbtBuilder(["radiance"])
        sbt.set_miss("radiance", miss)
        pipeline = wo.create_pipeline(ctx, trace, sbt=sbt, params_type=Params, payload_words=1)
        first = wp.empty(64, dtype=wp.uint32, device=ctx.device)
        second = wp.empty_like(first)
        wo.launch(pipeline, (64,), [params(37, first)])
        buffer = wo.create_launch_params_buffer(pipeline)
        wo.write_launch_params(buffer, params(37, second))
        wo.launch(pipeline, (64,), params_buffer=buffer)
        np.testing.assert_array_equal(first.numpy(), np.full(64, 37))
        np.testing.assert_array_equal(first.numpy(), second.numpy())


def test_automatic_storage_is_independent_between_streams():
    with wo.create_context("cuda") as ctx:
        pipeline = wo.create_pipeline(ctx, write)
        streams = [wp.Stream(ctx.device), wp.Stream(ctx.device)]
        outputs = []
        for value in range(20):
            stream = streams[value % 2]
            output = wp.empty(4096, dtype=wp.uint32, device=ctx.device)
            wo.launch(pipeline, (4096,), [value, output], stream=stream)
            outputs.append(output)
        assert len({b.data.ptr for b in pipeline.buffers.values()}) == 2
        wp.synchronize_device(ctx.device)
        for value, output in enumerate(outputs):
            np.testing.assert_array_equal(output.numpy(), np.full(4096, value))


def test_explicit_buffer_update_and_launch_across_streams():
    with wo.create_context("cuda") as ctx:
        pipeline = wo.create_pipeline(ctx, write)
        buffer = wo.create_launch_params_buffer(pipeline)
        first, second = wp.Stream(ctx.device), wp.Stream(ctx.device)
        outputs = []
        for value in range(12):
            output = wp.empty(4096, dtype=wp.uint32, device=ctx.device)
            wo.write_launch_params(buffer, [value, output], stream=first)
            wo.launch(pipeline, (4096,), params_buffer=buffer, stream=second)
            outputs.append(output)
        wp.synchronize_device(ctx.device)
        for value, output in enumerate(outputs):
            np.testing.assert_array_equal(output.numpy(), np.full(4096, value))


def test_invalid_buffers_schemas_and_stage_placements():
    with wo.create_context("cuda") as ctx:
        with pytest.raises(TypeError, match="declares MISS.*RAYGEN"):
            wo.create_pipeline(ctx, miss)
        with pytest.raises(TypeError, match="declares RAYGEN.*MISS"):
            wo.create_pipeline(ctx, write, miss=[trace])
        with pytest.raises(TypeError, match="Warp struct"):
            wo.create_pipeline(ctx, write, params_type=int)
        pipeline = wo.create_pipeline(ctx, write, params_type=Params)
        other = wo.create_pipeline(ctx, write)
        buffer = wo.create_launch_params_buffer(pipeline)
        with pytest.raises(ValueError, match="Write launch parameters"):
            wo.launch(pipeline, (1,), params_buffer=buffer)
        with pytest.raises(ValueError, match="exactly one"):
            wo.launch(pipeline, (1,), [], params_buffer=buffer)
        with pytest.raises(ValueError, match="exactly one"):
            wo.launch(pipeline, (1,))
        with pytest.raises(ValueError, match="belong to this pipeline"):
            wo.launch(other, (1,), params_buffer=buffer)
        with pytest.raises(TypeError, match="instance"):
            wo.write_launch_params(buffer, [])
        with pytest.raises(ValueError, match="same CUDA device"):
            wo.write_launch_params(buffer, params(1, wp.empty(1, dtype=wp.uint32, device="cpu")))
    with pytest.raises(RuntimeError, match="closed"):
        wo.write_launch_params(buffer, Params())


def test_submitted_launch_retains_explicit_storage_and_nested_inputs():
    import gc
    import weakref

    with wo.create_context("cuda") as ctx:
        pipeline = wo.create_pipeline(ctx, write, params_type=Params)
        output = wp.empty(128, dtype=wp.uint32, device=ctx.device)
        buffer = wo.create_launch_params_buffer(pipeline)
        storage_ref = weakref.ref(buffer.data)
        output_ref = weakref.ref(output)
        wo.write_launch_params(buffer, params(53, output))
        wo.launch(pipeline, (128,), params_buffer=buffer)
        del buffer, output
        gc.collect()
        # Pending operations retain storage even if the caller drops the buffer.
        assert storage_ref() is not None and output_ref() is not None
        np.testing.assert_array_equal(output_ref().numpy(), np.full(128, 53))
    gc.collect()
    assert storage_ref() is None and output_ref() is None
