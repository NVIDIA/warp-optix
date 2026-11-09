# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pipeline compile options and the Warp modules pipelines compile."""

import gc

import numpy as np
import pytest
import warp as wp

import warp_optix as wo
from warp_optix import native

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA unavailable")]


@wp.kernel(module="unique")
def sine(output: wp.array[float]):
    i = wp.tid()
    output[i] = wp.sin(float(i) + 0.5)


@wp.kernel(module="unique")
def miss(output: wp.array[float]):
    output[0] = wp.sin(output[1])


@wp.kernel(module="unique", module_options={"fast_math": True})
def fast_miss(output: wp.array[float]):
    output[0] = wp.sin(output[1])


@wp.kernel(module="unique")
def call_first(output: wp.array[wp.uint32]):
    wp.optix_direct_call(wp.uint32(0))


@wp.kernel(module="unique")
def first(output: wp.array[wp.uint32]):
    wp.optix_direct_call(wp.uint32(1))
    output[0] = output[1] + wp.uint32(1)


@wp.kernel(module="unique")
def second(output: wp.array[wp.uint32]):
    output[1] = wp.uint32(41)


@pytest.fixture
def compiled(monkeypatch):
    """Record the compile options and PTX of every OptiX module created during the test."""
    records = []
    original = native.DeviceContext.moduleCreate

    def create(self, options, pipeline_options, ptx):
        records.append((options, ptx.decode() if isinstance(ptx, bytes) else ptx))
        return original(self, options, pipeline_options, ptx)

    monkeypatch.setattr(native.DeviceContext, "moduleCreate", create)
    return records


def test_module_options_apply_to_every_program_and_must_agree(compiled):
    wo.Pipeline(sine, miss=[miss])
    assert "sin.approx" not in compiled[-1][1]
    pipeline = wo.Pipeline(sine, miss=[miss], module_options={"fast_math": True})
    assert "sin.approx" in compiled[-1][1]
    output = wp.zeros(4, dtype=float, device="cuda")
    wo.launch(pipeline, 4, [output])
    np.testing.assert_allclose(output.numpy(), np.sin(np.arange(4) + 0.5), rtol=1e-5)
    # The same programs without the override compile without it again.
    wo.Pipeline(sine, miss=[miss])
    assert "sin.approx" not in compiled[-1][1]

    # Kernels whose modules disagree are rejected before compiling, unless the pipeline decides.
    count = len(compiled)
    with pytest.raises(ValueError, match="'fast_math' is False for kernel 'sine' but True for kernel 'fast_miss'"):
        wo.Pipeline(sine, miss=[fast_miss])
    assert len(compiled) == count
    wo.Pipeline(sine, miss=[fast_miss], module_options={"fast_math": True})
    assert "sin.approx" in compiled[-1][1]
    with pytest.raises(ValueError, match="Unknown Warp module options: fast_maths"):
        wo.Pipeline(sine, module_options={"fast_maths": True})
    with pytest.raises(ValueError, match="enable_backward"):
        wo.Pipeline(sine, module_options={"enable_backward": True})

    # Warp's debug mode compiles debuggable OptiX programs: unoptimized, with full debug information.
    assert int(compiled[-1][0].debugLevel) == int(native.COMPILE_DEBUG_LEVEL_DEFAULT)
    pipeline = wo.Pipeline(sine, module_options={"mode": "debug"})
    assert int(compiled[-1][0].optLevel) == int(native.COMPILE_OPTIMIZATION_LEVEL_0)
    assert int(compiled[-1][0].debugLevel) == int(native.COMPILE_DEBUG_LEVEL_FULL)
    wo.launch(pipeline, 4, [output])
    np.testing.assert_allclose(output.numpy(), np.sin(np.arange(4) + 0.5), rtol=1e-6)


def test_nested_callables_get_stack_for_their_depth(monkeypatch):
    original = native.util.computeStackSizes
    depths = []

    def compute(stack_sizes, max_trace_depth, max_cc_depth, max_dc_depth):
        depths.append((max_cc_depth, max_dc_depth))
        return original(stack_sizes, max_trace_depth, max_cc_depth, max_dc_depth)

    monkeypatch.setattr(native.util, "computeStackSizes", compute)
    wo.Pipeline(call_first, callables=[first, second])
    pipeline = wo.Pipeline(call_first, callables=[first, second], max_callable_depth=2)
    assert depths == [(0, 1), (0, 2)]

    output = wp.zeros(2, dtype=wp.uint32, device="cuda")
    wo.launch(pipeline, 1, [output])
    assert tuple(output.numpy()) == (42, 41)


def test_closed_pipelines_leave_no_warp_modules_or_structs():
    modules = wp._src.context.user_modules

    def make(value):
        @wp.kernel(module="unique")
        def fill(output: wp.array[float]):
            i = wp.tid()
            output[i] = value

        return fill

    kernels = [make(float(i)) for i in range(2)]
    output = wp.zeros(1, dtype=float, device="cuda")
    count = len(modules)
    for kernel in kernels:
        # Equal pipelines compile the same code; closing one leaves the other usable.
        first, second = wo.Pipeline(kernel), wo.Pipeline(kernel)
        assert first.params_type is second.params_type
        first.close()
        wo.launch(second, 1, [output])
        second.close()
        assert not kernel.module.dependents
    wp.synchronize()
    gc.collect()
    assert len(modules) == count
    assert output.numpy()[0] == 1.0

    # A failure after registering the raygen program must detach its module as well.
    with pytest.raises(TypeError, match="type differs from launch parameters"):
        wo.Pipeline(sine, miss=[call_first])
    assert not sine.module.dependents
    assert len(modules) == count
