# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


import warp as wp  # noqa: E402
import warp_optix as woptix
from warp_optix._runtime import runtime as _runtime
from warp_optix._runtime.hit_kernels import HitKernel as _NativeHitKernel  # noqa: E402

Vec16h = wp.types.vector(length=16, dtype=wp.float16)


@wp.struct
class CoopVecParams:
    output: wp.array(dtype=wp.float32)
    matrix: wp.uint64


@woptix.optix_kernel(woptix.OptixKernelType.RAYGEN)
def coop_vec_raygen(params: CoopVecParams):
    a = wp.vec4h(wp.float16(1.0), wp.float16(2.0), wp.float16(3.0), wp.float16(4.0))
    b = wp.vec4h(wp.float16(2.0), wp.float16(3.0), wp.float16(4.0), wp.float16(5.0))
    summed = wp.vec4h()
    wp.optix_coop_vec_add(a, b, summed)
    for i in range(4):
        params.output[i] = wp.float32(summed[i])


@woptix.optix_kernel(woptix.OptixKernelType.RAYGEN)
def coop_vec_matmul_raygen(params: CoopVecParams):
    input_value = Vec16h(wp.float16(1.0))
    result = Vec16h()
    wp.optix_coop_vec_matmul_fp16(
        input_value, params.matrix, wp.uint32(0), wp.uint32(0), result
    )
    for i in range(16):
        params.output[i] = wp.float32(result[i])


@woptix.optix_kernel(woptix.OptixKernelType.RAYGEN)
def coop_vec_fp8_matmul_raygen(params: CoopVecParams):
    input_value = Vec16h(wp.float16(1.0))
    result = Vec16h()
    wp.optix_coop_vec_matmul_fp8_e4m3(
        input_value, params.matrix, wp.uint32(0), wp.uint32(0), result
    )
    for i in range(16):
        params.output[i] = wp.float32(result[i])


@woptix.optix_kernel(woptix.OptixKernelType.MISS)
def coop_vec_miss(params: CoopVecParams):
    _ = params.output


def _require_coop_vec():
    try:
        optix = _runtime.require_optix()
        wp.init()
    except Exception as error:
        pytest.skip(f"OptiX/Warp unavailable: {error}")
    if not wp.is_cuda_available():
        pytest.skip("CUDA device unavailable")
    device = wp.get_device("cuda:0")
    cuda_context = (
        device.context.value
        if hasattr(device.context, "value")
        else int(device.context)
    )
    context, _ = _runtime.create_optix_context(optix, int(cuda_context), log_level=1)
    try:
        flags = context.getProperty(optix.DEVICE_PROPERTY_COOP_VEC)
    except Exception as error:
        pytest.skip(f"OptiX cooperative vectors unavailable: {error}")
    if not flags:
        pytest.skip("GPU does not support OptiX cooperative vectors")
    return optix, context


def test_coop_vec_elementwise_executes_in_optix(tmp_path, monkeypatch):
    optix, context = _require_coop_vec()
    device_name = "cuda:0"
    monkeypatch.setattr(wp.config, "kernel_cache_dir", str(tmp_path / "warp_cache"))
    with wp.ScopedDevice(device_name):
        ptx = woptix.compile_warp_module_to_ptx(
            wp.get_module(__name__), "", "test_coop_vec", __file__, device=device_name
        )
        pipeline, sbt, resources = _runtime.create_pipeline_and_sbt(
            optix,
            context,
            ptx,
            coop_vec_raygen,
            coop_vec_miss,
            None,
            num_payload_values=0,
            num_attribute_values=0,
            device=device_name,
        )
        output = wp.zeros(4, dtype=wp.float32, device=device_name)
        params = CoopVecParams()
        params.output = output
        params_buffer = woptix.create_launch_params_buffer(CoopVecParams, device_name)
        params.matrix = wp.uint64(0)
        woptix.write_launch_params(params_buffer, params)
        _runtime.launch(optix, pipeline, sbt, 1, 1, params_buffer)
        wp.synchronize_device(device_name)
        np.testing.assert_array_equal(output.numpy(), [3.0, 5.0, 7.0, 9.0])

        # Keep all OptiX/CUDA resources alive until the launch has completed.
        assert resources


def test_coop_vec_fp16_matmul_executes_in_optix(tmp_path, monkeypatch):
    optix, context = _require_coop_vec()
    device_name = "cuda:0"
    monkeypatch.setattr(wp.config, "kernel_cache_dir", str(tmp_path / "warp_cache"))
    with wp.ScopedDevice(device_name):
        n = k = 16

        def description(layout, size):
            desc = optix.CoopVecMatrixDescription()
            desc.N = n
            desc.K = k
            desc.offsetInBytes = 0
            desc.elementType = optix.COOP_VEC_ELEM_TYPE_FLOAT16
            desc.layout = layout
            desc.rowColumnStrideInBytes = 0
            desc.sizeInBytes = size
            return desc

        row_size = context.coopVecMatrixComputeSize(
            n,
            k,
            optix.COOP_VEC_ELEM_TYPE_FLOAT16,
            optix.COOP_VEC_MATRIX_LAYOUT_ROW_MAJOR,
        )
        packed_size = context.coopVecMatrixComputeSize(
            n,
            k,
            optix.COOP_VEC_ELEM_TYPE_FLOAT16,
            optix.COOP_VEC_MATRIX_LAYOUT_INFERENCING_OPTIMAL,
        )
        source = wp.array(np.eye(n, dtype=np.float16).reshape(-1), device=device_name)
        packed = wp.empty(packed_size, dtype=wp.uint8, device=device_name)
        context.coopVecMatrixConvert(
            0,
            [description(optix.COOP_VEC_MATRIX_LAYOUT_ROW_MAJOR, row_size)],
            source.ptr,
            [
                description(
                    optix.COOP_VEC_MATRIX_LAYOUT_INFERENCING_OPTIMAL, packed_size
                )
            ],
            packed.ptr,
        )
        wp.synchronize_device(device_name)

        ptx = woptix.compile_warp_module_to_ptx(
            wp.get_module(__name__),
            "",
            "test_coop_vec_matmul",
            __file__,
            device=device_name,
        )
        pipeline, sbt, resources = _runtime.create_pipeline_and_sbt(
            optix,
            context,
            ptx,
            coop_vec_matmul_raygen,
            coop_vec_miss,
            None,
            num_payload_values=0,
            num_attribute_values=0,
            device=device_name,
        )
        output = wp.zeros(n, dtype=wp.float32, device=device_name)
        params = CoopVecParams()
        params.output = output
        params.matrix = wp.uint64(packed.ptr)
        params_buffer = woptix.create_launch_params_buffer(CoopVecParams, device_name)
        woptix.write_launch_params(params_buffer, params)
        _runtime.launch(optix, pipeline, sbt, 1, 1, params_buffer)
        wp.synchronize_device(device_name)
        np.testing.assert_array_equal(output.numpy(), np.ones(n, dtype=np.float32))
        assert resources and packed


def test_coop_vec_fp8_matmul_executes_in_optix(tmp_path, monkeypatch):
    optix, context = _require_coop_vec()
    device_name = "cuda:0"
    monkeypatch.setattr(wp.config, "kernel_cache_dir", str(tmp_path / "warp_cache"))
    with wp.ScopedDevice(device_name):
        n = k = 16

        def description(element_type, layout, size):
            desc = optix.CoopVecMatrixDescription()
            desc.N = n
            desc.K = k
            desc.offsetInBytes = 0
            desc.elementType = element_type
            desc.layout = layout
            desc.rowColumnStrideInBytes = 0
            desc.sizeInBytes = size
            return desc

        source_size = context.coopVecMatrixComputeSize(
            n,
            k,
            optix.COOP_VEC_ELEM_TYPE_FLOAT16,
            optix.COOP_VEC_MATRIX_LAYOUT_ROW_MAJOR,
        )
        packed_size = context.coopVecMatrixComputeSize(
            n,
            k,
            optix.COOP_VEC_ELEM_TYPE_FLOAT8_E4M3,
            optix.COOP_VEC_MATRIX_LAYOUT_INFERENCING_OPTIMAL,
        )
        source = wp.array(np.eye(n, dtype=np.float16).reshape(-1), device=device_name)
        packed = wp.empty(packed_size, dtype=wp.uint8, device=device_name)
        context.coopVecMatrixConvert(
            0,
            [
                description(
                    optix.COOP_VEC_ELEM_TYPE_FLOAT16,
                    optix.COOP_VEC_MATRIX_LAYOUT_ROW_MAJOR,
                    source_size,
                )
            ],
            source.ptr,
            [
                description(
                    optix.COOP_VEC_ELEM_TYPE_FLOAT8_E4M3,
                    optix.COOP_VEC_MATRIX_LAYOUT_INFERENCING_OPTIMAL,
                    packed_size,
                )
            ],
            packed.ptr,
        )
        wp.synchronize_device(device_name)
        ptx = woptix.compile_warp_module_to_ptx(
            wp.get_module(__name__),
            "",
            "test_coop_vec_fp8_matmul",
            __file__,
            device=device_name,
        )
        pipeline, sbt, resources = _runtime.create_pipeline_and_sbt(
            optix,
            context,
            ptx,
            coop_vec_fp8_matmul_raygen,
            coop_vec_miss,
            None,
            num_payload_values=0,
            num_attribute_values=0,
            device=device_name,
        )
        output = wp.zeros(n, dtype=wp.float32, device=device_name)
        params = CoopVecParams()
        params.output = output
        params.matrix = wp.uint64(packed.ptr)
        params_buffer = woptix.create_launch_params_buffer(CoopVecParams, device_name)
        woptix.write_launch_params(params_buffer, params)
        _runtime.launch(optix, pipeline, sbt, 1, 1, params_buffer)
        wp.synchronize_device(device_name)
        np.testing.assert_array_equal(output.numpy(), np.ones(n, dtype=np.float32))
        assert resources and packed


def test_coop_vec_matrix_convert_checks_allocation_bounds():
    optix, context = _require_coop_vec()
    device_name = "cuda:0"
    n = k = 16

    def description(layout, size, offset=0):
        desc = optix.CoopVecMatrixDescription()
        desc.N = n
        desc.K = k
        desc.offsetInBytes = offset
        desc.elementType = optix.COOP_VEC_ELEM_TYPE_FLOAT16
        desc.layout = layout
        desc.rowColumnStrideInBytes = 0
        desc.sizeInBytes = size
        return desc

    row_layout = optix.COOP_VEC_MATRIX_LAYOUT_ROW_MAJOR
    packed_layout = optix.COOP_VEC_MATRIX_LAYOUT_INFERENCING_OPTIMAL
    row_size = context.coopVecMatrixComputeSize(
        n, k, optix.COOP_VEC_ELEM_TYPE_FLOAT16, row_layout
    )
    packed_size = context.coopVecMatrixComputeSize(
        n, k, optix.COOP_VEC_ELEM_TYPE_FLOAT16, packed_layout
    )
    with wp.ScopedDevice(device_name):
        source = wp.zeros(n * k, dtype=wp.float16, device=device_name)
        packed = wp.empty(packed_size, dtype=wp.uint8, device=device_name)
        host = np.zeros(packed_size + 64, dtype=np.uint8)
        host_ptr = host.ctypes.data + (-host.ctypes.data % 64)
        past_end = (packed_size + 63) // 64 * 64

        cases = [
            # Output layer placed past the end of the destination allocation.
            ([description(row_layout, row_size)], source.ptr,
             [description(packed_layout, packed_size, offset=past_end)], packed.ptr),
            # Input layer larger than the source allocation.
            ([description(row_layout, 4 * row_size)], source.ptr,
             [description(packed_layout, packed_size)], packed.ptr),
            # Host memory instead of a CUDA allocation.
            ([description(row_layout, row_size)], source.ptr,
             [description(packed_layout, packed_size)], host_ptr),
        ]
        for inputs, input_ptr, outputs, output_ptr in cases:
            with pytest.raises(ValueError, match="Networks"):
                context.coopVecMatrixConvert(0, inputs, input_ptr, outputs, output_ptr)
