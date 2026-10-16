# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import array

import pytest
import warp as wp

pytestmark = pytest.mark.gpu


PAYLOAD_PTX = """
.version 7.0
.target sm_70
.address_size 64
.visible .entry __miss__payload()
{
    .reg .b32 %r, %index;
    mov.u32 %r, 7;
    mov.u32 %index, 0;
    call _optix_set_payload, (%index, %r);
    ret;
}
"""

PARAMS_PTX = """
.version 7.0
.target sm_70
.address_size 64
.visible .const .align 8 .b8 params[8];
.visible .entry __raygen__params()
{
    .reg .b64 %p;
    ld.const.u64 %p, [params];
    st.global.u32 [%p], 7;
    ret;
}
"""


def _context():
    from warp_optix import native as optix

    wp.init()
    if not wp.is_cuda_available():
        pytest.skip("CUDA device unavailable")
    if tuple(optix.version()) < (7, 4):
        pytest.skip("Payload metadata requires OptiX 7.4")
    context = optix.deviceContextCreate(wp.get_device("cuda").context, optix.DeviceContextOptions())
    return optix, context


def _compile(context, options, pipeline, ptx):
    create = getattr(context, "moduleCreate", None)
    if create is None:
        create = context.moduleCreateFromPTX
    return create(options, pipeline, ptx)[0]


@pytest.mark.parametrize("setter", [False, True])
def test_payload_metadata_reaches_native_compiler(setter):
    optix, context = _context()
    module = None
    try:
        semantics = optix.PAYLOAD_SEMANTICS_TRACE_CALLER_READ | optix.PAYLOAD_SEMANTICS_MS_WRITE
        payload = optix.PayloadType([semantics])
        options = optix.ModuleCompileOptions(payloadTypes=[payload])
        if setter:
            options.payloadTypes = [payload]
        pipeline = optix.PipelineCompileOptions(numPayloadValues=0)
        module = _compile(context, options, pipeline, PAYLOAD_PTX)
        assert module is not None
    finally:
        if module is not None:
            module.destroy()
        context.destroy()


def test_bound_values_reach_native_compiler():
    optix, context = _context()
    module = None
    try:
        # Beyond the eight-byte launch parameter block. Silently dropping
        # boundValues would incorrectly allow this module to compile.
        entry = optix.ModuleCompileBoundValueEntry(
            pipelineParamOffsetInBytes=16,
            boundValue=array.array("I", [7]),
            annotation="outside_params",
        )
        options = optix.ModuleCompileOptions(boundValues=[entry])
        pipeline = optix.PipelineCompileOptions(pipelineLaunchParamsVariableName="params")
        module = _compile(context, optix.ModuleCompileOptions(), pipeline, PARAMS_PTX)
        module.destroy()
        module = None
        with pytest.raises(RuntimeError):
            module = _compile(context, options, pipeline, PARAMS_PTX)
    finally:
        if module is not None:
            module.destroy()
        context.destroy()
