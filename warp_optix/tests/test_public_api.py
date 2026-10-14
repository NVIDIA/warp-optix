# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public ownership, namespace, and stream contracts."""

import gc
import subprocess
import sys
import weakref
from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp
import warp_optix as wo
from warp_optix import native
from warp_optix._api import _TrackedContext


@wp.struct
class Params:
    output: wp.array(dtype=wp.uint32)


@wo.optix_kernel(wo.OptixKernelType.RAYGEN)
def raygen(params: Params):
    params.output[wp.optix_get_launch_index()[0]] = wp.uint32(42)


@wo.optix_kernel(wo.OptixKernelType.MISS)
def miss(params: Params):
    pass


@pytest.fixture
def context():
    wp.init()
    if not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    with wo.Context("cuda:0", log_level=1) as result:
        yield result


@pytest.fixture
def ptx(context):
    return wo.compile_warp_module_to_ptx(
        wp.get_module(__name__), "", "public_api", __file__, context.device
    )


def test_import_boundary_in_fresh_process(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.util, sys
import warp_optix as wo
import warp as wp
import optix
assert wo.Context.__module__ == 'warp_optix._api'
assert 'warp_nn' not in sys.modules
assert 'pyglet' not in sys.modules
assert 'imgui_bundle' not in sys.modules
assert wp._src.context.runtime is None
assert not hasattr(wo, 'require_optix')
assert not hasattr(wo, 'create_context')
assert wo.__version__ == '0.2.0'
from warp_optix import native
assert native.version() == (9, 1, 0)
assert native.DeviceContext is optix.DeviceContext
assert native.Module is optix.Module
assert 'warp_optix._bindings' not in sys.modules
""",
        ],
        cwd=tmp_path,
        check=True,
    )


def test_owned_geometry_and_instance_lifetime(context):
    vertices = wp.array(
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
        device=context.device,
    )
    indices = np.array([[0, 1, 2]], dtype=np.uint32)
    geometry = context.create_triangle_gas(
        vertices, indices, build_flags=wo.BuildFlags.ALLOW_UPDATE
    )
    assert geometry.output_size_in_bytes > 0
    assert geometry.handle != 0
    with pytest.raises(TypeError):
        geometry.buffers["other"] = None
    stream = wp.Stream(context.device)
    geometry.refit(stream)
    reference = weakref.ref(geometry)
    instance = wo.Instance(geometry)
    ias = context.create_instance_acceleration_structure([instance])
    del vertices, geometry, instance
    gc.collect()
    assert reference() is not None
    assert ias.handle != 0
    context.close()
    context.close()
    with pytest.raises(RuntimeError, match="closed"):
        _ = ias.handle
    with pytest.raises(RuntimeError, match="closed"):
        context.create_triangle_gas([], [])


def test_context_and_device_validation(context):
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    indices = np.array([[0, 1, 2]], dtype=np.uint32)
    geometry = context.create_triangle_gas(vertices, indices)
    with wo.Context(context.device, log_level=1) as other:
        with pytest.raises(ValueError, match="belong"):
            other.create_instance_acceleration_structure([wo.Instance(geometry)])
    with pytest.raises(ValueError, match="same CUDA device"):
        context.create_triangle_gas(wp.array(vertices, device="cpu"), indices)
    with pytest.raises(TypeError, match="Warp Stream"):
        geometry.refit(stream=0)
    with pytest.raises(ValueError, match="ALLOW_UPDATE"):
        geometry.refit()
    with pytest.raises(ValueError, match="cannot be combined"):
        context.create_triangle_gas(
            vertices, indices, compact=True, build_flags=wo.BuildFlags.ALLOW_UPDATE
        )
    with pytest.raises(ValueError, match="CUDA device"):
        wo.Context("cpu")


def test_launch_nondefault_stream_and_reject_unsafe_capture(context, ptx):
    pipeline = context.create_pipeline(
        ptx, raygen, miss, num_payload_values=0, num_attribute_values=0
    )
    params = Params()
    params.output = wp.zeros(8, dtype=wp.uint32, device=context.device)
    buffer = wo.create_launch_params_buffer(Params, context.device)
    stream = wp.Stream(context.device)
    wo.write_launch_params(buffer, params, stream=stream)
    pipeline.launch(buffer, width=8, stream=stream)
    wp.synchronize_stream(stream)
    np.testing.assert_array_equal(
        params.output.numpy(), np.full(8, 42, dtype=np.uint32)
    )
    params.output.zero_()
    wp.synchronize_device(context.device)
    with wp.ScopedCapture(stream=stream) as capture:
        # Reject before calling OptiX so capture remains usable for Warp work.
        with pytest.raises(RuntimeError, match="capture is unsupported"):
            pipeline.launch(buffer, width=8, stream=stream)
        wo.write_launch_params(buffer, params, stream=stream)
    wp.capture_launch(capture.graph, stream=stream)
    pipeline.launch(buffer, width=8, stream=stream)
    wp.synchronize_stream(stream)
    np.testing.assert_array_equal(
        params.output.numpy(), np.full(8, 42, dtype=np.uint32)
    )
    cpu_buffer = wo.create_launch_params_buffer(Params, "cpu")
    with pytest.raises(ValueError, match="same CUDA device"):
        pipeline.launch(cpu_buffer, width=1)
    with pytest.raises(ValueError, match="positive"):
        pipeline.launch(buffer, width=0)
    context.close()
    with pytest.raises(RuntimeError, match="closed"):
        pipeline.launch(buffer, width=1)


def test_failed_pipeline_releases_allocations(context, ptx):
    before = len(context._native_context.objects)
    with pytest.raises(RuntimeError):
        context.create_pipeline(
            ptx,
            "__raygen__does_not_exist",
            miss,
            num_payload_values=0,
            num_attribute_values=0,
        )
    assert len(context._native_context.objects) == before
    # Construction can be retried on the same context.
    pipeline = context.create_pipeline(
        ptx, raygen, miss, num_payload_values=0, num_attribute_values=0
    )
    assert pipeline.native_handle is not None


def test_tracking_destroys_once_in_reverse_dependency_order():
    destroyed = []

    class Handle:
        def __init__(self, name):
            self.name = name

        def destroy(self):
            destroyed.append(self.name)

    module, group, pipeline = (Handle(n) for n in ("module", "group", "pipeline"))
    handle = SimpleNamespace(
        moduleCreate=lambda: (module, ""),
        builtinISModuleGet=lambda: module,
        programGroupCreate=lambda: ([group], ""),
        pipelineCreate=lambda: pipeline,
    )
    tracker = _TrackedContext(SimpleNamespace(_check_open=lambda: None), handle)
    tracker.moduleCreate()
    tracker.builtinISModuleGet()
    tracker.programGroupCreate()
    tracker.pipelineCreate()
    tracker.release_since(0)
    tracker.release_since(0)
    assert destroyed == ["pipeline", "group", "module"]


def test_neural_upload_uses_owned_context(context):
    from warp_optix.neural_texture import upload_asset
    from neural_texture.test_runtime import _asset

    if not context.native_handle.getProperty(native.DEVICE_PROPERTY_COOP_VEC):
        pytest.skip("cooperative vectors unsupported")
    stream = wp.Stream(context.device)
    result = upload_asset(_asset(), context, stream=stream)
    assert result.matrices.size > 0
    context.close()
    with pytest.raises(RuntimeError, match="closed"):
        upload_asset(_asset(), context)


def test_public_enums_match_bundled_sdk():
    assert int(wo.PrimitiveType.ROUND_LINEAR) == int(native.PRIMITIVE_TYPE_ROUND_LINEAR)
    flags = wo.BuildFlags.PREFER_FAST_TRACE | wo.BuildFlags.ALLOW_UPDATE
    assert int(flags) == int(native.BUILD_FLAG_PREFER_FAST_TRACE) | int(
        native.BUILD_FLAG_ALLOW_UPDATE
    )


@pytest.mark.parametrize("indices", [[[0, 1, 3]], [[0, 1, -1]], [[0.0, 1.0, 2.0]]])
def test_triangle_indices_must_reference_existing_vertices(context, indices):
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    with pytest.raises((ValueError, TypeError), match="indices"):
        context.create_triangle_gas(vertices, np.asarray(indices))


@pytest.mark.parametrize(
    ("curve_type", "segment_indices"),
    [
        (wo.PrimitiveType.ROUND_LINEAR, [3]),
        (wo.PrimitiveType.ROUND_CUBIC_BSPLINE, [1]),
        (wo.PrimitiveType.ROUND_QUADRATIC_BSPLINE, [-1]),
    ],
)
def test_curve_segments_must_fit_in_vertex_buffer(context, curve_type, segment_indices):
    vertices = np.zeros((4, 3), dtype=np.float32)
    widths = np.full(4, 0.1, dtype=np.float32)
    with pytest.raises(ValueError, match="segment_indices"):
        context.create_curve_gas(
            vertices, widths, np.asarray(segment_indices), curve_type=curve_type
        )


@pytest.mark.parametrize("module_tag", ["", "../escape", "a/b", ".hidden", "-opt"])
def test_compile_rejects_module_tags_outside_kernel_cache(module_tag):
    with pytest.raises(ValueError, match="module_tag"):
        wo.compile_warp_module_to_ptx(
            wp.get_module(__name__), "", module_tag, __file__
        )
